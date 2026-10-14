"""M11: monthly return: by how many percent a share's price changed in one calendar month, scored against the market.

The replacement for M6's noisy yearly direction (pilot D: one year, 2022, drove every model's gap). Twelve periods a
year instead of one, and the primary score asks whether the answer puts the share on the right side of that month's
median Nordic share: chance is 50% by construction, and knowing only the market's direction ("that month was a
crash") earns nothing. That market-wide knowledge is reported separately (absolute direction), against its own
measured baseline, because the market's direction alone gets about 60% right (scoring spec, 2026-09-30).
Fact type: return."""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from ..data import Data
from ..items import Item, ItemScore
from . import price_recall
from .annual_return import MAX_TOKENS, VARIANTS, parse  # noqa: F401  (same JSON answer: change_percent)
from .common import SYSTEM_MEMORY
from .price_level import eligible

NAME = "monthly_return"
PROMPT_VERSION = "p1"
P0 = 0.5          # relative direction: the median splits every month in half
FABRICATED_SHARE = 0.1
BAND_PP = 5.0
MONTHS = pd.period_range("2021-01", "2026-07", freq="M")
FILTERS = ("same eligible share listings as M1 (market-cap terciles); one calendar month per item, 2021-01..2026-07, "
           "at least 10 trading days in the month and a close in the month before; return = last adjusted close of "
           "the month / last adjusted close of the previous month - 1 (splits adjusted, dividends excluded); market "
           "= the median return of all eligible listings that month; blind guess = the previous month's return; item "
           "date = last trading day of the month; 10% fabricated companies")


def month_end_returns(data: Data, insrefs) -> pd.DataFrame:
    """One row per (insref, month): last adjusted close, its date, trading days, and the return over the month."""
    px = data.prices[data.prices["insref"].isin(insrefs) & data.prices["adj_close"].notna()].sort_values(["insref", "date"])
    px = px.assign(month=px["date"].dt.to_period("M"))
    me = px.groupby(["insref", "month"]).agg(adj=("adj_close", "last"), date=("date", "last"),
                                             days=("adj_close", "size")).reset_index()
    prev = me.assign(month=me["month"] + 1)[["insref", "month", "adj"]]
    me = me.merge(prev, on=["insref", "month"], how="left", suffixes=("", "_prev"))
    me["return_pct"] = np.where((me["days"] >= 10) & me["adj_prev"].notna(), (me["adj"] / me["adj_prev"] - 1) * 100,
                                np.nan)
    return me


def build_items(data: Data, n: int, seed: int) -> list[Item]:
    rng = np.random.default_rng(seed)
    elig = eligible(data)
    me = month_end_returns(data, elig["insref"])
    market = me.groupby("month")["return_pct"].median()
    lookup = {(int(r.insref), r.month): r for r in me.itertuples()}
    n_fake = round(n * FABRICATED_SHARE)
    sizes, months = ["large", "mid", "small"], list(MONTHS)
    items: list[Item] = []
    seen: set[tuple[int, str]] = set()
    attempts = 0
    while len(items) < n - n_fake:
        attempts += 1
        if attempts > 50 * n:
            raise RuntimeError("could not sample enough monthly-return items")
        size = sizes[len(items) % 3]
        pool = elig[elig["size"] == size]
        row = pool.iloc[int(rng.integers(len(pool)))]
        month = months[int(rng.integers(len(months)))]
        r = lookup.get((int(row["insref"]), month))
        if r is None or pd.isna(r.return_pct) or (int(row["insref"]), str(month)) in seen:
            continue
        seen.add((int(row["insref"]), str(month)))
        before = lookup.get((int(row["insref"]), month - 1))
        items.append(Item(
            item_id=f"mr-{len(items):04d}", method=NAME, date=r.date.date().isoformat(),
            cluster=f"c{int(row['company_key'])}",
            payload={"name": str(row["name"]), "symbol": str(row["symbol"]), "month": str(month),
                     "month_name": f"{calendar.month_name[month.month]} {month.year}"},
            truth={"return_pct": float(r.return_pct), "market_return_pct": float(market.get(month, np.nan)),
                   "baseline_return_pct": (float(before.return_pct) if before is not None
                                           and pd.notna(before.return_pct) else None),
                   "insref": int(row["insref"])},
            strata={"size": size, "currency": str(row["tradecurrency"])},
        ))
    for k in range(n_fake):
        name, symbol = price_recall.FAKE_COMPANIES[k % len(price_recall.FAKE_COMPANIES)]
        month = months[int(rng.integers(len(months)))]
        items.append(Item(
            item_id=f"mr-fake-{k:03d}", method=NAME, date=month.to_timestamp(how="end").date().isoformat(),
            cluster=f"fake{k}",
            payload={"name": name, "symbol": symbol, "month": str(month),
                     "month_name": f"{calendar.month_name[month.month]} {month.year}"},
            truth={"return_pct": None, "market_return_pct": None, "baseline_return_pct": None, "insref": None},
            strata={"size": "fabricated", "currency": "SEK"}, fabricated=True,
        ))
    return [items[i] for i in rng.permutation(len(items))]


def render(item: Item, mode: str, variant: str = "main") -> list[dict]:
    p = item.payload
    month = pd.Period(p["month"], "M")
    before = calendar.month_name[(month - 1).month]
    question = (f"By how many percent did the share price of {p['name']} (ticker: {p['symbol']}) change during "
                f"{p['month_name']}, from the last close of {before} to the last close of "
                f"{calendar.month_name[month.month]} (adjusted for splits, excluding dividends)? "
                "Give a signed percentage: negative if the price fell.")
    if mode == "forced":
        instr = ('You must give your best estimate, even if you are unsure. '
                 'Respond only with JSON: {"change_percent": <number>}')
    else:
        instr = 'If you do not know, answer null. Respond only with JSON: {"change_percent": <number or null>}'
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{question}\n{instr}"}]


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    """Primary: the right side of that month's market median. Extras: plain direction, error in percentage points."""
    value = parsed["main"].get("value")
    if item.fabricated:
        return ItemScore(0.0, refused=value is None, extra={"hallucinated": value is not None})
    if value is None:
        return ItemScore(0.0, refused=True)
    t, mkt = item.truth["return_pct"], item.truth["market_return_pct"]
    relative = (value - mkt > 0) == (t - mkt > 0)
    err = abs(value - t)
    no_direction = value == 0   # "0" is a hedge, not a claim about the market: chance (change of 2026-10-01)
    return ItemScore(0.5 if no_direction else float(relative), refused=False, extra={
        "abs_direction_hit": bool((value > 0) == (t > 0)), "market_up": bool(mkt > 0), "abs_err_pp": err,
        "within_5pp": bool(err <= BAND_PP), "no_direction": no_direction})
