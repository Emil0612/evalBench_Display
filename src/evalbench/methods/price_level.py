"""M5: approximate price level: the average closing price of a share in a given month, scored within ±10%.

Pilot C showed that exact daily closes of Nordic shares are too hard (0-6% within 1%) while answers were closer
before the cutoff than after. Rough knowledge of a price level is what could leak into a backtest, so this asks for
it directly. Fact type: price (approximate)."""
from __future__ import annotations

import calendar
import math

import numpy as np
import pandas as pd

from ..data import Data
from ..items import Item, ItemScore
from . import price_recall
from .common import SYSTEM_MEMORY

NAME = "price_level"
PROMPT_VERSION = "p1"
P0 = 0.0
VARIANTS = ["main"]
MAX_TOKENS = 60
TOLERANCE = 0.10
CLOSENESS_ZERO = 2.0          # closeness 0 once an answer is off by this factor (twice or half the truth)
FABRICATED_SHARE = 0.1
MONTHS = pd.period_range("2021-01", "2026-07", freq="M")
FILTERS = ("same eligible share listings as M1 (market-cap terciles); one calendar month per item, uniform over "
           "2021-01..2026-07, at least 10 trading days; truth = mean raw close and mean adjusted close of the month; "
           "item date = last trading day of the month; 10% fabricated companies")


def eligible(data: Data) -> pd.DataFrame:
    u = data.universe
    elig = u[u["is_primary"] & (u["instrumenttype"] == 4) & u["median_mcap_2015_2024"].notna()
             & (u["price_days"] >= 500)].copy()
    elig["size"] = pd.qcut(elig["median_mcap_2015_2024"], 3, labels=["small", "mid", "large"]).astype(str)
    return elig


def month_truth(m: pd.DataFrame) -> dict:
    """The month's average close (as traded and adjusted) and its own low-high range, for the spec's range rule."""
    adj = m["adj_close"].notna().all()
    return {"level": float(m["close"].mean()), "adj_level": float(m["adj_close"].mean()) if adj else None,
            "lo": float(m["close"].min()), "hi": float(m["close"].max()),
            "adj_lo": float(m["adj_close"].min()) if adj else None, "adj_hi": float(m["adj_close"].max()) if adj else None}


def build_items(data: Data, n: int, seed: int, months=MONTHS, exclude: set | None = None,
                fabricated_share: float = FABRICATED_SHARE, id_prefix: str = "pl") -> list[Item]:
    """n items (a share `fabricated_share` of them fake companies), months drawn uniformly from `months`, sizes in
    turn; (insref, "YYYY-MM") pairs in `exclude` are never asked (a fresh version)."""
    months = list(months)
    rng = np.random.default_rng(seed)
    elig = eligible(data)
    px = data.prices[data.prices["insref"].isin(elig["insref"])].copy()
    px["month"] = px["date"].dt.to_period("M")
    by_ins = {int(k): g for k, g in px.groupby("insref")}
    n_fake = round(n * fabricated_share)
    sizes = ["large", "mid", "small"]
    items: list[Item] = []
    seen: set[tuple[int, str]] = set(exclude or ())
    attempts = 0
    while len(items) < n - n_fake:
        attempts += 1
        if attempts > 50 * n:
            raise RuntimeError("could not sample enough price-level items")
        size = sizes[len(items) % 3]
        pool = elig[elig["size"] == size]
        row = pool.iloc[int(rng.integers(len(pool)))]
        month = months[int(rng.integers(len(months)))]
        g = by_ins.get(int(row["insref"]))
        if g is None or (int(row["insref"]), str(month)) in seen:
            continue
        m = g[g["month"] == month]
        if len(m) < 10:
            continue
        seen.add((int(row["insref"]), str(month)))
        prev = g[g["month"] == month - 12]
        items.append(Item(
            item_id=f"{id_prefix}-{len(items):04d}", method=NAME, date=m["date"].max().date().isoformat(),
            cluster=f"c{int(row['company_key'])}",
            payload={"name": str(row["name"]), "symbol": str(row["symbol"]), "currency": str(row["tradecurrency"]),
                     "month": str(month), "month_name": f"{calendar.month_name[month.month]} {month.year}"},
            truth={**month_truth(m), "baseline_level": float(prev["close"].mean()) if len(prev) else None,
                   "insref": int(row["insref"])},
            strata={"size": size, "currency": str(row["tradecurrency"])},
        ))
    for k in range(n_fake):
        name, symbol = price_recall.FAKE_COMPANIES[k % len(price_recall.FAKE_COMPANIES)]
        month = months[int(rng.integers(len(months)))]
        items.append(Item(
            item_id=f"{id_prefix}-fake-{k:03d}", method=NAME, date=(month.to_timestamp(how="end")).date().isoformat(),
            cluster=f"fake{k}",
            payload={"name": name, "symbol": symbol, "currency": "SEK", "month": str(month),
                     "month_name": f"{calendar.month_name[month.month]} {month.year}"},
            truth={"level": None, "adj_level": None, "lo": None, "hi": None, "adj_lo": None, "adj_hi": None,
                   "baseline_level": None, "insref": None},
            strata={"size": "fabricated", "currency": "SEK"}, fabricated=True,
        ))
    return [items[i] for i in rng.permutation(len(items))]


def render(item: Item, mode: str, variant: str = "main") -> list[dict]:
    p = item.payload
    question = (f"What was the average closing price of {p['name']} (ticker: {p['symbol']}) during "
                f"{p['month_name']}, in {p['currency']}? An approximate figure is fine.")
    if mode == "forced":
        instr = 'You must give your best numeric estimate, even if you are unsure. Respond only with JSON: {"price": <number>}'
    else:
        instr = 'If you do not know, answer null. Respond only with JSON: {"price": <number or null>}'
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{question}\n{instr}"}]


parse = price_recall.parse


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    value = parsed["main"].get("value")
    if item.fabricated:
        return ItemScore(0.0, refused=value is None, extra={"hallucinated": value is not None})
    if value is None:
        return ItemScore(0.0, refused=True)
    readings = price_recall._readings(parsed["main"])
    if not readings:
        # a price of 0 (or below): no share trades there, so it is "I don't know" (scoring spec §10)
        return ItemScore(0.0, refused=False, extra={"no_answer": True})
    rel_raw, log_raw = price_recall._distance(readings, item.truth["level"])
    rel_adj, log_adj = price_recall._distance(readings, item.truth.get("adj_level"))
    hit = min(e for e in (rel_raw, rel_adj) if e is not None) <= TOLERANCE + 1e-12
    log_error = min(e for e in (log_raw, log_adj) if e is not None)
    return ItemScore(float(hit), refused=False, extra={
        "log_error": log_error, "rel_err_raw": rel_raw, "rel_err_adj": rel_adj, "log_err_raw": log_raw,
        "log_err_adj": log_adj, "in_range": in_range(readings, item.truth),
        "closeness": max(0.0, 1.0 - log_error / math.log(CLOSENESS_ZERO))})


def in_range(readings: set[float], truth: dict) -> bool | None:
    """Did the model name a price that actually occurred that month (within the low-high range as traded, or as
    adjusted)? None for items built before the range was recorded."""
    ranges = [(truth.get(lo), truth.get(hi)) for lo, hi in (("lo", "hi"), ("adj_lo", "adj_hi"))]
    ranges = [(a, b) for a, b in ranges if a is not None and b is not None]
    if not ranges:
        return None
    return any(a - 1e-9 <= v <= b + 1e-9 for v in readings for a, b in ranges)


def build_monthly_grid(data: Data, per_month: int, seed: int, months=MONTHS, exclude: set | None = None,
                       id_prefix: str = "pl3") -> list[Item]:
    """The same question for exactly `per_month` companies in every month, split evenly over the three size groups
    (per_month // 3 each), for rolling-window views with even precision over time. No fabricated companies; no
    (insref, "YYYY-MM") pair in `exclude`. A month with too few eligible companies of a size gets what exists."""
    rng = np.random.default_rng(seed)
    elig = eligible(data)
    px = data.prices[data.prices["insref"].isin(elig["insref"])].copy()
    px["month"] = px["date"].dt.to_period("M")
    counts = px.groupby(["insref", "month"]).size()
    groups = {k: g for k, g in px.groupby(["insref", "month"])}
    exclude = set(exclude or ())
    items: list[Item] = []
    for month in months:
        for size in ("large", "mid", "small"):
            pool = elig[elig["size"] == size]
            order = rng.permutation(len(pool))
            taken = 0
            for i in order:
                if taken >= per_month // 3:
                    break
                row = pool.iloc[int(i)]
                key = (int(row["insref"]), month)
                if counts.get(key, 0) < 10 or (key[0], str(month)) in exclude:
                    continue
                m = groups[key]
                prev = groups.get((key[0], month - 12))
                items.append(Item(
                    item_id="", method=NAME, date=m["date"].max().date().isoformat(),
                    cluster=f"c{int(row['company_key'])}",
                    payload={"name": str(row["name"]), "symbol": str(row["symbol"]),
                             "currency": str(row["tradecurrency"]), "month": str(month),
                             "month_name": f"{calendar.month_name[month.month]} {month.year}"},
                    truth={**month_truth(m), "baseline_level": float(prev["close"].mean()) if prev is not None
                           and len(prev) else None, "insref": key[0]},
                    strata={"size": size, "currency": str(row["tradecurrency"])},
                ))
                taken += 1
    items = [items[i] for i in rng.permutation(len(items))]
    for k, it in enumerate(items):
        it.item_id = f"{id_prefix}-{k:04d}"
    return items
