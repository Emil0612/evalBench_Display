"""M6: annual return: by how many percent a share's (split-adjusted) price changed over a calendar year.

Research on financial memorization finds it largest for low-frequency returns; knowing a year's outcome is exactly
what would contaminate a backtest of that year. Scored on direction (primary), plus the error in percentage points
and whether the answer puts the company on the right side of that year's median share (which cancels
market-wide knowledge such as "2022 was a bad year"). Fact type: return."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..data import Data
from ..items import Item, ItemScore
from ..textutil import classify_non_answer, extract_json, is_year, parse_answer_number
from . import price_recall
from .common import SYSTEM_MEMORY
from .price_level import eligible

NAME = "annual_return"
PROMPT_VERSION = "p1"
P0 = 0.5          # direction; the measured baseline (the model's own after-cutoff level) is what labels rely on
VARIANTS = ["main"]
MAX_TOKENS = 60
YEARS = range(2021, 2026)
FABRICATED_SHARE = 0.1
FILTERS = ("same eligible share listings as M1; one calendar year 2021-2025 per item with a December close in both "
           "the year and the year before; return = last adjusted close of the year / last adjusted close of the "
           "previous year - 1 (splits and rights issues adjusted, dividends excluded); item date = last trading day "
           "of the year; 10% fabricated companies")
_NEG = re.compile(r"\b(fell|fall|falls|dropped|drop|declined|decline|decreased|decrease|down|lost|lower|negative|"
                  r"minskade|sjönk|föll|tappade|gick ned|nedgång|minus)\b", re.IGNORECASE)
_SIGNED = re.compile(r"(?<![\w.,])([+\-−–]?)\s?(\d+(?:[.,]\d+)?)")


def year_end_returns(data: Data, insrefs) -> pd.DataFrame:
    """One row per (insref, year): last adjusted close, its date, and the return from the previous year's."""
    px = data.prices[data.prices["insref"].isin(insrefs) & data.prices["adj_close"].notna()]
    px = px.sort_values(["insref", "date"])
    ye = px.groupby(["insref", px["date"].dt.year]).agg(adj=("adj_close", "last"), date=("date", "last"))
    ye.index = ye.index.set_names(["insref", "year"])
    ye = ye.reset_index()
    ye["complete"] = ye["date"].dt.month == 12
    prev = ye.assign(year=ye["year"] + 1)[["insref", "year", "adj", "complete"]]
    ye = ye.merge(prev, on=["insref", "year"], how="left", suffixes=("", "_prev"))
    ok = ye["complete"] & ye["complete_prev"].fillna(False).astype(bool)
    ye["return_pct"] = np.where(ok, (ye["adj"] / ye["adj_prev"] - 1) * 100, np.nan)
    return ye


def build_items(data: Data, n: int, seed: int) -> list[Item]:
    rng = np.random.default_rng(seed)
    elig = eligible(data)
    ye = year_end_returns(data, elig["insref"])
    median = ye.groupby("year")["return_pct"].median()
    lookup = {(int(r.insref), int(r.year)): r for r in ye.itertuples()}
    n_fake = round(n * FABRICATED_SHARE)
    sizes = ["large", "mid", "small"]
    years = list(YEARS)
    items: list[Item] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    while len(items) < n - n_fake:
        attempts += 1
        if attempts > 50 * n:
            raise RuntimeError("could not sample enough annual-return items")
        size = sizes[len(items) % 3]
        pool = elig[elig["size"] == size]
        row = pool.iloc[int(rng.integers(len(pool)))]
        year = years[len(items) % len(years)]          # balanced over the years
        r = lookup.get((int(row["insref"]), year))
        if r is None or pd.isna(r.return_pct) or (int(row["insref"]), year) in seen:
            continue
        seen.add((int(row["insref"]), year))
        before = lookup.get((int(row["insref"]), year - 1))
        items.append(Item(
            item_id=f"ar-{len(items):04d}", method=NAME, date=r.date.date().isoformat(),
            cluster=f"c{int(row['company_key'])}",
            payload={"name": str(row["name"]), "symbol": str(row["symbol"]), "year": year},
            truth={"return_pct": float(r.return_pct), "median_return_pct": float(median.get(year, np.nan)),
                   "baseline_return_pct": (float(before.return_pct) if before is not None
                                           and pd.notna(before.return_pct) else None),
                   "insref": int(row["insref"])},
            strata={"size": size, "currency": str(row["tradecurrency"])},
        ))
    for k in range(n_fake):
        name, symbol = price_recall.FAKE_COMPANIES[k % len(price_recall.FAKE_COMPANIES)]
        year = years[int(rng.integers(len(years)))]
        items.append(Item(
            item_id=f"ar-fake-{k:03d}", method=NAME, date=f"{year}-12-30", cluster=f"fake{k}",
            payload={"name": name, "symbol": symbol, "year": year},
            truth={"return_pct": None, "median_return_pct": None, "baseline_return_pct": None, "insref": None},
            strata={"size": "fabricated", "currency": "SEK"}, fabricated=True,
        ))
    return [items[i] for i in rng.permutation(len(items))]


def render(item: Item, mode: str, variant: str = "main") -> list[dict]:
    p, y = item.payload, item.payload["year"]
    question = (f"By how many percent did the share price of {p['name']} (ticker: {p['symbol']}) change during {y}, "
                f"from the last close of {y - 1} to the last close of {y} (adjusted for splits, excluding dividends)? "
                "Give a signed percentage: negative if the price fell.")
    if mode == "forced":
        instr = ('You must give your best estimate, even if you are unsure. '
                 'Respond only with JSON: {"change_percent": <number>}')
    else:
        instr = 'If you do not know, answer null. Respond only with JSON: {"change_percent": <number or null>}'
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{question}\n{instr}"}]


def _signed_value(text: str) -> float | None:
    """The last non-year number in the text, with its sign: an explicit minus, or a word like 'fell'/'sjönk'."""
    t = str(text).replace("−", "-").replace("–", "-")
    found = [m for m in _SIGNED.finditer(t) if not is_year(m.group(2))]
    if not found:
        return None
    sign, digits = found[-1].group(1), found[-1].group(2)
    v = parse_answer_number(digits)
    if v is None:
        return None
    if sign == "-" or (sign != "+" and _NEG.search(t)):
        v = -abs(v)
    return v


def parse(text: str) -> dict:
    obj = extract_json(text)
    if obj is not None and "change_percent" in obj:
        raw = obj["change_percent"]
        if raw is None:
            return _no_answer("abstain")
        v = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else _signed_value(raw)
        if v is None:
            return _no_answer(classify_non_answer(str(raw)))
        return {"value": v, "raw": str(raw), "refused": False, "outcome": "answer"}
    kind = classify_non_answer(text)
    if kind in ("abstain", "refusal"):
        return _no_answer(kind)
    v = _signed_value(text or "")
    return _no_answer(kind) if v is None else {"value": v, "raw": text, "refused": False, "outcome": "answer"}


def _no_answer(outcome: str) -> dict:
    return {"value": None, "raw": None, "refused": True, "outcome": outcome}


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    value = parsed["main"].get("value")
    if item.fabricated:
        return ItemScore(0.0, refused=value is None, extra={"hallucinated": value is not None})
    if value is None:
        return ItemScore(0.0, refused=True)
    t, med = item.truth["return_pct"], item.truth.get("median_return_pct")
    hit = (value > 0) == (t > 0) if t != 0 else abs(value) < 0.5
    # "0" claims no direction: chance, not "down" (change of 2026-10-01); a share that did not move keeps its rule
    no_direction = value == 0 and t != 0
    err = abs(value - t)
    vs_median = None if med is None or pd.isna(med) else bool((value - med > 0) == (t - med > 0))
    return ItemScore(0.5 if no_direction else float(hit), refused=False, extra={
        "abs_err_pp": err, "within_10pp": bool(err <= 10), "vs_median_hit": vs_median, "no_direction": no_direction})
