"""M1: direct recall of a closing price on a date, no price context (Lopez-Lira et al. 2025)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data import DATE_MAX, DATE_MIN, Data
from ..items import Item, ItemScore
from ..textutil import answer_number, classify_non_answer, extract_json, is_year, number_values, parse_answer_number
from .common import SYSTEM_MEMORY

NAME = "price_recall"
PROMPT_VERSION = "p1"
P0 = 0.0
VARIANTS = ["main"]
MAX_TOKENS = 60
TOLERANCE = 0.01
FABRICATED_SHARE = 0.1
FILTERS = ("primary share listings (instrumenttype 4, >=500 price days) split into market-cap terciles; "
           "one random trading day per item with the month uniform over the window; 10% fabricated companies")
FAKE_COMPANIES = [
    ("Norrvik Industri B", "NRVK B"), ("Solbacka Fastigheter", "SOLB"), ("Hjortdal Medical", "HJRT"),
    ("Kvarnholm Energi", "KVRN"), ("Östmark Logistik B", "OSTM B"), ("Lindeby Bank", "LNDB"),
    ("Almvik Gaming", "ALMV"), ("Brattfors Mining", "BRTF"), ("Sjöviks Bygg B", "SJOV B"),
    ("Fjällnäs Biotech", "FJLN"), ("Granlunda Tech", "GRNL"), ("Mossberga Foods", "MOSB"),
    ("Tallåsen Skog B", "TALL B"), ("Vretby Telecom", "VRTB"), ("Ekeröd Pharma", "EKRD"),
]


def build_items(data: Data, n: int, seed: int) -> list[Item]:
    rng = np.random.default_rng(seed)
    u = data.universe
    elig = u[u["is_primary"] & (u["instrumenttype"] == 4) & u["median_mcap_2015_2024"].notna()
             & (u["price_days"] >= 500)].copy()
    elig["size"] = pd.qcut(elig["median_mcap_2015_2024"], 3, labels=["small", "mid", "large"]).astype(str)
    px = data.prices[data.prices["insref"].isin(elig["insref"])]
    by_ins = {int(k): g.sort_values("date").reset_index(drop=True) for k, g in px.groupby("insref")}
    months = pd.period_range(DATE_MIN[:7], DATE_MAX[:7], freq="M")
    n_fake = round(n * FABRICATED_SHARE)
    sizes = ["large", "mid", "small"]
    items: list[Item] = []
    attempts = 0
    while len(items) < n - n_fake:
        attempts += 1
        if attempts > 50 * n:
            raise RuntimeError("could not sample enough price items")
        size = sizes[len(items) % 3]
        pool = elig[elig["size"] == size]
        row = pool.iloc[int(rng.integers(len(pool)))]
        g = by_ins.get(int(row["insref"]))
        month = months[int(rng.integers(len(months)))]
        if g is None:
            continue
        in_month = g[(g["date"].dt.year == month.year) & (g["date"].dt.month == month.month)]
        if in_month.empty:
            continue
        day = in_month.iloc[int(rng.integers(len(in_month)))]
        earlier = g[g["date"] <= day["date"] - pd.Timedelta(days=365)]
        items.append(Item(
            item_id=f"pr-{len(items):04d}", method=NAME, date=day["date"].date().isoformat(),
            cluster=f"c{int(row['company_key'])}",
            payload={"name": str(row["name"]), "symbol": str(row["symbol"]), "currency": str(row["tradecurrency"])},
            truth={"close": float(day["close"]),
                   "adj_close": float(day["adj_close"]) if pd.notna(day["adj_close"]) else None,
                   "baseline_close": float(earlier["close"].iloc[-1]) if len(earlier) else None,
                   "insref": int(row["insref"])},
            strata={"size": size, "currency": str(row["tradecurrency"])},
        ))
    for k in range(n_fake):
        name, symbol = FAKE_COMPANIES[k % len(FAKE_COMPANIES)]
        month = months[int(rng.integers(len(months)))]
        day = pd.Timestamp(year=month.year, month=month.month, day=int(rng.integers(1, 29))) + pd.offsets.BDay(0)
        items.append(Item(
            item_id=f"pr-fake-{k:03d}", method=NAME, date=day.date().isoformat(), cluster=f"fake{k}",
            payload={"name": name, "symbol": symbol, "currency": "SEK"},
            truth={"close": None, "adj_close": None, "baseline_close": None, "insref": None},
            strata={"size": "fabricated", "currency": "SEK"}, fabricated=True,
        ))
    return [items[i] for i in rng.permutation(len(items))]


def render(item: Item, mode: str, variant: str = "main") -> list[dict]:
    p = item.payload
    question = f"What was the closing price of {p['name']} (ticker: {p['symbol']}) on {item.date}, in {p['currency']}?"
    if mode == "forced":
        instr = 'You must give your best numeric estimate, even if you are unsure. Respond only with JSON: {"price": <number>}'
    else:
        instr = 'If you do not know, answer null. Respond only with JSON: {"price": <number or null>}'
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{question}\n{instr}"}]


def parse(text: str) -> dict:
    """value: best single reading; raw: the answer as written (scored under both sv and en conventions)."""
    obj = extract_json(text)
    if obj is not None and "price" in obj:
        raw = obj["price"]
        if raw is None:
            return _no_answer("abstain")
        num = parse_answer_number(raw)
        if num is None:
            return _no_answer(classify_non_answer(str(raw)))
        return {"value": num, "raw": str(raw), "refused": False, "outcome": "answer"}
    m = answer_number(text or "")
    if m and (num := parse_answer_number(m.group())) is not None:
        # a year-like number (e.g. "closed at 2010") is an answer unless the reply abstains or declines
        # ("I can't provide 2023 prices")
        kind = classify_non_answer(text) if is_year(m.group()) else "answer"
        if kind in ("answer", "malformed"):
            return {"value": num, "raw": m.group(), "refused": False, "outcome": "answer"}
        return _no_answer(kind)
    return _no_answer(classify_non_answer(text))


def _no_answer(outcome: str) -> dict:
    return {"value": None, "raw": None, "refused": True, "outcome": outcome}


def _readings(answer: dict) -> set[float]:
    raw = answer.get("raw")
    values = number_values(raw) if raw else set()
    if answer.get("value") is not None:
        values.add(answer["value"])
    return {v for v in values if v > 0}


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    value = parsed["main"].get("value")
    if item.fabricated:
        return ItemScore(0.0, refused=value is None, extra={"hallucinated": value is not None})
    if value is None:
        return ItemScore(0.0, refused=True)
    readings = _readings(parsed["main"])  # "0,125" may mean 0.125 (sv) or 125 (en): either reading may hit
    if not readings:
        return ItemScore(0.0, refused=False)
    truths = [t for t in (item.truth["close"], item.truth["adj_close"]) if t]
    hit = any(abs(v - t) / t <= TOLERANCE for v in readings for t in truths)
    log_error = min(abs(math.log(v / t)) for v in readings for t in truths)
    rel_raw, log_raw = _distance(readings, item.truth["close"])
    rel_adj, log_adj = _distance(readings, item.truth.get("adj_close"))
    # distance to each price convention separately: tolerance bands, the unadjusted-only convention and the Split
    # Clock (which convention an answer matches) are computed from these
    return ItemScore(float(hit), refused=False, extra={"log_error": log_error, "rel_err_raw": rel_raw,
                                                       "rel_err_adj": rel_adj, "log_err_raw": log_raw,
                                                       "log_err_adj": log_adj})


def _distance(readings: set[float], truth: float | None) -> tuple[float | None, float | None]:
    """(relative error, absolute log error) of the closest reading to one truth; None when the truth is missing."""
    if not truth:
        return None, None
    return (min(abs(v - truth) / truth for v in readings), min(abs(math.log(v / truth)) for v in readings))
