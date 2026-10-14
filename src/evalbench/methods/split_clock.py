"""M8: Split Clock. When a company splits its shares, price sites rewrite every older price (400 kr becomes 40 kr).
Asking a model the average price in a month *before* a big split shows which version it learned: an answer near
the rewritten (adjusted) price means it saw data from after the split. The item date is therefore the split date:
before/after a model's cutoff here means "was the split itself before or after the cutoff", and the usual Δκ
machinery tests whether the model's data reaches past its stated cutoff. Only a rough level is needed (the two
versions differ by at least 3x). Fact type: price."""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd

from ..data import Data
from ..items import Item, ItemScore
from . import price_level, price_recall

NAME = "split_clock"
PROMPT_VERSION = "p1"
P0 = 0.0
VARIANTS = ["main"]
MAX_TOKENS = 60
MIN_FACTOR = 3.0
LEAD_MONTHS = (6, 18)                  # the asked month lies this many months before the split
EVENT_WINDOW = (date(2016, 7, 1), date(2026, 6, 30))
FILTERS = (f"splits or reverse splits of eligible share listings with a price factor of at least {MIN_FACTOR:g}x "
           f"between {EVENT_WINDOW[0]} and {EVENT_WINDOW[1]}; one item per split asking the average price in a month "
           f"{LEAD_MONTHS[0]}-{LEAD_MONTHS[1]} months before it (at least 10 trading days); truth = the month's mean raw "
           "and mean adjusted close; item date = the split date")

render = price_level.render
parse = price_recall.parse


def adjustment_events(prices: pd.DataFrame, min_factor: float = 2.0) -> dict[int, list[tuple[date, float]]]:
    """insref -> [(date, factor)] where the raw/adjusted relation steps by at least min_factor (a split or reverse
    split). factor = old raw / adjusted, e.g. 10 for a 1:10 split."""
    p = prices[prices["price_type"] == 0] if "price_type" in prices else prices
    p = p.dropna(subset=["close", "adj_close"])
    p = p[(p["close"] > 0) & (p["adj_close"] > 0)].sort_values(["insref", "date"])
    ratio = p["adj_close"] / p["close"]
    step = ratio / ratio.groupby(p["insref"]).shift(1)
    hit = np.maximum(step, 1 / step) >= min_factor
    big = p[hit].assign(factor=step[hit])     # step[hit]: an empty frame must not take the full index
    out: dict[int, list[tuple[date, float]]] = {}
    for r in big.itertuples():
        out.setdefault(int(r.insref), []).append((pd.Timestamp(r.date).date(), float(r.factor)))
    return out


def build_items(data: Data, n: int, seed: int, kinds=("ordinary", "reverse")) -> list[Item]:
    """kinds: "ordinary" (one share becomes several, the old price is rewritten down) and/or "reverse"."""
    rng = np.random.default_rng(seed)
    elig = price_level.eligible(data).set_index("insref")
    px = data.prices[data.prices["insref"].isin(elig.index)].copy()
    px["month"] = px["date"].dt.to_period("M")
    by_ins = {int(k): g for k, g in px.groupby("insref")}
    events = adjustment_events(px, min_factor=MIN_FACTOR)
    candidates = [(ins, d, f) for ins, evs in events.items() for d, f in evs if EVENT_WINDOW[0] <= d <= EVENT_WINDOW[1]]
    items: list[Item] = []
    for i in rng.permutation(len(candidates)):
        ins, event_date, _ = candidates[int(i)]
        g, row = by_ins[ins], elig.loc[ins]
        month = pd.Period(event_date, "M") - int(rng.integers(LEAD_MONTHS[0], LEAD_MONTHS[1] + 1))
        m = g[g["month"] == month]
        if len(m) < 10 or m["adj_close"].isna().any():
            continue
        level, adj = float(m["close"].mean()), float(m["adj_close"].mean())
        k = level / adj
        if max(k, 1 / k) < MIN_FACTOR:
            continue                   # later events cancelled out: the two versions are too close to tell apart
        if ("ordinary" if k > 1 else "reverse") not in kinds:
            continue
        prev = g[g["month"] == month - 12]
        items.append(Item(
            item_id=f"sc-{len(items):04d}", method=NAME, date=event_date.isoformat(),
            cluster=f"c{int(row['company_key'])}",
            payload={"name": str(row["name"]), "symbol": str(row["symbol"]), "currency": str(row["tradecurrency"]),
                     "month": str(month), "month_name": f"{month.strftime('%B')} {month.year}"},
            truth={"level": level, "adj_level": adj, "factor": k, "event_date": event_date.isoformat(),
                   "baseline_level": float(prev["close"].mean()) if len(prev) else None, "insref": int(ins)},
            strata={"size": str(row["size"]), "currency": str(row["tradecurrency"])},
        ))
        if len(items) >= n:
            break
    return items


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    """1 when the answer is nearer the rewritten (adjusted) price, within a factor √k of it; kind says which."""
    value = parsed["main"].get("value")
    if value is None:
        return ItemScore(0.0, refused=True)
    readings = price_recall._readings(parsed["main"])
    if not readings:
        return ItemScore(0.0, refused=False, extra={"kind": "neither"})
    _, log_raw = price_recall._distance(readings, item.truth["level"])
    _, log_adj = price_recall._distance(readings, item.truth["adj_level"])
    half = abs(math.log(item.truth["level"] / item.truth["adj_level"])) / 2
    kind = "neither" if min(log_raw, log_adj) > half else ("adjusted" if log_adj < log_raw else "raw")
    return ItemScore(float(kind == "adjusted"), refused=False, extra={"kind": kind, "factor": item.truth["factor"],
                                                                     "log_err_raw": log_raw, "log_err_adj": log_adj})
