"""M9 (positive control): monthly price levels of the largest Nordic companies in 2016-2022.

The price-level twin of M7: Novo Nordisk, Equinor, Nordea, Volvo, Ericsson and the other largest listings are the
most widely reported Nordic share prices, and every month asked is before every model's cutoff. If a model cannot
name even these levels more often than the blind guesser (the same month a year earlier), the M5 question format is
too hard and a "none detected" on M5 says little. Same question and scoring as M5. Fact type: price."""
from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

from ..data import Data
from ..items import Item
from .known_return import N_COMPANIES
from .price_level import MAX_TOKENS, P0, VARIANTS, eligible, item_score, month_truth, parse, render  # noqa: F401

NAME = "known_level"
PROMPT_VERSION = "p1"
MONTHS = pd.period_range("2016-01", "2022-12", freq="M")
FILTERS = (f"the {N_COMPANIES} largest primary share listings by median market cap (as M7); one calendar month per "
           "item, 2016-01..2022-12, at least 10 trading days; truth = mean raw and adjusted close and the month's "
           "low-high range; same question and scoring as M5 (price_level); no fabricated companies (positive control)")


def build_items(data: Data, n: int, seed: int) -> list[Item]:
    rng = np.random.default_rng(seed)
    biggest = eligible(data).nlargest(N_COMPANIES, "median_mcap_2015_2024").set_index("insref")
    px = data.prices[data.prices["insref"].isin(biggest.index)].copy()
    px["month"] = px["date"].dt.to_period("M")
    candidates = []
    for (insref, month), m in px[px["month"].isin(MONTHS)].groupby(["insref", "month"]):
        if len(m) < 10:
            continue
        prev = px[(px["insref"] == insref) & (px["month"] == month - 12)]
        row = biggest.loc[insref]
        candidates.append(Item(
            item_id="", method=NAME, date=m["date"].max().date().isoformat(), cluster=f"c{int(row['company_key'])}",
            payload={"name": str(row["name"]), "symbol": str(row["symbol"]), "currency": str(row["tradecurrency"]),
                     "month": str(month), "month_name": f"{calendar.month_name[month.month]} {month.year}"},
            truth={**month_truth(m), "baseline_level": float(prev["close"].mean()) if len(prev) else None,
                   "insref": int(insref)},
            strata={"size": "large", "currency": str(row["tradecurrency"])},
        ))
    picked = [candidates[i] for i in rng.permutation(len(candidates))][:n]
    for k, it in enumerate(picked):
        it.item_id = f"kl-{k:04d}"
    return picked
