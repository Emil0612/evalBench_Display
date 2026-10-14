"""M7 (positive control): annual returns of the largest Nordic companies in 2016-2022.

These are among the most widely reported Nordic market facts, from well before every model's cutoff. If a model
cannot get even these right more often than chance, the question format is too hard and a "none detected" elsewhere
says little. Same question and scoring as M6; only the companies and years differ. Fact type: return."""
from __future__ import annotations

import numpy as np

from ..data import Data
from ..items import Item
from .annual_return import MAX_TOKENS, P0, VARIANTS, item_score, parse, render, year_end_returns  # noqa: F401
from .price_level import eligible

NAME = "known_return"
PROMPT_VERSION = "p1"
N_COMPANIES = 15
YEARS = range(2016, 2023)
FILTERS = (f"the {N_COMPANIES} largest primary share listings by median market cap; every calendar year 2016-2022 "
           "with a December close in the year and the year before; same question and scoring as M6 (annual_return); "
           "no fabricated companies (positive control)")


def build_items(data: Data, n: int, seed: int) -> list[Item]:
    rng = np.random.default_rng(seed)
    elig = eligible(data)
    biggest = elig.nlargest(N_COMPANIES, "median_mcap_2015_2024")
    median = year_end_returns(data, elig["insref"]).groupby("year")["return_pct"].median()
    ye = year_end_returns(data, biggest["insref"])
    rows = biggest.set_index("insref")
    items: list[Item] = []
    for r in ye[ye["year"].isin(YEARS) & ye["return_pct"].notna()].itertuples():
        row = rows.loc[r.insref]
        before = ye[(ye["insref"] == r.insref) & (ye["year"] == r.year - 1)]["return_pct"]
        items.append(Item(
            item_id="", method=NAME, date=r.date.date().isoformat(), cluster=f"c{int(row['company_key'])}",
            payload={"name": str(row["name"]), "symbol": str(row["symbol"]), "year": int(r.year)},
            truth={"return_pct": float(r.return_pct), "median_return_pct": float(median.get(r.year, np.nan)),
                   "baseline_return_pct": float(before.iloc[0]) if len(before) and before.notna().iloc[0] else None,
                   "insref": int(r.insref)},
            strata={"size": "large", "currency": str(row["tradecurrency"])},
        ))
    picked = [items[i] for i in rng.permutation(len(items))][:n]
    for k, it in enumerate(picked):
        it.item_id = f"kr-{k:04d}"
    return picked
