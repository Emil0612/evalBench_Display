"""M10 (private check): monthly price levels of famous US shares, 2021-01..2026-07.

Apple, Microsoft and the like are among the most discussed share prices in any training corpus, so a model should
name their monthly levels before its cutoff and fail after it. That makes this a check that the M5 question format
and the before/after analysis *can* detect memory where it certainly exists (a positive control).

PRIVATE: the prices come from Yahoo Finance (personal use only). They live in
data/private/ (gitignored), are fetched with `python -m evalbench.fetch_us_prices`, and nothing built from them may
be published. Shares with a split in the window are skipped, so the price as traded equals the split-adjusted one.
Same question and scoring as M5. Fact type: price."""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import DATA_DIR
from ..items import Item
from .price_level import MAX_TOKENS, P0, VARIANTS, item_score, parse, render  # noqa: F401

NAME = "us_level"
PROMPT_VERSION = "p1"
PRIVATE = True
US_PRICES = DATA_DIR / "private" / "us_prices.parquet"
MONTHS = pd.period_range("2021-01", "2026-07", freq="M")
FILTERS = ("famous US shares from data/private/us_prices.parquet (Yahoo Finance, private use only), skipping any "
           "share with a split inside the question window; one calendar month per item, 2021-01..2026-07, at least "
           "10 trading days; truth = mean close and the month's low-high range; blind guess = the same month a year "
           "earlier; same question and scoring as M5 (price_level)")


def build_items(data, n: int, seed: int, path: Path = US_PRICES) -> list[Item]:
    """`data` is unused (the prices are not Millistream's); kept for the builder's common signature."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `uv run --with yfinance python -m evalbench.fetch_us_prices`")
    rng = np.random.default_rng(seed)
    px = pd.read_parquet(path)
    px["date"] = pd.to_datetime(px["date"])
    px = px[px["date"] >= "2020-01-01"]
    # only a split inside the question window matters: after it, the traded price equals the split-adjusted one
    split = px[(px["split"].fillna(0) > 0) & (px["date"] >= MONTHS[0].start_time)].groupby("symbol").size()
    px = px[~px["symbol"].isin(split.index)].copy()
    px["month"] = px["date"].dt.to_period("M")
    candidates = []
    for (symbol, month), m in px[px["month"].isin(MONTHS)].groupby(["symbol", "month"]):
        if len(m) < 10:
            continue
        prev = px[(px["symbol"] == symbol) & (px["month"] == month - 12)]
        candidates.append(Item(
            item_id="", method=NAME, date=m["date"].max().date().isoformat(), cluster=f"us-{symbol}",
            payload={"name": str(m["name"].iloc[0]), "symbol": str(symbol), "currency": "USD", "month": str(month),
                     "month_name": f"{calendar.month_name[month.month]} {month.year}"},
            truth={"level": float(m["close"].mean()), "adj_level": None, "lo": float(m["close"].min()),
                   "hi": float(m["close"].max()), "adj_lo": None, "adj_hi": None,
                   "baseline_level": float(prev["close"].mean()) if len(prev) else None, "insref": None},
            strata={"size": "us-large", "currency": "USD"},
        ))
    picked = [candidates[i] for i in rng.permutation(len(candidates))][:n]
    for k, it in enumerate(picked):
        it.item_id = f"us-{k:04d}"
    return picked
