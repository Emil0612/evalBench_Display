"""M16: M14 (index level) for four US indices, the reference for the Nordic result: the same question, scoring and
months. GPT-4o recalls these indices almost exactly inside its training window (Lopez-Lira, Tang & Zhu 2025).
PRIVATE: daily closes from Yahoo Finance (src/evalbench/fetch_us_indices.py) are for personal use only; nothing built
from them may be published."""
from __future__ import annotations

from ..data import DATA_DIR
from . import index_level as _il
from .index_level import CLOSENESS_ZERO, MAX_TOKENS, P0, VARIANTS, item_score, parse, render  # noqa: F401

NAME = "us_index_level"
PROMPT_VERSION = "p1"
PRIVATE = True
NAMES = {"SPX": "S&P 500", "DJIA": "Dow Jones Industrial Average", "COMP": "Nasdaq Composite", "NDX": "Nasdaq-100"}
YAHOO = {"SPX": "^GSPC", "DJIA": "^DJI", "COMP": "^IXIC", "NDX": "^NDX"}
US_INDEX_FILE = DATA_DIR / "private" / "us_index_daily.parquet"
FILTERS = "as M14 (index_level), for S&P 500, Dow Jones, Nasdaq Composite and Nasdaq-100; no pilot block"


def build_items(daily, seed: int) -> list:
    return _il.build_items(daily, seed, method=NAME, id_prefix="ul", names=NAMES, pilot_symbol=None)


def load_index():
    return _il.load_index(US_INDEX_FILE)
