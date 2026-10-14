"""M17: M15 (index monthly change) for the four US indices of M16; the same question, scoring and months. PRIVATE."""
from __future__ import annotations

from . import index_level as _il
from .index_return import MAX_TOKENS, P0, VARIANTS, item_score, parse, render  # noqa: F401
from .us_index_level import NAMES, PRIVATE  # noqa: F401

NAME = "us_index_return"
PROMPT_VERSION = "p1"
FILTERS = "as M15 (index_return), the index-months of M16 (us_index_level), same order"


def build_items(daily, seed: int) -> list:
    return _il.build_items(daily, seed, method=NAME, id_prefix="ur", names=NAMES, pilot_symbol=None)
