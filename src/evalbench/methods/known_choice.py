"""M13: M12's multiple choice for the M9 positive-control questions (the 15 largest Nordic companies, 2016-2022,
before every model's cutoff). Validates the format: a model that fails recall on M9 but passes here holds the
knowledge and cannot produce it; one that fails both does not hold it in a usable form."""
from __future__ import annotations

from .price_choice import (LETTERS, MAX_TOKENS, P0, VARIANTS, audit, blind_choices, correct_letter,  # noqa: F401
                           item_score, option_values, options, parse, render)

NAME = "known_choice"
PROMPT_VERSION = "p1"
FILTERS = "the company-months of known_level (M9); options as M12 (price_choice)"
