"""M13p: M13 (the positive-control questions) as a probability probe; see price_choice_prob."""
from __future__ import annotations

from .price_choice import (LETTERS, MAX_TOKENS, P0, VARIANTS, correct_letter, option_values,  # noqa: F401
                           options, parse, render, with_logprobs)
from .price_choice import item_score_prob as item_score  # noqa: F401

NAME = "known_choice_prob"
PROMPT_VERSION = "p1"
LOGPROBS = 5
FILTERS = "the questions of known_choice (M13), same options and orderings"
