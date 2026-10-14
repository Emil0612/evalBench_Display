"""M12p: M12 as a probability probe. The same questions, asked the same four ways, with the provider's top token
log-probabilities requested: the score is the probability the model gives the correct letter, so a model that leans
toward the right price without choosing it still shows it (and letter habits average out). For models that will not
produce or pick prices (gpt-oss-20b). Providers without log-probabilities fall back to the chosen letter."""
from __future__ import annotations

from .price_choice import (LETTERS, MAX_TOKENS, P0, VARIANTS, correct_letter, option_values,  # noqa: F401
                           options, parse, render, with_logprobs)
from .price_choice import item_score_prob as item_score  # noqa: F401

NAME = "price_choice_prob"
PROMPT_VERSION = "p1"
LOGPROBS = 5
FILTERS = "the questions of price_choice (M12), same options and orderings"
