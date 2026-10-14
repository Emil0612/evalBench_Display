"""Registry of contamination-detection methods."""
from . import (annual_return, completion, decop, headline_date, index_forecast, index_level, index_return, known_choice,
               known_choice_prob, path_date,
               known_level, known_return, monthly_return, number_cloze, price_choice, price_choice_prob, price_level,
               price_recall, split_clock, us_index_level, us_index_return, us_level)

METHODS = {m.NAME: m for m in (price_recall, number_cloze, decop, completion, price_level, annual_return,
                               known_return, split_clock, known_level, us_level, monthly_return, price_choice,
                               known_choice, index_level, index_return, us_index_level, us_index_return,
                               price_choice_prob, known_choice_prob, headline_date, index_forecast, path_date)}
LABELS = {
    "price_recall": "M1 Exact price",
    "number_cloze": "M2 Hidden number",
    "decop": "M3 Spot the original",
    "completion": "M4 Guided completion",
    "price_level": "M5 Price level",
    "annual_return": "M6 Annual return",
    "known_return": "M7 Known returns (positive control)",
    "split_clock": "M8 Split Clock",
    "known_level": "M9 Known price levels (positive control)",
    "us_level": "M10 US price levels (private check)",
    "monthly_return": "M11 Monthly return",
    "price_choice": "M12 Which price fits (multiple choice)",
    "known_choice": "M13 Known price choice (positive control)",
    "index_level": "M14 Index level",
    "index_return": "M15 Index monthly change",
    "us_index_level": "M16 US index level (private check)",
    "us_index_return": "M17 US index monthly change (private check)",
    "price_choice_prob": "M12p Which price fits (probabilities)",
    "known_choice_prob": "M13p Known price choice (probabilities)",
    "headline_date": "M18 Headline date",
    "index_forecast": "M19 Index forecast (fake date)",
    "path_date": "M20 Date an unlabeled index path",
}
# kept, but not run or shown by default. M4: the "real article" hint changed nothing in pilot C (Δκ within ±0.03).
# M3: pilots D and E showed the original can be found without memory (blind rules 47-55%, quote marks); secondary
# until a DE-COP set passes the blind-baseline audit (decop_audit) and the Gemini gate test
# M6: its gaps come from one market-wide year (2022, pilot D), and the placebo flags it; secondary since 2026-10-05.
# M1, M2, M8 (2026-10-06): M1 is too hard for every model (0-6% within 1%), M2's numbers can mostly be worked out from
# the article, and M8 has too few items after the line; the findings list them under "what did not work"
SECONDARY = {"completion", "decop", "annual_return", "price_recall", "number_cloze", "split_clock",
             "index_forecast", "path_date"}   # M19, M20: exploratory leakage tests with their own reports
                                              # (fake_date, mask_date), off the default views
# every item is before every cutoff: a level to beat, not a before/after difference
POSITIVE_CONTROLS = {"known_return", "known_level", "known_choice"}
# built from data that may never be published (US prices from Yahoo Finance); never in the default views
PRIVATE_CHECKS = {"us_level", "us_index_level", "us_index_return"}
# M12/M13 scored by the probability of the right letter (for models that will not produce or pick a price): probes
PROBES = {"price_choice_prob", "known_choice_prob"}
# the main table (pilot D's E1 family): what summary views show by default
MAIN_METHODS = [m for m in METHODS if m not in SECONDARY | POSITIVE_CONTROLS | PRIVATE_CHECKS | PROBES]
