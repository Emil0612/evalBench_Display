"""The minor findings of the 2026-10-05 review: labels for registered statistics, the placebo note, Compare's
defaults and opposite-sign guard, and where a hand-entered cutoff comes from."""
from datetime import date

import pandas as pd

from evalbench import ui


def test_registered_cells_name_their_own_statistic():
    assert ui.stat_name({"registered": "model − blind, before"}) == "model − blind baseline (before the cutoff)"
    assert ui.stat_name({"registered": "correlation drop"}) == "correlation drop (before − after)"
    assert ui.stat_name({"registered": float("nan")}) == "Δκ"


def test_forest_axis_says_what_the_value_is():
    assert ui.forest_axis_title(pd.DataFrame({"registered": ["model − blind, before"] * 2})).startswith("model − blind")
    assert ui.forest_axis_title(pd.DataFrame({"dkappa": [0.1]})).startswith("Δκ")


def test_the_placebo_note_names_the_models_it_leaves_out():
    cut = {"openai/gpt-3.5-turbo": date(2021, 9, 30), "google/gemini-2.5-flash": date(2025, 1, 31)}
    note = ui.placebo_note(list(cut), cut)
    assert "GPT-3.5 Turbo" in note and "Gemini" not in note
    assert "every model shown" in ui.placebo_note(["google/gemini-2.5-flash"], cut)


def test_compare_opens_on_a_pair_that_both_answered_the_method():
    answered = {"price_recall": {"a/1", "b/2"}, "price_level": {"a/1", "b/2", "c/3"}}
    assert ui.compare_defaults(answered, ["a/1", "c/3", "b/2"], ["price_recall", "price_level"]) == \
        ("a/1", "c/3", "price_level")


def test_compare_flags_an_answered_only_result_in_the_opposite_direction():
    text, tone = ui.compare_sentence("A", "B", "M5", {"diff": 0.3, "ci_lo": 0.1, "ci_hi": 0.5},
                                     {"diff": -0.2, "ci_lo": -0.35, "ci_hi": -0.05})
    assert "reverses" in text and tone == "warn"


def test_the_source_of_a_hand_entered_cutoff_does_not_depend_on_the_live_listing():
    sources = {"openai/gpt-3.5-turbo": "listing", "qwen/qwen3.8-flash": "estimate"}
    over = {"openai/gpt-3.5-turbo": "2021-09-30", "qwen/qwen3.8-flash": "2026-03-31"}
    assert ui.cutoff_source("openai/gpt-3.5-turbo", over, {}, sources) == "OpenRouter listing"   # listing gone
    assert ui.cutoff_source("qwen/qwen3.8-flash", over, {}, sources) == "set by hand"
    assert ui.load_cutoff_sources()["openai/gpt-3.5-turbo"] == "listing"


def test_compare_side_labels_follow_the_line_and_the_registered_windows():
    assert ui.side_label(1, "release") == "before (in the test window)"
    assert ui.side_label(0, "release") == "after release"
    assert ui.side_label(0, "cutoff") == "after cutoff"
    assert ui.side_label(-1, "release") == "outside the test windows"


def test_mcnemar_is_not_reported_on_zero_shared_before_items():
    import numpy as np
    from evalbench import analysis
    rows = lambda g: pd.DataFrame({"item_id": ["x1", "x2"], "fabricated": False, "score": [1.0, 0.0],
                                   "group": g, "cluster": ["c1", "c2"], "date": "2021-01-01"})
    res, _ = analysis.compare(rows([1, 0]), rows([0, 0]), "price_recall", B=50)
    assert res["n_both_pre"] == 0 and np.isnan(res["mcnemar_p"])
