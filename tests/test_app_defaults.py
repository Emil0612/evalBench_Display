"""App audit step B (2026-10-05): what is shown by default and how it is labelled."""
from datetime import date

from evalbench import ui
from evalbench.methods import MAIN_METHODS, SECONDARY


def test_a_beats_the_blind_baseline_verdict_counts_as_a_detection_on_the_cards():
    assert ui.is_detection("memorization detected") and ui.is_detection("beats the blind baseline before the cutoff")
    assert not ui.is_detection("no significant difference") and not ui.is_detection("not confirmed: refusal-driven")


def test_m6_is_secondary_and_off_the_default_views():
    assert "annual_return" in SECONDARY and "annual_return" not in MAIN_METHODS


def test_profile_defaults_to_the_release_line():
    assert ui.PROFILE_DEFAULT_BASELINE == "release"


def test_rolling_charts_pool_versions_only_where_the_versions_share_a_design():
    assert ui.rolling_pools_versions("price_level")
    assert not ui.rolling_pools_versions("split_clock") and not ui.rolling_pools_versions("decop")


def test_the_placebo_keeps_only_models_that_had_seen_every_placebo_item():
    cut = {"gpt-3.5": date(2021, 9, 30), "gpt-4o-mini": date(2023, 10, 31), "llama-3.3": date(2023, 12, 31),
           "gemini": date(2025, 1, 31)}
    assert ui.placebo_models(list(cut), cut) == ["llama-3.3", "gemini"]


def test_models_that_fail_the_positive_controls_are_flagged():
    assert "openai/gpt-oss-20b" in ui.CONTROL_FAILS


def test_the_cutoff_source_says_when_it_was_set_by_hand():
    assert ui.cutoff_source("qwen/qwen3.8-flash", {"qwen/qwen3.8-flash": "2026-03-31"}) == "set by hand"
    assert ui.cutoff_source("google/gemini-2.5-flash", {}) == "OpenRouter listing"


def test_the_methods_that_found_nothing_usable_are_secondary():
    # M1 too hard for every model, M2 numbers computable from the text, M8 too few items after the line (2026-10-06)
    for m in ("price_recall", "number_cloze", "split_clock"):
        assert m in SECONDARY and m not in MAIN_METHODS


def test_a_cell_without_a_verdict_shows_no_number():
    from evalbench import ui
    assert ui.heat_label(-1.02, 0.9, "too few items") == "–"
    assert ui.heat_label(0.06, 0.01, "refusal-driven") == "+0.06"
    assert ui.heat_label(0.09, 0.01, "memorization detected").startswith("+0.09")


def test_an_unresolved_cell_shows_no_number_either():
    from evalbench import ui
    assert ui.heat_label(0.3, 0.001, "unresolved") == "?"


def test_the_heatmap_note_explains_starred_grey_cells_only_when_there_are_some():
    import pandas as pd
    from evalbench import ui
    t = pd.DataFrame({"method": ["index_level", "price_level"], "dkappa": [-0.27, 0.09], "p_holm": [0.001, 0.01],
                      "reading": ["none detected", "memorization detected"]})
    note = ui.heat_note(t)
    assert "other direction" in note and "M14" in note
    assert ui.heat_note(t.iloc[[1]]) == ""
    few = pd.DataFrame({"method": ["monthly_return"], "dkappa": [-1.0], "p_holm": [0.9], "reading": ["too few items"]})
    assert "–" in ui.heat_note(few) and "other direction" not in ui.heat_note(few)


def test_a_cutoff_from_the_model_card_is_labelled_so_and_is_not_provisional():
    over = {"google/gemma-4-31b-it": "2025-01-31"}
    sources = {"google/gemma-4-31b-it": "model card"}
    assert ui.cutoff_source("google/gemma-4-31b-it", over, {}, sources) == "model card"
