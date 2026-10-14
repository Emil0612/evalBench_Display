"""The backtest-readiness table: one row per model from the registered results (no new statistics)."""
from datetime import date

import pandas as pd

from evalbench import readiness as rd




def test_the_verdict_names_the_clean_window_and_every_kind_of_hindsight():
    row = {"model": "openai/gpt-3.5-turbo", "release": date(2024, 1, 25), "clean_from": "2024-02", "clean_months": 31,
           "controls": "passes", "m5": "detected", "m15": "not detected", "m18": "not detected",
           "habit": "2021-09", "dating_ends": "2020–2022"}
    text = rd.verdict(row)
    assert "2024-02" in text and "31 months" in text and "price levels" in text and "2020–2022" in text


def test_a_model_that_fails_the_controls_is_flagged_first():
    row = {"model": "openai/gpt-oss-20b", "release": date(2025, 8, 5), "clean_from": "2025-09", "clean_months": 12,
           "controls": "fails", "m5": "refusal-driven", "m15": "not detected", "m18": "not detected",
           "habit": "2023-03", "dating_ends": None}
    assert rd.verdict(row).startswith("Fails the positive controls")


def test_reading_maps_registered_report_readings_to_three_words():
    assert rd.reading("memorization detected: Δκ +0.09") == "detected"
    assert rd.reading("memory detected") == "detected"
    assert rd.reading("dates better than the text alone") == "detected"
    assert rd.reading("not confirmed: refusal-driven (memorization detected)") == "refusal-driven"
    assert rd.reading("none detected") == "not detected" and rd.reading("not shown") == "not detected"
    assert rd.reading(None) == "no data"


def test_controls_use_each_registered_rule():
    m9 = {"label": "M9 within the month's range", "lo": 0.02, "blind": 0.12}
    m9b = {"label": "M9 within ±10%", "lo": 0.03, "blind": 0.24}
    m7_oss = {"label": "M7 direction", "lo": 0.463, "blind": 0.408}       # above blind, not above 0.5 (pilot D E2)
    assert rd.controls_pass([m9, m9b, m7_oss]) is False
    m7_ok = {"label": "M7 direction", "lo": 0.55, "blind": 0.39}
    assert rd.controls_pass([m9, m9b, m7_ok]) is True
    assert rd.controls_pass([{**m9, "lo": 0.15}, m9b, m7_oss]) is True     # M9 above the blind guesser (pilot E E2)


def test_the_prompt_date_reading_follows_the_fake_date_gate_and_test():
    assert rd.prompt_date_reading(None, None) == "not tested"
    assert rd.prompt_date_reading({"passes": False}, None) == "did not forecast"
    assert rd.prompt_date_reading({"passes": True}, {"reading": "hindsight leaks through the date"}) == "leaks"
    assert rd.prompt_date_reading({"passes": True}, {"reading": "not shown"}) == "no leak shown"


def test_the_masking_reading_counts_only_dating_beyond_habits_as_a_leak():
    assert rd.masking_reading(None) == "not tested" and rd.masking_reading({"n": 0}) == "not tested"
    assert rd.masking_reading({"n": 255, "reading": "dates the unlabeled path"}) == "dates masked paths"
    assert rd.masking_reading({"n": 321, "reading": "not shown"}) == "not dated to the month"
    assert rd.masking_reading({"n": 312, "reading": "below its habits"}) == "not dated to the month"
    # the coarser re-score (exploratory) says when a model places paths within a few months instead
    assert rd.masking_reading({"n": 321, "reading": "not shown"}, coarse=6) == "within ±6 months (exploratory)"
    assert rd.masking_reading({"n": 255, "reading": "dates the unlabeled path"}, coarse=3) == "dates masked paths"


def test_the_coarse_window_is_the_narrowest_with_a_clear_positive_holm_result():
    by_window = {3: [{"model": "a", "x": 0.03, "p_holm": 0.20}, {"model": "b", "x": 0.05, "p_holm": 0.01}],
                 6: [{"model": "a", "x": 0.04, "p_holm": 0.04}, {"model": "b", "x": 0.07, "p_holm": 0.001}],
                 12: [{"model": "a", "x": -0.01, "p_holm": 0.01}, {"model": "b", "x": 0.08, "p_holm": 0.001}]}
    assert rd.coarse_windows(by_window) == {"a": 6, "b": 3}


def test_the_market_fade_names_the_half_year_and_how_firm_it_is():
    firm = {"stable": 0.89, "switch": "2024-H1", "at_40": "2024-H1", "at_60": "2024-H1"}
    assert rd.market_fade({"market": "2024-H1", "certainty": firm}) == "2024-H1 (firm, 89%)"
    moves = {"stable": 0.72, "switch": "2023-H1", "at_40": "2023-H1", "at_60": "2024-H1"}
    assert rd.market_fade({"market": "2023-H1", "certainty": moves}) == "2023-H1 (uncertain, 72%)"
    assert rd.market_fade({"market": "no clear switch"}) == "no clear switch"
    assert rd.market_fade(None) == "–"


def test_the_verdict_says_what_a_date_in_the_prompt_and_masking_do():
    row = {"model": "google/gemini-2.5-flash", "release": date(2025, 6, 17), "clean_from": "2025-07",
           "clean_months": 14, "controls": "passes", "m5": "detected", "m15": "detected", "m18": "detected",
           "habit": "2023-11", "dating_ends": "2024", "prompt_date": "leaks", "leak": 0.55,
           "masking": "dates masked paths"}
    text = rd.verdict(row)
    assert "date in the prompt does not stop" in text and "55%" in text and "March 2020" in text
    quiet = {**row, "prompt_date": "no leak shown", "leak": None, "masking": "not dated to the month"}
    text = rd.verdict(quiet)
    assert "no leak shown" in text.lower() and "could not date market paths to the month" in text
    coarse = rd.verdict({**quiet, "masking": "within ±6 months (exploratory)"})
    assert "within ±6 months" in coarse and "not a complete defence" in coarse
    untested = {**row, "prompt_date": "not tested", "masking": "not tested"}
    assert "prompt" not in rd.verdict(untested) and "mask" not in rd.verdict(untested).lower()


def test_market_recall_inside_the_window_counts_even_when_the_drop_is_not_shown():
    assert rd.market_recall("detected", 0.30) == "detected"
    assert rd.market_recall("not detected", 0.31) == "recall shown"
    assert rd.market_recall("not detected", -0.05) == "not detected"
    assert rd.market_recall("not detected", None) == "not detected"
    row = {"model": "meta-llama/llama-3.1-70b-instruct", "release": date(2024, 7, 23), "clean_from": "2024-08",
           "clean_months": 25, "controls": "passes", "m5": "not detected", "m15": "recall shown", "m15_before": 0.55,
           "m18": "detected", "habit": "2019-06", "dating_ends": "2023"}
    text = rd.verdict(row)
    assert "market moves" in text and "+0.55" in text and "drop after release" in text


def test_a_model_without_months_after_release_is_told_why_its_drop_cannot_be_tested():
    row = {"model": "qwen/qwen3.8-flash", "release": date(2026, 8, 26), "clean_from": "2026-09", "clean_months": 0,
           "controls": "passes", "m5": "not detected", "m15": "recall shown", "m15_before": 0.53,
           "m18": "not detected", "habit": "2023-01", "dating_ends": "2025"}
    assert "no months after release" in rd.verdict(row)
