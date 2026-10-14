"""Fixes from the full review of the app audit work (2026-10-05)."""
import numpy as np
import pandas as pd

from evalbench import analysis, stats, ui


def _frames():
    rng = np.random.default_rng(0)
    frames = {}
    for method in ("price_level", "monthly_return"):
        for k in range(3):
            frames[(f"m/{k}", method)] = pd.DataFrame({
                "item_id": [f"{method}-{i}" for i in range(60)], "date": "2021-01-01",
                "cluster": [f"c{i % 15}" for i in range(60)], "fabricated": False, "refused": False,
                "group": [1] * 30 + [0] * 30, "score": rng.uniform(0, 1, 60)})
    return frames


def test_the_pilot_reports_keep_holm_across_every_cell():
    out = analysis.summary_from_frames(_frames(), B=200, family="all")
    assert np.allclose(out["p_holm"].to_numpy(), stats.holm(out["p_value"].to_numpy(float)))


def test_dating_years_survives_a_model_that_refused_everything():
    df = pd.DataFrame({"item_id": ["hd-0200", "hd-0201"], "date": ["2021-03-01", "2021-04-01"], "refused": [True, True]})
    y = ui.dating_years(df)
    assert list(y["share"]) == [0.0]


def test_the_registered_label_does_not_need_a_registered_column():
    assert ui.registered_label(pd.DataFrame({"dkappa": [np.nan]})) == "its registered test"
    assert ui.registered_label(pd.DataFrame({"registered": [np.nan, "model − blind, before"]})) == "model − blind, before"


def test_m18_keeps_its_registered_refusal_guard_and_the_index_tests_do_not_need_one():
    row = {"dkappa": 0.1, "p_holm": 0.001, "n_pre": 1000, "n_post": 300, "malformed_rate": 0.0,
           "registered": "model − blind, before", "answered_p_holm": 0.3, "answered_dkappa": 0.02,
           "coverage_pre": 0.95, "coverage_post": 0.95}
    assert ui.cell_status(pd.Series({**row, "method": "headline_date"})) == "refusal-driven"
    assert ui.cell_status(pd.Series({**row, "method": "index_level"})) == "memorization detected"


def test_registered_methods_stay_out_of_the_staggered_table_and_the_breakdown():
    methods = ["price_level", "index_return", "headline_date", "monthly_return"]
    assert ui.generic_methods(methods) == ["price_level", "monthly_return"]
