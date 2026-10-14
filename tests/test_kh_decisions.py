"""Decisions of 2026-10-06 (scoring spec §8 and §9): refusals are left out of the app's own scores and reported as a
rate; market questions (M14-M17) are their own fact type, read with their registered statistics;
DE-COP (M3) is frozen at v7."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from evalbench import build_testsets, pilot, ui

CUT, REL = date(2021, 12, 31), date(2022, 6, 30)


def _frame(pre_score, post_score, post_refused, years=(2019, 2020, 2023, 2024), per_month=3):
    rows = []
    for y in years:
        for m in range(1, 13):
            for k in range(per_month):
                d = date(y, m, 15)
                pre = d <= date(2021, 9, 30)
                refused = (not pre) and k < round(per_month * post_refused)
                rows.append({"item_id": f"x{y}{m:02d}{k}", "date": d.isoformat(), "cluster": f"m{y}-{m:02d}",
                             "fabricated": False, "outcome": "refusal" if refused else "answer", "group": int(pre),
                             "refused": refused, "score": 0.0 if refused else (pre_score if pre else post_score)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- §8 refusals
def test_a_model_that_refuses_after_release_shows_no_memory_once_refusals_are_left_out():
    df = _frame(pre_score=0.6, post_score=0.6, post_refused=2 / 3)
    old = pilot.label_profile(pilot.profile_cells(df, "price_level", "m", CUT, REL, B=500))
    new = pilot.label_profile(pilot.profile_cells(df, "price_level", "m", CUT, REL, B=500, drop_refused=True))
    by_old, by_new = {c["period"]: c for c in old}, {c["period"]: c for c in new}
    assert by_old["2019"]["label_release"] == "recall_above_baseline"        # the refusal artefact
    assert by_new["2019"]["label_release"] == "none_detected"
    assert by_new["2019"]["baseline_release"] == pytest.approx(0.6)
    assert by_new["2019"]["refused_baseline_release"] == pytest.approx(2 / 3)  # reported, not scored
    assert by_new["2023"]["refused_share"] == pytest.approx(2 / 3)


def test_evidence_cells_of_generic_methods_read_the_answered_questions():
    out = pd.DataFrame({"model": ["a", "a"], "method": ["monthly_return", "index_return"], "dkappa": [0.6, 0.5],
                        "p_holm": [0.01, 0.01], "n_pre": [800, 90], "n_post": [200, 40], "coverage_pre": [0.94, 1.0],
                        "coverage_post": [0.38, 1.0], "ci_lo": [0.4, 0.3], "ci_hi": [0.8, 0.7]})
    ans = out.assign(dkappa=[0.1, 0.4], p_holm=[0.9, 0.02], n_pre=[790, 90], n_post=[80, 40], ci_lo=[-0.1, 0.2],
                     ci_hi=[0.3, 0.6])
    res = ui.answered_primary(out, ans)
    assert res.loc[0, "dkappa"] == 0.1 and res.loc[0, "n_post"] == 80 and res.loc[0, "coverage_post"] == 0.38
    assert res.loc[1, "dkappa"] == 0.5                                       # registered test: its own rule


def test_the_over_time_views_leave_refusals_out():
    df = _frame(0.6, 0.6, 2 / 3)
    assert not ui.over_time_score(df, "monthly_return")["refused"].any()


# ---------------------------------------------------------------- §9 market questions
def test_market_questions_are_their_own_fact_type():
    for m in ("index_level", "index_return", "us_index_level", "us_index_return"):
        assert pilot.FACT_TYPES[m] == "market"


def _index_frame():
    rng = np.random.default_rng(0)
    rows = []
    for y in (2018, 2019, 2020, 2023, 2024):
        for m in range(1, 13):
            for k, sym in enumerate(("A", "B", "C", "D")):
                d = date(y, m, 28)
                truth = float(rng.normal(0.8, 4.0))
                pre = d <= date(2021, 9, 30)
                answer = truth + float(rng.normal(0, 1.0)) if pre else 2.5          # after release: a stock answer
                rows.append({"item_id": f"i{y}{m:02d}{sym}", "date": d.isoformat(), "cluster": f"m{y}-{m:02d}",
                             "fabricated": False, "outcome": "answer", "group": int(pre), "refused": False,
                             "answer": answer, "truth": truth, "score": float((answer > 0) == (truth > 0))})
    return pd.DataFrame(rows)


def test_market_move_cells_read_the_correlation_with_the_real_moves():
    cells = pilot.label_profile(pilot.profile_cells(_index_frame(), "index_return", "m", CUT, REL, B=500,
                                                    corr=True))
    by = {c["period"]: c for c in cells}
    assert by["2019"]["recall"] > 0.9 and by["2019"]["baseline_release"] == pytest.approx(0.0)
    assert by["2019"]["label_release"] == "recall_above_baseline"
    assert by["2024"]["label"] == "after_cutoff"


def test_market_cells_use_the_registered_comparisons_and_formats():
    assert ui.CORR_METHODS == {"index_return", "us_index_return", "index_forecast"}
    assert {"index_level", "us_index_level", "headline_date"} <= set(ui.PAIRED_BLIND)
    assert ui.value_fmt("index_return")(0.8) == "+0.80" and ui.measure_name("index_return") == "Correlation"
    assert ui.value_fmt("index_level")(0.81) == "0.81" and ui.measure_name("index_level") == "Mean closeness"


# ---------------------------------------------------------------- M3 frozen
def test_new_decop_versions_are_refused_without_unfreeze(monkeypatch, capsys):
    monkeypatch.setattr(build_testsets, "run_decop_v3", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    monkeypatch.setattr(build_testsets, "load_data", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    build_testsets.main(["--methods", "decop", "--decop-v3"])
    build_testsets.main(["--methods", "decop", "--paraphraser", "some/model"])
    assert capsys.readouterr().out.count("frozen at v7") == 2


def test_the_refusal_note_needs_enough_questions_and_counts_read_naturally():
    c = {"stratum": "all", "method": "monthly_return", "baseline_release": 1.0, "n_baseline_release": 1,
         "refused_baseline_release": 0.5}
    text = ui.baseline_summary([c], "release")
    assert "1 answer after release" in text and "refused" not in text and text.endswith("too few for a reading")
    many = {**c, "n_baseline_release": 81, "refused_baseline_release": 0.62}
    assert "81 answers after release · 62% of the questions there refused, left out" in ui.baseline_summary([many], "release")
