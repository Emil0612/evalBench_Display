"""A return answer of exactly 0% is an abstention (scoring spec §7, 2026-10-06): the app's scores leave it out, like a
refusal under the refusal guard, and the share of such answers is shown on its own as an 'I don't know' rate."""
import pandas as pd
import pytest

from evalbench import analysis, ui


def _df():
    return pd.DataFrame({"item_id": list("abcde"), "date": ["2023-01-31"] * 5, "fabricated": [False] * 4 + [True],
                         "refused": [False, False, True, False, False], "group": 1,
                         "no_direction": [True, False, None, False, True], "score": [0.5, 1.0, 0.0, 0.0, 0.0]})


def test_hedges_are_left_out_of_the_scores_of_return_methods_only():
    out = analysis.drop_hedges(_df(), "monthly_return")
    assert list(out["item_id"]) == ["b", "c", "d"]                  # the refusal stays for the refusal guard
    assert analysis.drop_hedges(_df(), "known_return").equals(_df())  # registered positive control: unchanged
    assert {"monthly_return", "annual_return"} <= analysis.HEDGE_METHODS
    # M15/M17: the registered test correlates the answers themselves, where 0% is a prediction; they stay as registered
    assert analysis.drop_hedges(_df(), "index_return").equals(_df())


def test_the_hedge_frame_scores_one_for_a_zero_answer_among_real_answered_items():
    h = analysis.hedge_frame(_df())
    assert list(h["item_id"]) == ["a", "b", "d"] and list(h["score"]) == [1.0, 0.0, 0.0]


def test_a_frame_without_the_flag_has_no_hedges():
    df = _df().drop(columns="no_direction")
    assert len(analysis.drop_hedges(df, "monthly_return")) == 5
    assert analysis.hedge_frame(df)["score"].sum() == 0


def test_the_app_scores_leave_hedges_out_and_the_raw_frame_keeps_them():
    assert ui.app_scores(_df(), "monthly_return", drop_hedges=True)["item_id"].tolist() == ["b", "c", "d"]
    assert len(ui.app_scores(_df(), "monthly_return", drop_hedges=False)) == 5


def test_evidence_offers_the_zero_answer_view_for_return_methods():
    assert "hedge" in ui.over_time_views("monthly_return")
    assert "hedge" not in ui.over_time_views("index_return")
