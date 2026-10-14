"""An anchoring-free horizon from the M18 answers (exploratory): per publication year, the share of headlines dated to
the exact month minus the share the model's own answer habits would hit there anyway (its answers' spread over
months), leaving out the months next to its habitual answer; the horizon is the last year clearly above 0."""
import numpy as np
import pandas as pd

from evalbench import dating_horizon as dh

P = lambda s: pd.Period(s, "M")


def _answers(h0: str, habit: str, p_know: float, seed: int = 0) -> pd.DataFrame:
    """A model that knows the true month with probability p_know up to h0 and otherwise names its habit month."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in pd.period_range("2016-01", "2026-07", freq="M"):
        for k in range(12):
            knows = m <= P(h0) and rng.random() < p_know
            rows.append({"item_id": f"hd-{1000 + len(rows)}", "true": m, "answer": m if knows else P(habit),
                         "refused": False})
    return pd.DataFrame(rows)


def test_expected_hit_is_the_share_of_the_models_answers_near_the_true_month():
    answers = pd.Series([P("2021-09")] * 8 + [P("2019-01")] * 2)
    assert dh.expected_hit(answers, P("2021-07"), window=3) == 0.8
    assert dh.expected_hit(answers, P("2019-01"), window=0) == 0.2
    assert dh.expected_hit(answers, P("2017-01"), window=0) == 0.0


def test_the_habitual_month_earns_no_excess():
    df = _answers("2015-12", "2021-09", 0.0)          # knows nothing: always the habit month
    ex = dh.excess_by_year(df, B=200)
    assert ex["excess"].abs().max() < 1e-9


def test_a_mixed_model_gets_no_free_excess_next_to_its_habit():
    df = _answers("2018-12", "2021-09", 0.3)          # knows up to 2018, habit far later
    ex = dh.excess_by_year(df, B=200).set_index("year")
    assert ex.loc[2021, "excess"] < 0.02 and ex.loc[2018, "excess"] > 0.2


def test_the_horizon_is_the_last_year_with_real_dating():
    df = _answers("2021-09", "2021-09", 0.3)
    assert dh.horizon(dh.excess_by_year(df, B=300)) == 2021


def test_no_horizon_without_any_year_above_zero():
    ex = pd.DataFrame({"year": [2020, 2021], "lo": [-0.01, -0.02], "excess": [0.0, 0.0]})
    assert dh.horizon(ex) is None


def test_silent_from_is_the_first_year_after_which_no_headline_is_dated_exactly():
    ex = pd.DataFrame({"year": [2020, 2021, 2022, 2023, 2024], "observed": [0.04, 0.0, 0.02, 0.0, 0.0]})
    assert dh.silent_from(ex) == 2023
    assert dh.silent_from(ex.assign(observed=0.01)) is None


def test_a_model_not_asked_a_headline_set_gives_an_empty_frame():
    calls = pd.DataFrame({"model": ["openai/gpt-4o-mini-2024-07-18"], "testset": ["headline_date/v1"]})
    f = dh.frame(calls, "google/gemini-2.5-flash", None, {}, testset="headline_date/v2")
    assert f.empty and list(f.columns) == ["item_id", "true", "answer", "refused"]


def test_pooling_joins_a_models_answers_over_the_sets_it_was_asked_and_skips_the_others():
    a = _answers("2021-12", "2019-06", 0.3, seed=1)
    b = _answers("2021-12", "2019-06", 0.3, seed=2).assign(item_id=lambda d: "v2-" + d["item_id"])
    empty = dh.frame(pd.DataFrame({"model": [], "testset": []}), "x/y", None, {})
    both = dh.pooled([a, empty, b])
    assert len(both) == len(a) + len(b) and both["item_id"].is_unique
    ex = dh.excess_by_year(both)
    assert ex.set_index("year").loc[2020, "n"] == 2 * dh.excess_by_year(a).set_index("year").loc[2020, "n"]
    assert (dh.horizon(ex), dh.silent_from(ex)) == (2021, 2022)
    assert dh.pooled([empty]).empty


def test_the_dating_span_runs_from_the_last_clearly_dated_year_to_the_year_before_silence():
    assert dh.dating_span(2020, 2023) == "2020–2022"
    assert dh.dating_span(2023, 2024) == "2023"
    assert dh.dating_span(2025, None) == "2025 or later"
    assert dh.dating_span(None, 2025) is None          # no year clearly dated: no horizon to name
