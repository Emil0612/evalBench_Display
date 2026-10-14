"""The masking test's analysis (docs/preregistration/2026-10-06-masking.md): a hit is an answer within a month of the
true end month, minus how often the model's own answers land there anyway, the months next to its habit left out."""
import numpy as np
import pandas as pd

from evalbench import mask_date as md

P = lambda s: pd.Period(s, "M")


def _answers(knows_until, habit="2021-10", p_know=0.5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in pd.period_range("2017-01", "2026-07", freq="M"):
        for k in range(4):
            knows = m <= P(knows_until) and rng.random() < p_know
            rows.append({"item_id": f"pd-{len(rows):04d}", "true": m, "answer": m if knows else P(habit),
                         "refused": False, "cluster": f"m{m}"})
    return pd.DataFrame(rows)


def test_a_model_with_only_a_habit_scores_zero():
    x = md.excess(_answers("2015-12", p_know=0.0))
    assert x["x"].dropna().abs().max() < 1e-9


def test_a_model_that_knows_scores_above_zero_and_its_habit_months_are_left_out():
    x = md.excess(_answers("2023-12"))
    known = x[(x["true"] <= P("2023-12"))].dropna(subset=["x"])
    assert known["x"].mean() > 0.3
    assert x.loc[(x["true"] - P("2021-10")).map(lambda o: abs(o.n)) <= md.HABIT_GAP, "x"].isna().all()


def test_refused_paths_are_left_out():
    a = _answers("2023-12").assign(refused=lambda d: d.index % 5 == 0)
    a.loc[a["refused"], "answer"] = None
    assert md.excess(a).loc[a["refused"], "x"].isna().all()


def test_a_wider_window_counts_near_misses_as_hits():
    a = _answers("2023-12", p_know=0.0).copy()
    a["answer"] = a["true"].map(lambda t: t + 4)        # always four months late: a near miss, never exact
    a.loc[a.index % 3 == 0, "answer"] = P("2021-10")   # plus a habit, so the habit share is not everything
    assert md.excess(a)["x"].dropna().mean() < 0.01
    assert md.excess(a, window=6)["x"].dropna().mean() > 0.3
