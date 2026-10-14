"""The new-model plan (docs/preregistration/2026-10-07-new-models.md): models released in April 2026 have only a few
months after release in our data, so their primary tests need no after side: the same months against a model that
cannot know them (GPT-3.5 Turbo), and recall inside the window against a month-shuffle null."""
from datetime import date

import numpy as np
import pandas as pd

from evalbench import new_models as nm

P = lambda s: pd.Period(s, "M")


def _frame(knows_until: str, start="2016-01", end="2026-08", seed=0, noise=1.5):
    rng = np.random.default_rng(seed)
    rows = []
    truths = {(m, s): float(rng.normal(1, 4)) for m in pd.period_range(start, end, freq="M") for s in "ABCD"}
    for (m, s), t in truths.items():
        a = t + float(rng.normal(0, noise)) if m <= P(knows_until) else 2.5
        rows.append({"item_id": f"{s}-{m}", "month": m, "symbol": s, "truth": t, "answer": a})
    return pd.DataFrame(rows)


def test_recall_beats_the_month_shuffle_when_the_model_knows_and_not_when_it_does_not():
    knows = nm.recall_perm(_frame("2025-12"), P("2024-10"), B=400)
    assert knows["r"] > 0.8 and knows["p"] < 0.01
    flat = nm.recall_perm(_frame("2010-01"), P("2024-10"), B=400)
    assert flat["r"] == 0.0 and flat["p"] > 0.5


def test_recall_is_not_fooled_by_an_index_that_always_moves_more():
    # answers that only know which index is volatile, not any month: the shuffle keeps that, so no recall is shown
    df = _frame("2010-01")
    scale = {"A": 3.0, "B": 1.0, "C": 1.0, "D": 1.0}
    df["truth"] = df.apply(lambda r: r["truth"] * scale[r["symbol"]], axis=1)
    df["answer"] = df["symbol"].map({"A": 6.0, "B": 1.0, "C": 1.5, "D": 0.5})
    assert nm.recall_perm(df, P("2024-10"), B=400)["p"] > 0.05


def test_the_release_line_test_is_insufficient_with_fewer_than_six_months_after_release():
    assert nm.release_line_status(date(2026, 4, 2)) == "insufficient: 4 months after release (< 6)"
    assert nm.release_line_status(date(2025, 6, 17)) is None


def test_the_same_months_against_a_model_that_cannot_know_them_show_memory():
    late, early = _frame("2024-12", seed=1), _frame("2021-09", seed=1)
    row = nm.cross_row(late, early, cut_late=date(2025, 1, 31), cut_early=date(2021, 9, 30),
                       rel_early=date(2024, 1, 25), B=500)
    assert row["usable"] and row["between"] == "2024-02..2024-10" and row["D"] > 0.5 and row["p"] < 0.01


def test_a_price_of_zero_counts_as_not_answered_in_the_m5_guard():
    g = pd.DataFrame({"item_id": ["a", "b", "c"], "refused": [False, True, False], "no_answer": [True, None, False],
                      "score": [0.0, 0.0, 0.4]})
    assert nm.zero_as_refused(g)["refused"].tolist() == [True, True, False]
    assert nm.zero_as_refused(g.drop(columns="no_answer"))["refused"].tolist() == [False, True, False]
