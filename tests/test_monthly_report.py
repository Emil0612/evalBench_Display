"""M11 v2 report (docs/preregistration/2026-10-05-monthly-return-v2.md): the registered table reuses pilot E's pieces;
the one new piece is how jagged a rolling curve is, to compare v1 with v2."""
import numpy as np
import pandas as pd
import pytest

from evalbench import monthly_report as mr


def test_wobble_is_the_median_step_between_neighbouring_windows():
    curve = pd.DataFrame({"month": ["2021-01", "2021-02", "2021-03", "2021-04"], "mean": [0.5, 0.6, 0.4, 0.45]})
    assert mr.wobble(curve) == pytest.approx(0.1)


def test_wobble_skips_steps_across_a_gap():
    curve = pd.DataFrame({"month": ["2021-01", "2021-02", "2021-06", "2021-07"], "mean": [0.5, 0.52, 0.9, 0.94]})
    assert mr.wobble(curve) == pytest.approx(0.03)        # 2021-02 -> 2021-06 is not a neighbouring step


def test_wobble_of_a_curve_without_neighbours_is_nan():
    assert np.isnan(mr.wobble(pd.DataFrame({"month": ["2021-01"], "mean": [0.5]})))


def test_directional_only_drops_the_zero_percent_hedges_and_keeps_refusals_out():
    df = pd.DataFrame({"item_id": list("abcd"), "refused": [False, False, True, False],
                       "no_direction": [True, False, None, False], "score": [0.5, 1.0, 0.0, 0.0]})
    out = mr.directional_only(df)
    assert list(out["item_id"]) == ["b", "d"]
    assert mr.hedge_share(df) == pytest.approx(1 / 3)        # of the answered items


def test_hedge_by_half_year_counts_zero_answers_among_real_answered_items():
    df = pd.DataFrame({"date": ["2023-02-28", "2023-03-31", "2023-08-31", "2023-09-30", "2023-10-31"],
                       "fabricated": [False, False, False, False, True], "refused": [False, False, True, False, False],
                       "no_direction": [True, False, None, True, True]})
    out = mr.hedge_by_half(df)
    assert out.to_dict() == {"2023-H1": 0.5, "2023-H2": 1.0}
