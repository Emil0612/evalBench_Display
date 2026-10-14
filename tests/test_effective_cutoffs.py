"""Effective cutoffs (exploratory): per model and kind of data, the last year its answers clearly show knowledge, next
to the stated cutoff."""
import numpy as np
import pandas as pd

from evalbench import effective_cutoffs as ec


def _index_frame(know_until="2021-12", seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in pd.period_range("2016-01", "2025-12", freq="M"):
        for k in range(4):
            t = float(rng.normal(0.8, 4.0))
            a = t + float(rng.normal(0, 1.0)) if m <= pd.Period(know_until, "M") else 2.5
            rows.append({"month": str(m), "answer": a, "truth": t})
    return pd.DataFrame(rows)


def test_the_market_switch_is_where_answers_turn_uninformative():
    f = _index_frame("2021-12")                                   # after 2021: a stock answer of +2.5
    asked = f.assign(h=f["month"].map(ec.half_of)).groupby("h").size()
    assert ec.uninformative_switch(f, asked) == "2022-H1"
    refusing = f[f["month"] < "2023-01"]                          # from 2023 on: no answers at all (refusals)
    assert ec.uninformative_switch(refusing, asked) == "2022-H1"


def test_a_switch_after_the_release_is_not_a_horizon():
    assert ec.market_reading("2024-H1", pd.Timestamp("2024-07-18").date()) == "2024-H1"
    assert ec.market_reading("2026-H1", pd.Timestamp("2024-07-23").date()) == "no clear switch"
    assert ec.market_reading(None, pd.Timestamp("2024-07-23").date()) == "no clear switch"


def test_the_switch_to_zero_answers_is_the_first_half_year_from_which_most_answers_are_zero():
    halves = pd.Series({"2022-H1": 0.1, "2022-H2": 0.6, "2023-H1": 0.3, "2023-H2": 0.7, "2024-H1": 0.95})
    assert ec.zero_switch(halves) == "2023-H2"
    assert ec.zero_switch(pd.Series({"2022-H1": 0.0, "2022-H2": 0.1})) is None


def test_switch_certainty_reports_how_often_resampling_lands_on_the_same_half_year():
    flags = pd.DataFrame({"h": ["2021-H1"] * 20 + ["2021-H2"] * 20 + ["2022-H1"] * 20,
                          "u": [0.0] * 20 + [1.0] * 20 + [1.0] * 20})
    c = ec.switch_certainty(flags, B=200)
    assert c["switch"] == "2021-H2" and c["stable"] == 1.0 and c["at_40"] == c["at_60"] == "2021-H2"
