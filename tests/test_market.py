"""The Market page's numbers: answers next to the true index path, and the rolling correlation between them."""
import numpy as np
import pandas as pd
import pytest

from evalbench import market


def _frame(n_months=48, knows_until="2019-12"):
    rng = np.random.default_rng(0)
    rows = []
    for m in pd.period_range("2016-01", periods=n_months, freq="M"):
        for sym in ("OMXS30", "OMXN40"):
            truth = float(rng.normal(1, 4))
            answer = truth + float(rng.normal(0, 0.5)) if str(m) <= knows_until else 0.0
            rows.append({"month": m, "symbol": sym, "truth": truth, "answer": answer})
    return pd.DataFrame(rows)


def test_answers_that_never_vary_carry_no_information():
    assert market.corr([0.0, 0.0, 0.0], [1.0, -2.0, 3.0]) == 0.0
    assert market.corr([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert np.isnan(market.corr([1.0], [2.0]))


def test_the_rolling_correlation_is_high_while_the_model_knows_and_zero_after():
    curve = market.rolling_corr(_frame(), window=12, min_n=12, B=200)
    early = curve[curve["month"] <= "2018-12"]["r"]
    late = curve[curve["month"] >= "2020-07"]["r"]
    assert (early > 0.9).all() and (late == 0.0).all()
    assert {"month", "r", "lo", "hi", "n"} <= set(curve.columns)


def test_windows_with_too_few_answers_are_left_out():
    curve = market.rolling_corr(_frame(n_months=6), window=12, min_n=20, B=50)
    assert curve.empty


def test_the_path_chart_draws_one_panel_per_model_and_breaks_lines_at_gaps():
    from evalbench import ui
    truth = pd.DataFrame({"month": [str(m) for m in pd.period_range("2016-01", "2016-12", freq="M")], "value": range(12)})
    answers = pd.DataFrame({"model": ["a/one"] * 4 + ["b/two"] * 2,
                            "month": ["2016-01", "2016-02", "2016-07", "2016-08", "2016-03", "2016-04"],
                            "answer": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]})
    spec = ui.market_path_chart(truth, answers, ["a/one", "b/two"], [], [], "level").to_dict()
    assert len(spec["concat"]) == 2
    assert ui.with_segments(answers[answers.model == "a/one"])["segment"].nunique() == 2   # a gap Mar-Jun


def test_the_nordic_vs_us_chart_draws_both_markets_in_every_panel():
    from evalbench import ui
    rows = [{"model": m, "series": s, "month": f"2020-{k:02d}", "mean": 0.5, "lo": 0.4, "hi": 0.6, "n": 40}
            for m in ("a/one", "b/two") for s in ("Nordic", "US") for k in range(1, 13)]
    spec = ui.rolling_market_chart(pd.DataFrame(rows), ["a/one", "b/two"], [], [])
    d = spec.to_dict()
    assert len(d["concat"]) == 2
    assert '"Nordic"' in str(d).replace("'", '"') and '"US"' in str(d).replace("'", '"')
