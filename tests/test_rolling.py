"""The rolling-window view: a 6-month window moved one month at a time, with a cluster-bootstrap band; windows with
too few items are left empty; pooled methods are chance-corrected first."""
import numpy as np
import pandas as pd
import pytest

from evalbench import analysis, ui


def _frame(months, scores, clusters=None):
    dates = [pd.Period(m, "M").to_timestamp(how="end").date().isoformat() for m in months]
    return pd.DataFrame({"date": dates, "score": scores, "cluster": clusters or [f"c{i}" for i in range(len(months))],
                         "fabricated": False, "refused": False})


def test_a_step_drop_shows_as_a_falling_curve():
    months = [str(p) for p in pd.period_range("2022-01", "2024-12", freq="M") for _ in range(10)]
    scores = [1.0 if m < "2023-07" else 0.0 for m in months]
    curve = analysis.rolling_curve(_frame(months, scores), window=6, min_n=20, B=200)
    first, last = curve.iloc[0], curve.iloc[-1]
    assert first["mean"] == pytest.approx(1.0) and last["mean"] == pytest.approx(0.0)
    mid = curve[curve["month"] == "2023-07"].iloc[0]          # window 2023-05..2023-10: 2 months of 1s, 4 of 0s
    assert mid["mean"] == pytest.approx(2 / 6) and mid["n"] == 60
    assert (curve["lo"] <= curve["mean"]).all() and (curve["mean"] <= curve["hi"]).all()


def test_windows_with_too_few_items_are_left_out():
    months = ["2022-01"] * 30 + ["2024-06"] * 3
    curve = analysis.rolling_curve(_frame(months, [1.0] * 33), window=6, min_n=15, B=100)
    assert "2022-01" in set(curve["month"]) and "2024-06" not in set(curve["month"])


def test_fabricated_items_are_not_counted():
    df = _frame(["2022-01"] * 20, [1.0] * 20)
    df.loc[:9, "fabricated"] = True
    curve = analysis.rolling_curve(df, window=6, min_n=5, B=100)
    assert curve.iloc[0]["n"] == 10


def test_pooling_corrects_each_method_for_guessing():
    a = _frame(["2022-01"] * 20, [0.5] * 20).assign(p0=0.5)     # direction at chance
    b = _frame(["2022-01"] * 20, [0.0] * 20).assign(p0=0.0)
    pooled = analysis.chance_corrected(pd.concat([a, b]))
    assert pooled["score"].max() == pytest.approx(0.0)           # (0.5 - 0.5) / (1 - 0.5) = 0


def test_rolling_chart_draws_a_band_a_line_and_the_blind_reference():
    months = [str(p) for p in pd.period_range("2022-01", "2023-12", freq="M") for _ in range(5)]
    curve = analysis.rolling_curve(_frame(months, list(np.linspace(1, 0, len(months)))), window=6, min_n=10, B=100)
    frames = pd.concat([curve.assign(model="x/a"), curve.assign(model=ui.BLIND_ID)])
    spec = ui.rolling_chart(frames, ["x/a"], cut_rows=[{"model": "x/a", "month": "2023-01"}],
                            rel_rows=[{"model": "x/a", "month": "2023-06"}]).to_dict()
    panels = spec["concat"]                                    # one small panel per model
    assert len(panels) == 1
    marks = [(layer.get("mark") or {}).get("type") for layer in panels[0]["layer"]]
    assert "area" in marks and marks.count("line") == 2 and marks.count("rule") == 2   # model + blind; cutoff + release


def test_lines_break_where_windows_are_missing():
    curve = pd.DataFrame({"month": ["2024-01", "2024-02", "2024-06", "2024-07"], "mean": [0.1] * 4, "lo": [0.0] * 4,
                          "hi": [0.2] * 4, "n": [20] * 4})
    assert list(ui.with_segments(curve)["segment"]) == [0, 0, 1, 1]
