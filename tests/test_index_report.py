"""The M14/M15 report (docs/preregistration/2026-10-01-index.md §6.3)."""
from datetime import date

import numpy as np
import pandas as pd

from evalbench import index_report as xr


def test_before_runs_to_the_last_whole_month_before_cutoff_minus_91_days_and_after_starts_after_release():
    before_last, after_first = xr.sides(date(2025, 3, 31), date(2025, 8, 21))
    assert before_last == pd.Period("2024-11", "M") and after_first == pd.Period("2025-09", "M")


def _df(r_after_noise=None):
    rng = np.random.default_rng(3)
    rows = []
    for m in pd.period_range("2016-01", "2026-08", freq="M"):
        for sym in ("A", "B", "C"):
            t = float(rng.normal(1, 4))
            a = t + float(rng.normal(0, 1)) if m <= pd.Period("2023-12", "M") else 0.0
            rows.append({"month": m, "symbol": sym, "truth": t, "answer": a})
    return pd.DataFrame(rows)


def test_the_correlation_drop_is_detected_when_answers_go_flat_after_release():
    res = xr.corr_drop(_df(), pd.Period("2023-12", "M"), pd.Period("2024-06", "M"), B=500)
    assert res["r_before"] > 0.9 and res["r_after"] == 0.0 and res["p"] < 0.01
    assert res["lo"] > 0.5


def test_no_drop_when_the_model_knows_nothing_on_either_side():
    df = _df().assign(answer=0.0)
    res = xr.corr_drop(df, pd.Period("2023-12", "M"), pd.Period("2024-06", "M"), B=300)
    assert res["diff"] == 0.0 and res["p"] > 0.5


def test_recall_before_the_cutoff_gets_its_own_interval_beside_the_drop():
    res = xr.corr_drop(_df(), pd.Period("2023-12", "M"), pd.Period("2024-06", "M"), B=500)
    assert 0.8 < res["r_before_lo"] <= res["r_before"] <= res["r_before_hi"]
    flat = xr.corr_drop(_df().assign(answer=0.0), pd.Period("2023-12", "M"), pd.Period("2024-06", "M"), B=300)
    assert not flat["r_before_lo"] > 0          # no recall: the interval does not clear 0


def test_a_model_without_an_after_window_still_gets_an_interval_for_its_recall():
    lo, hi = xr.recall_interval(_df(), pd.Period("2023-12", "M"), B=300)
    assert 0.8 < lo <= hi
    lo0, _ = xr.recall_interval(_df().assign(answer=0.0), pd.Period("2023-12", "M"), B=300)
    assert not lo0 > 0
