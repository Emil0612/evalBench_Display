"""Estimating an unpublished knowledge cutoff from the US monthly-change answers (M17), as registered in
docs/preregistration/2026-10-07-new-models.md: the month where recall drops most steeply (6 months before vs after,
searched in the 36 months before release), plus the median offset of models whose cutoff is published."""
from datetime import date

import numpy as np
import pandas as pd

from evalbench import cutoff_estimate as ce

P = lambda s: pd.Period(s, "M")


def _frame(knows_until: str, start="2019-01", end="2026-08", seed=0):
    """Four indices a month; the model tracks the truth up to knows_until, then answers a stock value."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in pd.period_range(start, end, freq="M"):
        for k in range(4):
            t = float(rng.normal(1, 4))
            a = t + float(rng.normal(0, 1.5)) if m <= P(knows_until) else 2.5
            rows.append({"item_id": f"u-{len(rows)}", "month": m, "truth": t, "answer": a})
    return pd.DataFrame(rows)


def test_the_changepoint_finds_where_recall_stops():
    month, drop = ce.changepoint(_frame("2024-06"), release=date(2025, 6, 1))
    assert abs((month - P("2024-06")).n) <= 1 and drop > 0.5


def test_the_search_stays_in_the_36_months_before_release():
    # recall stops in 2020, outside the window: the estimate cannot land there
    month, drop = ce.changepoint(_frame("2020-03"), release=date(2025, 6, 1))
    assert month >= P("2022-06") and drop < 0.3


def test_the_offset_is_the_median_gap_to_published_cutoffs():
    gaps = {"a": -8, "b": -7, "c": 1, "d": -3, "e": -7}
    assert ce.offset(gaps) == 7


def test_the_estimate_adds_the_offset_and_falls_back_when_the_drop_is_weak():
    est = ce.estimate(P("2024-06"), 0.9, offset=7, release=date(2025, 8, 1))
    assert est == {"cutoff": date(2025, 1, 31), "rule": "changepoint + 7 months"}
    weak = ce.estimate(P("2024-06"), 0.1, offset=7, release=date(2025, 8, 1))
    assert weak == {"cutoff": date(2024, 8, 31), "rule": "fallback: release - 12 months (drop below 0.3)"}
