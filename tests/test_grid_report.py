"""The monthly-grid analysis (docs/preregistration/2026-09-30-monthly-grid.md): the near-line windows and the
break-month estimate."""
from datetime import date

import numpy as np
import pandas as pd

from evalbench import grid_report as gr


def test_before_ends_with_the_last_whole_month_before_cutoff_minus_91_days_and_after_starts_after_release():
    before, after = gr.windows(date(2025, 3, 31), date(2025, 8, 21), last_month=pd.Period("2026-07", "M"))
    assert before[-1] == pd.Period("2024-11", "M") and len(before) == 12 and before[0] == pd.Period("2023-12", "M")
    assert after[0] == pd.Period("2025-09", "M") and after[-1] == pd.Period("2026-07", "M") and len(after) == 11


def test_a_cutoff_minus_91_days_on_a_month_end_keeps_that_month():
    before, _ = gr.windows(date(2024, 3, 31), date(2025, 1, 1), last_month=pd.Period("2026-07", "M"))
    assert before[-1] == pd.Period("2023-12", "M")       # 2024-03-31 - 91 days = 2023-12-31, the month's last day


def test_after_needs_at_least_six_months():
    _, after = gr.windows(date(2025, 12, 31), date(2026, 3, 1), last_month=pd.Period("2026-07", "M"))
    assert after is None                                  # only 2026-04..07 exist


def test_the_break_month_is_found_at_a_clear_step():
    rng = np.random.default_rng(0)
    months = pd.period_range("2021-01", "2026-07", freq="M")
    rows = []
    for m in months:
        for k in range(12):
            level = 0.4 if m < pd.Period("2024-01", "M") else 0.1
            rows.append({"month": m, "score": level + rng.normal(0, 0.05), "cluster": f"c{rng.integers(300)}"})
    est = gr.break_month(pd.DataFrame(rows), B=100)
    assert est["month"] == pd.Period("2024-01", "M")
    assert est["lo"] <= pd.Period("2024-01", "M") <= est["hi"]


def test_several_grid_versions_are_pooled_by_item_and_testset(tmp_path):
    from evalbench.items import Item, save_testset

    def one(prefix, month):
        return Item(item_id=f"{prefix}-0000", method="price_level", date=f"{month}-28", cluster="c1",
                    payload={"month": month}, truth={"insref": 1})
    save_testset([one("pl3", "2024-01")], "price_level", "v3", {}, root=tmp_path)
    save_testset([one("pl4", "2024-02")], "price_level", "v4", {}, root=tmp_path)
    items, names = gr.set_items(("v3", "v4"), root=tmp_path)
    assert set(items) == {"pl3-0000", "pl4-0000"}
    assert names == ["price_level/v3", "price_level/v4"]
    assert gr.label(("v3",)) == "" and gr.label(("v4",)) == "v4" and gr.label(("v3", "v4")) == "v3+v4"


def _scores(levels, months, n=12, seed=0):
    """One model's per-question closeness: level by month, the same question ids and companies for every model."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in months:
        for k in range(n):
            rows.append({"item_id": f"{m}-{k}", "month": m, "cluster": f"c{k}", "fabricated": False,
                         "refused": False, "score": min(1, max(0, levels(m) + rng.normal(0, 0.02)))})
    return pd.DataFrame(rows)


def test_staggered_windows_run_from_after_the_release_to_the_late_models_last_month_before_its_cutoff():
    between, baseline = gr.staggered_months(early_cutoff=date(2024, 6, 30), early_release=date(2025, 4, 14),
                                            late_cutoff=date(2026, 3, 31))
    assert between[0] == pd.Period("2025-05", "M") and between[-1] == pd.Period("2025-11", "M")
    assert baseline[-1] == pd.Period("2024-03", "M") and len(baseline) == 12


def test_staggered_check_separates_memory_from_a_time_effect():
    months = list(pd.period_range("2023-04", "2025-11", freq="M"))
    between, baseline = gr.staggered_months(date(2024, 6, 30), date(2025, 4, 14), date(2026, 3, 31))
    late = _scores(lambda m: 0.30, months, seed=1)
    memory = _scores(lambda m: 0.10 if m in between else 0.30, months, seed=2)      # only the early model drops
    time_effect = _scores(lambda m: 0.10 if m in between else 0.30, months, seed=3)
    late_too = _scores(lambda m: 0.10 if m in between else 0.30, months, seed=4)    # everyone drops
    hit = gr.staggered_pair(late, memory, between, baseline, B=500)
    null = gr.staggered_pair(late_too, time_effect, between, baseline, B=500)
    assert hit["dkappa"] > 0.15 and hit["p_value"] < 0.01
    assert abs(null["dkappa"]) < 0.03
