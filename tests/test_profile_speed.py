"""The Profile's cells are screened with fewer bootstrap draws and recomputed with the full number only where the
reading can still change (p <= 0.1): Holm never lowers a p-value, so a cell above 0.1 can never be a detection."""
from datetime import date

import numpy as np
import pandas as pd

from evalbench import pilot

CUT, REL = date(2021, 12, 31), date(2022, 6, 30)


def _frame(pre_mean, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in (2019, 2020, 2023, 2024):
        for m in range(1, 13):
            for k in range(4):
                d = date(y, m, 15)
                pre = d <= date(2021, 9, 30)
                rows.append({"item_id": f"x{y}{m:02d}{k}", "date": d.isoformat(), "cluster": f"c{(m * 7 + k) % 40}",
                             "fabricated": False, "outcome": "answer", "group": int(pre), "refused": False,
                             "score": float(np.clip(rng.normal(pre_mean if pre else 0.3, 0.15), 0, 1))})
    return pd.DataFrame(rows)


def test_a_clear_cell_gets_the_full_resolution_and_a_null_cell_the_screen():
    strong = {c["period"]: c for c in pilot.profile_cells(_frame(0.7), "price_level", "m", CUT, REL, B=20000,
                                                          screen=500)}
    assert strong["2019"]["p_release"] < 1 / 501                    # below the screen's floor: recomputed at 20,000
    weak = {c["period"]: c for c in pilot.profile_cells(_frame(0.3), "price_level", "m", CUT, REL, B=20000,
                                                        screen=500)}
    assert weak["2019"]["p_release"] > 0.1


def test_screening_gives_the_same_readings_as_the_full_bootstrap():
    for mean in (0.3, 0.34, 0.45, 0.7):
        full = pilot.label_profile(pilot.profile_cells(_frame(mean, 1), "price_level", "m", CUT, REL, B=4000))
        fast = pilot.label_profile(pilot.profile_cells(_frame(mean, 1), "price_level", "m", CUT, REL, B=4000,
                                                       screen=500))
        assert [c.get("label_release") for c in full] == [c.get("label_release") for c in fast]
