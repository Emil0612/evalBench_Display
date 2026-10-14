"""M18 on the Profile: each year's cell is compared with the text-only guess on the same headlines (the registered
M18 comparison), not with the model's own after-cutoff level, which is near 0 by construction for news dating."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from evalbench import pilot, ui
from evalbench.items import Item
from evalbench.methods import headline_date as hd

CUT = date(2023, 12, 31)


def _frame(model_score, blind_score, years=(2019, 2020, 2021, 2024, 2025), per_month=4):
    rows, blind = [], {}
    for y in years:
        for m in range(1, 13):
            for k in range(per_month):
                i = f"hd-{y}{m:02d}{k}"
                d = f"{y}-{m:02d}-15"
                rows.append({"item_id": i, "date": d, "cluster": f"{y}-{m:02d}", "fabricated": False,
                             "outcome": "answer", "group": int(date.fromisoformat(d) <= date(2023, 9, 30)),
                             "score": model_score(y)})
                blind[i] = blind_score(y)
    return pd.DataFrame(rows), blind


def test_a_cell_is_compared_with_the_text_only_guess_on_its_own_headlines():
    df, blind = _frame(lambda y: 0.9 if y <= 2020 else 0.4, lambda y: 0.4)
    cells = pilot.label_profile(pilot.profile_cells(df, "headline_date", "m", CUT, date(2024, 7, 23), blind,
                                                    B=300, paired=True))
    by = {c["period"]: c for c in cells}
    assert by["2019"]["baseline"] == pytest.approx(0.4) and by["2019"]["n_baseline"] == 48
    assert by["2019"]["label"] == "recall_above_baseline"
    assert by["2021"]["label"] == "none_detected"                       # dates no better than the text alone
    assert by["2019"]["label_release"] == by["2019"]["label"]           # the after-cutoff/release switch does not apply
    assert by["2024"]["label"] == "after_cutoff" and by["2024"]["baseline"] == pytest.approx(0.4)


def test_the_comparison_ignores_how_low_the_model_scores_after_its_cutoff():
    # a model at the text-only level everywhere before its cutoff, near 0 after: no cell is above its baseline
    df, blind = _frame(lambda y: 0.4 if y <= 2021 else 0.02, lambda y: 0.4)
    cells = pilot.label_profile(pilot.profile_cells(df, "headline_date", "m", CUT, None, blind, B=300, paired=True))
    assert not any(c["label"] == "recall_above_baseline" for c in cells)


def test_blind_score_is_the_m18_score_of_the_text_only_month():
    it = Item("hd-1", "headline_date", "2021-05-14", "2021-05", {}, {"month": "2021-05", "blind_month": "2021-11"})
    assert hd.blind_score(it) == pytest.approx(0.5)


def test_m18_is_on_the_profile_with_its_own_baseline_wording():
    assert "headline_date" in ui.PAIRED_BLIND
    cells = [{"stratum": "all", "method": "headline_date", "paired_blind": True, "baseline": 0.31,
              "n_baseline": 48, "baseline_release": 0.31, "n_baseline_release": 48}]
    text = ui.baseline_summary(cells, "release")
    assert "text-only" in text and "same headlines" in text


def test_the_habitual_year_is_marked_on_the_grid():
    c = {"stratum": "all", "period": "2021", "method": "headline_date", "label": "none_detected",
         "label_release": "none_detected", "facts": 48, "responses": 48, "recall": 0.4, "ci": [0.3, 0.5],
         "baseline": 0.4, "n_baseline": 48, "baseline_release": 0.4, "n_baseline_release": 48}
    html = ui.profile_grid_html([c], ["2021"], ["all"], habit="2021-09")
    assert "habit" in html and "2021-09" in html
    assert "habit" not in ui.profile_grid_html([c], ["2021"], ["all"])


def test_m18_cells_show_the_score_as_a_number_not_a_share_of_headlines():
    assert ui.value_fmt("headline_date")(0.55) == "0.55"
    assert ui.measure_name("headline_date") == "Mean score"
    assert ui.measure_name("price_level") == "Mean closeness" and ui.measure_name("price_recall") == "Measured recall"
