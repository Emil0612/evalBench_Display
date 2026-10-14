"""App audit step C (2026-10-05): Compare and the staggered table use the release line, the registered windows and a
refusal guard, word their results neutrally, and leave out models whose cutoff we set by hand."""
from datetime import date

import numpy as np
import pandas as pd

from evalbench import pilot, ui


def _frames(refuse_late_window: bool = False):
    rows_a, rows_b = [], []
    for k, d in enumerate(pd.date_range("2020-01-15", "2024-12-15", freq="MS") + pd.Timedelta(days=14)):
        for j in range(3):
            iid = f"pl-{k}-{j}"
            in_window = date(2023, 6, 1) < d.date() < date(2024, 9, 1)
            base = {"item_id": iid, "date": d.date().isoformat(), "cluster": f"c{j}", "fabricated": False}
            rows_a.append({**base, "score": 0.2, "refused": False})
            refused = refuse_late_window and in_window and j == 0
            rows_b.append({**base, "score": 0.0 if refused else (0.6 if in_window else 0.2), "refused": refused})
    return {"price_level": (pd.DataFrame(rows_a), pd.DataFrame(rows_b))}


def test_staggered_window_can_start_after_the_earlier_models_release():
    frames = _frames()
    by_cutoff = pilot.staggered(frames, date(2022, 9, 30), date(2024, 12, 31), B=200)["pooled"]
    by_release = pilot.staggered(frames, date(2022, 9, 30), date(2024, 12, 31), B=200,
                                 window_start=date(2023, 5, 31))["pooled"]
    assert by_release["n_window"] < by_cutoff["n_window"]


def test_staggered_answered_only_drops_questions_either_model_refused():
    frames = _frames(refuse_late_window=True)
    allq = pilot.staggered(frames, date(2022, 9, 30), date(2024, 12, 31), B=200)["pooled"]
    ans = pilot.staggered(frames, date(2022, 9, 30), date(2024, 12, 31), B=200, answered_only=True)["pooled"]
    assert ans["n_window"] < allq["n_window"]


def test_a_frozen_listing_cutoff_is_not_called_hand_set():
    meta = {"openai/gpt-3.5-turbo": {"knowledge_cutoff": "2021-09-30"}, "qwen/qwen3.8-flash": {}}
    over = {"openai/gpt-3.5-turbo": "2021-09-30", "qwen/qwen3.8-flash": "2026-03-31"}
    assert ui.cutoff_source("openai/gpt-3.5-turbo", over, meta) == "OpenRouter listing"
    assert ui.cutoff_source("qwen/qwen3.8-flash", over, meta) == "set by hand"
    assert ui.provisional_models(list(over), over, meta) == ["qwen/qwen3.8-flash"]


def test_assign_groups_applies_the_registered_windows_and_drops_pilot_blocks():
    months = pd.period_range("2016-01", "2026-07", freq="M")
    df = pd.DataFrame({"item_id": [f"x-{i}" for i in range(len(months))],
                       "date": [m.to_timestamp().strftime("%Y-%m-15") for m in months], "group": 1})
    g = ui.assign_groups(df, "price_level", date(2023, 12, 31), date(2024, 12, 6), "release")
    assert int((g["group"] == 1).sum()) == 12 and int((g["group"] == 0).sum()) == 12
    pilots = pd.DataFrame({"item_id": ["ir-0003", "hd-0005", "hd-0300"], "date": "2018-05-15", "group": 1})
    out = ui.assign_groups(pilots, "headline_date", date(2023, 12, 31), date(2024, 12, 6), "release")
    assert list(out["group"]) == [-1, -1, 1]


def test_compare_sentence_is_neutral_and_flags_a_refusal_driven_difference():
    res = {"diff": 0.88, "ci_lo": 0.4, "ci_hi": 1.3}
    clean = ui.compare_sentence("Gemini", "GPT-4.1 Mini", "M11", res, {"diff": 0.5, "ci_lo": 0.2, "ci_hi": 0.8})
    driven = ui.compare_sentence("Gemini", "GPT-4.1 Mini", "M11", res, {"diff": 0.1, "ci_lo": -0.2, "ci_hi": 0.4})
    assert "more memorization" not in clean[0] and "larger" in clean[0] and clean[1] == "bad"
    assert "refusal" in driven[0] and driven[1] == "warn"
