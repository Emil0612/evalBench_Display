"""The cutoff ladder extended with later-cutoff models (docs/preregistration/2026-10-06-ladder-extension.md): a gate on
months outside the test window, then the registered staggered test against GPT-3.5 Turbo on more months."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from evalbench import ladder_extend as le


def _frame(months, r_target, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for m in months:
        for k in range(4):
            t = float(rng.normal(0.8, 4.0))
            a = t * r_target + float(rng.normal(0, 4.0 * np.sqrt(max(1 - r_target ** 2, 1e-6))))
            rows.append({"item_id": f"i{m}{k}", "month": m, "answer": a, "truth": t})
    return pd.DataFrame(rows)


def test_the_window_runs_from_after_gpt35s_release_to_the_late_models_before_edge():
    months = [pd.Period(m, "M") for m in pd.period_range("2016-01", "2026-07", freq="M")]
    between, baseline = le.windows(date(2025, 7, 31), months)
    assert between[0] == pd.Period("2024-02", "M") and between[-1] == pd.Period("2025-04", "M")
    assert baseline[-1] == pd.Period("2021-06", "M")


def test_the_gate_reads_only_months_outside_the_test_window():
    months = [pd.Period(m, "M") for m in pd.period_range("2016-01", "2026-07", freq="M")]
    strong = pd.concat([_frame([m for m in months if m < pd.Period("2024-02", "M")], 0.9, 1),
                        _frame([m for m in months if m >= pd.Period("2024-02", "M")], 0.0, 2)])
    row = le.gate_row(strong, asked=len(months) * 4, late_cutoff=date(2025, 7, 31))
    assert row["passes"] and row["r_old"] > 0.6 and row["r_recent"] > 0.6
    assert "r_window" not in row                                      # the test months stay unseen
    weak = _frame(months, 0.3, 3)
    assert not le.gate_row(weak, asked=len(months) * 4, late_cutoff=date(2025, 7, 31))["passes"]


def test_a_model_that_answers_too_little_fails_the_gate():
    months = [pd.Period(m, "M") for m in pd.period_range("2016-01", "2023-12", freq="M")]
    f = _frame(months, 0.9, 4).iloc[::2]
    assert not le.gate_row(f, asked=len(months) * 4, late_cutoff=date(2025, 7, 31))["passes"]


def test_the_ladder_candidates_are_off_by_default_in_the_app():
    from evalbench import ui
    assert set(le.CANDIDATES) <= ui.AUX_MODELS
    opts = ["openai/gpt-3.5-turbo", "deepseek/deepseek-v3.2-exp", "google/gemini-2.5-flash"]
    assert ui.default_visible(opts) == ["openai/gpt-3.5-turbo", "google/gemini-2.5-flash"]
    assert ui.default_visible(["deepseek/deepseek-v3.2-exp"]) == ["deepseek/deepseek-v3.2-exp"]   # never empty
