"""App correctness fixes (2026-10-05 audit, step A): the refusal guard with registered rows on screen, one p-value
rule and Holm within each method, the registered M5 windows, the same default models on Overview and Evidence, and
the reading of model - blind cells."""
from datetime import date

import numpy as np
import pandas as pd

from evalbench import analysis, market, stats, ui


def _row(**kw):
    base = {"dkappa": 0.79, "p_holm": 0.001, "n_pre": 188, "n_post": 58, "malformed_rate": 0.0,
            "answered_p_holm": 0.70, "answered_dkappa": 0.10, "coverage_pre": 0.94, "coverage_post": 0.40}
    return pd.Series({**base, **kw})


def test_the_refusal_guard_applies_when_the_table_also_holds_registered_rows():
    assert ui.cell_status(_row(registered=np.nan)) == "refusal-driven"        # a generic row in a mixed table
    assert ui.cell_status(_row()) == "refusal-driven"
    assert ui.cell_status(_row(registered="correlation drop")) == "memorization detected"


def test_holm_runs_within_each_method():
    rng = np.random.default_rng(0)
    frames = {}
    for method in ("price_level", "monthly_return"):
        for k in range(3):
            frames[(f"m/{k}", method)] = pd.DataFrame({
                "item_id": [f"{method}-{i}" for i in range(80)], "date": "2021-01-01", "cluster": [f"c{i % 20}" for i in range(80)],
                "fabricated": False, "refused": False, "group": [1] * 40 + [0] * 40,
                "score": np.r_[rng.uniform(0.4, 0.9, 40), rng.uniform(0.0, 0.5, 40)] if k == 0 else rng.uniform(0, 1, 80)})
    out = analysis.summary_from_frames(frames, B=300)
    for method, part in out.groupby("method"):
        assert np.allclose(part["p_holm"].to_numpy(), stats.holm(part["p_value"].to_numpy(float)))


def test_registered_p_values_follow_the_generic_floor():
    rng = np.random.default_rng(1)
    t = rng.normal(0, 4, 400)
    before = pd.DataFrame({"month": np.repeat(np.arange(100), 4), "answer": t + rng.normal(0, 0.5, 400), "truth": t})
    after = pd.DataFrame({"month": np.repeat(np.arange(100), 4), "answer": rng.normal(0, 1, 400), "truth": t})
    r = market.corr_drop_test(before, after, B=500)
    assert r["p"] >= 2 / 501 and r["p"] == stats.boot_p(np.array([1.0] * 500))


def test_m5_uses_the_registered_twelve_month_windows():
    months = pd.period_range("2021-01", "2026-07", freq="M")
    df = pd.DataFrame({"date": [m.to_timestamp().strftime("%Y-%m-15") for m in months]})
    g = analysis.grid_groups(df, date(2023, 12, 31), date(2024, 12, 6))       # Llama 3.3
    before = [str(m) for m, x in zip(months, g) if x == 1]
    after = [str(m) for m, x in zip(months, g) if x == 0]
    assert (before[0], before[-1], len(before)) == ("2022-10", "2023-09", 12)
    assert (after[0], after[-1], len(after)) == ("2025-01", "2025-12", 12)


def test_overview_and_evidence_share_their_default_models():
    cut = {"a/early": date(2021, 9, 30), "b/mid": date(2023, 12, 31), "c/late": date(2026, 3, 31)}
    assert ui.default_models(list(cut), cut, k=2) == ["a/early", "b/mid"]


def test_a_model_minus_blind_detection_reads_as_beating_the_blind_baseline():
    row = _row(registered="model − blind, before", dkappa=0.125, p_holm=0.001)
    assert ui.verdict(row) == "beats the blind baseline before the cutoff"
