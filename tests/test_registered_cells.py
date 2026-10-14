"""App cells for the index methods use the registered tests (index plan §6.3, US plan §4), not the generic before/after
score: M15/M17 the correlation drop, M14/M16 model − blind on the months before the cutoff (2026-10-01)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from evalbench import analysis, market
from evalbench.methods import index_return


def _change_frame(seed=0, months=60, split=30, after_constant=True):
    rng = np.random.default_rng(seed)
    rows = []
    for k, m in enumerate(pd.period_range("2016-01", periods=months, freq="M")):
        move = rng.normal(0, 4)
        group = 1 if k < split else (0 if k >= split + 6 else -1)
        for s in range(4):
            t = move + rng.normal(0, 1)
            a = 0.0 if (after_constant and group == 0) else 0.8 * t + rng.normal(0, 1.5)
            rows.append({"item_id": f"{m}-{s}", "date": f"{m}-28", "cluster": f"m{m}", "group": group,
                         "fabricated": False, "refused": False, "outcome": "answer", "score": float(a * t > 0),
                         "answer": a, "truth": t})
    return pd.DataFrame(rows)


def test_index_return_scores_keep_the_answer_and_the_truth():
    from evalbench.items import Item
    it = Item(item_id="x", method="index_return", date="2020-03-31", cluster="m2020-03",
              payload={"symbol": "OMXS30", "index_name": "x", "month": "2020-03", "month_name": "March 2020"},
              truth={"level": 1, "return_pct": -12.0, "prev_level": 1, "baseline_level": None, "prev_return_pct": None})
    sc = index_return.item_score(it, {"main": index_return.parse('{"change_percent": -10}')})
    assert sc.extra["answer"] == -10.0 and sc.extra["truth"] == -12.0


def test_change_cell_is_the_correlation_drop():
    df = _change_frame()
    cell = analysis.registered_index_cell(df, "index_return", B=1000)
    pre, post = df[df["group"] == 1], df[df["group"] == 0]
    r_b = market.corr(pre["answer"], pre["truth"])
    assert cell["dkappa"] == pytest.approx(r_b - 0.0)
    assert cell["mean_pre"] == pytest.approx(r_b) and cell["mean_post"] == 0.0
    assert cell["ci_lo"] < cell["dkappa"] < cell["ci_hi"] and cell["p_value"] < 0.01
    assert cell["n_pre"] == len(pre) and cell["n_post"] == len(post)
    assert cell["registered"] == "correlation drop"


def test_change_cell_ignores_refusals_in_the_correlation():
    df = _change_frame()
    refuse = df.index[(df["group"] == 1)][:10]
    df.loc[refuse, ["refused", "answer"]] = [True, np.nan]
    df.loc[refuse, "outcome"] = "refusal"
    cell = analysis.registered_index_cell(df, "index_return", B=200)
    ok = df[(df["group"] == 1) & ~df["refused"]]
    assert cell["mean_pre"] == pytest.approx(market.corr(ok["answer"], ok["truth"]))
    assert cell["coverage_pre"] == pytest.approx(1 - 10 / (df["group"] == 1).sum())


def test_level_cell_is_model_minus_blind_before_the_cutoff():
    rng = np.random.default_rng(1)
    months = pd.period_range("2016-01", periods=40, freq="M")
    df = pd.DataFrame([{"item_id": f"{m}-{s}", "date": f"{m}-28", "cluster": f"m{m}", "fabricated": False,
                        "group": 1 if k < 25 else 0, "refused": False, "outcome": "answer",
                        "score": float(np.clip(rng.normal(0.8, 0.1), 0, 1))}
                       for k, m in enumerate(months) for s in range(3)])
    blind = {i: 0.5 for i in df["item_id"]}
    cell = analysis.registered_index_cell(df, "index_level", blind=blind, B=1000)
    pre = df[df["group"] == 1]
    assert cell["dkappa"] == pytest.approx(pre["score"].mean() - 0.5)
    assert cell["p_value"] < 0.01 and cell["registered"] == "model − blind, before"


def test_summary_from_frames_uses_the_registered_cell_for_index_methods_only():
    df = _change_frame()
    out = analysis.summary_from_frames({("m", "index_return"): df, ("m", "monthly_return"): df.assign(
        method="monthly_return")}, B=200)
    idx = out[out["method"] == "index_return"].iloc[0]
    assert idx["registered"] == "correlation drop"
    assert pd.isna(out[out["method"] == "monthly_return"].iloc[0].get("registered"))


def test_registered_groups_follow_the_plans_windows():
    """BEFORE: up to the last whole month before cutoff − 91 days; AFTER: from the month after the release month."""
    df = pd.DataFrame({"date": ["2024-02-29", "2024-03-28", "2024-04-30", "2024-12-31", "2025-01-31"]})
    g = analysis.registered_groups(df, date(2024, 6, 30), date(2024, 12, 6))
    # cutoff − 91 days = 2024-03-31, a month end: March is the last BEFORE month; release Dec 2024 → AFTER from Jan
    assert g.tolist() == [1, 1, -1, -1, 0]


def test_the_nordic_micro_pilot_questions_are_left_out():
    from evalbench.methods import index_level
    assert index_level.is_pilot("il-0023") and index_level.is_pilot("ir-0000")
    assert not index_level.is_pilot("il-0024") and not index_level.is_pilot("ul-0003") and not index_level.is_pilot("ur-0010")


def test_a_level_cell_below_the_blind_guesser_is_not_called_better_after_the_cutoff():
    from evalbench import ui
    row = {"dkappa": -0.27, "p_holm": 0.0001, "n_pre": 300, "n_post": 60, "malformed_rate": 0.0,
           "coverage_pre": 1.0, "coverage_post": 1.0, "registered": "model − blind, before"}
    assert ui.cell_status(row) == "none detected"
    assert "below the blind guesser" in ui.verdict(row)
