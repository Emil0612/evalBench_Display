"""M16/M17 report (docs/preregistration/2026-10-01-us-index.md §4–§6): the US tests, the paired US − Nordic comparison,
E4 "Nordic beyond US", and the vectorised month bootstraps (identical draws to the original loops, so the published
Nordic numbers do not move)."""
import numpy as np
import pandas as pd
import pytest

from evalbench import index_report as ir
from evalbench import market


def _frame(months, per=4, seed=0, slope=0.7, noise=2.0, shift=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in months:
        move = rng.normal(0, 4)
        for s in range(per):
            t = move + rng.normal(0, 1) + shift
            rows.append({"item_id": f"{m}-{s}", "month": m, "symbol": f"S{s}", "truth": t,
                         "answer": slope * t + rng.normal(0, noise)})
    return pd.DataFrame(rows)


MONTHS = list(pd.period_range("2016-01", periods=60, freq="M"))


def _old_corr_drop(df, before_last, after_first, B, seed=0):
    """The original loop (index_report before vectorising), kept here as the reference."""
    bef, aft = df[df["month"] <= before_last], df[df["month"] >= after_first]
    rng = np.random.default_rng(seed)
    gb, ga = [g for _, g in bef.groupby("month")], [g for _, g in aft.groupby("month")]
    diffs = []
    for _ in range(B):
        sb = pd.concat([gb[int(i)] for i in rng.integers(0, len(gb), len(gb))])
        sa = pd.concat([ga[int(i)] for i in rng.integers(0, len(ga), len(ga))])
        d = market.corr(sb["answer"], sb["truth"]) - market.corr(sa["answer"], sa["truth"])
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return lo, hi, min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))


def _old_mean_test(values, clusters, B, seed=0):
    rng = np.random.default_rng(seed)
    groups = [g.to_numpy(float) for _, g in values.groupby(clusters)]
    draws = np.array([np.concatenate([groups[int(i)] for i in rng.integers(0, len(groups), len(groups))]).mean()
                      for _ in range(B)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi), float(min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())))


def test_corr_drop_draws_exactly_the_original_months():
    df = _frame(MONTHS)
    df.loc[df["month"] >= MONTHS[40], "answer"] = 0.0          # constant after: r_after = 0
    last, first = MONTHS[30], MONTHS[40]
    new = ir.corr_drop(df, last, first, B=400)
    lo, hi, p = _old_corr_drop(df, last, first, B=400)
    assert new["lo"] == pytest.approx(lo, abs=1e-9) and new["hi"] == pytest.approx(hi, abs=1e-9)
    assert new["p"] == pytest.approx(p, abs=1e-12) and new["r_after"] == 0.0


def test_mean_test_draws_exactly_the_original_months():
    rng = np.random.default_rng(1)
    v = pd.Series(rng.normal(0.1, 0.3, 300))
    cl = pd.Series([f"m{i // 5}" for i in range(300)])
    assert ir._mean_test(v, cl, B=500) == pytest.approx(_old_mean_test(v, cl, B=500), abs=1e-12)


def test_paired_difference_resamples_the_same_months_for_both_markets():
    us = _frame(MONTHS, seed=2, noise=0.5)          # tight memory
    nord = _frame(MONTHS, seed=3, noise=4.0)        # loose memory
    out = ir.paired_corr_diff(us, nord, B=2000)
    r_us, r_n = market.corr(us["answer"], us["truth"]), market.corr(nord["answer"], nord["truth"])
    assert out["diff"] == pytest.approx(r_us - r_n)
    assert out["lo"] < out["diff"] < out["hi"] and out["p"] < 0.01 and out["n_months"] == len(MONTHS)


def test_partial_correlation_matches_residuals_on_the_us_move():
    rng = np.random.default_rng(4)
    df = _frame(MONTHS, seed=5)
    us_move = pd.Series({m: rng.normal(0, 4) for m in MONTHS})
    u = df["month"].map(us_move).to_numpy(float)

    def resid(y):
        X = np.column_stack([np.ones(len(u)), u])
        return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    expected = market.corr(resid(df["answer"].to_numpy(float)), resid(df["truth"].to_numpy(float)))
    assert ir.partial_corr(df, us_move) == pytest.approx(expected, abs=1e-9)


def test_partial_correlation_is_zero_when_the_answers_only_know_the_us_move():
    """A model that answers with the US move (global knowledge only) tracks the Nordic market but nothing beyond it."""
    rng = np.random.default_rng(6)
    us_move = pd.Series({m: rng.normal(0, 4) for m in MONTHS})
    rows = []
    for m in MONTHS:
        for s in range(4):
            rows.append({"month": m, "truth": 0.9 * us_move[m] + rng.normal(0, 1.5), "answer": us_move[m]})
    df = pd.DataFrame(rows)
    assert market.corr(df["answer"], df["truth"]) > 0.8
    assert ir.partial_corr(df, us_move) == pytest.approx(0.0, abs=1e-9)
    t = ir.partial_test(df, us_move, B=500)
    assert t["lo"] <= 0 <= t["hi"] or abs(t["r"]) < 1e-9


def test_partial_correlation_constant_answers_give_zero():
    df = _frame(MONTHS).assign(answer=1.5)
    us_move = pd.Series({m: 1.0 * k for k, m in enumerate(MONTHS)})
    assert ir.partial_corr(df, us_move) == 0.0


def test_answer_style_counts_distinct_and_repeated_values():
    df = pd.DataFrame({"answer": [-1.5] * 6 + [2.0] * 3 + [0.3]})
    s = ir.answer_style(df)
    assert s["n"] == 10 and s["distinct"] == 3
    assert s["top"][0] == (-1.5, 0.6) and s["top"][1] == (2.0, 0.3)
