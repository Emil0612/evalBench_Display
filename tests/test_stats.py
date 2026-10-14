import math

import numpy as np
import pytest

from evalbench import stats


def test_delta_ignores_excluded_and_needs_both_sides():
    assert stats.delta([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0
    assert stats.delta([1, 0, 5], [1, 0, -1]) == 1.0
    assert math.isnan(stats.delta([1, 1], [1, 1]))


def test_kappa_delta():
    assert stats.kappa_delta(0.375, 0.25) == pytest.approx(0.5)


def test_bootstrap_ci_degenerate_and_null():
    s = [1] * 20 + [0] * 20
    g = [1] * 20 + [0] * 20
    assert stats.bootstrap_delta_ci(s, g, list(range(40)), B=300) == (1.0, 1.0)
    s2 = np.tile([0, 1], 100)
    g2 = np.repeat([1, 0], 100)
    lo, hi = stats.bootstrap_delta_ci(s2, g2, np.arange(200), B=500)
    assert lo < 0 < hi


def test_bootstrap_ci_empty_is_nan():
    lo, hi = stats.bootstrap_delta_ci([], [], [], B=10)
    assert math.isnan(lo) and math.isnan(hi)


def test_permutation_pvalue():
    s = [1] * 20 + [0] * 20
    g = [1] * 20 + [0] * 20
    assert stats.permutation_pvalue(s, g, n_perm=500) < 0.01
    assert stats.permutation_pvalue([1] * 10, [1] * 5 + [0] * 5, n_perm=200) == 1.0


def test_binom_greater():
    assert stats.binom_greater(10, 10, 0.25) < 1e-5


def test_mcnemar_exact():
    a = np.array([1, 1, 1, 1, 1, 1, 0], bool)
    b = np.zeros(7, bool)
    assert stats.mcnemar_exact(a, b) == pytest.approx(0.03125)
    assert stats.mcnemar_exact(a, a) == 1.0


def test_paired_bootstrap_diff():
    g = np.array([1] * 10 + [0] * 10)
    a = np.array([1.0] * 10 + [0.0] * 10)
    b = np.full(20, 0.5)
    point, lo, hi = stats.paired_bootstrap_diff(a, g, b, g, B=200)
    assert point == pytest.approx(1.0) and lo <= point <= hi


def test_holm():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    out = stats.holm([0.01, float("nan")])
    assert out[0] == pytest.approx(0.01) and math.isnan(out[1])


def test_bootstrap_pvalue_agrees_with_ci():
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = 40
        g = np.repeat([1, 0], n // 2)
        s = (rng.random(n) < np.where(g == 1, rng.uniform(0.2, 0.8), 0.4)).astype(float)
        clusters = rng.integers(0, 12, n)
        lo, hi, p = stats.bootstrap_delta(s, g, clusters, B=400)
        assert (p < 0.05) == (lo > 0 or hi < 0)


def test_bootstrap_delta_degenerate():
    lo, hi, p = stats.bootstrap_delta([1] * 20 + [0] * 20, [1] * 20 + [0] * 20, list(range(40)), B=300)
    assert (lo, hi) == (1.0, 1.0) and p < 0.01
    lo, hi, p = stats.bootstrap_delta([1] * 10, [1] * 5 + [0] * 5, list(range(10)), B=100)
    assert p == 1.0


def _dates(n, seed=0):
    rng = np.random.default_rng(seed)
    start = np.datetime64("2021-01-01").astype("datetime64[D]").astype(int)
    end = np.datetime64("2026-06-30").astype("datetime64[D]").astype(int)
    return np.sort(rng.integers(start, end, n)).astype(float)


def test_step_change_finds_the_drop():
    x = _dates(300)
    cut = float(np.datetime64("2023-07-01").astype("datetime64[D]").astype(int))
    y = np.where(x < cut, 1.0, 0.0)
    r = stats.step_change(x, y, np.arange(300), B=200)
    # breakpoint = first item date after the drop, so it (and its interval) can sit a few days after the true cut
    assert abs(r["breakpoint"] - cut) <= 45 and r["lo"] - 45 <= cut <= r["hi"] + 45
    assert r["step"] == pytest.approx(1.0) and r["step_lo"] > 0


def test_step_change_without_signal_has_no_clear_drop():
    x = _dates(300, seed=1)
    y = np.random.default_rng(2).integers(0, 2, 300).astype(float)
    assert stats.step_change(x, y, np.arange(300), B=200)["step_lo"] <= 0


def test_step_change_needs_enough_items():
    assert stats.step_change(np.arange(10.0), np.ones(10), np.arange(10), min_side=15) is None


# ---------------------------------------------------------------- vectorised bootstrap: same numbers, much faster
def _reference_draws(scores, group, clusters, B, seed):
    """The original one-draw-at-a-time implementation, kept here as the reference."""
    s, g, c = np.asarray(scores, float), np.asarray(group, int), np.asarray(clusters)
    keep = g >= 0
    s, g, c = s[keep], g[keep], c[keep]
    uniq, inv = np.unique(c, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        pre, post = s[idx][g[idx] == 1], s[idx][g[idx] == 0]
        if len(pre) and len(post):
            out.append(round(float(pre.mean() - post.mean()), 12))
    return np.asarray(out)


def test_vectorised_cluster_bootstrap_matches_the_reference():
    rng = np.random.default_rng(3)
    s = rng.random(120).round(2)
    g = rng.choice([1, 0, -1], 120, p=[0.5, 0.4, 0.1])
    c = rng.integers(0, 25, 120)
    new = stats._cluster_draws(s, g, c, B=400, seed=7)
    ref = _reference_draws(s, g, c, B=400, seed=7)
    assert len(new) == len(ref) and np.allclose(new, ref, atol=1e-9)


def test_vectorised_bootstrap_mean_matches_the_reference():
    rng = np.random.default_rng(4)
    s, c = rng.random(80), rng.integers(0, 15, 80)
    uniq, inv = np.unique(c, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    r = np.random.default_rng(0)
    ref = [s[np.concatenate([members[k] for k in r.integers(0, len(uniq), len(uniq))])].mean() for _ in range(300)]
    lo, hi = stats.bootstrap_mean(s, c, B=300)
    assert lo == pytest.approx(np.quantile(ref, 0.025)) and hi == pytest.approx(np.quantile(ref, 0.975))
