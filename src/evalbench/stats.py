"""Pre-registered statistics for memorization scores.

group arrays: 1 = PRE (before the model's cutoff), 0 = POST, -1 = excluded (buffer / no cutoff).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, fisher_exact

NAN = float("nan")


def _arrays(scores, group) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(scores, dtype=float), np.asarray(group, dtype=int)


def delta(scores, group) -> float:
    """Mean score on PRE items minus mean score on POST items."""
    s, g = _arrays(scores, group)
    pre, post = s[g == 1], s[g == 0]
    if len(pre) == 0 or len(post) == 0:
        return NAN
    # rounded, so that equal means give exactly 0 and not a float residue that a bootstrap would read as a sign
    return round(float(pre.mean() - post.mean()), 12)


def kappa_delta(d: float, p0: float) -> float:
    """Chance-corrected delta: (acc_pre - p0)/(1 - p0) - (acc_post - p0)/(1 - p0) = delta / (1 - p0)."""
    return d / (1.0 - p0)


def _resample_counts(n_clusters: int, B: int, seed: int) -> np.ndarray:
    """B x K matrix: how often each of K clusters is drawn in each bootstrap replicate. Uses the same random stream as
    drawing K cluster indices B times in a row, so results match the original one-draw-at-a-time implementation."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, (B, n_clusters))
    flat = (idx + np.arange(B)[:, None] * n_clusters).ravel()
    return np.bincount(flat, minlength=B * n_clusters).reshape(B, n_clusters).astype(float)


def _cluster_draws(scores, group, clusters, B: int, seed: int) -> np.ndarray:
    """Bootstrap distribution of delta, resampling whole clusters (companies/articles). Vectorised: per-cluster sums
    and counts times the resampling counts."""
    s, g = _arrays(scores, group)
    c = np.asarray(clusters)
    keep = g >= 0
    s, g, c = s[keep], g[keep], c[keep]
    if len(s) == 0:
        return np.array([])
    uniq, inv = np.unique(c, return_inverse=True)
    k = len(uniq)
    pre, post = (g == 1).astype(float), (g == 0).astype(float)
    sums = np.stack([np.bincount(inv, weights=s * pre, minlength=k), np.bincount(inv, weights=s * post, minlength=k)])
    ns = np.stack([np.bincount(inv, weights=pre, minlength=k), np.bincount(inv, weights=post, minlength=k)])
    counts = _resample_counts(k, B, seed)
    tot, n = counts @ sums.T, counts @ ns.T                     # B x 2 (pre, post)
    ok = (n[:, 0] > 0) & (n[:, 1] > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = tot[ok, 0] / n[ok, 0] - tot[ok, 1] / n[ok, 1]
    return np.round(d, 12)


def bootstrap_delta(scores, group, clusters, B: int = 2000, seed: int = 0, alpha: float = 0.05):
    """(ci_lo, ci_hi, p) from one cluster bootstrap. p is two-sided, so p < alpha exactly when the
    (1 - alpha) percentile CI excludes 0: a significance star can never disagree with the interval."""
    draws = _cluster_draws(scores, group, clusters, B, seed)
    if len(draws) == 0:
        return NAN, NAN, NAN
    lo, hi = float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))
    return lo, hi, boot_p(draws, alpha)


def boot_p(draws, alpha: float = 0.05) -> float:
    """Two-sided bootstrap p for 'the statistic differs from 0', the same rule for every test the app shows: never 0
    (at least 2/(B+1)), and below alpha exactly when the (1 - alpha) percentile interval excludes 0."""
    draws = np.asarray(draws, dtype=float)
    draws = draws[~np.isnan(draws)]
    if not len(draws):
        return NAN
    n = len(draws)
    below, above = int((draws <= 0).sum()), int((draws >= 0).sum())
    lo, hi = float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))
    if lo > 0 or hi < 0:
        return float(min(2 * (min(below, above) + 1) / (n + 1), alpha * (1 - 1e-9)))
    return float(max(min(1.0, 2 * min(below, above) / n), alpha))


def bootstrap_mean(scores, clusters, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI of a mean, resampling whole clusters (companies/articles)."""
    s, c = np.asarray(scores, dtype=float), np.asarray(clusters)
    if len(s) == 0:
        return NAN, NAN
    uniq, inv = np.unique(c, return_inverse=True)
    k = len(uniq)
    counts = _resample_counts(k, B, seed)
    draws = (counts @ np.bincount(inv, weights=s, minlength=k)) / (counts @ np.bincount(inv, minlength=k))
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def bootstrap_delta_ci(scores, group, clusters, B: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of delta from a cluster bootstrap (resample whole companies/articles)."""
    lo, hi, _ = bootstrap_delta(scores, group, clusters, B, seed, alpha)
    return lo, hi


def permutation_pvalue(scores, group, n_perm: int = 2000, seed: int = 0) -> float:
    """One-sided p for delta > 0 by shuffling PRE/POST labels: (1 + #{perm >= obs}) / (1 + n_perm)."""
    s, g = _arrays(scores, group)
    keep = g >= 0
    s, g = s[keep], g[keep]
    obs = delta(s, g)
    if np.isnan(obs):
        return NAN
    n_pre = int((g == 1).sum())
    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((n_perm, len(s))), axis=1)  # vectorised: one permutation per row
    shuffled = s[perms]
    d = shuffled[:, :n_pre].mean(axis=1) - shuffled[:, n_pre:].mean(axis=1)
    count = int((d >= obs - 1e-12).sum())
    return (1 + count) / (1 + n_perm)


def binom_greater(k: int, n: int, p0: float) -> float:
    if n == 0:
        return NAN
    return float(binomtest(int(k), int(n), p0, alternative="greater").pvalue)


def fisher_greater(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided Fisher exact test that the rate k1/n1 exceeds k2/n2."""
    if n1 == 0 or n2 == 0:
        return NAN
    return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]], alternative="greater").pvalue)


def mcnemar_exact(a_correct, b_correct) -> float:
    """Two-sided exact McNemar test on paired binary outcomes."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(min(n01, n10), n01 + n10, 0.5, alternative="two-sided").pvalue)


def paired_bootstrap_diff(scores_a, group_a, scores_b, group_b, B: int = 2000, seed: int = 0, alpha: float = 0.05):
    """delta_A - delta_B on the same items (each model keeps its own split), resampling items jointly."""
    sa, ga = _arrays(scores_a, group_a)
    sb, gb = _arrays(scores_b, group_b)
    point = delta(sa, ga) - delta(sb, gb)
    n = len(sa)
    if n == 0:
        return NAN, NAN, NAN
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        d = delta(sa[idx], ga[idx]) - delta(sb[idx], gb[idx])
        if not np.isnan(d):
            draws.append(d)
    if not draws:
        return point, NAN, NAN
    return point, float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values; NaNs stay NaN and do not count towards m."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), NAN)
    valid = np.flatnonzero(~np.isnan(p))
    m = len(valid)
    running = 0.0
    for rank, i in enumerate(valid[np.argsort(p[valid])]):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def _best_split(x: np.ndarray, y: np.ndarray, min_side: int):
    """Two-level step fit: the split (between distinct x values) minimising squared error.
    Returns (last x before, first x after, mean before - mean after) or None if there are too few points.
    The drop is only located to within that gap between neighbouring items."""
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = len(ys)
    if n < 2 * min_side:
        return None
    cs, cs2 = np.cumsum(ys), np.cumsum(ys * ys)
    k = np.arange(min_side, n - min_side + 1)          # size of the "before" part
    left, left2 = cs[k - 1], cs2[k - 1]
    right, right2 = cs[-1] - left, cs2[-1] - left2
    sse = (left2 - left * left / k) + (right2 - right * right / (n - k))
    sse = np.where(xs[k - 1] < xs[np.minimum(k, n - 1)], sse, np.inf)  # never split inside a tie
    if not np.isfinite(sse).any():
        return None
    j = int(np.argmin(sse))
    kk = int(k[j])
    return float(xs[kk - 1]), float(xs[kk]), float(left[j] / kk - right[j] / (n - kk))


def step_change(x, y, clusters, min_side: int = 15, B: int = 500, seed: int = 0, alpha: float = 0.05):
    """Where does the score drop? Best single breakpoint in x (e.g. item date in days) with cluster-bootstrap
    intervals for the breakpoint and for the size of the drop. None if fewer than 2 x min_side points."""
    x, y, c = np.asarray(x, float), np.asarray(y, float), np.asarray(clusters)
    best = _best_split(x, y, min_side)
    if best is None:
        return None
    uniq, inv = np.unique(c, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    befores, afters, steps = [], [], []
    for _ in range(B):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        r = _best_split(x[idx], y[idx], min_side)
        if r is not None:
            befores.append(r[0])
            afters.append(r[1])
            steps.append(r[2])
    q = lambda v, a: float(np.quantile(v, a)) if v else NAN  # noqa: E731
    return {"breakpoint": (best[0] + best[1]) / 2, "last_before": best[0], "first_after": best[1],
            "lo": q(befores, alpha / 2), "hi": q(afters, 1 - alpha / 2), "step": best[2],
            "step_lo": q(steps, alpha / 2), "step_hi": q(steps, 1 - alpha / 2), "n": len(y)}
