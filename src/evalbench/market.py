"""Market-level memory (M14/M15) for the Market page and the report: each model's answers next to the true index path,
and how closely they track it over time (docs/preregistration/2026-10-01-index.md §6.3)."""
from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import stats
from .cutoffs import BUFFER_DAYS
from .items import Item

LEVEL_METHODS = ("index_level", "us_index_level")


def sides(cutoff: date, release: date) -> tuple[pd.Period, pd.Period]:
    """(last BEFORE month, first AFTER month) of the index plans: the last whole month before cutoff - 91 days; the
    month after the release month."""
    edge = cutoff - timedelta(days=BUFFER_DAYS)
    last = pd.Period(edge, "M") if (edge + timedelta(days=1)).day == 1 else pd.Period(edge, "M") - 1
    return last, pd.Period(release, "M") + 1


def corr_drop_test(before: pd.DataFrame, after: pd.DataFrame, B: int = 2000, seed: int = 0) -> dict:
    """r_before - r_after with a cluster bootstrap by month on each side (frames: month, answer, truth)."""
    nan = float("nan")
    r_b = corr(before["answer"], before["truth"]) if len(before) else nan
    r_a = corr(after["answer"], after["truth"]) if len(after) else nan
    if before.empty or after.empty:
        return {"r_before": r_b, "r_after": r_a, "diff": nan, "lo": nan, "hi": nan, "p": nan}
    d = boot_corr(before, B=B, seed=seed) - boot_corr(after, B=B, seed=seed + 1)
    d = d[~np.isnan(d)]
    if not len(d):
        return {"r_before": r_b, "r_after": r_a, "diff": r_b - r_a, "lo": nan, "hi": nan, "p": nan}
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"r_before": r_b, "r_after": r_a, "diff": r_b - r_a, "lo": float(lo), "hi": float(hi), "p": stats.boot_p(d)}


def answers_frame(calls: pd.DataFrame, items: list[Item], model: str, method: str) -> pd.DataFrame:
    """One row per answered question: month, symbol, truth (level or % change) and the model's answer."""
    by_id = {it.item_id: it for it in items}
    key = "level" if method in LEVEL_METHODS else "return_pct"
    sub = calls[(calls["model"] == model) & (calls["method"] == method)]
    rows = []
    for item_id, parsed in zip(sub["item_id"], sub["parsed"]):
        it = by_id.get(item_id)
        value = json.loads(parsed).get("value")
        if it is None or value is None:
            continue
        rows.append({"item_id": item_id, "month": pd.Period(it.payload["month"], "M"), "symbol": it.payload["symbol"],
                     "truth": it.truth[key], "answer": float(value)})
    return pd.DataFrame(rows, columns=["item_id", "month", "symbol", "truth", "answer"])


def corr(answers, truths) -> float:
    """Pearson correlation; answers that never vary carry no month-specific information (r = 0); NaN below 3 pairs."""
    a, t = np.asarray(answers, dtype=float), np.asarray(truths, dtype=float)
    if len(a) < 3:
        return float("nan")
    if a.std() == 0 or t.std() == 0:
        return 0.0
    return float(np.corrcoef(a, t)[0, 1])


def _month_sums(df: pd.DataFrame) -> np.ndarray:
    """Per month: n, sum a, sum t, sum a², sum t², sum a·t (the sufficient statistics of a correlation)."""
    a, t = df["answer"].to_numpy(float), df["truth"].to_numpy(float)
    parts = pd.DataFrame({"m": df["month"].to_numpy(), "n": 1.0, "a": a, "t": t, "aa": a * a, "tt": t * t, "at": a * t})
    return parts.groupby("m", sort=True)[["n", "a", "t", "aa", "tt", "at"]].sum().to_numpy()


def _corr_from_sums(s: np.ndarray) -> np.ndarray:
    """Correlations from summed statistics (one row per draw), with corr()'s rules: answers (or truths) that never
    vary give 0, fewer than 3 pairs give NaN."""
    n, sa, st, saa, stt, sat = s.T
    with np.errstate(invalid="ignore", divide="ignore"):
        va, vt = saa - sa * sa / n, stt - st * st / n
        cov = sat - sa * st / n
        flat = (va <= 1e-9 * np.maximum(saa, 1.0)) | (vt <= 1e-9 * np.maximum(stt, 1.0))
        r = np.where(flat, 0.0, cov / np.sqrt(np.where(flat, 1.0, va * vt)))
    return np.where(n < 3, np.nan, np.clip(r, -1.0, 1.0))


def boot_corr(df: pd.DataFrame, B: int = 500, seed: int = 0) -> np.ndarray:
    """B bootstrap correlations, resampling whole months with replacement (each month's questions together).
    Vectorised: a draw is a vector of month counts, so its sums are counts @ per-month sums."""
    sums = _month_sums(df)
    k = len(sums)
    counts = np.random.default_rng(seed).multinomial(k, np.full(k, 1.0 / k), size=B)
    return _corr_from_sums(counts @ sums)


def rolling_corr(df: pd.DataFrame, window: int = 12, min_n: int = 20, B: int = 500, seed: int = 0) -> pd.DataFrame:
    """The correlation over the questions in a `window`-month window around each month (m-5..m+6 for 12), with a 95%
    band from resampling whole months. Windows with fewer than min_n answers are left out."""
    if df.empty:
        return pd.DataFrame(columns=["month", "r", "lo", "hi", "n"])
    rng = np.random.default_rng(seed)
    offsets = range(-(window // 2 - 1), window // 2 + 1)
    by_month = {m: g for m, g in df.groupby("month")}
    sums = {m: s for m, s in zip(sorted(by_month), _month_sums(df))}
    rows = []
    for m in pd.period_range(df["month"].min(), df["month"].max(), freq="M"):
        months = [m + k for k in offsets if (m + k) in sums]
        n = int(sum(sums[x][0] for x in months))
        if n < min_n:
            continue
        part = pd.concat([by_month[x] for x in months])
        k = len(months)
        counts = rng.multinomial(k, np.full(k, 1.0 / k), size=B)
        draws = _corr_from_sums(counts @ np.array([sums[x] for x in months]))
        draws = draws[~np.isnan(draws)]
        lo, hi = np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan)
        rows.append({"month": str(m), "r": corr(part["answer"], part["truth"]), "lo": lo, "hi": hi, "n": n})
    return pd.DataFrame(rows, columns=["month", "r", "lo", "hi", "n"])


def repeat_share(df: pd.DataFrame, window: int = 12, min_n: int = 20) -> pd.DataFrame:
    """Stock answers: in a `window`-month window around each month (as rolling_corr), the share of the answers that
    are the window's most repeated value (rounded to 0.01), that value, and how many different answers there were.
    A model that knows the months gives varied answers (low share); one that does not repeats a default."""
    cols = ["month", "share", "value", "distinct", "n"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    offsets = range(-(window // 2 - 1), window // 2 + 1)
    rounded = df.assign(a=df["answer"].astype(float).round(2))
    rows = []
    for m in pd.period_range(df["month"].min(), df["month"].max(), freq="M"):
        part = rounded[rounded["month"].isin([m + k for k in offsets])]
        if len(part) < min_n:
            continue
        vc = part["a"].value_counts()
        rows.append({"month": str(m), "share": float(vc.iloc[0]) / len(part), "value": float(vc.index[0]),
                     "distinct": int(len(vc)), "n": len(part)})
    return pd.DataFrame(rows, columns=cols)


PATH_ROOM = {"level": 1.25, "change": 10.0}   # level: 25% off, where M14/M16 closeness reaches 0


def path_scale(truth: pd.Series, answers: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, float, float]:
    """The y range of an answers-vs-path chart and where to draw each answer. The range covers the real path and the
    answers, but reaches at most PATH_ROOM beyond the path (levels: 25%, where an answer scores 0; changes: 10 points), so one wild
    answer (a misread "500", a wrong scale) cannot flatten the path. Answers beyond are drawn at the edge
    ('shown') and marked ('off': 'above' / 'below'); 'answer' keeps the real value for the tooltip."""
    t_lo, t_hi = float(truth.min()), float(truth.max())
    a = answers["answer"].astype(float)
    room = PATH_ROOM[kind]
    floor, ceil = (t_lo / room, t_hi * room) if kind == "level" else (t_lo - room, t_hi + room)
    lo = max(min(t_lo, float(a.min())) if len(a) else t_lo, floor)
    hi = min(max(t_hi, float(a.max())) if len(a) else t_hi, ceil)
    off = np.where(a > hi, "above", np.where(a < lo, "below", ""))
    return answers.assign(shown=a.clip(lo, hi), off=off), lo, hi
