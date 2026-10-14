"""The cutoff ladder (docs/preregistration/2026-10-05-cutoff-ladder.md): models whose knowledge ends at different
dates answer the same questions. A month one model may know and another cannot separates memory from a time effect
shared by both (purpose A); the earliest models give the longest clean backtest windows (purpose B)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import analysis, market
from .index_report import MARKETS, PILOT
from .items import load_testset
from .pilot_d_report import LLAMA, RELEASE as PILOT_RELEASE
from .pilot_e import with_score

GPT35 = "openai/gpt-3.5-turbo"
GPT4OMINI = "openai/gpt-4o-mini-2024-07-18"
LLAMA31 = "meta-llama/llama-3.1-70b-instruct"
NEW = [GPT35, GPT4OMINI, LLAMA31]
# late references: each detected on M5 (grid v4) and M15 in its registered report
REFERENCES = [LLAMA, "openai/gpt-4.1-mini", "google/gemini-2.5-flash", "deepseek/deepseek-chat-v3.1"]
REFERENCE_DETECTED = {(m, method) for m in REFERENCES for method in ("price_level", "index_return")}
# GPT-3.5: the snapshot served now (gpt-3.5-turbo-0125), not the 2023-05-28 listing (plan §2)
RELEASE = {**PILOT_RELEASE, GPT35: date(2024, 1, 25), GPT4OMINI: date(2024, 7, 18), LLAMA31: date(2024, 7, 23)}
LISTED = {GPT35: date(2023, 5, 28)}      # sensitivity analysis only (plan §5)
MIN_MONTHS = 6
DATA_END = pd.Period("2026-08", "M")


def last_before(cutoff: date) -> pd.Period:
    """The last whole month before cutoff - 91 days (the BEFORE edge of every registered report)."""
    return market.sides(cutoff, cutoff)[0]


def first_after(release: date) -> pd.Period:
    return pd.Period(release, "M") + 1


def pair_months(early_cutoff: date, early_release: date, late_cutoff: date, months) -> tuple[list, list]:
    """between: after the early model's release month, up to the late model's last BEFORE month (the late model may
    know them, the early one cannot). baseline: months both may know."""
    months = sorted(set(months))
    hi = last_before(late_cutoff)
    between = [m for m in months if first_after(early_release) <= m <= hi]
    baseline = [m for m in months if m <= min(last_before(early_cutoff), hi)]
    return between, baseline


def usable(between, baseline, min_months: int = MIN_MONTHS) -> bool:
    return len(between) >= min_months and len(baseline) >= min_months


def release_months(early_release: date, late_release: date, cutoff: date, months) -> tuple[list, list]:
    """Two models with the same cutoff, released at different dates: between = after the early release, before the
    late release month; baseline = both models' BEFORE months."""
    months = sorted(set(months))
    between = [m for m in months if first_after(early_release) <= m < pd.Period(late_release, "M")]
    baseline = [m for m in months if m <= last_before(cutoff)]
    return between, baseline


def clean_months(release: date, data_end: pd.Period = DATA_END) -> int:
    """Whole months after the release month that our data covers: the clean backtest window."""
    return max(0, data_end.ordinal - first_after(release).ordinal + 1)


def grey_bins(cutoff: date, release: date, size: int = 6) -> list[list[pd.Period]]:
    """The months between the BEFORE edge and the AFTER edge, in bins of `size` months."""
    months = list(pd.period_range(last_before(cutoff) + 1, first_after(release) - 1, freq="M"))
    return [months[i:i + size] for i in range(0, len(months), size)]


def grid_months() -> list[pd.Period]:
    return sorted({pd.Period(it.date[:7], "M") for it in load_testset("price_level", "v4")})


def _index_items(method: str) -> list:
    """An index set's questions without its pilot items (the 24 Nordic pilot items of 2026-10-01; the US sets had no
    pilot), as index_report.MARKETS registers them."""
    skip = next(s for level, ret, s in MARKETS.values() if method in (level, ret))
    return load_testset(method, "v1")[skip:]


def index_months(method: str = "index_return") -> list[pd.Period]:
    return sorted({pd.Period(it.payload["month"], "M") for it in _index_items(method)})


def index_frame(calls: pd.DataFrame, model: str, method: str = "index_return") -> pd.DataFrame:
    """One row per answered index question (pilot items left out)."""
    items = _index_items(method)
    return market.answers_frame(calls[calls["testset"] == f"{method}/v1"], items, model, method)


def grid_scores(calls: pd.DataFrame, model: str, cutoff: date) -> pd.DataFrame:
    """Closeness per price_level/v4 question (refused = 0), with its month; fabricated items left out."""
    items = {it.item_id: it for it in load_testset("price_level", "v4")}
    sub = calls[(calls["testset"] == "price_level/v4") & (calls["model"] == model)]
    df = analysis.item_scores(sub, items, "price_level", cutoff)
    if df.empty:
        return df
    df = with_score(df[~df["fabricated"].astype(bool)], "price_level", "primary")
    return df.assign(month=pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M"))


def _aligned_sums(df: pd.DataFrame, months: list) -> np.ndarray:
    """market._month_sums on a fixed month order; a month without answers is a row of zeros."""
    out = np.zeros((len(months), 6))
    pos = {m: i for i, m in enumerate(months)}
    sub = df[df["month"].isin(pos)]
    if len(sub):
        for m, row in zip(sorted(sub["month"].unique()), market._month_sums(sub)):
            out[pos[m]] = row
    return out


def staggered_corr(late: pd.DataFrame, early: pd.DataFrame, between: list, baseline: list, B: int = 20000,
                   seed: int = 0) -> dict:
    """D = (r_late - r_early) on between months - (r_late - r_early) on baseline months, over the questions both
    models answered (frames as from index_frame). Months are resampled together for both models (paired cluster
    bootstrap); two-sided p = 2 x the smaller tail beyond 0. Memory predicts D > 0; a time effect shared by both
    cancels."""
    both = set(late["item_id"]) & set(early["item_id"])
    frames = {"late": late[late["item_id"].isin(both)], "early": early[early["item_id"].isin(both)]}
    sides = {"between": list(between), "baseline": list(baseline)}
    sums = {(w, s): _aligned_sums(frames[w], sides[s]) for w in frames for s in sides}
    r = {k: float(market._corr_from_sums(v.sum(axis=0, keepdims=True))[0]) for k, v in sums.items()}
    out = {"r_late_between": r[("late", "between")], "r_early_between": r[("early", "between")],
           "r_late_baseline": r[("late", "baseline")], "r_early_baseline": r[("early", "baseline")],
           "n_between": int(sums[("late", "between")][:, 0].sum()),
           "n_baseline": int(sums[("late", "baseline")][:, 0].sum())}
    nan = float("nan")
    if not between or not baseline:
        return {**out, "D": nan, "lo": nan, "hi": nan, "p": nan}
    out["D"] = (r[("late", "between")] - r[("early", "between")]) - (r[("late", "baseline")] - r[("early", "baseline")])
    rng = np.random.default_rng(seed)
    counts = {s: rng.multinomial(len(m), np.full(len(m), 1.0 / len(m)), size=B) for s, m in sides.items()}
    rb = {k: market._corr_from_sums(counts[k[1]] @ v) for k, v in sums.items()}
    draws = (rb[("late", "between")] - rb[("early", "between")]) - (rb[("late", "baseline")] - rb[("early", "baseline")])
    draws = draws[~np.isnan(draws)]
    if not len(draws):
        return {**out, "lo": nan, "hi": nan, "p": nan}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return {**out, "lo": float(lo), "hi": float(hi), "p": float(p)}
