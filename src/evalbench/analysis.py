"""From stored calls to item scores, per-model summaries, model comparisons and time profiles."""
from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from . import market, stats
from .cutoffs import BUFFER_DAYS, group_for
from .items import Item
from .methods import METHODS

NAN = float("nan")
BINARY_METHODS = ("price_recall", "number_cloze", "price_level", "annual_return", "known_return", "split_clock",
                  "known_level", "us_level", "monthly_return")


# Return questions where an answer of exactly 0% is an abstention, not a direction (scoring spec §7, 2026-10-06). M7
# (known returns), a registered positive control where it is rare (0-3%), keeps its registered score; M15/M17 keep
# theirs because their registered test correlates the answers themselves, where 0% is a prediction.
# Price questions where a price of 0 is an abstention (no share trades at 0; scoring spec §10, 2026-10-07): M5 and
# M10. M9 (known levels), a registered positive control, keeps its registered score.
HEDGE_METHODS = {"monthly_return", "annual_return", "price_level", "us_level"}
ZERO_LABEL = {"price_level": "Answers of 0", "us_level": "Answers of 0"}


def _hedged(df: pd.DataFrame) -> pd.Series:
    flag = pd.Series(False, index=df.index)
    for col in ("no_direction", "no_answer"):       # a 0% return (§7) or a price of 0 (§10)
        if col in df:
            flag |= df[col].fillna(False).astype(bool)
    return flag & ~df["refused"].astype(bool)


def drop_hedges(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Leave out the answers of exactly 0% ("I don't know"), as the refusal guard leaves out refusals: scoring them 0.5
    pins a score at chance and makes its interval look tighter than the evidence. Refusals stay (the guard needs
    them)."""
    if method not in HEDGE_METHODS or not len(df):
        return df
    return df[~_hedged(df)]


def hedge_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Real answered items with score = 1 for an answer of exactly 0%, else 0: the 'I don't know' rate over time."""
    if not len(df):
        return df
    a = df[~df["fabricated"].astype(bool) & ~df["refused"].astype(bool)]
    return a.assign(score=_hedged(a).astype(float))


def item_scores(calls: pd.DataFrame, items_by_id: dict[str, Item], method: str, cutoff,
                extra_strata: dict[str, dict] | None = None) -> pd.DataFrame:
    """One row per item with all variants answered; calls must be for a single model.
    extra_strata: {name: {cluster: value}} adds s_<name> columns (default "unknown")."""
    if calls.empty:
        return pd.DataFrame()
    mod = METHODS[method]
    rows = []
    sub = calls[calls["method"] == method]
    # grouped in plain lists: iterating pandas groups one item at a time took ~40 s for the Overview's 54 cells
    n = len(sub)
    samples = sub["sample"].tolist() if "sample" in sub else [0] * n
    responses_all = sub["response"].tolist() if "response" in sub else [""] * n
    by_item: dict = {}
    for item_id, sample, variant, p, r in zip(sub["item_id"].tolist(), samples, sub["variant"].tolist(),
                                              sub["parsed"].tolist(), responses_all):
        by_item.setdefault(item_id, {}).setdefault(sample, []).append((variant, p, r))
    for item_id in sorted(by_item):
        item = items_by_id.get(item_id)
        if item is None:
            continue
        attempts = []  # (sample, score, parsed) for every sample with all variants answered
        for sample in sorted(by_item[item_id]):
            parsed = {v: _with_outcome(mod, json.loads(p), r) for v, p, r in by_item[item_id][sample]}
            if set(mod.VARIANTS) <= parsed.keys():
                attempts.append((sample, mod.item_score(item, parsed), parsed))
        if not attempts or attempts[0][0] != 0:
            continue  # the headline is always sample 0 (the first attempt); without it the item is incomplete
        _, sc, parsed = attempts[0]
        scores = [a[1].score for a in attempts]
        rows.append({"item_id": item_id, "date": item.date, "cluster": item.cluster, "fabricated": item.fabricated,
                     "group": group_for(item.date, cutoff), "score": sc.score, "refused": sc.refused,
                     "score_best": max(scores), "score_mean": sum(scores) / len(scores), "n_samples": len(scores),
                     "outcome": item_outcome(method, parsed), **sc.extra,
                     **{f"s_{k}": v for k, v in item.strata.items()},
                     **{f"s_{k}": m.get(item.cluster, "unknown") for k, m in (extra_strata or {}).items()}})
    return pd.DataFrame(rows)


OUTCOMES = ("answer", "abstain", "refusal", "malformed")


def _with_outcome(mod, parsed: dict, response) -> dict:
    """Rows stored before outcomes existed: answered rows are answers; others are re-parsed from the raw reply."""
    if "outcome" in parsed:
        return parsed
    if not parsed.get("refused"):
        return {**parsed, "outcome": "answer"}
    try:
        outcome = mod.parse(response if isinstance(response, str) else "").get("outcome", "malformed")
    except Exception:  # a reply the parser cannot handle is malformed, never a crash of the analysis
        outcome = "malformed"
    return {**parsed, "outcome": outcome}


def item_outcome(method: str, parsed_by_variant: dict[str, dict]) -> str:
    """One outcome per item. M4 follows its guided answer; otherwise any answered variant makes it an answer,
    else the most common non-answer."""
    if method == "completion":
        return parsed_by_variant["guided"].get("outcome", "malformed")
    outs = [p.get("outcome", "malformed") for p in parsed_by_variant.values()]
    if "answer" in outs:
        return "answer"
    return max(OUTCOMES[1:], key=outs.count)


def _empty_summary() -> dict:
    keys = ("mean_pre", "mean_post", "delta", "dkappa", "ci_lo", "ci_hi", "p_value", "p_perm", "coverage_pre",
            "coverage_post", "answered_mean_pre", "answered_mean_post", "hallucination_rate", "known_p",
            "abstain_rate", "refusal_rate", "malformed_rate", "dkappa_best", "ci_lo_best", "ci_hi_best")
    return {"n_pre": 0, "n_post": 0, "n_excluded": 0, "n_samples": 1, "n_best": 0, **{k: NAN for k in keys}}


def summarize(df: pd.DataFrame, method: str, B: int = 2000, seed: int = 0) -> dict:
    out = _empty_summary()
    if df.empty:
        return out
    mod = METHODS[method]
    real = df[~df["fabricated"]]
    s, g = real["score"].to_numpy(float), real["group"].to_numpy(int)
    pre, post = real[real["group"] == 1], real[real["group"] == 0]
    d = stats.delta(s, g)
    lo, hi, p = stats.bootstrap_delta(s, g, real["cluster"].to_numpy(), B=B, seed=seed)
    out.update({
        "n_pre": len(pre), "n_post": len(post), "n_excluded": int((real["group"] == -1).sum()),
        "mean_pre": pre["score"].mean() if len(pre) else NAN,
        "mean_post": post["score"].mean() if len(post) else NAN,
        "delta": d, "dkappa": stats.kappa_delta(d, mod.P0),
        "ci_lo": stats.kappa_delta(lo, mod.P0), "ci_hi": stats.kappa_delta(hi, mod.P0),
        "p_value": p,  # cluster bootstrap, consistent with the CI
        "p_perm": stats.permutation_pvalue(s, g, n_perm=B, seed=seed),  # item-level, for reference
        "coverage_pre": 1 - pre["refused"].mean() if len(pre) else NAN,
        "coverage_post": 1 - post["refused"].mean() if len(post) else NAN,
        "answered_mean_pre": _answered_mean(pre),
        "answered_mean_post": _answered_mean(post),
    })
    if "outcome" in real and len(real):
        for name in ("abstain", "refusal", "malformed"):
            out[f"{name}_rate"] = float((real["outcome"] == name).mean())
    k = int(real["n_samples"].max()) if "n_samples" in real and len(real) else 1
    out["n_samples"] = k
    if k > 1:  # best of k, only over items that really have k attempts, reported next to the first attempt
        full = real[real["n_samples"] == k]
        gb = full["group"].to_numpy(int)
        best = full["score_best"].to_numpy(float)
        lo_b, hi_b, _ = stats.bootstrap_delta(best, gb, full["cluster"].to_numpy(), B=B, seed=seed)
        out.update({"dkappa_best": stats.kappa_delta(stats.delta(best, gb), mod.P0), "n_best": len(full),
                    "ci_lo_best": stats.kappa_delta(lo_b, mod.P0), "ci_hi_best": stats.kappa_delta(hi_b, mod.P0)})
    else:
        out.update({"dkappa_best": out["dkappa"], "ci_lo_best": out["ci_lo"], "ci_hi_best": out["ci_hi"],
                    "n_best": len(real)})
    fab = df[df["fabricated"]]
    if len(fab) and "hallucinated" in fab:
        out["hallucination_rate"] = float(fab["hallucinated"].astype(bool).mean())
    if method == "decop" and len(pre) and len(post):
        # a model with a consistent style preference can look "known" at any date, so compare against its own POST rate
        out["known_p"] = stats.fisher_greater(int(pre["known"].astype(bool).sum()), len(pre),
                                              int(post["known"].astype(bool).sum()), len(post))
    return out


def _answered_mean(df: pd.DataFrame) -> float:
    answered = df[~df["refused"].astype(bool)]
    return float(answered["score"].mean()) if len(answered) else NAN


BREAKDOWN_COLS = ["stratum", "n_pre", "n_post", "mean_pre", "mean_post", "dkappa", "ci_lo", "ci_hi", "p_value"]


def breakdown(df: pd.DataFrame, method: str, key: str, B: int = 2000) -> pd.DataFrame:
    """The memorization index within each value of one stratum (e.g. company size, language)."""
    col = f"s_{key}"
    if df.empty or col not in df:
        return pd.DataFrame(columns=BREAKDOWN_COLS)
    rows = [{"stratum": value, **summarize(part, method, B=B)}
            for value, part in df[~df["fabricated"]].groupby(col)]
    return pd.DataFrame(rows)[BREAKDOWN_COLS]


def summary_table(calls: pd.DataFrame, testsets: dict, models, methods, cutoffs: dict, B: int = 2000) -> pd.DataFrame:
    frames = {}
    for model in models:
        mc = calls[calls["model"] == model] if len(calls) else calls
        for method in methods:
            name, items = testsets[method]
            sub = mc[mc["testset"] == name] if len(mc) else mc
            frames[(model, method)] = item_scores(sub, {it.item_id: it for it in items}, method, cutoffs.get(model))
    return summary_from_frames(frames, B=B, family="all")   # pilot C: Holm across all 16 cells


CHANGE_REGISTERED = ("index_return", "us_index_return")
LEVEL_REGISTERED = ("index_level", "us_index_level")
INDEX_REGISTERED = CHANGE_REGISTERED + LEVEL_REGISTERED
# M18 (headline dating plan §3): model - blind baseline on BEFORE months, like the index levels. Its generic
# before/after comparison is biased by construction: after its horizon a model cannot name a later month.
DATING_REGISTERED = ("headline_date",)
BLIND_REGISTERED = LEVEL_REGISTERED + DATING_REGISTERED
REGISTERED = INDEX_REGISTERED + DATING_REGISTERED
# M5 in the app uses the registered windows of the grid plan (§2): the 12 months ending at the BEFORE edge against the
# first 12 months after the release month, not every month on each side.
GRID_REGISTERED = ("price_level",)


def grid_groups(df: pd.DataFrame, cutoff, release) -> pd.Series:
    """1 / 0 / -1 by the grid plan's windows (12 months each side of the line), from each item's date."""
    last, first = market.sides(cutoff, release)
    months = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    before = (months >= last - 11) & (months <= last)
    after = (months >= first) & (months <= first + 11)
    return pd.Series(np.where(before, 1, np.where(after, 0, -1)), index=df.index)


def registered_window(cutoff, release, line: str) -> tuple:
    """(cutoff, end of the grey zone) for registered_groups: on the release line the release date; on the cutoff
    line cutoff + 91 days, so AFTER starts the month after it (the cutoff line's whole-month analogue)."""
    if line == "release":
        return cutoff, release
    return cutoff, cutoff + timedelta(days=BUFFER_DAYS)


def registered_groups(df: pd.DataFrame, cutoff, release) -> pd.Series:
    """1 / 0 / -1 (before / after / left out) by the index plans' windows (market.sides), from each item's date."""
    last, first = market.sides(cutoff, release)
    months = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    return pd.Series(np.where(months <= last, 1, np.where(months >= first, 0, -1)), index=df.index)


def registered_index_cell(df: pd.DataFrame, method: str, blind: dict | None = None, B: int = 2000,
                          seed: int = 0) -> dict:
    """A summary row by the registered test (index plan §6.3, US plan §4) instead of the generic before/after score.
    Changes (M15/M17): dkappa = r_before - r_after, the correlation of the answers with the true changes (answered
    questions only; answers that never vary: 0). Levels (M14/M16): dkappa = mean(model - blind closeness) on BEFORE
    months; the after side is reported (d_after) but not tested. Cluster bootstrap by month."""
    out = _empty_summary()
    if df.empty:
        return out
    real = df[~df["fabricated"]]
    pre, post = real[real["group"] == 1], real[real["group"] == 0]
    out.update({"n_pre": len(pre), "n_post": len(post), "n_excluded": int((real["group"] == -1).sum()),
                "coverage_pre": 1 - pre["refused"].mean() if len(pre) else NAN,
                "coverage_post": 1 - post["refused"].mean() if len(post) else NAN})
    if "outcome" in real and len(real):
        for name in ("abstain", "refusal", "malformed"):
            out[f"{name}_rate"] = float((real["outcome"] == name).mean())
    if method in CHANGE_REGISTERED:
        def answered(d: pd.DataFrame) -> pd.DataFrame:
            d = d[~d["refused"].astype(bool)]
            if "answer" not in d:          # no answered question at all: the column never got created
                return pd.DataFrame(columns=["month", "answer", "truth"])
            return d.dropna(subset=["answer"]).assign(month=d["cluster"])
        r = market.corr_drop_test(answered(pre), answered(post), B=B, seed=seed)
        out.update({"registered": "correlation drop", "mean_pre": r["r_before"], "mean_post": r["r_after"],
                    "delta": r["diff"], "dkappa": r["diff"], "ci_lo": r["lo"], "ci_hi": r["hi"], "p_value": r["p"]})
        return out
    blind = blind or {}
    d_pre = (pre["score"] - pre["item_id"].map(blind)).dropna()
    if not len(d_pre):
        return out
    rng = np.random.default_rng(seed)
    g = d_pre.groupby(pre.loc[d_pre.index, "cluster"])
    sums, ns = g.sum().to_numpy(), g.size().to_numpy(float)
    counts = rng.multinomial(len(sums), np.full(len(sums), 1.0 / len(sums)), size=B)
    draws = (counts @ sums) / (counts @ ns)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    d_post = (post["score"] - post["item_id"].map(blind)).dropna()
    out.update({"registered": "model − blind, before", "mean_pre": pre["score"].mean(),
                "mean_post": post["score"].mean() if len(post) else NAN, "delta": float(d_pre.mean()),
                "dkappa": float(d_pre.mean()), "ci_lo": float(lo), "ci_hi": float(hi),
                "p_value": stats.boot_p(draws),
                "d_after": float(d_post.mean()) if len(d_post) else NAN})
    return out


def summary_from_frames(frames: dict, B: int = 2000, blind: dict | None = None, family: str = "method") -> pd.DataFrame:
    """One row per (model, method) from already computed item scores; Holm across all rows. The index methods use
    their registered test (registered_index_cell); blind: method -> {item_id: blind-guesser score} for the levels."""
    def row(method: str, df: pd.DataFrame) -> dict:
        if method in REGISTERED:
            return registered_index_cell(df, method, blind=(blind or {}).get(method), B=B)
        return summarize(df, method, B=B)
    rows = [{"model": model, "method": method, **row(method, df)} for (model, method), df in frames.items()]
    out = pd.DataFrame(rows)
    if len(out) and family == "all":   # pilots C and D registered Holm across every cell of their table
        out["p_holm"] = stats.holm(out["p_value"].to_numpy(float))
    elif len(out):   # Holm within each method across the models shown, as in the later registered reports
        out["p_holm"] = out.groupby("method")["p_value"].transform(lambda s: stats.holm(s.to_numpy(float)))
    return out


def compare(df_a: pd.DataFrame, df_b: pd.DataFrame, method: str, B: int = 2000) -> tuple[dict, pd.DataFrame]:
    """delta_A - delta_B on shared items (chance-corrected), plus McNemar on items PRE for both."""
    p0 = METHODS[method].P0
    m = df_a.merge(df_b, on="item_id", suffixes=("_a", "_b"))
    m = m[~m["fabricated_a"]]
    point, lo, hi = stats.paired_bootstrap_diff(m["score_a"], m["group_a"], m["score_b"], m["group_b"], B=B)
    res = {"n_items": len(m), "diff": stats.kappa_delta(point, p0), "ci_lo": stats.kappa_delta(lo, p0),
           "ci_hi": stats.kappa_delta(hi, p0), "mcnemar_p": NAN, "n_both_pre": 0}
    if method in BINARY_METHODS:
        both = (m["group_a"] == 1) & (m["group_b"] == 1)
        res["n_both_pre"] = int(both.sum())
        col = "hit_10pct" if "hit_10pct_a" in m else "score"   # price levels: right/wrong is the ±10% hit
        if res["n_both_pre"]:   # no shared BEFORE items: no test, not p = 1
            res["mcnemar_p"] = stats.mcnemar_exact(m.loc[both, f"{col}_a"] >= 0.5, m.loc[both, f"{col}_b"] >= 0.5)
    return res, m


def by_quarter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["quarter", "mean", "n"])
    real = df[~df["fabricated"]].copy()
    real["quarter"] = pd.PeriodIndex(pd.to_datetime(real["date"]), freq="Q").astype(str)
    q = real.groupby("quarter")["score"].agg(["mean", "count"]).reset_index()
    return q.rename(columns={"count": "n"})[["quarter", "mean", "n"]]


def reading(row) -> str:
    """Plain-language result for one model x method cell; 'at most X' when nothing is detected."""
    d, lo, hi, p = row["dkappa"], row["ci_lo"], row["ci_hi"], row["p_holm"]
    if pd.isna(d):
        return "not scorable"
    if p < 0.05 and d > 0:
        return f"memorization detected: Δκ {d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    if p < 0.05:
        return f"better after the cutoff: Δκ {d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    return f"none detected: at most {hi:+.2f}"


def effective_cutoff(parts: list[tuple[pd.DataFrame, float]], min_side: int = 15, B: int = 500) -> dict:
    """Where a model's scores actually drop, pooling methods on the chance-corrected scale (s - p0) / (1 - p0).
    Independent of the stated cutoff. 'detected' only when the drop's 95% interval is above zero."""
    frames = [df[~df["fabricated"]].assign(y=(df.loc[~df["fabricated"], "score"] - p0) / (1 - p0))
              for df, p0 in parts if len(df)]
    if not frames:
        return {"detected": False, "n": 0}
    d = pd.concat(frames, ignore_index=True)
    x = pd.to_datetime(d["date"]).map(pd.Timestamp.toordinal).to_numpy(float)
    r = stats.step_change(x, d["y"].to_numpy(float), d["cluster"].to_numpy(), min_side=min_side, B=B)
    if r is None:
        return {"detected": False, "n": len(d)}
    as_date = lambda v: pd.Timestamp.fromordinal(int(v)).date() if pd.notna(v) else None  # noqa: E731
    return {"detected": bool(r["step_lo"] > 0), "breakpoint": as_date(r["breakpoint"]),
            "last_known": as_date(r["last_before"]), "first_unknown": as_date(r["first_after"]),
            "lo": as_date(r["lo"]), "hi": as_date(r["hi"]), "step": r["step"], "step_lo": r["step_lo"],
            "step_hi": r["step_hi"], "n": r["n"]}


def versus_stated(eff: dict, stated) -> str | None:
    """Compare the estimated knowledge boundary with the stated cutoff. The boundary lies somewhere in the gap
    between the last item still known and the first not known; 'lo'/'hi' widen that gap to a 95% interval."""
    if not eff.get("detected") or stated is None:
        return None
    if eff["hi"] < stated:
        return "earlier than stated"
    if eff["lo"] > stated:
        return "later than stated"
    return "consistent with stated"


def chance_corrected(df: pd.DataFrame) -> pd.DataFrame:
    """Scores on a common scale for pooling methods: (score - p0) / (1 - p0); 0 = guessing, 1 = always right.
    Needs a p0 column (each item's method chance level)."""
    return df.assign(score=(df["score"] - df["p0"]) / (1 - df["p0"]))


def rolling_curve(df: pd.DataFrame, window: int = 6, min_n: int = 15, B: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean item score in a `window`-month window moved one month at a time (the window for month m covers
    m-2..m+3 for 6 months), with a cluster-bootstrap 95% band. Windows with fewer than min_n real items are left
    out. Columns: month ('YYYY-MM', the window's anchor), mean, lo, hi, n."""
    if df.empty:
        return pd.DataFrame(columns=["month", "mean", "lo", "hi", "n"])
    real = df[~df["fabricated"].astype(bool)].copy()
    if real.empty:
        return pd.DataFrame(columns=["month", "mean", "lo", "hi", "n"])
    real["month"] = pd.PeriodIndex(pd.to_datetime(real["date"]), freq="M")
    offsets = range(-(window // 2 - 1), window // 2 + 1)
    rows = []
    for m in pd.period_range(real["month"].min(), real["month"].max(), freq="M"):
        part = real[real["month"].isin([m + k for k in offsets])]
        if len(part) < min_n:
            continue
        lo, hi = stats.bootstrap_mean(part["score"], part["cluster"], B=B, seed=seed)
        rows.append({"month": str(m), "mean": float(part["score"].mean()), "lo": lo, "hi": hi, "n": len(part)})
    return pd.DataFrame(rows, columns=["month", "mean", "lo", "hi", "n"])
