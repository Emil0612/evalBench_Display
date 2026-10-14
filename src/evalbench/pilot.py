"""Pre-registered pilot C analyses (docs/preregistration/2026-09-29-pilot-c.md): the primary reading (§2),
S1 tolerance bands, S2 staggered-cutoff comparison, S3 placebo cutoff, S4 boilerplate split, S5 Split Clock first
look, S7 per-period profile and S8 unadjusted-price convention. Everything works on item-score frames from
analysis.item_scores; nothing here calls a model."""
from __future__ import annotations

import math
import re
from collections import Counter
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import stats
from .analysis import summarize
from .cutoffs import BUFFER_DAYS
from .items import Item
from .methods import METHODS
from .methods import split_clock as split_clock_method

NAN = float("nan")
BANDS = (0.01, 0.03, 0.05, 0.10)
PLACEBO_SPLIT, PLACEBO_UPPER = date(2022, 7, 1), date(2023, 10, 1)
MIN_PRIMARY = 20      # §2: fewer real items than this in PRE or POST -> insufficient coverage
MIN_CELL = 15         # S7: fewer PRE items in a period cell -> insufficient coverage
MIN_BASE = 20         # S7: fewer items behind a baseline -> insufficient coverage
MAX_MALFORMED = 0.20  # §2 and S7: more malformed answers than this -> unresolved
# the fact types; M3/M4 measure whether the text itself was seen, so they get their own type
FACT_TYPES = {"price_recall": "price", "number_cloze": "event", "decop": "document", "completion": "document",
              "price_level": "price", "annual_return": "return", "known_return": "return", "split_clock": "price",
              "known_level": "price", "us_level": "price", "monthly_return": "return", "price_choice": "price",
              "known_choice": "price", "price_choice_prob": "price", "known_choice_prob": "price",
              "index_level": "market", "index_return": "market", "us_index_level": "market",
              "us_index_return": "market", "headline_date": "event", "index_forecast": "market",
              "path_date": "market"}
# "market" (scoring spec §9, 2026-10-06): questions about an index, not a company; until then M14/M16 were filed under
# price and M15/M17 under return
# S4, as registered (case-insensitive). "AGM" only as a whole word, so that e.g. "Magma" does not match.
BOILERPLATE = ("återköp", "buyback", "repurchase", "kallelse", "notice of", "årsstämma", "annual general meeting",
               "bolagsstämma", "extra general meeting", "insynshandel", "managers' transactions", "flaggning",
               "major shareholding", "valberedning", "nomination committee")
_AGM = re.compile(r"\bagm\b")


# ---------------------------------------------------------------- §2 primary reading
def primary_reading(row) -> str:
    """The registered reading of one model x method cell."""
    d, lo, hi, p = row["dkappa"], row["ci_lo"], row["ci_hi"], row["p_holm"]
    if (row.get("malformed_rate") or 0) > MAX_MALFORMED:
        return f"unresolved: {row['malformed_rate']:.0%} malformed answers"
    if pd.isna(d) or row["n_pre"] < MIN_PRIMARY or row["n_post"] < MIN_PRIMARY:
        return f"insufficient coverage: {row['n_pre']} before / {row['n_post']} after the cutoff"
    if p < 0.05 and d > 0:
        return f"memorization detected: Δκ {d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    if p < 0.05:
        return f"better after the cutoff: Δκ {d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
    return f"none detected: Δκ {d:+.2f}, at most {hi:+.2f}"


def _real(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["fabricated"].astype(bool)]


def _stat(s: dict) -> dict:
    return {k: s[k] for k in ("n_pre", "n_post", "mean_pre", "mean_post", "dkappa", "ci_lo", "ci_hi", "p_value")}


# ---------------------------------------------------------------- S1 tolerance bands, S8 unadjusted convention
def _err(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(NAN, index=df.index)


def tolerance_bands(df: pd.DataFrame, bands=BANDS, B: int = 2000, method: str = "price_recall") -> pd.DataFrame:
    """Hit rates and Δκ when a hit is within each band of the raw or the adjusted price (M1, M5)."""
    real = _real(df)
    err = np.fmin(_err(real, "rel_err_raw"), _err(real, "rel_err_adj"))
    rows = []
    for band in bands:
        s = summarize(real.assign(score=(err <= band + 1e-12).astype(float)), method, B=B)
        rows.append({"band": band, **_stat(s), "hit_pre": s["mean_pre"], "hit_post": s["mean_post"]})
    return pd.DataFrame(rows)


def median_log_error(df: pd.DataFrame) -> dict:
    """Median absolute log error of answered real items, PRE and POST (M1)."""
    real = _real(df)
    err = pd.to_numeric(real["log_error"], errors="coerce") if "log_error" in real else pd.Series(dtype=float)
    return {"pre": float(err[real["group"] == 1].median()), "post": float(err[real["group"] == 0].median())}


def raw_convention(df: pd.DataFrame, B: int = 2000, tolerance: float = 0.01, method: str = "price_recall") -> dict:
    """S8: a hit only within the tolerance of the unadjusted price (the reporting convention); M1 1%, M5 10%."""
    real = _real(df)
    return summarize(real.assign(score=(_err(real, "rel_err_raw") <= tolerance + 1e-12).astype(float)), method, B=B)


# ---------------------------------------------------------------- S2 staggered-cutoff comparison
def staggered(frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]], cutoff_a: date, cutoff_b: date,
              B: int = 2000, min_window: int = 10, window_start: date | None = None,
              answered_only: bool = False) -> dict:
    """frames: method -> (earlier-cutoff model's item scores, later-cutoff model's). On the same items,
    (later - earlier) chance-corrected score in the window between their cutoffs minus the same difference on items
    before both. Per method (None below min_window window items) and pooled over methods. window_start: start the
    window after this date instead of the earlier cutoff + 91 days (the release line: the earlier model's release).
    answered_only: only questions both models answered (the refusal guard)."""
    buf = timedelta(days=BUFFER_DAYS)
    lo_w, hi_w, ref = window_start or cutoff_a + buf, cutoff_b - buf, cutoff_a - buf
    parts, out = [], {}
    for method, (a, b) in frames.items():
        p0 = METHODS[method].P0
        cols = ["item_id", "date", "cluster", "score"] + (["refused"] if "refused" in a and "refused" in b else [])
        m = _real(a)[cols].merge(_real(b)[[c for c in cols if c not in ("date", "cluster")]], on="item_id",
                                 suffixes=("_a", "_b"))
        if answered_only and "refused_a" in m:
            m = m[~(m["refused_a"].astype(bool) | m["refused_b"].astype(bool))]
        d = (m["score_b"].to_numpy(float) - m["score_a"].to_numpy(float)) / (1 - p0)
        dates = pd.to_datetime(m["date"]).dt.date
        group = np.where((dates > lo_w) & (dates < hi_w), 1, np.where(dates < ref, 0, -1))
        part = pd.DataFrame({"d": d, "group": group, "cluster": m["cluster"].to_numpy()})
        parts.append(part)
        n_w, n_r = int((group == 1).sum()), int((group == 0).sum())
        out[method] = _did(part, B) if n_w >= min_window and n_r else None
    out["pooled"] = _did(pd.concat(parts, ignore_index=True), B) if parts else None
    return out


def _did(part: pd.DataFrame, B: int) -> dict:
    lo, hi, p = stats.bootstrap_delta(part["d"], part["group"], part["cluster"], B=B)
    return {"n_window": int((part["group"] == 1).sum()), "n_ref": int((part["group"] == 0).sum()),
            "did": stats.delta(part["d"], part["group"]), "ci_lo": lo, "ci_hi": hi, "p_value": p}


# ---------------------------------------------------------------- S3 placebo cutoff
def placebo(df: pd.DataFrame, method: str, split: date = PLACEBO_SPLIT, upper: date = PLACEBO_UPPER,
            B: int = 2000) -> dict:
    """Δκ at a fake cutoff (no buffer) among items every model had seen; expected ≈ 0."""
    real = _real(df)
    dates = pd.to_datetime(real["date"]).dt.date
    part = real[dates < upper].assign(group=np.where(dates[dates < upper] < split, 1, 0))
    return summarize(part, method, B=B)


# ---------------------------------------------------------------- S4 boilerplate
def is_boilerplate(headline: str | None) -> bool:
    h = (headline or "").lower().replace("’", "'")
    return any(k in h for k in BOILERPLATE) or bool(_AGM.search(h))


def boilerplate_split(df: pd.DataFrame, flags: dict[str, bool], method: str, B: int = 2000) -> dict:
    real = _real(df)
    flagged = real["item_id"].map(lambda i: flags.get(i, False)).astype(bool)
    return {"flagged": summarize(real[flagged], method, B=B), "unflagged": summarize(real[~flagged], method, B=B)}


# ---------------------------------------------------------------- S5 Split Clock first look
adjustment_events = split_clock_method.adjustment_events   # one implementation, shared with M8


def split_clock(df: pd.DataFrame, items: dict[str, Item], events: dict, cutoff: date) -> dict[tuple[str, str], int]:
    """Counts of answered M1 items with a ≥2x adjustment after the item date, by which convention the answer is
    nearer to (within a factor √k, else 'neither') and whether that adjustment came before or after the cutoff."""
    counts: Counter = Counter()
    answered = _real(df)
    if "outcome" in answered:
        answered = answered[answered["outcome"] == "answer"]
    for r in answered.itertuples():
        it = items.get(r.item_id)
        close, adj = (it.truth.get("close"), it.truth.get("adj_close")) if it else (None, None)
        if not close or not adj or max(close / adj, adj / close) < 2:
            continue
        d = date.fromisoformat(it.date)
        later = [(ed, f) for ed, f in events.get(it.truth.get("insref"), []) if ed > d]
        if not later:
            continue
        event_date = max(later, key=lambda e: abs(math.log(e[1])))[0]
        side = "event_before_cutoff" if event_date <= cutoff else "event_after_cutoff"
        lr, la, half = float(r.log_err_raw), float(r.log_err_adj), abs(math.log(close / adj)) / 2
        kind = "neither" if min(lr, la) > half else ("adjusted" if la < lr else "raw")
        counts[(side, kind)] += 1
    return dict(counts)


# ---------------------------------------------------------------- S7 per-period profile
def profile_cells(df: pd.DataFrame, method: str, model: str, cutoff: date, release: date | None = None,
                  blind: dict[str, float] | None = None, B: int = 2000, paired: bool = False,
                  drop_refused: bool = False, corr: bool = False, screen: int | None = None) -> list[dict]:
    """One cell per calendar year in the per-year cell format, with measured baselines: B1 = the model's own POST items,
    B1r = its items after release, B2 = the blind baseline on the same items. Unlabelled until label_profile.
    paired: the cell is compared with the blind baseline on its own items instead (M18, M14/M16, whose registered tests
    compare with a blind baseline): model minus blind per item, resampling the items' clusters; B1 and B1r are then the
    same. corr (M15/M17): the cell is the correlation of the answers with the true moves, against the correlation after
    the cutoff (B1) or release (B1r), resampling months, as the registered test. drop_refused (scoring spec §8):
    refused questions are left out of every number and reported as a share (refused_share, refused_baseline,
    refused_baseline_release). screen (the app, for speed): each test first runs with `screen` draws and is repeated
    with all B only where p <= SCREEN_P, the only cells Holm can still turn into a detection; a cell's own interval
    uses `screen` draws. The pilot reports keep the defaults."""
    Bc = screen or B
    real = _real(df).copy()
    real["_d"] = pd.to_datetime(real["date"]).dt.date
    full = real
    if drop_refused and len(real):
        real = real[~real["refused"].astype(bool)]
    post = real[real["group"] == 0]
    after_release = real[real["_d"] > release] if release else real.iloc[0:0]
    full_post = full[full["group"] == 0]
    full_after = full[full["_d"] > release] if release else full.iloc[0:0]
    n_var = len(METHODS[method].VARIANTS)
    cells = []
    for year in sorted({d.year for d in full["_d"]}):
        yr = real[real["_d"].map(lambda d, y=year: d.year == y)]
        fyr = full[full["_d"].map(lambda d, y=year: d.year == y)]
        pre = yr[yr["group"] == 1]
        cell = {"model": model, "method": method, "fact_type": FACT_TYPES[method], "region": "SE", "stratum": "all",
                "elicitation": "first_attempt", "period": str(year),
                "after_availability": bool(release and date(year, 1, 1) > release),
                "after_cutoff": bool((fyr["group"] == 1).sum() == 0 and (fyr["group"] == 0).any()),
                "baseline": _mean(post), "n_baseline": len(post),
                "baseline_release": _mean(after_release), "n_baseline_release": len(after_release)}
        shown = yr[yr["group"] == 0] if cell["after_cutoff"] else pre
        cell.update(facts=len(shown), responses=len(shown) * n_var, recall=_mean(shown),
                    ci=list(stats.bootstrap_mean(shown["score"], shown["cluster"], B=Bc)) if len(shown) else [NAN, NAN])
        if drop_refused:
            fshown = fyr[fyr["group"] == (0 if cell["after_cutoff"] else 1)]
            cell.update(refused_share=_refused(fshown), refused_baseline=_refused(full_post),
                        refused_baseline_release=_refused(full_after))
        if corr:
            sf = _corr_frame(shown)
            draws = _boot_corr(sf, Bc) if len(sf) >= 3 else np.array([])
            draws = draws[~np.isnan(draws)]
            cell.update(corr=True, recall=_r(sf), facts=len(sf), responses=len(sf) * n_var,
                        ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))] if len(draws)
                        else [NAN, NAN],
                        baseline=_r(_corr_frame(post)), n_baseline=len(_corr_frame(post)),
                        baseline_release=_r(_corr_frame(after_release)),
                        n_baseline_release=len(_corr_frame(after_release)))
        if blind is not None:
            cell["baseline_blind"] = _mean(pd.DataFrame({"score": [blind[i] for i in shown["item_id"] if i in blind]}))
        if paired:
            n = int(shown["item_id"].isin(blind or {}).sum())
            cell.update(paired_blind=True, baseline=cell.get("baseline_blind", NAN), n_baseline=n,
                        baseline_release=cell.get("baseline_blind", NAN), n_baseline_release=n)
        if cell["after_cutoff"]:
            cell["label"] = "after_cutoff"      # descriptive: this is the model's own "cannot know" level
        elif pre.empty:
            cell["label"] = "insufficient_coverage"
        elif "outcome" in pre and (pre["outcome"] == "malformed").mean() > MAX_MALFORMED:
            cell["label"] = "unresolved"
        elif paired:
            v = _paired(pre, blind or {}, B, screen)
            cell.update({**v, **{k + "_release": val for k, val in v.items()}})
        elif corr:
            cell.update(_corr_versus(pre, post, "", B, screen), **_corr_versus(pre, after_release, "_release", B,
                                                                               screen))
        else:
            cell.update(_versus(pre, post, "", B, screen), **_versus(pre, after_release, "_release", B, screen))
        cells.append(cell)
    return cells


def _refused(df: pd.DataFrame) -> float:
    return float(df["refused"].astype(bool).mean()) if len(df) else NAN


def _corr_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Answered questions as the correlation test needs them: month (the cluster), answer, truth."""
    if not len(d) or "answer" not in d:
        return pd.DataFrame(columns=["month", "answer", "truth"])
    d = d[~d["refused"].astype(bool)].dropna(subset=["answer", "truth"])
    return pd.DataFrame({"month": d["cluster"].to_numpy(), "answer": d["answer"].astype(float).to_numpy(),
                         "truth": d["truth"].astype(float).to_numpy()})


def _r(f: pd.DataFrame) -> float:
    from . import market
    return market.corr(f["answer"], f["truth"]) if len(f) else NAN


def _boot_corr(f: pd.DataFrame, B: int) -> np.ndarray:
    from . import market
    return market.boot_corr(f, B=B)


SCREEN_P = 0.10     # Holm never lowers a p-value: above this a cell cannot become a detection, whatever the draws


def _screened(run, B: int, screen: int | None, p_of):
    """run(n_draws) with the screen first; all B draws only where the screened p could still matter."""
    if screen and screen < B:
        first = run(screen)
        if not (p_of(first) <= SCREEN_P):
            return first
    return run(B)


def _corr_versus(pre: pd.DataFrame, base: pd.DataFrame, suffix: str, B: int, screen: int | None = None) -> dict:
    """(correlation in the cell - correlation of the baseline), resampling months on each side as the registered M15
    test does; 'pending' until Holm in label_profile."""
    from . import market
    p, b = _corr_frame(pre), _corr_frame(base)
    if len(p) < MIN_CELL or len(b) < MIN_BASE:
        return {f"label{suffix}": "insufficient_coverage"}
    r = _screened(lambda n: market.corr_drop_test(p, b, B=n), B, screen, lambda r: r["p"])
    if pd.isna(r["p"]):
        return {f"label{suffix}": "insufficient_coverage"}
    return {f"label{suffix}": "pending", f"excess{suffix}": r["diff"], f"excess_ci{suffix}": [r["lo"], r["hi"]],
            f"p{suffix}": r["p"]}


def _mean(df: pd.DataFrame) -> float:
    return float(df["score"].mean()) if len(df) else NAN


def _paired(pre: pd.DataFrame, blind: dict[str, float], B: int = 20000, screen: int | None = None) -> dict:
    """(cell - blind baseline on the same items), resampling clusters as M18's registered test does (20,000 draws);
    'pending' until Holm in label_profile."""
    from .index_report import _mean_test
    d = pre[pre["item_id"].isin(blind)]
    if len(d) < MIN_CELL:
        return {"label": "insufficient_coverage"}
    diff = d["score"].astype(float) - d["item_id"].map(blind)
    v, c = diff.reset_index(drop=True), d["cluster"].astype(str).reset_index(drop=True)
    mean, lo, hi, p = _screened(lambda n: _mean_test(v, c, B=n), B, screen, lambda r: r[3])
    return {"label": "pending", "excess": mean, "excess_ci": [lo, hi], "p": p}


def _versus(pre: pd.DataFrame, base: pd.DataFrame, suffix: str, B: int, screen: int | None = None) -> dict:
    """(cell - baseline) with a cluster bootstrap; 'pending' until Holm across the family in label_profile."""
    if len(pre) < MIN_CELL or len(base) < MIN_BASE:
        return {f"label{suffix}": "insufficient_coverage"}
    scores = np.r_[pre["score"].to_numpy(float), base["score"].to_numpy(float)]
    group = np.r_[np.ones(len(pre), int), np.zeros(len(base), int)]
    clusters = np.r_[pre["cluster"].to_numpy(), base["cluster"].to_numpy()]
    lo, hi, p = _screened(lambda n: stats.bootstrap_delta(scores, group, clusters, B=n), B, screen, lambda r: r[2])
    return {f"label{suffix}": "pending", f"excess{suffix}": stats.delta(scores, group),
            f"excess_ci{suffix}": [lo, hi], f"p{suffix}": p}


def label_profile(cells: list[dict]) -> list[dict]:
    """Holm across every tested cell (separately for B1 and B1r), then the registered reading."""
    for suffix in ("", "_release"):
        tested = [c for c in cells if c.get(f"label{suffix}") == "pending"]
        adjusted = stats.holm([c[f"p{suffix}"] for c in tested]) if tested else []
        for c, p_holm in zip(tested, adjusted):
            c[f"p_holm{suffix}"] = float(p_holm)
            if p_holm < 0.05 and c[f"excess{suffix}"] > 0:
                c[f"label{suffix}"] = "recall_above_baseline"
            else:
                c[f"label{suffix}"] = "none_detected"
                c[f"detection_upper_bound{suffix}"] = c[f"excess_ci{suffix}"][1]
    return cells
