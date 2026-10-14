"""Pilot D's registered analyses that pilot C did not have (docs/preregistration/2026-09-29-pilot-d.md): the M7
positive control (E2), Split Clock beyond the cutoff (E3), M5 bands and convention, the fixed exclusions and the
abstain-mode hallucination measure (E6). Everything works on item-score frames; nothing calls a model."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import stats

LEVEL_BANDS = (0.05, 0.10, 0.25)
# M3 v2 items where one rewrite changes the company name (Freemelt, Konecranes): asked, never analysed
EXCLUDED = {"decop": {"dc-0003", "dc-0035"}}


def drop_excluded(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df[~df["item_id"].isin(EXCLUDED.get(method, set()))] if len(df) else df


def _real(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["fabricated"].astype(bool)]


def positive_control(df: pd.DataFrame, blind_mean: float, B: int = 20000) -> dict:
    """E2: the share of years whose direction the model gets right, with a cluster-bootstrap interval. Sensitive
    when the lower bound is above 0.5 and above the blind baseline."""
    real = _real(df)
    if not len(real):
        return {"n": 0, "accuracy": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "sensitive": False}
    lo, hi = stats.bootstrap_mean(real["score"], real["cluster"], B=B)
    out = {"n": len(real), "accuracy": float(real["score"].mean()), "ci_lo": lo, "ci_hi": hi,
           "blind": blind_mean, "sensitive": bool(lo > 0.5 and lo > blind_mean)}
    for col in ("within_10pp", "vs_median_hit"):
        if col in real:
            vals = real[col].dropna().astype(float)
            out[col] = float(vals.mean()) if len(vals) else float("nan")
    return out


def split_clock_beyond(model: pd.DataFrame, blind: pd.DataFrame, cutoff: date, B: int = 20000) -> dict:
    """E3: for splits after the model's cutoff, the share of answers near the rewritten price against the blind
    baseline's share on the same items. The interval is for (model share - blind share), clustered by company."""
    def after(df: pd.DataFrame) -> pd.DataFrame:
        real = _real(df)
        return real[pd.to_datetime(real["date"]).dt.date > cutoff]

    m, b = after(model), after(blind)
    if not len(m):
        return {"n": 0, "share": float("nan"), "blind_share": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "counts": {}}
    scores = np.r_[m["score"].to_numpy(float), b["score"].to_numpy(float)]
    group = np.r_[np.ones(len(m), int), np.zeros(len(b), int)]
    lo, hi, p = stats.bootstrap_delta(scores, group, np.r_[m["cluster"].to_numpy(), b["cluster"].to_numpy()], B=B)
    counts = m["kind"].value_counts().to_dict() if "kind" in m else {}
    return {"n": len(m), "share": float(m["score"].mean()), "blind_share": float(b["score"].mean()) if len(b) else
            float("nan"), "ci_lo": lo, "ci_hi": hi, "p_value": p, "counts": {k: int(v) for k, v in counts.items()}}


def typical_guesser(items: dict, median_by_month: dict) -> pd.DataFrame:
    """E3's null (plan deviation 2026-09-29): a knowledge-free guesser that always answers the median share price of
    the month asked about. A split brings the price back to a normal-looking level, so this guess often lands nearer
    the rewritten price; a model's rewritten-price share must be compared with this, not with the blind baseline."""
    rows = []
    for item_id, it in items.items():
        guess = median_by_month.get(it.payload["month"])
        level, adj = it.truth["level"], it.truth["adj_level"]
        if not guess or not level or not adj:
            continue
        lr, la = abs(np.log(guess / level)), abs(np.log(guess / adj))
        kind = "neither" if min(lr, la) > abs(np.log(level / adj)) / 2 else ("adjusted" if la < lr else "raw")
        rows.append({"item_id": item_id, "date": it.date, "cluster": it.cluster, "fabricated": False, "refused": False,
                     "outcome": "answer", "score": float(kind == "adjusted"), "kind": kind,
                     "type": "reverse" if it.truth["factor"] < 1 else "ordinary"})
    return pd.DataFrame(rows)


def hallucination(df: pd.DataFrame) -> dict:
    """E6 (abstain mode): how often a fabricated company gets an answer, and abstention before vs after the cutoff."""
    fab = df[df["fabricated"].astype(bool)]
    real = _real(df)

    def abstain(part: pd.DataFrame) -> float:
        return float((part["outcome"] == "abstain").mean()) if len(part) else float("nan")

    hall = fab["hallucinated"].dropna().astype(bool) if "hallucinated" in fab else pd.Series(dtype=bool)
    return {"n_fabricated": len(fab), "hallucination_rate": float(hall.mean()) if len(hall) else float("nan"),
            "abstain_pre": abstain(real[real["group"] == 1]), "abstain_post": abstain(real[real["group"] == 0])}
