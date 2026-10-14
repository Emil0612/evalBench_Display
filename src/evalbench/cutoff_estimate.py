"""Estimating an unpublished knowledge cutoff (registered in docs/preregistration/2026-10-07-new-models.md before any
answer of the new models): Qwen3.6 27B has none, and the new-model tests need one for their BEFORE window.

The rule, from the US monthly-change answers (M17; not part of the new models' Nordic tests): for each month t in the
36 months before the model's release, recall is the correlation of the answers with the true moves over the 6 months
ending at t, minus the same over the 6 months after t (answers that never vary count as r = 0); the changepoint is the
t with the largest drop. Recall fades before a stated cutoff, so the estimate adds the median gap of the reference
models, whose makers state a cutoff. A weak drop (below 0.3) falls back to release - 12 months. The Gemma 4 models,
whose model card states January 2025, are the out-of-sample check. No model calls, no cost.

uv run python -X utf8 -m evalbench.cutoff_estimate      # -> results/new_models/cutoff_estimate.md
"""
from __future__ import annotations

import calendar
from datetime import date

import numpy as np
import pandas as pd

from . import ladder, market, store
from .cutoffs import resolve_cutoff, resolve_release
from .items import ROOT
from .llm.openrouter import fetch_models, models_by_id
from .pilot_report import _table, short

METHOD = "us_index_return"
K = 6               # months on each side of a candidate changepoint
SPAN = 36           # months before release searched
MIN_DROP = 0.3      # a weaker drop is no estimate: fall back
FALLBACK = 12       # months before release
# Models whose maker states a cutoff and that answered the US questions. Left out: GPT-3.5 Turbo (the served snapshot,
# released 2024-01-25, postdates its stated 2021-09 cutoff; cutoff-ladder plan §0, deviation 1), DeepSeek V3.1 (no
# official cutoff) and Qwen3.8 Flash (our own estimate).
REFERENCES = ["openai/gpt-4o-mini-2024-07-18", "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.3-70b-instruct",
              "openai/gpt-4.1-mini", "google/gemini-2.5-flash", "openai/gpt-oss-20b"]
NEW = ["google/gemma-4-31b-it", "google/gemma-4-26b-a4b-it", "qwen/qwen3.6-27b"]
MODEL_CARD = {"google/gemma-4-31b-it": date(2025, 1, 31), "google/gemma-4-26b-a4b-it": date(2025, 1, 31)}
OUT = ROOT / "results" / "new_models"


def _corr0(w: pd.DataFrame) -> float:
    if len(w) < 8:
        return float("nan")
    r = market.corr(w["answer"], w["truth"])
    return 0.0 if pd.isna(r) else float(r)


def changepoint(frame: pd.DataFrame, release: date, k: int = K, span: int = SPAN) -> tuple[pd.Period | None, float]:
    """(month, drop): the month in the `span` months before release where recall drops most (k months each side)."""
    if not len(frame):
        return None, float("nan")
    rel = pd.Period(release, "M")
    best, best_d = None, float("-inf")
    for t in pd.period_range(rel - span, min(rel, frame["month"].max() - 1), freq="M"):
        pre = frame[(frame["month"] > t - k) & (frame["month"] <= t)]
        post = frame[(frame["month"] > t) & (frame["month"] <= t + k)]
        d = _corr0(pre) - _corr0(post)
        if pd.notna(d) and d > best_d:
            best, best_d = t, d
    return best, (best_d if best is not None else float("nan"))


def gap(a: pd.Period, b: pd.Period) -> int:
    return (a.year - b.year) * 12 + (a.month - b.month)


def offset(gaps: dict[str, int]) -> int:
    """Months to add to a changepoint: minus the median gap (changepoint - stated cutoff) of the reference models."""
    return -int(round(float(np.median(list(gaps.values())))))


def _month_end(p: pd.Period) -> date:
    return date(p.year, p.month, calendar.monthrange(p.year, p.month)[1])


def estimate(month: pd.Period | None, drop: float, offset: int, release: date) -> dict:
    """The estimated cutoff (end of month), never after the release month; a weak drop falls back."""
    rel = pd.Period(release, "M")
    if month is None or pd.isna(drop) or drop < MIN_DROP:
        return {"cutoff": _month_end(rel - FALLBACK), "rule": f"fallback: release - {FALLBACK} months (drop below {MIN_DROP})"}
    return {"cutoff": _month_end(min(month + offset, rel)), "rule": f"changepoint + {offset} months"}


def compute() -> dict:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    refs = []
    for m in REFERENCES:
        cut, rel = resolve_cutoff(m, meta), resolve_release(m, meta)
        t, d = changepoint(ladder.index_frame(calls, m, METHOD), rel)
        if t is not None:
            refs.append({"model": m, "cutoff": cut, "release": rel, "changepoint": t, "drop": d,
                         "gap": gap(t, pd.Period(cut, "M"))})
    off = offset({r["model"]: r["gap"] for r in refs})
    for r in refs:   # leave-one-out: the offset from the other references, as if this model's cutoff were unknown
        loo = offset({o["model"]: o["gap"] for o in refs if o is not r})
        r["loo_error"] = gap(r["changepoint"] + loo, pd.Period(r["cutoff"], "M"))
    new = []
    for m in NEW:
        rel = resolve_release(m, meta)
        t, d = changepoint(ladder.index_frame(calls, m, METHOD), rel)
        row = {"model": m, "release": rel, "changepoint": t, "drop": d, **estimate(t, d, off, rel)}
        if m in MODEL_CARD:
            row["model_card"] = MODEL_CARD[m]
            row["error"] = gap(pd.Period(row["cutoff"], "M"), pd.Period(MODEL_CARD[m], "M"))
        new.append(row)
    return {"references": refs, "offset": off, "new": new}


def render(res: dict) -> str:
    refs = res["references"]
    mae = np.mean([abs(r["loo_error"]) for r in refs]) if refs else float("nan")
    L = ["# Estimated knowledge cutoffs for the new models\n",
         "Registered in `docs/preregistration/2026-10-07-new-models.md` before any answer of the new models. From the US "
         "monthly-change answers (M17): the month in the 36 months before release where recall drops most (correlation "
         f"with the true moves, 6 months before vs after), plus {res['offset']} months, the median gap of the reference "
         "models whose makers state a cutoff.\n",
         "## Reference models (stated cutoffs)\n",
         _table(["model", "stated cutoff", "changepoint", "drop", "gap", "error, leave-one-out"],
                [[short(r["model"]), str(r["cutoff"]), str(r["changepoint"]), f"{r['drop']:.2f}", f"{r['gap']:+d}",
                  f"{r['loo_error']:+d}"] for r in refs]),
         f"\nMean absolute error, leave-one-out: {mae:.1f} months.\n",
         "## New models\n",
         _table(["model", "release", "changepoint", "drop", "estimated cutoff", "rule", "model card", "error"],
                [[short(r["model"]), str(r["release"]), str(r["changepoint"] or "–"),
                  "–" if pd.isna(r["drop"]) else f"{r['drop']:.2f}", str(r["cutoff"]), r["rule"],
                  str(r.get("model_card", "–")), f"{r['error']:+d}" if "error" in r else "–"] for r in res["new"]]),
         "\nThe Gemma 4 rows are the out-of-sample check: their model card states the cutoff. The Qwen3.6 27B estimate is "
         "frozen into `config/model_cutoffs.yaml` (dated addendum to the plan) before its Nordic answers are analysed.\n"]
    return "\n".join(L)


def main() -> None:
    text = render(compute())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cutoff_estimate.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
