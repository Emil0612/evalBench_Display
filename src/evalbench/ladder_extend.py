"""The cutoff ladder on more months: memory or a time effect? (docs/preregistration/2026-10-06-ladder-extension.md)

The registered ladder (2026-10-05) compared GPT-3.5 Turbo (no knowledge after January 2024) with Gemini 2.5 Flash on the
9 months 2024-02..2024-10 of the Nordic index questions (M15): D = +0.81 on 20 paired answers. Here later-cutoff models
take Gemini's place, for 14-17 months. Step 1, the gate (on months outside every test window): a candidate must answer
most questions and track the real monthly index moves on older and recent months. Step 2, the test: the registered
staggered statistic (ladder.staggered_corr) for each candidate that passed, Holm across them. No model calls here.

uv run python -X utf8 -m evalbench.ladder_extend gate      # step 1: only months outside the test windows
uv run python -X utf8 -m evalbench.ladder_extend report    # step 2: the registered test, for the models that passed
"""
from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from . import ladder, market, stats, store
from .cutoffs import resolve_cutoff
from .items import ROOT
from .llm.openrouter import fetch_models, models_by_id
from .pilot_report import _f, _table, short

EARLY = ladder.GPT35
CANDIDATES = ["deepseek/deepseek-v3.2-exp", "qwen/qwen3-235b-a22b-2507", "qwen/qwen3-next-80b-a3b-instruct"]
METHOD = "index_return"
RECENT = (pd.Period("2022-01", "M"), pd.Period("2024-01", "M"))   # after GPT-3.5's window, before every test window
MIN_ANSWERED, MIN_R = 0.80, 0.60                                    # calibrated on the nine models (plan §3)
OUT = ROOT / "results" / "ladder"


EARLY_CUTOFF = date(2021, 9, 30)      # GPT-3.5 Turbo's frozen cutoff (config/model_cutoffs.yaml), as in the ladder


def windows(late_cutoff: date, months, early_cutoff: date = EARLY_CUTOFF) -> tuple[list, list]:
    """between: months after GPT-3.5 Turbo's release month up to the late model's BEFORE edge; baseline: months both
    may know (GPT-3.5's BEFORE months), exactly as the registered ladder."""
    return ladder.pair_months(early_cutoff, ladder.RELEASE[EARLY], late_cutoff, months)


def _periods(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(pd.PeriodIndex(frame["month"].astype(str), freq="M"), index=frame.index)


def gate_row(frame: pd.DataFrame, asked: int, late_cutoff: date) -> dict:
    """The gate for one candidate (frame: its answered index questions: month, answer, truth). Only months outside its
    test window are read for the correlations."""
    months = sorted(set(_periods(frame)))
    between, baseline = windows(late_cutoff, months)
    p = _periods(frame)
    old = frame[p.isin(baseline)]
    recent = frame[(p >= RECENT[0]) & (p <= RECENT[1])]
    row = {"answered": len(frame) / asked if asked else float("nan"),
           "r_old": market.corr(old["answer"], old["truth"]) if len(old) else float("nan"),
           "r_recent": market.corr(recent["answer"], recent["truth"]) if len(recent) else float("nan"),
           "n_old": len(old), "n_recent": len(recent), "window": f"{between[0]}..{between[-1]}" if between else "–",
           "window_months": len(between)}
    row["passes"] = bool(row["answered"] >= MIN_ANSWERED and row["r_old"] >= MIN_R and row["r_recent"] >= MIN_R)
    return row


def gate(calls, meta) -> list[dict]:
    asked = len(ladder._index_items(METHOD))
    rows = []
    for m in CANDIDATES:
        f = ladder.index_frame(calls, m, METHOD)
        rows.append({"model": m, "cutoff": str(resolve_cutoff(m, meta)), **(gate_row(f, asked, resolve_cutoff(m, meta))
                     if len(f) else {"answered": 0.0, "passes": False})})
    return rows


def test(calls, meta, passing: list[str], B: int = 20000) -> list[dict]:
    """The registered staggered test against GPT-3.5 Turbo for each candidate that passed the gate; Holm across them."""
    months = ladder.index_months(METHOD)
    early = ladder.index_frame(calls, EARLY, METHOD)
    rows = []
    for m in passing:
        between, baseline = windows(resolve_cutoff(m, meta), months, resolve_cutoff(EARLY, meta))
        r = ladder.staggered_corr(ladder.index_frame(calls, m, METHOD), early, between, baseline, B=B)
        rows.append({"model": m, "between": f"{between[0]}..{between[-1]}", "months": len(between), **r})
    for r, p in zip(rows, stats.holm([r["p"] for r in rows])):
        r["p_holm"] = float(p)
        r["reading"] = "memory, not a time effect" if (r["D"] > 0 and p < 0.05) else \
            "the opposite of memory" if (r["D"] < 0 and p < 0.05) else "not shown"
    return rows


def render_gate(rows: list[dict]) -> str:
    return "\n".join([
        "# Cutoff ladder extension: step 1, the gate\n",
        f"Months outside every test window only. Passes: answers at least {MIN_ANSWERED:.0%} of the questions and "
        f"tracks the real monthly index moves with r ≥ {MIN_R} on the months both models may know (to 2021-06) and on "
        f"{RECENT[0]}..{RECENT[1]}.\n",
        _table(["candidate", "stated cutoff", "test window (months)", "answered", "r 2016–2021-06", "r 2022–2024-01",
                "passes"],
               [[short(r["model"]), r["cutoff"], f"{r.get('window', '–')} ({r.get('window_months', 0)})",
                 _f(r["answered"], "{:.0%}"), _f(r.get("r_old"), "{:+.2f}"), _f(r.get("r_recent"), "{:+.2f}"),
                 "yes" if r["passes"] else "no"] for r in rows])]) + "\n"


def render_test(rows: list[dict]) -> str:
    if not rows:
        return "# Cutoff ladder extension: step 2\n\nNo candidate passed the gate: the extension is not run (plan §4).\n"
    return "\n".join([
        "# Cutoff ladder extension: step 2, the registered test\n",
        "D = (r_late − r_GPT-3.5) on the test months − (r_late − r_GPT-3.5) on the months both may know, over the "
        "questions both answered; months resampled together (20,000 draws); Holm across the candidates that passed "
        "the gate. Memory predicts D > 0; a time effect shared by both models cancels.\n",
        _table(["late model", "test months", "paired answers (test / both know)", "r late / GPT-3.5 on test months",
                "r late / GPT-3.5 on months both know", "D [95%]", "p (Holm)", "reading"],
               [[short(r["model"]), f"{r['between']} ({r['months']})", f"{r['n_between']} / {r['n_baseline']}",
                 f"{_f(r['r_late_between'], '{:+.2f}')} / {_f(r['r_early_between'], '{:+.2f}')}",
                 f"{_f(r['r_late_baseline'], '{:+.2f}')} / {_f(r['r_early_baseline'], '{:+.2f}')}",
                 f"{_f(r['D'], '{:+.2f}')} [{_f(r['lo'], '{:+.2f}')}, {_f(r['hi'], '{:+.2f}')}]",
                 _f(r["p_holm"], "{:.3f}"), r["reading"]] for r in rows])]) + "\n"


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    rows = gate(calls, meta)
    text = render_gate(rows)
    if argv[:1] == ["report"]:
        text += "\n" + render_test(test(calls, meta, [r["model"] for r in rows if r["passes"]]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("extension.md" if argv[:1] == ["report"] else "extension_gate.md")).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
