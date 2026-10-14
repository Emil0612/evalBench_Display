"""The fake-date test (M19): does telling a model the date stop its hindsight?
Registered in docs/preregistration/2026-10-06-fake-date.md before any answer.

Each Nordic index question of M15 is asked as a forecast made at the end of the previous month. Step 1, the gate (the
24 micro-pilot questions, which no test reads): a model must answer at least 80% and give varied forecasts (at least
half not refused, not 0% and not its single most frequent value); one that will not forecast cannot leak. Step 2, the
test, per model that passed: the correlation of its forecasts with the real moves on months before its cutoff minus
the same after its release (the registered M15 windows and statistic, months resampled), Holm across the models.
Next to it, descriptive: the same model's recall correlation on M15 (no date given) and the leak share (forecast r /
recall r before the cutoff). No model calls here.

uv run python -X utf8 -m evalbench.fake_date build     # builds testsets/index_forecast/v1 from index_return/v1 (free)
uv run python -X utf8 -m evalbench.fake_date           # gate and the registered test -> results/index/fake_date.md
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import market, stats, store
from .cutoffs import resolve_cutoff, resolve_release
from .index_report import PILOT
from .items import ROOT, load_testset, save_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods import index_forecast as fd
from .pilot_report import _f, _table, short

BASE = ("index_return", "v1")
VERSION = "v1"
MODELS = ["openai/gpt-4o-mini-2024-07-18", "meta-llama/llama-3.3-70b-instruct", "openai/gpt-4.1-mini",
          "google/gemini-2.5-flash", "deepseek/deepseek-chat-v3.1"]          # the five with M15 memory detected
MIN_ANSWERED, MIN_INFORMATIVE = 0.80, 0.50
B = 20000
OUT = ROOT / "results" / "index"


def build() -> None:
    items = fd.build_items(load_testset(*BASE))
    path = save_testset(items, fd.NAME, VERSION, {"prompt_version": fd.PROMPT_VERSION, "filters": fd.FILTERS,
                                                  "derived_from": "/".join(BASE)})
    print(f"{len(items)} items -> {path}")


def gate_row(frame: pd.DataFrame, asked: int) -> dict:
    """frame: the answered pilot questions (answer). Answered share, and the share of questions with an informative
    forecast: answered, not 0%, not the model's single most frequent value."""
    a = frame["answer"].astype(float)
    stock = a.round(2).mode().iloc[0] if len(a) else None
    informative = int(((a.round(2) != stock) & (a != 0)).sum()) if len(a) else 0
    row = {"answered": len(a) / asked if asked else float("nan"), "informative": informative / asked if asked else 0.0}
    row["passes"] = bool(row["answered"] >= MIN_ANSWERED and row["informative"] >= MIN_INFORMATIVE)
    return row


def leak_share(r_forecast: float, r_recall: float, min_recall: float = 0.2) -> float:
    """How much of the model's recall survives the fake date: forecast r / recall r (undefined without recall)."""
    if pd.isna(r_forecast) or pd.isna(r_recall) or r_recall < min_recall:
        return float("nan")
    return float(r_forecast / r_recall)


def _frames(calls, model: str, method: str, version: str):
    items = load_testset(method, version)
    pilot, main = items[:PILOT], items[PILOT:]
    sub = calls[calls["testset"] == f"{method}/{version}"]
    return market.answers_frame(sub, pilot, model, method), market.answers_frame(sub, main, model, method)


def _sides(frame: pd.DataFrame, cutoff, release):
    last, first = market.sides(cutoff, release)
    p = pd.PeriodIndex(frame["month"].astype(str), freq="M")
    return frame[p <= last], frame[p >= first]


def compute() -> dict:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    gate, rows = [], []
    for m in MODELS:
        pilot, main = _frames(calls, m, fd.NAME, VERSION)
        g = {"model": m, **gate_row(pilot, PILOT)}
        gate.append(g)
        if not g["passes"]:
            continue
        cut, rel = resolve_cutoff(m, meta), resolve_release(m, meta)
        before, after = _sides(main, cut, rel)
        r = market.corr_drop_test(before, after, B=B)
        _, recall = _frames(calls, m, *BASE)
        recall_before, _ = _sides(recall, cut, rel)
        r_recall = market.corr(recall_before["answer"], recall_before["truth"]) if len(recall_before) else np.nan
        rows.append({"model": m, "n_before": len(before), "n_after": len(after), **r, "r_recall": r_recall,
                     "leak": leak_share(r["r_before"], r_recall)})
    for r, p in zip(rows, stats.holm([r["p"] for r in rows])):
        r["p_holm"] = float(p)
        r["reading"] = "hindsight leaks through the date" if (r["diff"] > 0 and p < 0.05) else \
            "the opposite" if (r["diff"] < 0 and p < 0.05) else "not shown"
    return {"gate": gate, "rows": rows}


def render(res: dict) -> str:
    L = ["# Fake-date test (M19): does telling a model the date stop its hindsight?\n",
         "Registered in `docs/preregistration/2026-10-06-fake-date.md` before any answer. Each Nordic index question "
         "of M15 asked as a forecast made at the end of the previous month (\"Today is 31 March 2023 ... how will the "
         "OMX Stockholm 30 change during April 2023?\").\n",
         "## Step 1: the gate (the 24 micro-pilot questions)\n",
         _table(["model", "answered", "informative forecasts", "passes"],
                [[short(r["model"]), _f(r["answered"], "{:.0%}"), _f(r["informative"], "{:.0%}"),
                  "yes" if r["passes"] else "no"] for r in res["gate"]]),
         f"\nPasses: answers at least {MIN_ANSWERED:.0%} and at least {MIN_INFORMATIVE:.0%} informative forecasts (not "
         "0%, not its single most frequent value).\n",
         "## Step 2: the registered test\n",
         "Correlation of the forecasts with the real monthly moves on months before the cutoff (last whole month before "
         "cutoff − 91 days) minus after the release; months resampled (20,000 draws); Holm across the models that "
         "passed. An honest forecaster scores about 0 on both sides; hindsight shows as a positive difference.\n"]
    if not res["rows"]:
        L.append("No model passed the gate: the test is not run (plan §3).\n")
        return "\n".join(L)
    L += [_table(["model", "forecasts (before / after)", "r forecast before → after", "difference [95%]", "p (Holm)",
                  "reading", "recall r before (M15, no date)", "leak share"],
                 [[short(r["model"]), f"{r['n_before']} / {r['n_after']}",
                   f"{_f(r['r_before'], '{:+.2f}')} → {_f(r['r_after'], '{:+.2f}')}",
                   f"{_f(r['diff'], '{:+.2f}')} [{_f(r['lo'], '{:+.2f}')}, {_f(r['hi'], '{:+.2f}')}]",
                   _f(r["p_holm"], "{:.3f}"), r["reading"], _f(r["r_recall"], "{:+.2f}"), _f(r["leak"], "{:.0%}")]
                  for r in res["rows"]]),
          "\nLeak share: the forecast correlation before the cutoff over the recall correlation on the same months "
          "(descriptive): 100% = the date instruction removes nothing; 0% = it removes all.\n"]
    return "\n".join(L)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["build"]:
        build()
        return
    text = render(compute())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fake_date.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
