"""The masking test (M20): can a model date an unlabeled market path?
Registered in docs/preregistration/2026-10-06-masking.md before any answer.

Each question shows the 12 monthly changes of an unnamed Nordic index (no dates, no name, no levels) and asks for the
end month. A hit is an answer within one month of the truth; the measure per question is the hit minus how often the
model's own answers land within a month of that truth anyway (its answer habits), with the months next to its habitual
answer left out and refused questions left out. A model without memory scores about 0 whatever its habits. The test,
per model: the mean on paths ending on or before its BEFORE edge (last whole month before cutoff − 91 days) is above
0, months resampled, Holm across the models. No model calls here.

uv run python -X utf8 -m evalbench.mask_date build     # builds testsets/path_date/v1 from index_return/v1 (free)
uv run python -X utf8 -m evalbench.mask_date           # the registered test -> results/index/masking.md
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import analysis, ladder, stats, store
from .cutoffs import resolve_cutoff
from .fake_date import MODELS
from .index_report import _mean_test
from .items import ROOT, load_testset, save_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods import path_date as pdm
from .pilot_report import _f, _table, short

VERSION = "v1"
WINDOW = 1          # a hit: within one month of the true end month
HABIT_GAP = 2       # months within this distance of the habitual answer are left out
CONTROLS = ["mock/oracle@2024-06-30", "mock/random@2024-06-30", "baseline/blind@2024-06-30"]
CRASH = pd.Period("2020-03", "M")
B = 20000
OUT = ROOT / "results" / "index"


def build() -> None:
    items = pdm.build_items(load_testset("index_return", "v1"))
    path = save_testset(items, pdm.NAME, VERSION, {"prompt_version": pdm.PROMPT_VERSION, "filters": pdm.FILTERS,
                                                   "derived_from": "index_return/v1"})
    print(f"{len(items)} items -> {path}")


def excess(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """df: item_id, true (Period), answer (Period or None), refused, cluster. Adds x: hit within `window` months minus
    the share of the model's answers within `window` months of the truth; NaN when refused or near its habitual
    answer. The registered window is WINDOW (one month); wider ones are an exploratory re-score."""
    out = df.copy()
    answered = out[~out["refused"].astype(bool) & out["answer"].notna()]
    ords = np.array([a.ordinal for a in answered["answer"]])
    habit = pd.Series(ords).mode().iloc[0] if len(ords) else None
    xs = []
    for r in out.itertuples():
        if r.refused or r.answer is None or pd.isna(r.answer):
            xs.append(np.nan)
            continue
        t = r.true.ordinal
        if habit is not None and abs(t - habit) <= HABIT_GAP:
            xs.append(np.nan)
            continue
        xs.append(float(abs(r.answer.ordinal - t) <= window) - float((np.abs(ords - t) <= window).mean()))
    return out.assign(x=xs)


def frame(calls, model: str, items: dict, cutoff) -> pd.DataFrame:
    sub = calls[(calls["model"] == model) & (calls["testset"] == f"{pdm.NAME}/{VERSION}")]
    df = analysis.item_scores(sub, items, pdm.NAME, cutoff)
    if df.empty:
        return pd.DataFrame(columns=["item_id", "true", "answer", "refused", "cluster"])
    answer = df["answer"].map(lambda v: pd.Period(v, "M") if isinstance(v, str) else None) if "answer" in df else None
    return pd.DataFrame({"item_id": df["item_id"], "true": df["item_id"].map(lambda i: pd.Period(items[i].truth["month"], "M")),
                         "answer": answer, "refused": df["refused"].astype(bool), "cluster": df["cluster"],
                         "first": df["item_id"].map(lambda i: pd.Period(items[i].truth["first_month"], "M"))})


def model_row(df: pd.DataFrame, cutoff, window: int = WINDOW) -> dict:
    x = excess(df, window)
    last = ladder.last_before(cutoff)
    before = x[x["true"] <= last].dropna(subset=["x"])
    mean, lo, hi, p = _mean_test(before["x"].reset_index(drop=True), before["cluster"].reset_index(drop=True), B=B) \
        if len(before) else (np.nan,) * 4
    a = x[~x["refused"]]
    hit = (a["answer"].notna()) & a.apply(lambda r: r["answer"] is not None and abs((r["answer"] - r["true"]).n) <= window
                                          if r["answer"] is not None else False, axis=1)
    crash = before[(before["first"] <= CRASH) & (before["true"] >= CRASH)]
    return {"n": len(before), "x": mean, "lo": lo, "hi": hi, "p": p,
            "answered": float(1 - x["refused"].mean()) if len(x) else np.nan,
            "hit_before": float(hit[a["true"] <= last].mean()) if (a["true"] <= last).any() else np.nan,
            "x_crash": float(crash["x"].mean()) if len(crash) else np.nan,
            "x_other": float(before.drop(crash.index)["x"].mean()) if len(before) > len(crash) else np.nan,
            "distinct": int(a["answer"].dropna().nunique())}


def compute(window: int = WINDOW) -> dict:
    """The registered test at the registered window; another window is an exploratory re-score (same rule, Holm
    across the same models)."""
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    items = {it.item_id: it for it in load_testset(pdm.NAME, VERSION)}
    rows = []
    for m in MODELS:
        df = frame(calls, m, items, resolve_cutoff(m, meta))
        rows.append({"model": m, **(model_row(df, resolve_cutoff(m, meta), window) if len(df) else {"n": 0})})
    tested = [r for r in rows if r["n"]]
    for r, p in zip(tested, stats.holm([r["p"] for r in tested])):
        r["p_holm"] = float(p)
        r["reading"] = "dates the unlabeled path" if (r["x"] > 0 and p < 0.05) else \
            "below its habits" if (r["x"] < 0 and p < 0.05) else "not shown"
    controls = []
    for m in CONTROLS:
        df = frame(calls, m, items, resolve_cutoff(m, meta))
        if len(df):
            controls.append({"model": m, **model_row(df, resolve_cutoff(m, meta), window)})
    return {"rows": rows, "controls": controls}


def render(res: dict) -> str:
    L = ["# Masking test (M20): can a model date an unlabeled market path?\n",
         "Registered in `docs/preregistration/2026-10-06-masking.md` before any answer. Each question: the 12 monthly "
         "changes of an unnamed Nordic index, no dates, no name, no levels; \"in which month does the period end?\" "
         "Measure: an answer within one month of the truth, minus how often the model's own answers land there anyway "
         "(the months next to its habitual answer left out; refusals left out). About 0 without memory. Test: the mean "
         "on paths ending before the model's cutoff edge, months resampled (20,000 draws), Holm across the models.\n",
         _table(["model", "paths before the edge (scored)", "answered", "different months named",
                 "within a month (raw)", "beyond its habits [95%]", "p (Holm)", "reading",
                 "paths through March 2020 / others"],
                [[short(r["model"]), r["n"], _f(r.get("answered"), "{:.0%}"), r.get("distinct", "–"),
                  _f(r.get("hit_before"), "{:.0%}"),
                  f"{_f(r.get('x'), '{:+.3f}')} [{_f(r.get('lo'), '{:+.3f}')}, {_f(r.get('hi'), '{:+.3f}')}]",
                  _f(r.get("p_holm"), "{:.3f}"), r.get("reading", "no answers"),
                  f"{_f(r.get('x_crash'), '{:+.3f}')} / {_f(r.get('x_other'), '{:+.3f}')}"] for r in res["rows"]]),
         "\n## Controls (local, cutoff 2024-06-30)\n",
         _table(["control", "paths", "beyond its habits [95%]"],
                [[r["model"], r["n"], f"{_f(r['x'], '{:+.3f}')} [{_f(r['lo'], '{:+.3f}')}, {_f(r['hi'], '{:+.3f}')}]"]
                 for r in res["controls"]]),
         "\nThe oracle (perfect memory before 2024-06-30) must date the paths; the random and blind controls must not. "
         "Paths through March 2020 (the pandemic crash) are shown apart as a descriptive check on famous events.\n"]
    return "\n".join(L)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["build"]:
        build()
        return
    text = render(compute())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "masking.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
