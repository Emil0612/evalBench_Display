"""M18 headline dating report: every analysis registered in docs/preregistration/2026-10-05-headline-dating.md, from
stored answers only (no model calls, no cost). Writes results/headline/pilot.md (--pilot) or report.md and
results.json.

uv run python -X utf8 -m evalbench.headline_report --pilot
uv run python -X utf8 -m evalbench.headline_report
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import analysis, headline_feasibility, ladder, stats, store
from .cutoffs import resolve_cutoff
from .index_report import _mean_test
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods import headline_date as hd
from .pilot_report import _f, _jsonable, _table, short

OUT = ROOT / "results" / "headline"
PILOT_MODELS = ["openai/gpt-3.5-turbo", "meta-llama/llama-3.3-70b-instruct", "google/gemini-2.5-flash",
                "deepseek/deepseek-chat-v3.1"]
TESTSET = "headline_date/v1"


def scored(calls: pd.DataFrame, model: str, cutoff, items: dict, testset: str = TESTSET, release=None) -> pd.DataFrame:
    """One row per item the model was asked: score, refused, signed error, the blind baseline's score, side
    (1 BEFORE: on or before the last whole month before cutoff - 91 days; 0 AFTER: after the release month; -1 gap).
    release: for a model outside the registered list (ladder.RELEASE), e.g. the new models of 2026-10-07."""
    sub = calls[(calls["model"] == model) & (calls["testset"] == testset)]
    df = analysis.item_scores(sub, items, hd.NAME, cutoff)
    if df.empty:
        return df
    df["month"] = pd.PeriodIndex(df["item_id"].map(lambda i: items[i].truth["month"]), freq="M")
    df["blind_score"] = df["item_id"].map(lambda i: hd.blind_score(items[i]))
    last, first = ladder.last_before(cutoff), ladder.first_after(release or ladder.RELEASE[model])
    df["side"] = np.where(df["month"] <= last, 1, np.where(df["month"] >= first, 0, -1))
    return df


def gate_row(df: pd.DataFrame) -> dict:
    before = df[df["side"] == 1]
    return {"answer_rate": float(1 - df["refused"].astype(bool).mean()), "n_before": len(before),
            "model_before": float(before["score"].mean()), "blind_before": float(before["blind_score"].mean())}


def gate_decision(rows: list[dict]) -> bool:
    answering = sum(r["answer_rate"] >= 0.70 for r in rows)
    beating = sum(r["model_before"] > r["blind_before"] for r in rows)
    return answering >= 3 and beating >= 2


def pilot_gate(calls, meta) -> dict:
    items = {it.item_id: it for it in load_testset(hd.NAME, "v1") if hd.is_pilot(it.item_id)}
    rows = []
    for m in PILOT_MODELS:
        df = scored(calls, m, resolve_cutoff(m, meta), items)
        rows.append({"model": m, **(gate_row(df) if len(df) else
                                    {"answer_rate": 0.0, "n_before": 0, "model_before": float("nan"),
                                     "blind_before": float("nan")})})
    return {"rows": rows, "go": gate_decision(rows)}


def render_pilot(g: dict) -> str:
    return "\n".join([
        "# M18 §0: micro-pilot gate\n",
        _table(["model", "answer rate", "BEFORE items", "model score (BEFORE)", "blind score (same items)"],
               [[short(r["model"]), _f(r["answer_rate"], "{:.0%}"), r["n_before"], _f(r["model_before"], "{:.3f}"),
                 _f(r["blind_before"], "{:.3f}")] for r in g["rows"]]),
        f"\n**Gate:** {'GO' if g['go'] else 'NO-GO'} (needs 3 of 4 answering ≥ 70% and 2 of 4 above the blind "
        "baseline).\n"])


MODELS = ladder.NEW + ladder.REFERENCES + ["openai/gpt-oss-20b", "qwen/qwen3.8-flash"]


def size_rows(df: pd.DataFrame, B: int = 20000) -> list[dict]:
    """Exploratory: the registered H1 comparison (model minus text-only baseline on BEFORE months, a refusal scoring 0)
    within each company-size group (df has a 'size' column), resampling months."""
    rows = []
    before = df[df["side"] == 1]
    for size, part in before.groupby("size"):
        mean, lo, hi, p = _mean_test(part["score"] - part["blind_score"], part["month"].astype(str), B=B)
        rows.append({"size": size, "n": len(part), "d": mean, "lo": lo, "hi": hi, "p": p,
                     "model_score": float(part["score"].mean()), "blind_score": float(part["blind_score"].mean())})
    return rows


def main_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["item_id"].map(hd.is_pilot)]


def h1_row(df: pd.DataFrame, B: int = 20000) -> dict:
    before = df[df["side"] == 1]
    d = before["score"] - before["blind_score"]
    mean, lo, hi, p = _mean_test(d, before["month"].astype(str), B=B)
    ans = before[~before["refused"].astype(bool)]
    a_mean, _, _, a_p = _mean_test(ans["score"] - ans["blind_score"], ans["month"].astype(str), B=B)
    return {"d": mean, "lo": lo, "hi": hi, "p": p, "answered_d": a_mean, "answered_p": a_p, "n": len(before),
            "model_score": float(before["score"].mean()), "blind_score": float(before["blind_score"].mean()),
            "answer_rate": float(1 - before["refused"].astype(bool).mean())}


def read_h1(rows: list[dict]) -> list[dict]:
    tested = [r for r in rows if pd.notna(r["p"])]
    for key, out in (("p", "p_holm"), ("answered_p", "answered_p_holm")):
        for r, p in zip(tested, stats.holm([r[key] for r in tested])):
            r[out] = float(p)
    for r in rows:
        if "p_holm" not in r:
            r["reading"] = "no estimate"
        elif r["p_holm"] < 0.05 and r["d"] > 0:
            ok = r["answered_p_holm"] < 0.05 and r["answered_d"] > 0
            r["reading"] = "dates better than the text alone" if ok else \
                "not confirmed: refusal-driven (dates better than the text alone)"
        else:
            r["reading"] = "not shown"
    return rows


def place(lo, hi, cutoff, release) -> str:
    """Where the horizon's interval falls (plan §4)."""
    if lo is None or hi is None:
        return "undetermined"
    last, rel = ladder.last_before(cutoff), pd.Period(release, "M")
    if hi <= last:
        return "at or before the cutoff"
    if lo > last and hi <= rel:
        return "in the gap"
    if lo > rel:
        return "after release"
    return "undetermined"


def horizon_row(model: str, cutoff, release, k: dict) -> dict:
    """One horizon row with its months as text (results.json is plain JSON)."""
    s = lambda p: str(p) if p is not None else None
    return {"model": model, "cutoff": str(cutoff), "release": str(release), "H": s(k["H"]), "lo": s(k.get("lo")),
            "hi": s(k.get("hi")), "slope": k.get("slope"),
            "place": place(k.get("lo"), k.get("hi"), cutoff, release) if k["H"] is not None else "no estimate"}


def compute(calls, meta) -> dict:
    items = {it.item_id: it for it in load_testset(hd.NAME, "v1")}
    res = {"gate": pilot_gate(calls, meta), "h1": [], "horizon": [], "e1": [], "e2": [], "e3": []}
    for m in MODELS:
        cut, rel = resolve_cutoff(m, meta), ladder.RELEASE[m]
        df = scored(calls, m, cut, items)
        if df.empty:
            continue
        df = main_only(df)
        res["h1"].append({"model": m, **h1_row(df)})
        ans = df[~df["refused"].astype(bool)]
        res["horizon"].append(horizon_row(m, cut, rel, headline_feasibility.boot_knee(ans[["month", "err_months"]])))
        for side, name in ((1, "BEFORE"), (-1, "gap"), (0, "AFTER")):
            part = df[df["side"] == side]
            if not len(part):
                continue
            a = part[~part["refused"].astype(bool)]
            blind_err = part["item_id"].map(
                lambda i: hd.month_error(items[i].truth["blind_month"], items[i].truth["month"]))
            res["e1"].append({"model": m, "side": name, "n": len(part), "exact": float(a["exact"].mean()),
                              "within_3": float(a["within_3"].mean()), "same_year": float(a["same_year"].mean()),
                              "blind_exact": float((blind_err == 0).mean()),
                              "blind_within_3": float((blind_err.abs() <= 3).mean())})
        before = df[df["side"] == 1]
        for col, groups in (("language", ("sv", "en")), ("newstype", ("macro", "editorial", "press_release"))):
            for g in groups:
                part = before[before["item_id"].map(lambda i: items[i].strata[col]) == g]
                if len(part) >= 30:
                    res["e2"].append({"model": m, "group": f"{col}={g}", "n": len(part),
                                      "d": float((part["score"] - part["blind_score"]).mean())})
        mode = ans["answer"].mode()
        res["e3"].append({"model": m, "mode": str(mode.iloc[0]) if len(mode) else "–",
                          "share": float((ans["answer"] == mode.iloc[0]).mean()) if len(mode) else 0.0})
    res["h1"] = read_h1(res["h1"])
    return res


def render(res: dict) -> str:
    L = ["# M18 headline dating report (plan docs/preregistration/2026-10-05-headline-dating.md)\n",
         render_pilot(res["gate"]), "## H1: inside its window, better than the text alone? (confirmatory)\n",
         _table(["model", "BEFORE items", "answered", "model / blind score", "d", "95% CI", "p (Holm)",
                 "d, answered (Holm p)", "reading"],
                [[short(r["model"]), r["n"], _f(r["answer_rate"], "{:.0%}"),
                  f"{_f(r['model_score'], '{:.3f}')} / {_f(r['blind_score'], '{:.3f}')}", _f(r["d"], "{:+.3f}"),
                  f"[{_f(r['lo'], '{:+.3f}')}, {_f(r['hi'], '{:+.3f}')}]", _f(r.get("p_holm"), "{:.4f}"),
                  f"{_f(r['answered_d'], '{:+.3f}')} ({_f(r.get('answered_p_holm'), '{:.4f}')})", r["reading"]]
                 for r in res["h1"]]),
         "\n## The horizon (primary estimate)\n",
         "The knee fit assumes a model's answers track the truth before its horizon. Read it next to E1 and E3: a "
         "model that falls back on a default month for most headlines has a sloping error from the start, and its "
         "knee then marks where its answers stop following the truth, not a knowledge horizon.\n",
         _table(["model", "stated cutoff", "release", "horizon H", "95% interval", "slope after H", "where"],
                [[short(r["model"]), r["cutoff"], r["release"], str(r["H"]), f"{r['lo']}..{r['hi']}",
                  _f(r["slope"], "{:+.2f}"), r["place"]] for r in res["horizon"]]),
         "\n## E1: exact / ±3 months / same year (answered), model vs blind\n",
         _table(["model", "side", "n", "exact", "±3 months", "same year", "blind exact", "blind ±3"],
                [[short(r["model"]), r["side"], r["n"], _f(r["exact"], "{:.0%}"), _f(r["within_3"], "{:.0%}"),
                  _f(r["same_year"], "{:.0%}"), _f(r["blind_exact"], "{:.0%}"), _f(r["blind_within_3"], "{:.0%}")]
                 for r in res["e1"]]),
         "\n## E2: H1's d by language and type (BEFORE items)\n",
         _table(["model", "group", "n", "d"], [[short(r["model"]), r["group"], r["n"], _f(r["d"], "{:+.3f}")]
                                               for r in res["e2"]]),
         "\n## E3: most frequent answer month\n",
         _table(["model", "month", "share of answers"], [[short(r["model"]), r["mode"], _f(r["share"], "{:.0%}")]
                                                        for r in res["e3"]])]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    OUT.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        text = render_pilot(pilot_gate(calls, meta))
        (OUT / "pilot.md").write_text(text, encoding="utf-8")
        print(text)
        return
    res = compute(calls, meta)
    (OUT / "results.json").write_text(json.dumps(_jsonable(res), indent=2), encoding="utf-8")
    text = render(res)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
