"""Pilot C report: every analysis registered in docs/preregistration/2026-09-29-pilot-c.md, from the stored
answers only (no model calls, no cost). Writes results/pilot_c/results.json and results/pilot_c/report.md.

uv run python -m evalbench.pilot_report            # the pilot's four models
uv run python -m evalbench.pilot_report --controls # dry run on the three control models (checks the pipeline)
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd

from . import analysis, pilot, stats, store
from .cutoffs import resolve_cutoff
from .data import company_size_map, load_data
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods import LABELS, METHODS

OUT = ROOT / "results" / "pilot_c"
LLAMA, GPT = "meta-llama/llama-3.3-70b-instruct", "openai/gpt-4.1-mini"
GEMINI, DEEPSEEK = "google/gemini-2.5-flash", "deepseek/deepseek-chat-v3.1"
MODELS = [LLAMA, GPT, GEMINI, DEEPSEEK]
PAIRS = [(LLAMA, GEMINI), (LLAMA, DEEPSEEK), (GPT, GEMINI), (GPT, DEEPSEEK)]      # S2, as registered
RELEASE = {LLAMA: date(2024, 12, 6), GPT: date(2025, 4, 14), GEMINI: date(2025, 6, 17), DEEPSEEK: date(2025, 8, 21)}
CONTROLS = ["mock/oracle@2024-06-30", "mock/random@2024-06-30", "baseline/blind@2024-06-30"]
BLIND = "baseline/blind@2024-06-30"
B_MAIN, B_PROFILE = 2000, 20000
# the test sets registered in the plan: always these, even after newer versions are built
PILOT_SETS = {"price_recall": "v1", "number_cloze": "v1", "decop": "v1", "completion": "v1"}


def short(model: str) -> str:
    return model.split("/")[-1].split("@")[0]


def compute(models: list[str], pairs: list[tuple[str, str]], release: dict) -> dict:
    ts = {m: (f"{m}/{v}", load_testset(m, v)) for m, v in PILOT_SETS.items()}
    methods = list(ts)
    conn = store.connect()
    calls = store.load_calls(conn, mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    cutoffs = {m: resolve_cutoff(m, meta) for m in [*models, BLIND]}
    data = load_data()
    sizes = company_size_map(data.universe)
    items = {m: {it.item_id: it for it in ts[m][1]} for m in methods}

    def scored(model: str, method: str) -> pd.DataFrame:
        sub = calls[(calls["model"] == model) & (calls["testset"] == ts[method][0])]
        return analysis.item_scores(sub, items[method], method, cutoffs[model], extra_strata={"size": sizes})

    sc = {(m, me): scored(m, me) for m in models for me in methods}
    blind = {me: dict(zip(df["item_id"], df["score"])) for me in methods if len(df := scored(BLIND, me))}
    res: dict = {"models": models, "cutoffs": {m: str(cutoffs[m]) for m in models},
                 "release": {m: str(release.get(m)) for m in models}, "methods": methods}

    # §2 primary: 16 cells, Holm across them
    primary = analysis.summary_table(calls, ts, models, methods, cutoffs, B=B_MAIN)
    primary["reading"] = primary.apply(pilot.primary_reading, axis=1)
    res["primary"] = primary.to_dict("records")

    # S1 tolerance bands + median log error, S8 unadjusted convention (M1)
    res["s1"] = {m: {"bands": pilot.tolerance_bands(sc[(m, "price_recall")], B=B_MAIN).to_dict("records"),
                     "median_log_error": pilot.median_log_error(sc[(m, "price_recall")])} for m in models}
    s8 = [{"model": m, **pilot.raw_convention(sc[(m, "price_recall")], B=B_MAIN)} for m in models]
    for row, p in zip(s8, stats.holm([r["p_value"] for r in s8])):
        row["p_holm"] = float(p)
    res["s8"] = s8

    # S2 staggered-cutoff comparison, Holm across pairs (pooled)
    s2 = []
    for a, b in pairs:
        r = pilot.staggered({me: (sc[(a, me)], sc[(b, me)]) for me in methods}, cutoffs[a], cutoffs[b], B=B_MAIN)
        s2.append({"earlier": a, "later": b, **r})
    for row, p in zip(s2, stats.holm([r["pooled"]["p_value"] for r in s2])):
        row["pooled"]["p_holm"] = float(p)
    res["s2"] = s2

    # S3 placebo, Holm across all model x method cells
    s3 = [{"model": m, "method": me, **pilot.placebo(sc[(m, me)], me, B=B_MAIN)} for m in models for me in methods]
    for row, p in zip(s3, stats.holm([r["p_value"] for r in s3])):
        row["p_holm"] = float(p)
    res["s3"] = s3

    # S4 boilerplate split (M2, M3, M4)
    headline = dict(zip(data.news["newsid"], data.news["headline"]))
    flags = {me: {i: pilot.is_boilerplate(it.payload.get("headline") or headline.get(it.truth.get("newsid")))
                  for i, it in items[me].items()} for me in ("number_cloze", "decop", "completion")}
    res["s4"] = [{"model": m, "method": me, "n_flagged_items": sum(flags[me].values()),
                  **{k: _brief(v) for k, v in pilot.boilerplate_split(sc[(m, me)], flags[me], me, B=B_MAIN).items()}}
                 for m in models for me in flags]

    # S5 Split Clock first look (M1)
    events = pilot.adjustment_events(data.prices, min_factor=2.0)
    res["s5"] = [{"model": m, **{f"{side}|{kind}": n for (side, kind), n in
                                 pilot.split_clock(sc[(m, "price_recall")], items["price_recall"], events,
                                                   cutoffs[m]).items()}} for m in models]

    # S6 built-in: knowledge boundary and company-size breakdown
    res["s6_boundary"] = {m: _jsonable(analysis.effective_cutoff([(sc[(m, me)], METHODS[me].P0) for me in methods]))
                          for m in models}
    res["s6_size"] = [{"model": m, "method": me, **r} for m in models for me in methods
                      for r in analysis.breakdown(sc[(m, me)], me, "size", B=B_MAIN).to_dict("records")]

    # S7 per-period profile (B1, B1r, B2), one Holm family per baseline; M1 also under the unadjusted convention
    cells = []
    for m in models:
        for me in methods:
            cells += pilot.profile_cells(sc[(m, me)], me, m, cutoffs[m], release.get(m), blind.get(me), B=B_PROFILE)
    res["s7"] = pilot.label_profile(cells)
    raw_cells = []
    for m in models:
        df = sc[(m, "price_recall")]
        df = df.assign(score=(pd.to_numeric(df.get("rel_err_raw"), errors="coerce") <= 0.01 + 1e-12).astype(float))
        raw_cells += pilot.profile_cells(df, "price_recall", m, cutoffs[m], release.get(m), None, B=B_PROFILE)
    res["s7_raw_price"] = pilot.label_profile(raw_cells)
    return res


def _brief(s: dict) -> dict:
    return {k: s.get(k) for k in ("n_pre", "n_post", "dkappa", "ci_lo", "ci_hi", "p_value")}


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, float) and math.isnan(v):
        return None
    if hasattr(v, "item"):   # numpy scalars
        return _jsonable(v.item())
    if isinstance(v, date):
        return v.isoformat()
    return v


# ---------------------------------------------------------------- markdown
def _f(v, fmt="{:+.2f}") -> str:
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v)


def _table(header: list[str], rows: list[list]) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                      *("| " + " | ".join(str(c) for c in r) + " |" for r in rows)])


def render(res: dict) -> str:
    L = [f"# Pilot C report (exploratory)\n\nModels and cutoffs: " +
         ", ".join(f"{short(m)} {res['cutoffs'][m]}" for m in res["models"]) + "\n"]
    L.append("## §2 Primary: Δκ per model × method (Holm across 16)\n")
    L.append(_table(["model", "method", "PRE", "POST", "Δκ", "95% CI", "p (Holm)", "reading"],
                    [[short(r["model"]), LABELS[r["method"]], r["n_pre"], r["n_post"], _f(r["dkappa"]),
                      f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]", _f(r["p_holm"], "{:.3f}"), r["reading"]]
                     for r in res["primary"]]))
    L.append("\n**Outcomes and hallucination (S6):**\n")
    L.append(_table(["model", "method", "abstain", "refusal", "malformed", "hallucination (fake cos)"],
                    [[short(r["model"]), LABELS[r["method"]], _f(r.get("abstain_rate"), "{:.1%}"),
                      _f(r.get("refusal_rate"), "{:.1%}"), _f(r.get("malformed_rate"), "{:.1%}"),
                      _f(r.get("hallucination_rate"), "{:.0%}")] for r in res["primary"]]))
    L.append("\n## S1 M1 tolerance bands (hit rate PRE → POST, Δκ)\n")
    rows = []
    for m, v in res["s1"].items():
        rows.append([short(m)] + [f"{_f(b['hit_pre'], '{:.0%}')} → {_f(b['hit_post'], '{:.0%}')} ({_f(b['dkappa'])})"
                                  for b in v["bands"]] +
                    [f"{_f(v['median_log_error']['pre'], '{:.2f}')} / {_f(v['median_log_error']['post'], '{:.2f}')}"])
    L.append(_table(["model", "±1%", "±3%", "±5%", "±10%", "median |log err| PRE / POST"], rows))
    L.append("\n## S8 M1 under the unadjusted convention (Holm across 4)\n")
    L.append(_table(["model", "Δκ", "95% CI", "p (Holm)"],
                    [[short(r["model"]), _f(r["dkappa"]), f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]",
                      _f(r["p_holm"], "{:.3f}")] for r in res["s8"]]))
    L.append("\n## S2 Staggered-cutoff comparison (later − earlier, window minus reference; chance-corrected)\n")
    rows = []
    for r in res["s2"]:
        per = "; ".join(f"{LABELS[me]} {_f(v['did'])} (n={v['n_window']})" if v else f"{LABELS[me]} thin"
                        for me, v in r.items() if me in res["methods"])
        p = r["pooled"]
        rows.append([f"{short(r['earlier'])} → {short(r['later'])}", p["n_window"], p["n_ref"], _f(p["did"]),
                     f"[{_f(p['ci_lo'])}, {_f(p['ci_hi'])}]", _f(p.get("p_holm"), "{:.3f}"), per])
    L.append(_table(["pair", "window items", "reference items", "pooled DiD", "95% CI", "p (Holm)", "per method"],
                    rows))
    L.append("\n## S3 Placebo cutoff 2022-07-01 (expected ≈ 0; Holm across 16)\n")
    L.append(_table(["model", "method", "Δκ", "95% CI", "p (Holm)"],
                    [[short(r["model"]), LABELS[r["method"]], _f(r["dkappa"]), f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]",
                      _f(r["p_holm"], "{:.3f}")] for r in res["s3"]]))
    L.append("\n## S4 Boilerplate split (Δκ flagged vs unflagged)\n")
    L.append(_table(["model", "method", "flagged items", "Δκ flagged", "Δκ unflagged [CI]"],
                    [[short(r["model"]), LABELS[r["method"]], r["n_flagged_items"], _f(r["flagged"]["dkappa"]),
                      f"{_f(r['unflagged']['dkappa'])} [{_f(r['unflagged']['ci_lo'])}, {_f(r['unflagged']['ci_hi'])}]"]
                     for r in res["s4"]]))
    L.append("\n## S5 Split Clock first look (answered M1 items with a ≥2× adjustment after the item date)\n")
    kinds = ["raw", "adjusted", "neither"]
    L.append(_table(["model"] + [f"event {s} cutoff: {k}" for s in ("before", "after") for k in kinds],
                    [[short(r["model"])] + [r.get(f"event_{s}_cutoff|{k}", 0) for s in ("before", "after") for k in kinds]
                     for r in res["s5"]]))
    L.append("\n## S6 Estimated knowledge boundary (all methods pooled)\n")
    L.append(_table(["model", "stated cutoff", "detected", "breakpoint", "last known", "first unknown"],
                    [[short(m), res["cutoffs"][m], b.get("detected"), b.get("breakpoint", "–"), b.get("last_known", "–"),
                      b.get("first_unknown", "–")] for m, b in res["s6_boundary"].items()]))
    L.append("\n## S7 Per-period profile (recall | B1 own POST baseline | reading; B1r reading)\n")
    for me in res["methods"]:
        L.append(f"\n**{LABELS[me]}** (fact type: {pilot.FACT_TYPES[me]})\n")
        years = sorted({c["period"] for c in res["s7"] if c["method"] == me})
        rows = []
        for m in res["models"]:
            row = [short(m)]
            for y in years:
                c = next((c for c in res["s7"] if c["model"] == m and c["method"] == me and c["period"] == y), None)
                if c is None:
                    row.append("·")
                    continue
                tag = {"recall_above_baseline": "● above", "none_detected": "○ none", "insufficient_coverage": "– n/a",
                       "unresolved": "? unresolved", "after_cutoff": "after cutoff"}[c["label"]]
                extra = f" ≤{_f(c.get('detection_upper_bound'))}" if c["label"] == "none_detected" else ""
                rel = {"recall_above_baseline": " / B1r ●", "none_detected": " / B1r ○"}.get(c.get("label_release"), "")
                row.append(f"{_f(c['recall'], '{:.2f}')} (n={c['facts']}) {tag}{extra}{rel}")
            rows.append(row)
        L.append(_table(["model"] + years, rows))
        L.append(f"\nB1 (own POST level): " + ", ".join(
            f"{short(m)} {_f(next(c['baseline'] for c in res['s7'] if c['model'] == m and c['method'] == me), '{:.2f}')}"
            for m in res["models"]))
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", action="store_true", help="dry run on the control models")
    args = ap.parse_args(argv)
    models, pairs, release = (CONTROLS, [], {m: date(2024, 12, 31) for m in CONTROLS}) if args.controls \
        else (MODELS, PAIRS, RELEASE)
    res = _jsonable(compute(models, pairs, release))
    out = OUT / ("controls" if args.controls else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "report.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
