"""Pilot D report: every analysis registered in docs/preregistration/2026-09-29-pilot-d.md, from the stored answers
only (no model calls, no cost). Writes results/pilot_d/results.json and results/pilot_d/report.md.

uv run python -m evalbench.pilot_d_report            # the pilot's six models
uv run python -m evalbench.pilot_d_report --controls # dry run on the three control models (checks the pipeline)
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from itertools import combinations

import pandas as pd

from . import analysis, pilot, pilot_d, stats, store
from .cutoffs import resolve_cutoff
from .data import company_size_map, load_data
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods import LABELS, METHODS
from .pilot_report import _f, _jsonable, _table, short

OUT = ROOT / "results" / "pilot_d"
LLAMA = "meta-llama/llama-3.3-70b-instruct"
MODELS = [LLAMA, "openai/gpt-4.1-mini", "openai/gpt-oss-20b", "google/gemini-2.5-flash",
          "deepseek/deepseek-chat-v3.1", "qwen/qwen3.8-flash"]
PROVISIONAL = {"qwen/qwen3.8-flash"}
RELEASE = {LLAMA: date(2024, 12, 6), "openai/gpt-4.1-mini": date(2025, 4, 14), "openai/gpt-oss-20b": date(2025, 8, 5),
           "google/gemini-2.5-flash": date(2025, 6, 17), "deepseek/deepseek-chat-v3.1": date(2025, 8, 21),
           "qwen/qwen3.8-flash": date(2026, 8, 26)}
CONTROLS = ["mock/oracle@2024-06-30", "mock/random@2024-06-30", "baseline/blind@2024-06-30"]
BLIND = "baseline/blind@2024-06-30"
# the sets registered in the plan: always these, even after newer versions are built
SETS = {"price_recall": "v1", "number_cloze": "v2", "decop": "v2", "completion": "v1", "price_level": "v1",
        "annual_return": "v1", "known_return": "v1", "split_clock": "v1"}
E1_METHODS = ["price_recall", "number_cloze", "decop", "price_level", "annual_return", "split_clock"]
N_ITEMS, B_MAIN, B_H1 = 150, 2000, 20000


def compute(models: list[str], release: dict, controls: bool = False) -> dict:
    ts = {m: (f"{m}/{v}", load_testset(m, v)[:N_ITEMS]) for m, v in SETS.items()}
    items = {m: {it.item_id: it for it in ts[m][1]} for m in ts}
    conn = store.connect()
    forced = store.load_calls(conn, mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    abstain = store.load_calls(conn, mode="abstain", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    cutoffs = {m: resolve_cutoff(m, meta) for m in [*models, BLIND]}
    data = load_data()
    sizes = company_size_map(data.universe)

    def scored(calls: pd.DataFrame, model: str, method: str) -> pd.DataFrame:
        sub = calls[(calls["model"] == model) & (calls["testset"] == ts[method][0])]
        df = analysis.item_scores(sub, items[method], method, cutoffs[model], extra_strata={"size": sizes})
        return pilot_d.drop_excluded(df, method)

    sc = {(m, me): scored(forced, m, me) for m in models for me in ts}
    blind = {me: scored(forced, BLIND, me) for me in ts}
    res: dict = {"models": models, "cutoffs": {m: str(cutoffs[m]) for m in models}, "provisional": sorted(PROVISIONAL),
                 "sets": {m: ts[m][0] for m in ts}, "excluded": {k: sorted(v) for k, v in pilot_d.EXCLUDED.items()}}

    # H1 (confirmatory): Llama on M3 v2, 20,000 draws, no multiplicity correction
    if LLAMA in models:
        h = analysis.summarize(sc[(LLAMA, "decop")], "decop", B=B_H1)
        res["h1"] = {**{k: h[k] for k in ("n_pre", "n_post", "mean_pre", "mean_post", "dkappa", "ci_lo", "ci_hi",
                                         "p_value")}, "confirmed": bool(h["p_value"] < 0.05 and h["dkappa"] > 0)}

    # E1: 36 cells, Holm across them; M4 separately
    e1 = analysis.summary_from_frames({(m, me): sc[(m, me)] for m in models for me in E1_METHODS}, B=B_MAIN,
                                      family="all")   # registered: Holm across the 36 cells
    e1["reading"] = e1.apply(pilot.primary_reading, axis=1)
    e1["provisional_cutoff"] = e1["model"].isin(PROVISIONAL)
    res["e1"] = e1.to_dict("records")
    m4 = analysis.summary_from_frames({(m, "completion"): sc[(m, "completion")] for m in models}, B=B_MAIN,
                                      family="all")
    m4["reading"] = m4.apply(pilot.primary_reading, axis=1)
    res["m4"] = m4.to_dict("records")

    # E2: positive control
    blind_kr = float(blind["known_return"]["score"].mean()) if len(blind["known_return"]) else float("nan")
    res["e2"] = [{"model": m, **pilot_d.positive_control(sc[(m, "known_return")], blind_kr, B=B_H1)} for m in models]

    # E3: Split Clock beyond the cutoff, and the split-type / size breakdown of the E1 cell
    elig = data.universe[data.universe["is_primary"] & (data.universe["instrumenttype"] == 4)]["insref"]
    px = data.prices[data.prices["insref"].isin(elig)]
    median_by_month = px.groupby(px["date"].dt.to_period("M").astype(str))["close"].median().to_dict()
    guess = pilot_d.typical_guesser(items["split_clock"], median_by_month)
    reverse = {i for i, it in items["split_clock"].items() if it.truth["factor"] < 1}
    e3 = []
    for m in models:
        df = sc[(m, "split_clock")]
        if not cutoffs[m] or not len(df):
            continue
        for name, keep in (("all", lambda ids: ids == ids), ("ordinary", lambda ids: ~ids.isin(reverse)),
                           ("reverse", lambda ids: ids.isin(reverse))):
            r = pilot_d.split_clock_beyond(df[keep(df["item_id"])], guess[keep(guess["item_id"])], cutoffs[m], B=B_H1)
            ref = pilot_d.split_clock_beyond(df[keep(df["item_id"])], blind["split_clock"][keep(blind["split_clock"]
                                                                                                   ["item_id"])],
                                             cutoffs[m], B=B_MAIN)
            e3.append({"model": m, "type": name, **r, "guesser_share": r["blind_share"],
                       "blind_share": ref["blind_share"]})
    res["e3"] = e3
    kinds = []
    for m in models:
        df = sc[(m, "split_clock")]
        if not len(df):
            continue
        reverse = df["item_id"].map(lambda i: items["split_clock"][i].truth["factor"] < 1)
        for name, part in (("ordinary split", df[~reverse]), ("reverse split", df[reverse])):
            kinds.append({"model": m, "type": name, **analysis.summarize(part, "split_clock", B=B_MAIN)})
    res["e3_types"] = kinds

    # E4: M3 quality, after-cutoff pick rate v2 against pilot C's v1
    v1 = {it.item_id: it for it in load_testset("decop", "v1")}
    e4 = []
    for m in models:
        old = analysis.item_scores(forced[(forced["model"] == m) & (forced["testset"] == "decop/v1")], v1, "decop",
                                   cutoffs[m])
        new = sc[(m, "decop")]
        e4.append({"model": m, "v1_after": float(old.loc[old["group"] == 0, "score"].mean()) if len(old) else None,
                   "v2_after": float(new.loc[new["group"] == 0, "score"].mean()) if len(new) else None,
                   "v1_before": float(old.loc[old["group"] == 1, "score"].mean()) if len(old) else None,
                   "v2_before": float(new.loc[new["group"] == 1, "score"].mean()) if len(new) else None})
    res["e4"] = e4

    # E5: pilot C's checks
    res["s1"] = {m: {"m1": pilot.tolerance_bands(sc[(m, "price_recall")], B=B_MAIN).to_dict("records"),
                     "m5": pilot.tolerance_bands(sc[(m, "price_level")], bands=pilot_d.LEVEL_BANDS, method="price_level",
                                                 B=B_MAIN).to_dict("records"),
                     "log_error_m1": pilot.median_log_error(sc[(m, "price_recall")]),
                     "log_error_m5": pilot.median_log_error(sc[(m, "price_level")])} for m in models}
    s8 = [{"model": m, "method": me, **pilot.raw_convention(sc[(m, me)], B=B_MAIN, tolerance=tol, method=me)}
          for m in models for me, tol in (("price_recall", 0.01), ("price_level", 0.10))]
    for row, p in zip(s8, stats.holm([r["p_value"] for r in s8])):
        row["p_holm"] = float(p)
    res["s8"] = s8
    pair_models = [m for m in models if m not in PROVISIONAL and cutoffs[m]]
    s2 = []
    for a, b in combinations(sorted(pair_models, key=lambda m: cutoffs[m]), 2):
        if cutoffs[a] < cutoffs[b]:
            r = pilot.staggered({me: (sc[(a, me)], sc[(b, me)]) for me in E1_METHODS}, cutoffs[a], cutoffs[b], B=B_MAIN)
            if r["pooled"] and r["pooled"]["n_window"]:
                s2.append({"earlier": a, "later": b, **r})
    for row, p in zip(s2, stats.holm([r["pooled"]["p_value"] for r in s2])):
        row["pooled"]["p_holm"] = float(p)
    res["s2"] = s2
    s3 = [{"model": m, "method": me, **pilot.placebo(sc[(m, me)], me, B=B_MAIN)} for m in models for me in E1_METHODS]
    for row, p in zip(s3, stats.holm([r["p_value"] for r in s3])):
        row["p_holm"] = float(p)
    res["s3"] = s3
    headline = dict(zip(data.news["newsid"], data.news["headline"]))
    flags = {me: {i: pilot.is_boilerplate(it.payload.get("headline") or headline.get(it.truth.get("newsid")))
                  for i, it in items[me].items()} for me in ("number_cloze", "decop", "completion")}
    res["s4"] = [{"model": m, "method": me, "n_flagged_items": sum(flags[me].values()),
                  **{k: {x: v.get(x) for x in ("n_pre", "n_post", "dkappa", "ci_lo", "ci_hi")}
                     for k, v in pilot.boilerplate_split(sc[(m, me)], flags[me], me, B=B_MAIN).items()}}
                 for m in models for me in flags]
    res["s6"] = {m: _jsonable(analysis.effective_cutoff([(sc[(m, me)], METHODS[me].P0) for me in E1_METHODS]))
                 for m in models}
    cells = []
    for m in models:
        for me in E1_METHODS:
            bl = dict(zip(blind[me]["item_id"], blind[me]["score"])) if len(blind[me]) else None
            cells += pilot.profile_cells(sc[(m, me)], me, m, cutoffs[m], release.get(m), bl, B=B_H1)
    res["s7"] = pilot.label_profile(cells)
    vs_median = []
    for m in models:
        df = sc[(m, "annual_return")]
        if len(df) and "vs_median_hit" in df:
            part = df[df["vs_median_hit"].notna()].assign(score=lambda d: d["vs_median_hit"].astype(float))
            vs_median.append({"model": m, **analysis.summarize(part, "annual_return", B=B_MAIN)})
    res["m6_vs_median"] = vs_median

    # E6: hallucination in abstain mode (the second run)
    res["e6"] = [{"model": m, "method": me, **pilot_d.hallucination(scored(abstain, m, me))}
                 for m in models for me in ("price_recall", "price_level", "annual_return")
                 if len(abstain) and len(scored(abstain, m, me))]
    return res


# ---------------------------------------------------------------- markdown
def render(res: dict) -> str:
    prov = set(res.get("provisional", []))
    L = ["# Pilot D report\n", "Models and cutoffs: " + ", ".join(
        f"{short(m)} {res['cutoffs'][m]}" + (" (provisional)" if m in prov else "") for m in res["models"]),
         "\nSets: " + ", ".join(f"{LABELS[m]} {s}" for m, s in res["sets"].items()),
         f"\nExcluded in advance: {res['excluded']}\n"]
    if "h1" in res:
        h = res["h1"]
        L.append("## H1 (confirmatory): Llama 3.3 on M3 v2\n")
        L.append(f"**{'CONFIRMED' if h['confirmed'] else 'NOT CONFIRMED'}**: Δκ {_f(h['dkappa'])} "
                 f"[{_f(h['ci_lo'])}, {_f(h['ci_hi'])}], p = {_f(h['p_value'], '{:.4f}')}, "
                 f"pick rate {_f(h['mean_pre'], '{:.0%}')} before vs {_f(h['mean_post'], '{:.0%}')} after "
                 f"({h['n_pre']} / {h['n_post']} items; 20,000 draws, no multiplicity correction).\n")
    L.append("## E1: main table (Holm across 36 cells)\n")
    L.append(_table(["model", "method", "PRE", "POST", "Δκ", "95% CI", "p (Holm)", "reading"],
                    [[short(r["model"]) + (" *" if r.get("provisional_cutoff") else ""), LABELS[r["method"]],
                      r["n_pre"], r["n_post"], _f(r["dkappa"]), f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]",
                      _f(r["p_holm"], "{:.3f}"), r["reading"]] for r in res["e1"]]))
    L.append("\n\\* provisional cutoff.\n\n**M4 (secondary, Holm across 6):** " + "; ".join(
        f"{short(r['model'])} {r['reading']}" for r in res["m4"]) + "\n")
    L.append("\n**Outcomes (forced mode):**\n")
    L.append(_table(["model", "method", "abstain", "refusal", "malformed"],
                    [[short(r["model"]), LABELS[r["method"]], _f(r.get("abstain_rate"), "{:.1%}"),
                      _f(r.get("refusal_rate"), "{:.1%}"), _f(r.get("malformed_rate"), "{:.1%}")] for r in res["e1"]]))
    L.append("\n## E2: positive control (M7, direction right)\n")
    L.append(_table(["model", "n", "accuracy", "95% CI", "blind", "within 10 pp", "vs median", "sensitive"],
                    [[short(r["model"]), r["n"], _f(r["accuracy"], "{:.0%}"),
                      f"[{_f(r['ci_lo'], '{:.0%}')}, {_f(r['ci_hi'], '{:.0%}')}]", _f(r.get("blind"), "{:.0%}"),
                      _f(r.get("within_10pp"), "{:.0%}"), _f(r.get("vs_median_hit"), "{:.0%}"),
                      "yes" if r["sensitive"] else "no"] for r in res["e2"]]))
    L.append("\n## E3: Split Clock, splits after the cutoff (rewritten-price share vs the typical-price guesser)\n")
    L.append(_table(["model", "split type", "splits after cutoff", "share rewritten", "typical-price guesser",
                     "difference 95% CI", "blind (reference)", "counts"],
                    [[short(r["model"]), r["type"], r["n"], _f(r["share"], "{:.0%}"), _f(r.get("guesser_share"), "{:.0%}"),
                      f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]", _f(r["blind_share"], "{:.0%}"), r["counts"]]
                     for r in res["e3"]]))
    L.append("\n" + _table(["model", "split type", "PRE", "POST", "Δκ", "95% CI"],
                           [[short(r["model"]), r["type"], r["n_pre"], r["n_post"], _f(r["dkappa"]),
                             f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]"] for r in res["e3_types"]]))
    L.append("\n## E4: M3 quality (pick rate of the original; chance 25%)\n")
    L.append(_table(["model", "v1 before", "v1 after", "v2 before", "v2 after"],
                    [[short(r["model"])] + [_f(r[k], "{:.0%}") for k in ("v1_before", "v1_after", "v2_before",
                                                                          "v2_after")] for r in res["e4"]]))
    L.append("\n## E5: checks\n\n**S3 placebo (Holm across 36):** " + (
        ", ".join(f"{short(r['model'])} {LABELS[r['method']]} {_f(r['dkappa'])} (p {_f(r['p_holm'], '{:.3f}')})"
                  for r in res["s3"] if r.get("p_holm", 1) < 0.05) or "no significant placebo cell") + "\n")
    L.append("\n**S2 staggered cutoffs (pooled, Holm across pairs):**\n")
    L.append(_table(["pair", "window", "reference", "DiD", "95% CI", "p (Holm)"],
                    [[f"{short(r['earlier'])} → {short(r['later'])}", r["pooled"]["n_window"], r["pooled"]["n_ref"],
                      _f(r["pooled"]["did"]), f"[{_f(r['pooled']['ci_lo'])}, {_f(r['pooled']['ci_hi'])}]",
                      _f(r["pooled"].get("p_holm"), "{:.3f}")] for r in res["s2"]]))
    L.append("\n**S1 tolerance bands (hit rate before → after):**\n")
    rows = []
    for m, v in res["s1"].items():
        rows.append([short(m)] + [f"{_f(b['hit_pre'], '{:.0%}')} → {_f(b['hit_post'], '{:.0%}')}" for b in v["m1"]]
                    + [f"{_f(b['hit_pre'], '{:.0%}')} → {_f(b['hit_post'], '{:.0%}')}" for b in v["m5"]])
    L.append(_table(["model", "M1 ±1%", "±3%", "±5%", "±10%", "M5 ±5%", "±10%", "±25%"], rows))
    L.append("\n**S8 unadjusted convention (Holm across 12):** " + "; ".join(
        f"{short(r['model'])} {LABELS[r['method']].split()[0]} {_f(r['dkappa'])} (p {_f(r['p_holm'], '{:.3f}')})"
        for r in res["s8"]) + "\n")
    L.append("\n**M6 vs median (cancels market-wide knowledge):** " + "; ".join(
        f"{short(r['model'])} {_f(r['dkappa'])} [{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]" for r in res["m6_vs_median"])
             + "\n")
    L.append("\n**S6 estimated knowledge boundary:**\n")
    L.append(_table(["model", "stated cutoff", "detected", "breakpoint"],
                    [[short(m), res["cutoffs"][m], b.get("detected"), b.get("breakpoint", "–")]
                     for m, b in res["s6"].items()]))
    L.append("\n**S7 profile, cells above baseline (B1 / B1r):** " + (", ".join(
        f"{short(c['model'])} {LABELS[c['method']].split()[0]} {c['period']}"
        + (" (also B1r)" if c.get("label_release") == "recall_above_baseline" else "")
        for c in res["s7"] if c["label"] == "recall_above_baseline") or "none") + "\n")
    if res.get("e6"):
        L.append("\n## E6: hallucination (abstain mode)\n")
        L.append(_table(["model", "method", "fake companies", "answered anyway", "abstain before", "abstain after"],
                        [[short(r["model"]), LABELS[r["method"]], r["n_fabricated"],
                          _f(r["hallucination_rate"], "{:.0%}"), _f(r["abstain_pre"], "{:.0%}"),
                          _f(r["abstain_post"], "{:.0%}")] for r in res["e6"]]))
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", action="store_true", help="dry run on the control models")
    args = ap.parse_args(argv)
    models, release = (CONTROLS, {m: date(2024, 12, 31) for m in CONTROLS}) if args.controls else (MODELS, RELEASE)
    res = _jsonable(compute(models, release, controls=args.controls))
    out = OUT / ("controls" if args.controls else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "report.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
