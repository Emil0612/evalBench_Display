"""Pilot E report: every analysis registered in docs/preregistration/2026-09-30-pilot-e.md (including deviation 1,
the refusal guard), from the stored answers only (no model calls, no cost). Writes results/pilot_e/results.json and
results/pilot_e/report.md.

uv run python -m evalbench.pilot_e_report            # the pilot's six models
uv run python -m evalbench.pilot_e_report --controls # dry run on the three control models (checks the pipeline)
"""
from __future__ import annotations

import argparse
import json
from datetime import date

import numpy as np
import pandas as pd

from . import analysis, pilot, pilot_d, stats, store
from .cutoffs import resolve_cutoff
from .data import load_data
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods import LABELS
from .pilot_d_report import CONTROLS, MODELS, PROVISIONAL, RELEASE
from .pilot_e import LEVEL_METHODS, guarded, refusal_affected, release_groups, with_score
from .pilot_report import _f, _jsonable, _table, short

OUT = ROOT / "results" / "pilot_e"
LLAMA, DEEPSEEK, BLIND = "meta-llama/llama-3.3-70b-instruct", "deepseek/deepseek-chat-v3.1", "baseline/blind@2024-06-30"
SETS = {"price_level": "v2", "known_level": "v1", "us_level": "v1", "split_clock": "v2", "monthly_return": "v1",
        "known_return": "v1"}
E1_METHODS = ("price_level", "split_clock", "monthly_return")
BANDS = (0.03, 0.05, 0.10, 0.25)
B_MAIN, B_H = 2000, 20000


def compute(models: list[str], release: dict, h_models: tuple[str, str]) -> dict:
    ts = {m: (f"{m}/{v}", load_testset(m, v)) for m, v in SETS.items()}
    items = {m: {it.item_id: it for it in ts[m][1]} for m in ts}
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    cutoffs = {m: resolve_cutoff(m, meta) for m in [*models, BLIND]}

    def scored(model: str, method: str) -> pd.DataFrame:
        sub = calls[(calls["model"] == model) & (calls["testset"] == ts[method][0])]
        return analysis.item_scores(sub, items[method], method, cutoffs[model])

    sc = {(m, me): scored(m, me) for m in models for me in ts}
    blind = {me: scored(BLIND, me) for me in ts}

    def line(df: pd.DataFrame, model: str) -> pd.DataFrame:
        """The release line (plan §2): replaces the cutoff grouping of item_scores."""
        return df.assign(group=release_groups(df["date"], cutoffs[model], release[model]).to_numpy()) if len(df) else df

    def cell(model: str, method: str, B: int, use_line: bool = True) -> dict:
        df = with_score(sc[(model, method)], method, "primary")
        df = line(df, model) if use_line else df
        reg = analysis.summarize(df, method, B=B)
        ans = analysis.summarize(df[~df["refused"].astype(bool)], method, B=B) if len(df) else analysis.summarize(df, method)
        return {"model": model, "method": method, **reg, "answered_dkappa": ans["dkappa"], "answered_p": ans["p_value"],
                "answered_n_pre": ans["n_pre"], "answered_n_post": ans["n_post"],
                "refusal_affected": refusal_affected(reg["coverage_pre"], reg["coverage_post"])}

    def read(rows: list[dict]) -> list[dict]:
        """Holm across the family (registered and answered-only separately), then the reading and the guard."""
        for key, out in (("p_value", "p_holm"), ("answered_p", "answered_p_holm")):
            for r, p in zip(rows, stats.holm([r[key] for r in rows])):
                r[out] = float(p)
        for r in rows:
            r["reading"] = guarded(pilot.primary_reading(r), r["answered_p_holm"], r["answered_dkappa"])
        return rows

    res: dict = {"models": models, "cutoffs": {m: str(cutoffs[m]) for m in models},
                 "release": {m: str(release[m]) for m in models}, "provisional": sorted(PROVISIONAL),
                 "sets": {m: ts[m][0] for m in ts}}
    res["missing"] = {f"{short(m)} {me}": len(items[me]) - len(sc[(m, me)]) for m in models for me in ts
                      if me != "known_return" and len(items[me]) - len(sc[(m, me)])}

    # H1 and H2 (confirmatory, 20,000 draws, no multiplicity correction; guarded)
    for key, (model, method) in (("h1", (h_models[0], "price_level")), ("h2", (h_models[1], "us_level"))):
        if model in models:
            c = cell(model, method, B_H)
            reg_ok = bool(c["p_value"] < 0.05 and c["dkappa"] > 0)
            ans_ok = bool(c["answered_p"] < 0.05 and c["answered_dkappa"] > 0)
            head = with_score(line(sc[(model, method)], model), method, "headline")
            real = head[~head["fabricated"].astype(bool)]
            res[key] = {**c, "confirmed": reg_ok and ans_ok, "refusal_driven": reg_ok and not ans_ok,
                        "headline_pre": float(real.loc[real["group"] == 1, "score"].mean()),
                        "headline_post": float(real.loc[real["group"] == 0, "score"].mean())}

    # E1: release line, Holm across the cells
    res["e1"] = read([cell(m, me, B_MAIN) for m in models for me in E1_METHODS])

    # E2: positive controls
    e2 = []
    for m in models:
        row = {"model": m}
        for rule, name in (("headline", "in_range"), ("tenpct", "pm10")):
            df = sc[(m, "known_level")]
            bl = blind["known_level"]
            if rule == "headline":
                df, bl = with_score(df, "known_level", "headline"), with_score(bl, "known_level", "headline")
            real = df[~df["fabricated"].astype(bool)] if len(df) else df
            if not len(real):
                continue
            ids = set(real["item_id"])
            blind_rate = float(bl[bl["item_id"].isin(ids)]["score"].mean()) if len(bl) else float("nan")
            lo, hi = stats.bootstrap_mean(real["score"], real["cluster"], B=B_H)
            row.update({f"{name}_rate": float(real["score"].mean()), f"{name}_lo": lo, f"{name}_hi": hi,
                        f"{name}_blind": blind_rate, f"{name}_sensitive": bool(lo > blind_rate)})
        row["n"] = int((~sc[(m, "known_level")]["fabricated"].astype(bool)).sum()) if len(sc[(m, "known_level")]) else 0
        e2.append(row)
    res["e2_m9"] = e2
    res["e2_m10"] = read([cell(m, "us_level", B_MAIN) for m in models if m not in PROVISIONAL])
    blind_kr = float(blind["known_return"]["score"].mean()) if len(blind["known_return"]) else float("nan")
    res["e2_m7"] = [{"model": m, **pilot_d.positive_control(sc[(m, "known_return")], blind_kr, B=B_H)} for m in models]

    # E3: Split Clock on ordinary splits: before the cutoff vs after release, and after release vs the guesser
    data = load_data()
    elig = data.universe[data.universe["is_primary"] & (data.universe["instrumenttype"] == 4)]["insref"]
    px = data.prices[data.prices["insref"].isin(elig)]
    median_by_month = px.groupby(px["date"].dt.to_period("M").astype(str))["close"].median().to_dict()
    guess = pilot_d.typical_guesser(items["split_clock"], median_by_month)
    e3 = []
    for m in models:
        df = sc[(m, "split_clock")]
        if not len(df):
            continue
        real = df[~df["fabricated"].astype(bool)]
        before = real[real["group"] == 1]
        beyond = pilot_d.split_clock_beyond(df, guess, release[m], B=B_H)
        e3.append({"model": m, "n_before": len(before), "share_before": float(before["score"].mean()),
                   "counts_before": before["kind"].value_counts().to_dict() if "kind" in before else {}, **beyond})
    res["e3"] = e3

    # E4: monthly returns beyond the primary score
    e4 = []
    for m in models:
        df = line(sc[(m, "monthly_return")], m)
        if not len(df):
            continue
        real = df[~df["fabricated"].astype(bool)].copy()
        truth_up = real["item_id"].map(lambda i: items["monthly_return"][i].truth["return_pct"] > 0)
        market_up = real["item_id"].map(lambda i: items["monthly_return"][i].truth["market_return_pct"] > 0)
        real["guesser"] = (truth_up == market_up).astype(float)
        row = {"model": m}
        for g, side in ((1, "pre"), (0, "post")):
            part = real[real["group"] == g]
            answered = part[~part["refused"].astype(bool)]
            row.update({f"n_{side}": len(part),
                        f"abs_direction_{side}": float(answered["abs_direction_hit"].astype(float).mean())
                        if len(answered) else float("nan"),
                        f"guesser_{side}": float(part["guesser"].mean()) if len(part) else float("nan"),
                        f"median_err_{side}": float(answered["abs_err_pp"].median()) if len(answered) else float("nan"),
                        f"within_5pp_{side}": float(answered["within_5pp"].astype(float).mean())
                        if len(answered) else float("nan")})
        e4.append(row)
    res["e4"] = e4

    # E5: the cutoff line, and the grey zone
    res["e5_cutoff"] = read([cell(m, me, B_MAIN, use_line=False) for m in models for me in E1_METHODS])
    res["e5_h_cutoff"] = [cell(model, method, B_H, use_line=False) for model, method in
                          ((h_models[0], "price_level"), (h_models[1], "us_level")) if model in models]
    grey = []
    for m in models:
        for me in ("price_level", "monthly_return"):
            df = line(with_score(sc[(m, me)], me, "primary"), m)
            if not len(df):
                continue
            real = df[~df["fabricated"].astype(bool)]
            grey.append({"model": m, "method": me, **{f"{k}_{s}": v for s, g in (("pre", 1), ("grey", -1), ("post", 0))
                                                      for k, v in (("n", int((real["group"] == g).sum())),
                                                                   ("mean", float(real.loc[real["group"] == g,
                                                                                           "score"].mean())))}})
    res["e5_grey"] = grey

    # E6: checks
    bands = []
    for m in models:
        for me in LEVEL_METHODS:
            df = line(sc[(m, me)], m)
            if not len(df):
                continue
            real = df[~df["fabricated"].astype(bool)]
            err = np.fmin(pd.to_numeric(real.get("rel_err_raw"), errors="coerce"),
                          pd.to_numeric(real.get("rel_err_adj"), errors="coerce"))
            row = {"model": m, "method": me}
            for g, side in ((1, "pre"), (0, "post")):
                mask = real["group"] == g
                row[f"n_{side}"] = int(mask.sum())
                for b in BANDS:
                    row[f"band_{int(b * 100)}_{side}"] = float((err[mask] <= b + 1e-12).mean()) if mask.any() else None
                inr = with_score(real[mask], me, "headline")["score"] if mask.any() else pd.Series(dtype=float)
                row[f"in_range_{side}"] = float(inr.mean()) if len(inr) else None
                logerr = pd.to_numeric(real.loc[mask, "log_error"], errors="coerce") if "log_error" in real else None
                row[f"log_err_{side}"] = float(logerr.median()) if logerr is not None and len(logerr) else None
            bands.append(row)
    res["e6_bands"] = bands
    raw = [{"model": m, **pilot.raw_convention(line(sc[(m, "price_level")], m), B=B_MAIN, tolerance=0.10,
                                                method="price_level")} for m in models]
    for r, p in zip(raw, stats.holm([r["p_value"] for r in raw])):
        r["p_holm"] = float(p)
    res["e6_raw"] = raw
    old = [{"model": m, **analysis.summarize(line(sc[(m, "price_level")], m), "price_level", B=B_MAIN)} for m in models]
    res["e6_pm10"] = old
    placebo = [{"model": m, "method": me, **pilot.placebo(with_score(sc[(m, me)], me, "primary"), me, B=B_MAIN)}
               for m in models for me in ("price_level", "monthly_return")]
    for r, p in zip(placebo, stats.holm([r["p_value"] for r in placebo])):
        r["p_holm"] = float(p)
    res["e6_placebo"] = placebo
    outcomes = []
    for m in models:
        for me in ("price_level", "known_level", "us_level", "split_clock", "monthly_return"):
            df = sc[(m, me)]
            if len(df):
                real = df[~df["fabricated"].astype(bool)]
                outcomes.append({"model": m, "method": me, **real["outcome"].value_counts(normalize=True).to_dict()})
    res["outcomes"] = outcomes
    return res


# ---------------------------------------------------------------- markdown
def _cell_rows(rows: list[dict]) -> list[list]:
    return [[short(r["model"]) + (" *" if r["model"] in PROVISIONAL else ""), LABELS[r["method"]], r["n_pre"],
             r["n_post"], _f(r["dkappa"]), f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]", _f(r["p_holm"], "{:.3f}"),
             f"{_f(r['coverage_pre'], '{:.0%}')} / {_f(r['coverage_post'], '{:.0%}')}"
             + (" ⚠" if r["refusal_affected"] else ""), _f(r["answered_dkappa"]), r["reading"]] for r in rows]


CELL_HEADER = ["model", "method", "PRE", "POST", "Δκ", "95% CI", "p (Holm)", "answered before / after",
               "Δκ answered only", "reading"]


def _h(res: dict, key: str, title: str, what: str) -> list[str]:
    if key not in res:
        return []
    h = res[key]
    verdict = "CONFIRMED" if h["confirmed"] else ("NOT CONFIRMED: refusal-driven" if h["refusal_driven"]
                                                  else "NOT CONFIRMED")
    return [f"## {title}\n",
            f"**{verdict}**: Δκ {_f(h['dkappa'])} [{_f(h['ci_lo'])}, {_f(h['ci_hi'])}], p = {_f(h['p_value'], '{:.4f}')}; "
            f"{what} closeness {_f(h['mean_pre'], '{:.2f}')} before the cutoff vs {_f(h['mean_post'], '{:.2f}')} after "
            f"release ({h['n_pre']} / {h['n_post']} items; 20,000 draws, no multiplicity correction). Answered items only: "
            f"Δκ {_f(h['answered_dkappa'])}, p = {_f(h['answered_p'], '{:.4f}')}. Headline (a price that occurred that "
            f"month): {_f(h['headline_pre'], '{:.0%}')} before vs {_f(h['headline_post'], '{:.0%}')} after.\n"]


def render(res: dict) -> str:
    prov = set(res.get("provisional", []))
    L = ["# Pilot E report\n", "Models (cutoff · release): " + ", ".join(
        f"{short(m)} {res['cutoffs'][m]} · {res['release'][m]}" + (" (provisional)" if m in prov else "")
        for m in res["models"]), "\nSets: " + ", ".join(f"{LABELS[m]} {s}" for m, s in res["sets"].items()),
         "\nLine: release (before = more than 91 days before the cutoff; after = after release; the grey zone is left "
         "out). Scores: closeness for price levels, rewritten-price share for Split Clock, right side of the market "
         "for monthly returns. A refusal scores 0; deviation 1 (the refusal guard) applies.\n",
         f"Missing answers: {res['missing'] or 'none'}\n"]
    L += _h(res, "h1", "H1 (confirmatory): DeepSeek V3.1 on M5 v2", "price-level")
    L += _h(res, "h2", "H2 (confirmatory, method validity): Llama 3.3 on M10", "US price-level")
    L.append("## E1: main table, release line (Holm across the cells)\n")
    L.append(_table(CELL_HEADER, _cell_rows(res["e1"])))
    L.append("\n\\* provisional cutoff. ⚠ answer rates differ by more than 10 points between the two sides.\n")
    L.append("\n## E2: positive controls\n\n**M9 known Nordic price levels** (all before every cutoff):\n")
    L.append(_table(["model", "n", "in range", "95% CI", "blind", "sensitive", "±10%", "blind ±10%", "sensitive ±10%"],
                    [[short(r["model"]), r["n"], _f(r.get("in_range_rate"), "{:.0%}"),
                      f"[{_f(r.get('in_range_lo'), '{:.0%}')}, {_f(r.get('in_range_hi'), '{:.0%}')}]",
                      _f(r.get("in_range_blind"), "{:.0%}"), "yes" if r.get("in_range_sensitive") else "no",
                      _f(r.get("pm10_rate"), "{:.0%}"), _f(r.get("pm10_blind"), "{:.0%}"),
                      "yes" if r.get("pm10_sensitive") else "no"] for r in res["e2_m9"]]))
    L.append("\n**M10 US price levels** (release line, closeness, Holm across the five models):\n")
    L.append(_table(CELL_HEADER, _cell_rows(res["e2_m10"])))
    L.append("\n**M7 known returns** (pilot D's answers, direction right):\n")
    L.append(_table(["model", "n", "accuracy", "95% CI", "blind", "sensitive"],
                    [[short(r["model"]), r["n"], _f(r["accuracy"], "{:.0%}"),
                      f"[{_f(r['ci_lo'], '{:.0%}')}, {_f(r['ci_hi'], '{:.0%}')}]", _f(r.get("blind"), "{:.0%}"),
                      "yes" if r["sensitive"] else "no"] for r in res["e2_m7"]]))
    L.append("\n## E3: Split Clock, ordinary splits\n")
    L.append(_table(["model", "splits before cutoff", "rewritten share before", "splits after release",
                     "rewritten share after", "typical-price guesser", "difference 95% CI", "counts after"],
                    [[short(r["model"]), r["n_before"], _f(r["share_before"], "{:.0%}"), r["n"],
                      _f(r["share"], "{:.0%}"), _f(r["blind_share"], "{:.0%}"),
                      f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]", r["counts"]] for r in res["e3"]]))
    L.append("\n## E4: monthly returns beyond the market\n")
    L.append(_table(["model", "n before / after", "direction right before / after", "market-direction guesser",
                     "median error (pp) before / after", "within ±5 pp before / after"],
                    [[short(r["model"]), f"{r['n_pre']} / {r['n_post']}",
                      f"{_f(r['abs_direction_pre'], '{:.0%}')} / {_f(r['abs_direction_post'], '{:.0%}')}",
                      f"{_f(r['guesser_pre'], '{:.0%}')} / {_f(r['guesser_post'], '{:.0%}')}",
                      f"{_f(r['median_err_pre'], '{:.1f}')} / {_f(r['median_err_post'], '{:.1f}')}",
                      f"{_f(r['within_5pp_pre'], '{:.0%}')} / {_f(r['within_5pp_post'], '{:.0%}')}"] for r in res["e4"]]))
    L.append("\n## E5: the cutoff line and the grey zone\n\n**E1 on the cutoff line** (after = more than 91 days after "
             "the cutoff):\n")
    L.append(_table(CELL_HEADER, _cell_rows(res["e5_cutoff"])))
    L.append("\n**H1 / H2 on the cutoff line:** " + "; ".join(
        f"{short(r['model'])} {LABELS[r['method']].split()[0]} Δκ {_f(r['dkappa'])} [{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] "
        f"p {_f(r['p_value'], '{:.4f}')}" for r in res["e5_h_cutoff"]) + "\n")
    L.append("\n**Grey zone** (mean primary score before the cutoff / between cutoff and release / after release):\n")
    L.append(_table(["model", "method", "before", "grey zone", "after release"],
                    [[short(r["model"]), LABELS[r["method"]], f"{_f(r['mean_pre'], '{:.2f}')} (n={r['n_pre']})",
                      f"{_f(r['mean_grey'], '{:.2f}')} (n={r['n_grey']})",
                      f"{_f(r['mean_post'], '{:.2f}')} (n={r['n_post']})"] for r in res["e5_grey"]]))
    L.append("\n## E6: checks\n\n**Hit rates before → after release** (M5 v2, M9, M10):\n")
    L.append(_table(["model", "method", "±3%", "±5%", "±10%", "±25%", "in range", "median log error"],
                    [[short(r["model"]), LABELS[r["method"]].split()[0]]
                     + [f"{_f(r.get(f'band_{b}_pre'), '{:.0%}')} → {_f(r.get(f'band_{b}_post'), '{:.0%}')}"
                        for b in (3, 5, 10, 25)]
                     + [f"{_f(r.get('in_range_pre'), '{:.0%}')} → {_f(r.get('in_range_post'), '{:.0%}')}",
                        f"{_f(r.get('log_err_pre'), '{:.2f}')} → {_f(r.get('log_err_post'), '{:.2f}')}"]
                     for r in res["e6_bands"]]))
    L.append("\n**Pilot D's ±10% rule on M5 v2 (release line):** " + "; ".join(
        f"{short(r['model'])} Δκ {_f(r['dkappa'])} [{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] p {_f(r['p_value'], '{:.3f}')}"
        for r in res["e6_pm10"]) + "\n")
    L.append("\n**Unadjusted price only, ±10% (Holm across 6):** " + "; ".join(
        f"{short(r['model'])} {_f(r['dkappa'])} (p {_f(r['p_holm'], '{:.3f}')})" for r in res["e6_raw"]) + "\n")
    L.append("\n**Placebo cutoff 2022-07-01 (Holm across 12):** " + (", ".join(
        f"{short(r['model'])} {LABELS[r['method']].split()[0]} {_f(r['dkappa'])} (p {_f(r['p_holm'], '{:.3f}')})"
        for r in res["e6_placebo"] if (r.get("p_holm") or 1) < 0.05) or "no significant placebo cell") + "\n")
    L.append("\n**Outcomes (share of real items):**\n")
    L.append(_table(["model", "method", "answer", "abstain", "refusal", "malformed"],
                    [[short(r["model"]), LABELS[r["method"]].split()[0]]
                     + [_f(r.get(k), "{:.0%}") for k in ("answer", "abstain", "refusal", "malformed")]
                     for r in res["outcomes"]]))
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", action="store_true", help="dry run on the control models")
    args = ap.parse_args(argv)
    if args.controls:
        models, release = CONTROLS, {m: date(2024, 12, 31) for m in CONTROLS}
        h_models = ("mock/oracle@2024-06-30", "mock/oracle@2024-06-30")
    else:
        models, release, h_models = MODELS, RELEASE, (DEEPSEEK, LLAMA)
    res = _jsonable(compute(models, release, h_models))
    out = OUT / ("controls" if args.controls else "")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "report.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
