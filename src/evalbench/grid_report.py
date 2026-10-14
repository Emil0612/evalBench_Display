"""Monthly price-level grid report: every analysis registered in docs/preregistration/2026-09-30-monthly-grid.md,
from the stored answers only (no model calls, no cost). Writes results/grid/results.json and results/grid/report.md.

uv run python -m evalbench.grid_report                      # v3 (the registered set)
uv run python -m evalbench.grid_report --versions v4         # the replication set (§6)
uv run python -m evalbench.grid_report --versions v3 v4      # pooled, 36 per month (§6)
uv run python -m evalbench.grid_report --staggered           # the staggered check on v3 + v4 (§7)
"""
from __future__ import annotations

import argparse
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import analysis, pilot, stats, store
from .cutoffs import BUFFER_DAYS, resolve_cutoff
from .items import ROOT, TESTSETS, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .pilot_d_report import MODELS, PROVISIONAL, RELEASE
from .pilot_e import guarded, refusal_affected, with_score
from .pilot_report import _f, _jsonable, _table, short

OUT = ROOT / "results" / "grid"
METHOD = "price_level"
BLIND = "baseline/blind@2024-06-30"
LAST_MONTH = pd.Period("2026-07", "M")
B_H, B_BREAK = 20000, 500
LATE = "qwen/qwen3.8-flash"   # before its cutoff for all of 2025 (plan §7)


def set_items(versions, root=TESTSETS) -> tuple[dict, list[str]]:
    """The questions of one or more grid versions, by item id, and their testset names."""
    items = {it.item_id: it for v in versions for it in load_testset(METHOD, v, root)}
    return items, [f"{METHOD}/{v}" for v in versions]


def label(versions) -> str:
    """Output subfolder: '' for the registered v3 alone, else the versions joined by '+'."""
    return "" if tuple(versions) == ("v3",) else "+".join(versions)


def windows(cutoff: date, release: date, last_month: pd.Period = LAST_MONTH):
    """(BEFORE, AFTER) month lists (plan §2). BEFORE: the 12 months ending with the last whole month before
    cutoff - 91 days. AFTER: the first 12 months after the release month, as far as the data goes; None if fewer
    than 6."""
    edge = cutoff - timedelta(days=BUFFER_DAYS)
    last_whole = pd.Period(edge, "M") if (edge + timedelta(days=1)).day == 1 else pd.Period(edge, "M") - 1
    before = list(pd.period_range(last_whole - 11, last_whole, freq="M"))
    first_after = pd.Period(release, "M") + 1
    after = list(pd.period_range(first_after, min(first_after + 11, last_month), freq="M")) \
        if first_after <= last_month else []
    return before, (after if len(after) >= 6 else None)


def break_month(df: pd.DataFrame, B: int = B_BREAK, min_side: int = 6, seed: int = 0) -> dict:
    """The month that best splits the monthly mean score into two levels (least squares, one change; the break
    month is the first month of the second level), with a cluster-bootstrap 95% interval (plan §4, E3).
    df: month (Period), score, cluster."""
    def best(frame):
        means = frame.groupby("month")["score"].mean().sort_index()
        if len(means) < 2 * min_side:
            return None
        v = means.to_numpy()
        sse = [((v[:k] - v[:k].mean()) ** 2).sum() + ((v[k:] - v[k:].mean()) ** 2).sum()
               for k in range(min_side, len(v) - min_side + 1)]
        k = int(np.argmin(sse)) + min_side
        return means.index[k], float(v[:k].mean()), float(v[k:].mean())

    point = best(df)
    if point is None:
        return {"month": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    clusters = df["cluster"].unique()
    by = {c: g for c, g in df.groupby("cluster")}
    draws = []
    for _ in range(B):
        pick = rng.choice(clusters, size=len(clusters), replace=True)
        r = best(pd.concat([by[c] for c in pick], ignore_index=True))
        if r is not None:
            draws.append(r[0].ordinal)
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)
    return {"month": point[0], "level_before": point[1], "level_after": point[2],
            "lo": pd.Period(ordinal=int(round(lo)), freq="M"), "hi": pd.Period(ordinal=int(round(hi)), freq="M")}


def staggered_months(early_cutoff: date, early_release: date, late_cutoff: date):
    """(between, baseline) months for the staggered check (plan §7). between: after the early model's release month, up
    to the late model's last whole month before its cutoff - 91 days. baseline: the early model's BEFORE window."""
    baseline, _ = windows(early_cutoff, early_release)
    late_last = windows(late_cutoff, late_cutoff)[0][-1]
    first = pd.Period(early_release, "M") + 1
    between = list(pd.period_range(first, late_last, freq="M")) if first <= late_last else []
    return between, baseline


def staggered_pair(late: pd.DataFrame, early: pd.DataFrame, between, baseline, B: int = B_H) -> dict:
    """D = mean(late - early) on between months - the same on baseline months, over the questions both answered
    (late, early: item_id, month, cluster, score, refused). Memory predicts D > 0; a time effect shared by both models
    cancels."""
    pair = late[["item_id", "month", "cluster", "score", "refused"]].merge(
        early[["item_id", "score", "refused"]], on="item_id", suffixes=("_late", "_early"))
    refused = pair["refused_late"].astype(bool) | pair["refused_early"].astype(bool)
    pair = pair.assign(score=pair["score_late"] - pair["score_early"], refused=refused, fabricated=False,
                       group=np.where(pair["month"].isin(between), 1, np.where(pair["month"].isin(baseline), 0, -1)))
    reg = analysis.summarize(pair, "price_level", B=B)
    ans = analysis.summarize(pair[~pair["refused"]], "price_level", B=B)
    return {**{k: reg[k] for k in ("n_pre", "n_post", "mean_pre", "mean_post", "dkappa", "ci_lo", "ci_hi", "p_value")},
            "answered_dkappa": ans["dkappa"], "answered_p": ans["p_value"]}


def compute_staggered(versions=("v3", "v4"), models: list[str] = MODELS, B: int = B_H) -> dict:
    items, names = set_items(versions)
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    calls = calls[calls["testset"].isin(names)]
    meta = models_by_id(fetch_models())

    def scored(model):
        df = analysis.item_scores(calls[calls["model"] == model], items, "price_level", resolve_cutoff(model, meta))
        df = with_score(df[~df["fabricated"].astype(bool)], "price_level", "primary")
        return df.assign(month=pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M"))

    late = scored(LATE)
    rows = []
    for m in [x for x in models if x != LATE]:
        between, baseline = staggered_months(resolve_cutoff(m, meta), RELEASE[m], resolve_cutoff(LATE, meta))
        rows.append({"model": m, "between": f"{between[0]}..{between[-1]}" if between else "–",
                     "baseline": f"{baseline[0]}..{baseline[-1]}",
                     **staggered_pair(late, scored(m), between, baseline, B=B)})
    for key, out in (("p_value", "p_holm"), ("answered_p", "answered_p_holm")):
        for r, p in zip(rows, stats.holm([r[key] for r in rows])):
            r[out] = float(p)
    return {"set": " + ".join(names), "late": LATE, "rows": rows}


def render_staggered(res: dict) -> str:
    L = ["# Staggered check on the monthly grid (plan §7, exploratory)\n",
         f"Set: {res['set']}. Per question d = closeness({short(res['late'])}) − closeness(model). D = mean d on "
         f"*between* months (after the model's release, before {short(res['late'])}'s cutoff − 91 days) − mean d on "
         "*baseline* months (the model's BEFORE window). Memory predicts D > 0; a time effect shared by both models "
         "cancels. Holm across the models.\n"]
    L.append(_table(["model", "between", "baseline", "d between / baseline", "D", "95% CI", "p (Holm)",
                     "D, both answered", "p (Holm)"],
                    [[short(r["model"]), r["between"], r["baseline"],
                      f"{_f(r['mean_pre'], '{:+.3f}')} / {_f(r['mean_post'], '{:+.3f}')}", _f(r["dkappa"]),
                      f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]", _f(r["p_holm"], "{:.4f}"),
                      _f(r["answered_dkappa"]), _f(r["answered_p_holm"], "{:.4f}")] for r in res["rows"]]))
    L.append("\nn between / baseline: " + ", ".join(f"{short(r['model'])} {r['n_pre']} / {r['n_post']}"
                                                   for r in res["rows"]))
    return "\n".join(L) + "\n"


def compute(models: list[str] = MODELS, versions=("v3",), b_break: int = B_BREAK, release: dict = RELEASE) -> dict:
    items, names = set_items(versions)
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    calls = calls[calls["testset"].isin(names)]
    meta = models_by_id(fetch_models())

    def scored(model):
        df = analysis.item_scores(calls[calls["model"] == model], items, "price_level", resolve_cutoff(model, meta))
        if df.empty:
            return df
        df = df[~df["fabricated"].astype(bool)].copy()
        df["month"] = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
        return df

    blind = with_score(scored(BLIND), "price_level", "primary")
    blind_score = dict(zip(blind["item_id"], blind["score"])) if len(blind) else {}
    res = {"set": " + ".join(names), "per_month": round(len(items) / 67), "models": models, "missing": {}, "h": [], "e2": [], "e3": [], "e4": []}
    for m in models:
        raw = scored(m)
        res["missing"][short(m)] = len(items) - len(raw)
        cut, rel = resolve_cutoff(m, meta), release[m]
        before, after = windows(cut, rel)
        prim = with_score(raw, "price_level", "primary")
        res["e3"].append({"model": m, "cutoff": str(cut), "release": str(rel),
                          **{k: (str(v) if isinstance(v, pd.Period) else v)
                             for k, v in break_month(prim[["month", "score", "cluster"]], B=b_break).items()}})
        if after is None:
            res["h"].append({"model": m, "note": "no AFTER window (released after the data ends)",
                             "mean_pre": float(prim[prim["month"].isin(before)]["score"].mean())})
            continue
        grp = np.where(prim["month"].isin(before), 1, np.where(prim["month"].isin(after), 0, -1))
        df = prim.assign(group=grp)
        reg = analysis.summarize(df, "price_level", B=B_H)
        ans = analysis.summarize(df[~df["refused"].astype(bool)], "price_level", B=B_H)
        res["h"].append({"model": m, "before": f"{before[0]}..{before[-1]}", "after": f"{after[0]}..{after[-1]}",
                         **{k: reg[k] for k in ("n_pre", "n_post", "mean_pre", "mean_post", "dkappa", "ci_lo", "ci_hi",
                                                "p_value", "coverage_pre", "coverage_post", "malformed_rate")},
                         "answered_dkappa": ans["dkappa"], "answered_p": ans["p_value"],
                         "refusal_affected": refusal_affected(reg["coverage_pre"], reg["coverage_post"])})
        diff = df.assign(score=df["score"] - df["item_id"].map(blind_score)).dropna(subset=["score"])
        e2 = analysis.summarize(diff, "price_level", B=B_H)
        res["e2"].append({"model": m, **{k: e2[k] for k in ("n_pre", "n_post", "mean_pre", "mean_post", "dkappa",
                                                              "ci_lo", "ci_hi", "p_value")}})
        row = {"model": m}
        for rule, name in (("headline", "in_range"), ("score", "pm10")):
            s = with_score(df, "price_level", "headline") if rule == "headline" else df
            row[f"{name}_before"] = float(s[s["group"] == 1]["score"].mean())
            row[f"{name}_after"] = float(s[s["group"] == 0]["score"].mean())
        res["e4"].append(row)
    tested = [r for r in res["h"] if "p_value" in r]
    for key, out in (("p_value", "p_holm"), ("answered_p", "answered_p_holm")):
        for r, p in zip(tested, stats.holm([r[key] for r in tested])):
            r[out] = float(p)
    for r in tested:
        r["reading"] = guarded(pilot.primary_reading(r), r["answered_p_holm"], r["answered_dkappa"])
        r["confirmed"] = r["reading"].startswith("memorization detected")
    for r, p in zip(res["e2"], stats.holm([r["p_value"] for r in res["e2"]])):
        r["p_holm"] = float(p)
    return res


def render(res: dict) -> str:
    L = ["# Monthly price-level grid report\n",
         f"Set: {res['set']} ({res['per_month']} questions per month, 2021-01..2026-07). Score: closeness (1 = exact, 0 = off by 2× or "
         f"more). BEFORE: the 12 months ending 3 months before the stated cutoff; AFTER: the first 12 months after "
         f"release. Missing answers: {res['missing']}\n",
         "## Confirmatory test (Holm across the models with an AFTER window; refusal guard)\n"]
    L.append(_table(["model", "BEFORE", "AFTER", "closeness before → after", "Δ", "95% CI", "p (Holm)",
                     "answered before / after", "Δ answered only", "reading"],
                    [[short(r["model"]), r.get("before", "–"), r.get("after", "–"),
                      f"{_f(r.get('mean_pre'), '{:.3f}')} → {_f(r.get('mean_post'), '{:.3f}')}", _f(r.get("dkappa")),
                      f"[{_f(r.get('ci_lo'))}, {_f(r.get('ci_hi'))}]" if "ci_lo" in r else "–",
                      _f(r.get("p_holm"), "{:.4f}"),
                      f"{_f(r.get('coverage_pre'), '{:.0%}')} / {_f(r.get('coverage_post'), '{:.0%}')}"
                      + (" ⚠" if r.get("refusal_affected") else ""), _f(r.get("answered_dkappa")),
                      r.get("reading", r.get("note", ""))] for r in res["h"]]))
    L.append("\n## E2: difficulty-adjusted (model − blind guesser on the same question; Holm across models)\n")
    L.append(_table(["model", "difference before → after", "Δ", "95% CI", "p (Holm)"],
                    [[short(r["model"]), f"{_f(r['mean_pre'], '{:+.3f}')} → {_f(r['mean_post'], '{:+.3f}')}",
                      _f(r["dkappa"]), f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]", _f(r["p_holm"], "{:.4f}")]
                     for r in res["e2"]]))
    L.append("\n## E3: where the monthly curve breaks (least squares, one change, cluster-bootstrap 95% interval)\n")
    L.append(_table(["model", "stated cutoff", "release", "break month", "95% interval", "level before → after"],
                    [[short(r["model"]), r["cutoff"], r["release"], r.get("month") or "–",
                      f"{r.get('lo') or '–'} .. {r.get('hi') or '–'}",
                      f"{_f(r.get('level_before'), '{:.3f}')} → {_f(r.get('level_after'), '{:.3f}')}"]
                     for r in res["e3"]]))
    L.append("\n## E4: headline rates, BEFORE → AFTER\n")
    L.append(_table(["model", "a price that occurred that month", "within ±10%"],
                    [[short(r["model"]), f"{_f(r['in_range_before'], '{:.0%}')} → {_f(r['in_range_after'], '{:.0%}')}",
                      f"{_f(r['pm10_before'], '{:.0%}')} → {_f(r['pm10_after'], '{:.0%}')}"] for r in res["e4"]]))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--versions", nargs="+", default=["v3"], help="price_level grid versions, pooled")
    ap.add_argument("--staggered", action="store_true", help="the staggered check on v3 + v4 (plan §7)")
    args = ap.parse_args()
    if args.staggered:
        res = _jsonable(compute_staggered())
        out = OUT / "staggered"
        out.mkdir(parents=True, exist_ok=True)
        (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
        (out / "report.md").write_text(render_staggered(res), encoding="utf-8")
        print(f"wrote {out / 'report.md'}")
        return
    pooled = len(args.versions) > 1
    res = _jsonable(compute(versions=tuple(args.versions), b_break=1000 if pooled else B_BREAK))
    out = OUT / label(args.versions)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (out / "report.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {out / 'report.md'}")


if __name__ == "__main__":
    main()
