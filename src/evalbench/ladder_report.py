"""Cutoff ladder report: every analysis registered in docs/preregistration/2026-10-05-cutoff-ladder.md, from the
stored answers only (no model calls, no cost). Writes results/ladder/pilot.md (--pilot) or results/ladder/report.md
and results.json.

uv run python -X utf8 -m evalbench.ladder_report --pilot     # §0: the GPT-3.5 gate and the served-model check
uv run python -X utf8 -m evalbench.ladder_report             # §1-§7
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import analysis, grid_report, index_report, ladder, market, stats, store
from .cutoffs import resolve_cutoff
from .grid_report import staggered_pair
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .pilot_d_report import BLIND
from .pilot_e import with_score
from .pilot_report import _f, _jsonable, _table, short

OUT = ROOT / "results" / "ladder"
B_H = 20000
GATE = (("known_level", "headline", "M9 within the month's range"), ("known_level", "score", "M9 within ±10%"),
        ("known_return", "score", "M7 direction"))
LATE = ["google/gemini-2.5-flash", "deepseek/deepseek-chat-v3.1"]
MEASURES = {"M15 correlation": "index_return", "M5 closeness": "price_level"}


def sensitive_row(real: pd.DataFrame, blind_rate: float, B: int = B_H) -> dict:
    lo, hi = stats.bootstrap_mean(real["score"], real["cluster"], B=B)
    return {"n": len(real), "rate": float(real["score"].mean()), "lo": lo, "hi": hi, "blind": blind_rate,
            "sensitive": bool(lo > blind_rate)}


def snapshot_check(returned: list[str], expected: str) -> dict:
    """Every answer came from the one model asked for (OpenRouter reports its own id, not the provider's dated
    snapshot; plan §0, deviation 1)."""
    snaps = sorted(set(returned))
    return {"snapshots": snaps, "ok": snaps == [expected]}


def _scored(calls, model, method, cutoff, rule):
    items = {it.item_id: it for it in load_testset(method, "v1")}
    df = analysis.item_scores(calls[(calls["model"] == model) & (calls["testset"] == f"{method}/v1")], items, method,
                              cutoff)
    if df.empty:
        return df
    df = df[~df["fabricated"].astype(bool)]
    df = with_score(df, method, "headline") if rule == "headline" else df
    return df.assign(month=pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M"))


def pilot_gate(calls, meta, model: str = ladder.GPT35, B: int = B_H) -> dict:
    cut = resolve_cutoff(model, meta)
    last = ladder.last_before(cut)
    rows = []
    for method, rule, label in GATE:
        df = _scored(calls, model, method, cut, rule)
        bl = _scored(calls, BLIND, method, resolve_cutoff(BLIND, {}), rule)
        if df.empty:
            rows.append({"label": label, "n": 0, "sensitive": False})
            continue
        real = df[df["month"] <= last]
        blind_rate = float(bl[bl["item_id"].isin(set(real["item_id"]))]["score"].mean()) if len(bl) else float("nan")
        rows.append({"label": label, **sensitive_row(real, blind_rate, B)})
    returned = calls[calls["model"] == model]["model_returned"].dropna().tolist()
    return {"model": model, "rows": rows, "go": any(r["sensitive"] for r in rows),
            "snapshot": snapshot_check(returned, model)}


def render_pilot(g: dict) -> str:
    L = [f"# Cutoff ladder §0: micro-pilot gate for {short(g['model'])}\n",
         "Items on or before the last whole month before cutoff − 91 days; sensitive = the 95% interval's lower end "
         "above the blind guesser on the same items (pilot E's E2 rule).\n",
         _table(["control", "n", "rate", "95% CI", "blind", "sensitive"],
                [[r["label"], r["n"], _f(r.get("rate"), "{:.0%}"),
                  f"[{_f(r.get('lo'), '{:.0%}')}, {_f(r.get('hi'), '{:.0%}')}]",
                  _f(r.get("blind"), "{:.0%}"), "yes" if r["sensitive"] else "no"] for r in g["rows"]]),
         f"\n**Gate:** {'GO' if g['go'] else 'NO-GO'}. **Served model:** "
         f"{', '.join(g['snapshot']['snapshots']) or 'none'} "
         f"({'one model, as asked' if g['snapshot']['ok'] else 'NOT as asked: decide before §1-§7'}).\n"]
    return "\n".join(L)


def family_rows(calls, cut: dict, release: dict, pairs: list[tuple[str, str]], method_key: str = "index_return",
                B: int = B_H, measures=("M15 correlation", "M5 closeness")) -> list[dict]:
    """One row per pair and measure: months, D with its interval and p, usable (§2's rule)."""
    rows = []
    for early, late in pairs:
        for measure in measures:
            if measure == "M15 correlation":
                months = ladder.index_months(method_key)
                between, baseline = ladder.pair_months(cut[early], release[early], cut[late], months)
                r = ladder.staggered_corr(ladder.index_frame(calls, late, method_key),
                                          ladder.index_frame(calls, early, method_key), between, baseline, B=B)
                stat = {"D": r["D"], "lo": r["lo"], "hi": r["hi"], "p": r["p"], "n_between": r["n_between"],
                        "n": f"{r['n_between']} / {r['n_baseline']}",
                        "detail": f"r_L {r['r_late_between']:+.2f} / {r['r_late_baseline']:+.2f}, "
                                  f"r_E {r['r_early_between']:+.2f} / {r['r_early_baseline']:+.2f}"}
            else:
                between, baseline = ladder.pair_months(cut[early], release[early], cut[late], ladder.grid_months())
                r = staggered_pair(ladder.grid_scores(calls, late, cut[late]),
                                   ladder.grid_scores(calls, early, cut[early]), between, baseline, B=B)
                stat = {"D": r["dkappa"], "lo": r["ci_lo"], "hi": r["ci_hi"], "p": r["p_value"],
                        "answered_D": r["answered_dkappa"], "answered_p": r["answered_p"], "n_between": r["n_pre"],
                        "n": f"{r['n_pre']} / {r['n_post']}", "detail": ""}
            rows.append({"early": early, "late": late, "measure": measure,
                         "between": f"{between[0]}..{between[-1]}" if between else "–",
                         "usable": ladder.usable(between, baseline)
                         and (late, MEASURES[measure]) in ladder.REFERENCE_DETECTED, **stat})
    return rows


def read_family(rows: list[dict]) -> list[dict]:
    """Holm across the usable rows; the refusal guard on closeness; unusable rows are exploratory."""
    tested = [r for r in rows if r["usable"] and pd.notna(r["p"])]
    for r, p in zip(tested, stats.holm([r["p"] for r in tested])):
        r["p_holm"] = float(p)
    guard = [r for r in tested if "answered_p" in r]
    for r, p in zip(guard, stats.holm([r["answered_p"] for r in guard])):
        r["answered_p_holm"] = float(p)
    for r in rows:
        if "p_holm" not in r:
            r["reading"] = "exploratory: too few months" if not r["usable"] else "no estimate"
            continue
        base = "memory, not a time effect" if r["p_holm"] < 0.05 and r["D"] > 0 else "not shown"
        if base.startswith("memory") and "answered_p_holm" in r:
            ok = r["answered_p_holm"] < 0.05 and r["answered_D"] > 0
            base = base if ok else f"not confirmed: refusal-driven ({base})"
        r["reading"] = base
    return rows


MIN_PAIRS = 10


def mark_small(rows: list[dict], min_pairs: int = MIN_PAIRS) -> list[dict]:
    """Exploratory correlation rows on fewer than min_pairs paired answers in the between months: no estimate (a
    correlation on a handful of points, or of flat answers, only reflects the baseline side)."""
    for r in rows:
        if r["measure"] == "M15 correlation" and r.get("n_between", 0) < min_pairs:
            r["reading"] = f"no estimate: {r.get('n_between', 0)} paired answers between (< {min_pairs})"
    return rows


def compute(calls, meta) -> dict:
    gate = pilot_gate(calls, meta)
    new = [m for m in ladder.NEW if m != ladder.GPT35 or gate["go"]]
    models = new + ladder.REFERENCES
    cut = {m: resolve_cutoff(m, meta) for m in models}
    rel = ladder.RELEASE
    res = {"gate": gate, "new": new}
    s_pairs = [(ladder.GPT35, late) for late in LATE] if ladder.GPT35 in new else []
    m15 = ("M15 correlation",)
    res["s"] = read_family(family_rows(calls, cut, rel, s_pairs, measures=m15))      # §3: M15 only (§3, E7)
    grid = grid_report.compute(new, ("v4",), release=rel)
    idx = index_report.compute(new, release=rel)
    res["r_m5"], res["r_m15"] = grid["h"], idx["m15"]
    res["e1_us"] = mark_small(family_rows(calls, cut, rel, s_pairs, "us_index_return", measures=m15))
    short_pairs = [(e, late) for e in (ladder.GPT4OMINI, ladder.LLAMA31) if e in new for late in LATE]
    res["e2_short"] = mark_small(family_rows(calls, cut, rel, short_pairs))
    res["e3_release"] = _release_effect(calls, cut, rel) if ladder.LLAMA31 in new else []
    res["e4_grey"] = _grey_zone(calls, cut, rel) if ladder.GPT35 in new else []
    if ladder.GPT35 in new:
        listed = {**rel, ladder.GPT35: ladder.LISTED[ladder.GPT35]}
        res["e5_listed"] = read_family(family_rows(calls, cut, listed, [(ladder.GPT35, late) for late in ladder.REFERENCES],
                                                   measures=m15))
    res["e6_clean"] = [{"model": m, "cutoff": str(cut[m]), "release": str(rel[m]),
                        "clean_months": ladder.clean_months(rel[m])} for m in models]
    res["e7_m5"] = family_rows(calls, cut, rel, s_pairs, measures=("M5 closeness",))
    return res


def _release_effect(calls, cut, rel) -> list[dict]:
    out = []
    a, b = ladder.LLAMA31, ladder.LLAMA          # same cutoff (2023-12); released 2024-07 and 2024-12
    for measure, months in (("M15 correlation", ladder.index_months()), ("M5 closeness", ladder.grid_months())):
        between, baseline = ladder.release_months(rel[a], rel[b], cut[a], months)
        if measure == "M15 correlation":
            r = ladder.staggered_corr(ladder.index_frame(calls, b), ladder.index_frame(calls, a), between, baseline)
            out.append({"measure": measure, "between": f"{between[0]}..{between[-1]}" if between else "–",
                        "D": r["D"], "lo": r["lo"], "hi": r["hi"], "p": r["p"], "n_between": r["n_between"]})
        else:
            r = staggered_pair(ladder.grid_scores(calls, b, cut[b]), ladder.grid_scores(calls, a, cut[a]), between,
                               baseline)
            out.append({"measure": measure, "between": f"{between[0]}..{between[-1]}" if between else "–",
                        "D": r["dkappa"], "lo": r["ci_lo"], "hi": r["ci_hi"], "p": r["p_value"]})
    return out


def _grey_zone(calls, cut, rel) -> list[dict]:
    m = ladder.GPT35
    idx, grid = ladder.index_frame(calls, m), ladder.grid_scores(calls, m, cut[m])
    last, first = ladder.last_before(cut[m]), ladder.first_after(rel[m])
    spans = [("BEFORE", idx["month"] <= last, grid["month"] <= last)]
    spans += [(f"{b[0]}..{b[-1]}", idx["month"].isin(b), grid["month"].isin(b)) for b in ladder.grey_bins(cut[m], rel[m])]
    spans += [("AFTER", idx["month"] >= first, grid["month"] >= first)]
    rows = []
    for label, im, gm in spans:
        part = idx[im]
        r = market.corr(part["answer"], part["truth"]) if len(part) >= MIN_PAIRS else float("nan")
        rows.append({"span": label, "r": r, "n_index": len(part),
                     "closeness": float(grid[gm]["score"].mean()), "n_grid": int(gm.sum())})
    return rows


def _rows(rows: list[dict], holm: bool = True) -> str:
    """A table of pair rows; rows with a both-answered variant (closeness) get its D and p as an extra column."""
    answered = any("answered_D" in r for r in rows)
    head = ["early", "late", "measure", "between", "n", "D", "95% CI", "p (Holm)" if holm else "p"] +         (["D, both answered (p)"] if answered else []) + ["reading", "detail"]
    return _table(head, [[short(r["early"]), short(r["late"]), r["measure"], r["between"], r["n"], _f(r["D"]),
                          f"[{_f(r['lo'])}, {_f(r['hi'])}]", _f(r.get("p_holm", r["p"]), "{:.4f}")]
                         + ([f"{_f(r['answered_D'])} ({_f(r['answered_p'], '{:.4f}')})" if "answered_D" in r else ""]
                            if answered else [])
                         + [r.get("reading", ""), r.get("detail", "")] for r in rows])


def render(res: dict) -> str:
    L = ["# Cutoff ladder report (plan docs/preregistration/2026-10-05-cutoff-ladder.md)\n",
         render_pilot(res["gate"]), "## Family S: memory, not a time effect? (confirmatory, M15)\n", _rows(res["s"]),
         "\n## Family R: do the new models remember? (confirmatory)\n",
         _table(["model", "M5 before → after", "Δ", "95% CI", "p (Holm)", "reading"],
                [[short(r["model"]), f"{_f(r.get('mean_pre'), '{:.3f}')} → {_f(r.get('mean_post'), '{:.3f}')}",
                  _f(r.get("dkappa")), f"[{_f(r.get('ci_lo'))}, {_f(r.get('ci_hi'))}]", _f(r.get("p_holm"), "{:.4f}"),
                  r.get("reading", r.get("note", ""))] for r in res["r_m5"]]),
         "",
         _table(["model", "M15 r before → after", "difference", "95% CI", "p (Holm)", "reading"],
                [[short(r["model"]), f"{_f(r.get('r_before'))} → {_f(r.get('r_after'))}", _f(r.get("diff")),
                  f"[{_f(r.get('lo'))}, {_f(r.get('hi'))}]", _f(r.get("p_holm"), "{:.4f}"), r.get("reading", "")]
                 for r in res["r_m15"]]),
         "\n## E1: US (PRIVATE, exploratory)\n", _rows(res["e1_us"], holm=False),
         "\n## E2: short pairs (exploratory)\n", _rows(res["e2_short"], holm=False),
         "\n## E3: release effect, Llama 3.1 vs 3.3 (exploratory)\n",
         _table(["measure", "between", "D", "95% CI", "p"],
                [[r["measure"], r["between"], _f(r["D"]), f"[{_f(r['lo'])}, {_f(r['hi'])}]", _f(r["p"], "{:.4f}")]
                 for r in res["e3_release"]]),
         "\n## E4: GPT-3.5's grey zone (exploratory)\n",
         _table(["span", "M15 r", "n", "M5 closeness", "n"],
                [[r["span"], _f(r["r"]), r["n_index"], _f(r["closeness"], "{:.3f}"), r["n_grid"]]
                 for r in res["e4_grey"]]),
         "\n## E5: GPT-3.5 at its listing date (sensitivity)\n", _rows(res.get("e5_listed", [])),
         "\n## E6: clean backtest windows\n",
         _table(["model", "cutoff", "release", "clean months to 2026-08"],
                [[short(r["model"]), r["cutoff"], r["release"], r["clean_months"]] for r in res["e6_clean"]]),
         "\n## E7: the price-level staggered test (exploratory)\n", _rows(res["e7_m5"], holm=False)]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="§0 only: the GPT-3.5 gate and the served-model check")
    args = ap.parse_args()
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    OUT.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        g = pilot_gate(calls, meta)
        text = render_pilot(g)
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
