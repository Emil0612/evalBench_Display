"""Free feasibility check for the cutoff ladder (docs/preregistration/2026-10-05-cutoff-ladder.md), from stored
answers and the public model list only: which early/late pairs have enough months, how strong each late reference is
on the months that matter, the projected power, and the cost ceilings of the micro-pilot and the full run. No model is
called.

uv run python -X utf8 -m evalbench.ladder_feasibility
"""
from __future__ import annotations

from datetime import date
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import ladder, market, runner, stats, store
from .cutoffs import resolve_cutoff
from .items import ROOT, load_current_testsets
from .llm.openrouter import fetch_models, load_pins, models_by_id
from .pilot_report import _f, _table, short

OUT = ROOT / "results" / "ladder"
ALPHA = 0.05 / 4                 # Holm's first step across family S's four expected tests
CONTROLS = ["known_return", "known_level"]
FULL = ["price_level", "index_level", "index_return", "us_index_level", "us_index_return"]


def projected_power(effect: float, se: float, alpha: float) -> float:
    """Two-sided power of a z-test at the expected effect; NaN without a usable standard error."""
    if not se or se <= 0 or np.isnan(se):
        return float("nan")
    return float(NormalDist().cdf(abs(effect) / se - NormalDist().inv_cdf(1 - alpha / 2)))


def _span(months) -> str:
    return f"{months[0]}..{months[-1]}" if months else "–"


def coverage_rows(dates: dict, early: list, late: list, months_by_set: dict) -> list[dict]:
    rows = []
    for e in early:
        for l in late:
            for name, months in months_by_set.items():
                between, baseline = ladder.pair_months(dates[e][0], dates[e][1], dates[l][0], months)
                rows.append({"early": e, "late": l, "set": name, "between": _span(between),
                             "baseline": _span(baseline), "n_between": len(between), "n_baseline": len(baseline),
                             "usable": ladder.usable(between, baseline)})
    return rows


def _se_corr(df: pd.DataFrame) -> float:
    return float(np.nanstd(market.boot_corr(df, B=2000))) if len(df) >= 3 else float("nan")


def _se_mean(df: pd.DataFrame) -> float:
    if len(df) < 3:
        return float("nan")
    lo, hi = stats.bootstrap_mean(df["score"], df["cluster"], B=2000)
    return (hi - lo) / 3.92


def reference_strength(calls, ref: str, cutoff: date, early_dates: tuple[date, date]) -> dict:
    """The late reference's own answers on the pair's between and baseline months: M15 correlation and M5 closeness,
    their bootstrap standard errors, and its answer rate on the between index questions."""
    out = {"late": ref}
    idx = ladder.index_frame(calls, ref)
    between, baseline = ladder.pair_months(early_dates[0], early_dates[1], cutoff, ladder.index_months())
    b, a = idx[idx["month"].isin(between)], idx[idx["month"].isin(baseline)]
    asked = sum(1 for it in ladder.load_testset("index_return", "v1")[ladder.PILOT:]
                if pd.Period(it.payload["month"], "M") in between)   # OMXHPI has a 2022-11..2025-08 gap
    out.update({"r_between": market.corr(b["answer"], b["truth"]), "r_baseline": market.corr(a["answer"], a["truth"]),
                "se_r_between": _se_corr(b), "se_r_baseline": _se_corr(a),
                "answered_between": len(b) / asked if asked else float("nan")})
    g = ladder.grid_scores(calls, ref, cutoff)
    between, baseline = ladder.pair_months(early_dates[0], early_dates[1], cutoff, ladder.grid_months())
    gb, ga = g[g["month"].isin(between)], g[g["month"].isin(baseline)]
    out.update({"c_between": float(gb["score"].mean()), "c_baseline": float(ga["score"].mean()),
                "se_c_between": _se_mean(gb), "se_c_baseline": _se_mean(ga)})
    return out


def power_rows(strength: dict) -> list[dict]:
    """Scenarios: the early model's baseline correlation 0.3 or 0.6 and 0 on between months; its closeness drop
    after release 0.05 or 0.09 (the range of the grid v4 replication). SE of D ~ sqrt(2) x the reference's."""
    se_r = np.sqrt(2 * (strength["se_r_between"] ** 2 + strength["se_r_baseline"] ** 2))
    se_c = np.sqrt(2 * (strength["se_c_between"] ** 2 + strength["se_c_baseline"] ** 2))
    rows = []
    for r0 in (0.3, 0.6):
        e = strength["r_between"] - (strength["r_baseline"] - r0)
        rows.append({"measure": "M15 correlation", "scenario": f"early r_baseline {r0}", "expected_D": e, "se": se_r,
                     "power": projected_power(e, se_r, ALPHA)})
    for drop in (0.05, 0.09):
        e = (strength["c_between"] - strength["c_baseline"]) + drop
        rows.append({"measure": "M5 closeness", "scenario": f"early drop {drop}", "expected_D": e, "se": se_c,
                     "power": projected_power(e, se_c, ALPHA)})
    return rows


def ceilings(meta: dict) -> dict:
    ts, pins = load_current_testsets(), load_pins()
    assert ts["price_level"][0] == "price_level/v4", ts["price_level"][0]
    pilot = runner.plan([ladder.GPT35], CONTROLS, 10_000, "forced", ts, pins=pins)
    full = runner.plan(ladder.NEW, FULL, 10_000, "forced", ts, pins=pins) + \
        runner.plan([ladder.GPT4OMINI, ladder.LLAMA31], CONTROLS, 10_000, "forced", ts, pins=pins)
    return {"pilot_calls": len(pilot), "pilot_ceiling": runner.estimate_cost(pilot, meta),
            "full_calls": len(full), "full_ceiling": runner.estimate_cost(full, meta)}


def main() -> None:
    meta = models_by_id(fetch_models())
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    dates = {m: (resolve_cutoff(m, meta), ladder.RELEASE[m]) for m in ladder.NEW + ladder.REFERENCES}
    cov = coverage_rows(dates, ladder.NEW, ladder.REFERENCES,
                        {"grid v4": ladder.grid_months(), "index": ladder.index_months()})
    strong, power = [], []
    for row in cov:
        if row["usable"] and row["set"] == "index":
            s = reference_strength(calls, row["late"], dates[row["late"]][0], dates[row["early"]])
            strong.append({"early": row["early"], **s})
            power += [{"early": row["early"], "late": row["late"], **p} for p in power_rows(s)]
    cost = ceilings(meta)
    L = ["# Cutoff ladder: feasibility (free; stored answers and the public model list)\n", "## Coverage\n",
         _table(["early", "late", "set", "between", "baseline", "months", "usable"],
                [[short(r["early"]), short(r["late"]), r["set"], r["between"], r["baseline"],
                  f"{r['n_between']} / {r['n_baseline']}", "yes" if r["usable"] else "no"] for r in cov]),
         "\n## The late references on the pair's months (their stored answers)\n",
         _table(["early", "late", "r between", "r baseline", "answered between", "closeness between / baseline"],
                [[short(s["early"]), short(s["late"]), _f(s["r_between"]), _f(s["r_baseline"]),
                  _f(s["answered_between"], "{:.0%}"),
                  f"{_f(s['c_between'], '{:.3f}')} / {_f(s['c_baseline'], '{:.3f}')}"] for s in strong]),
         f"\n## Projected power (alpha {ALPHA:.4f}, Holm's first step)\n",
         _table(["early", "late", "measure", "scenario", "expected D", "SE", "power"],
                [[short(p["early"]), short(p["late"]), p["measure"], p["scenario"], _f(p["expected_D"]),
                  _f(p["se"], "{:.3f}"), _f(p["power"], "{:.0%}")] for p in power]),
         "\n## Cost ceilings (upper bounds; real cost has run at 35–55% of the ceiling)\n",
         f"- Micro-pilot (GPT-3.5 Turbo, M7 + M9): {cost['pilot_calls']} calls, ceiling ${cost['pilot_ceiling']:.3f}",
         f"- Full run (three models, M5 v4 + M14–M17, M7/M9 for GPT-4o-mini and Llama 3.1): {cost['full_calls']} "
         f"calls, ceiling ${cost['full_ceiling']:.2f}\n"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feasibility.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
