"""The new models of 2026-10-07 (Gemma 4 31B, Gemma 4 26B A4B, Qwen3.6 27B), registered in
docs/preregistration/2026-10-07-new-models.md before their answers. Released in April 2026, they have only about four
months after release in our data (to 2026-08), so their primary tests need no after side:

- **Same months, a model that cannot know them** (the cutoff ladder's design): M15 correlation and M5 closeness on the
  months after GPT-3.5 Turbo's release up to the new model's BEFORE edge, minus the same difference on months both may
  know; months resampled together. Memory predicts D > 0; a time effect shared by both cancels.
- **Recall inside the window:** M15 correlation on BEFORE months against a month-shuffle null (the true moves of each
  index reassigned across months, so knowing which index moves more earns nothing).
- **M18:** model minus the text-only baseline on BEFORE months (headline_date/v2).
- **M19 fake date:** after the gate, do the dated "forecasts" on BEFORE months track the real moves (month shuffle)?
- **M20 masking:** the registered measure on paths ending before the BEFORE edge.
Holm across the three models within each test. The release-line tests read "insufficient" below six months after
release. No model calls, no cost.

uv run python -X utf8 -m evalbench.new_models      # -> results/new_models/report.md
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from . import (cutoff_estimate, dating_horizon, fake_date, headline_report, index_report, ladder, ladder_report,
               mask_date, readiness, stats, store)
from .cutoffs import resolve_cutoff, resolve_release
from .grid_report import staggered_pair
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods import headline_date as hd
from .methods import index_forecast as fd
from .methods import path_date as pdm
from .pilot_report import _f, _table, short

NEW = cutoff_estimate.NEW
EARLY = ladder.GPT35
MIN_AFTER = ladder.MIN_MONTHS
B = 20000
OUT = ROOT / "results" / "new_models"


def _corr0(answer: np.ndarray, truth: np.ndarray) -> float:
    ok = ~(np.isnan(answer) | np.isnan(truth))
    a, t = answer[ok], truth[ok]
    if len(a) < 3 or a.std() == 0 or t.std() == 0:
        return 0.0
    return float(np.corrcoef(a, t)[0, 1])


def recall_perm(frame: pd.DataFrame, last: pd.Period, B: int = B, seed: int = 0) -> dict:
    """r of the answers with the true moves on months up to `last`, and the one-sided p of a month-shuffle null: each
    index keeps its own true moves, reassigned across months, so only month-specific knowledge beats it."""
    f = frame[frame["month"] <= last]
    if not len(f):
        return {"r": float("nan"), "p": float("nan"), "n": 0}
    truth = f.pivot_table(index="month", columns="symbol", values="truth", aggfunc="first")
    answer = f.pivot_table(index="month", columns="symbol", values="answer", aggfunc="first").reindex_like(truth)
    T, A = truth.to_numpy(float), answer.to_numpy(float)
    r = _corr0(A.ravel(), T.ravel())
    rng = np.random.default_rng(seed)
    hits = sum(_corr0(A.ravel(), T[rng.permutation(len(T))].ravel()) >= r for _ in range(B))
    return {"r": r, "p": (1 + hits) / (B + 1), "n": len(f)}


def release_line_status(release: date) -> str | None:
    """None when the release-line tests can run; else why they read 'insufficient'."""
    months = ladder.clean_months(release)
    return None if months >= MIN_AFTER else f"insufficient: {months} months after release (< {MIN_AFTER})"


def cross_row(late: pd.DataFrame, early: pd.DataFrame, cut_late: date, cut_early: date, rel_early: date,
              B: int = B) -> dict:
    """M15 correlation, same months: the new model (late) against one that cannot know them (early)."""
    months = sorted(set(late["month"]) | set(early["month"]))
    between, baseline = ladder.pair_months(cut_early, rel_early, cut_late, months)
    r = ladder.staggered_corr(late, early, between, baseline, B=B)
    return {"between": f"{between[0]}..{between[-1]}" if between else "–", "usable": ladder.usable(between, baseline),
            "D": r["D"], "lo": r["lo"], "hi": r["hi"], "p": r["p"], "n": r["n_between"],
            "detail": f"new {r['r_late_between']:+.2f}, GPT-3.5 {r['r_early_between']:+.2f} on those months"}


def zero_as_refused(g: pd.DataFrame) -> pd.DataFrame:
    """A price of 0 counts as not answered in the refusal guard (scoring spec §10); it still scores 0 in D."""
    if "no_answer" not in g:
        return g
    return g.assign(refused=g["refused"].astype(bool) | g["no_answer"].fillna(False).astype(bool))


def cross_m5(calls, model: str, cut: date, cut_e: date, rel_e: date, B: int = B) -> dict:
    between, baseline = ladder.pair_months(cut_e, rel_e, cut, ladder.grid_months())
    r = staggered_pair(zero_as_refused(ladder.grid_scores(calls, model, cut)),
                       zero_as_refused(ladder.grid_scores(calls, EARLY, cut_e)), between, baseline, B=B)
    return {"between": f"{between[0]}..{between[-1]}" if between else "–", "usable": ladder.usable(between, baseline),
            "D": r["dkappa"], "lo": r["ci_lo"], "hi": r["ci_hi"], "p": r["p_value"], "n": r["n_pre"],
            "answered_D": r["answered_dkappa"], "answered_p": r["answered_p"]}


def _holm(rows: list[dict], key: str, ok=lambda r: True) -> None:
    tested = [r for r in rows if key in r and ok(r[key]) and pd.notna(r[key].get("p"))]
    for r, p in zip(tested, stats.holm([r[key]["p"] for r in tested])):
        r[key]["p_holm"] = float(p)


def _reading(t: dict | None, yes: str) -> str:
    if not t or "p_holm" not in t:
        return "–" if not t else ("too few months" if t.get("usable") is False else "no estimate")
    ok = t["p_holm"] < 0.05 and (t.get("D", t.get("r", t.get("x", t.get("d", 0)))) or 0) > 0
    if ok and "answered_p_holm" in t and not (t["answered_p_holm"] < 0.05 and t["answered_D"] > 0):
        return f"not confirmed: refusal-driven ({yes})"
    return yes if ok else "not shown"


def compute(models: list[str] = NEW) -> dict:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    cut_e, rel_e = resolve_cutoff(EARLY, meta), resolve_release(EARLY, meta)
    early_idx = ladder.index_frame(calls, EARLY)
    heads = {it.item_id: it for it in load_testset(hd.NAME, "v2")}
    paths = {it.item_id: it for it in load_testset(pdm.NAME, mask_date.VERSION)}
    us_move = index_report.us_move_by_month()
    rows = []
    for m in models:
        cut, rel = resolve_cutoff(m, meta), resolve_release(m, meta)
        last = ladder.last_before(cut)
        idx = ladder.index_frame(calls, m)
        row = {"model": m, "cutoff": cut, "release": rel, "clean_months": ladder.clean_months(rel), "n_index": len(idx)}
        if len(idx):
            row["cross_m15"] = cross_row(idx, early_idx, cut, cut_e, rel_e)
            row["recall"] = recall_perm(idx, last)
            before = idx[idx["month"] <= last]
            row["beyond_us"] = index_report.partial_test(before, us_move) if len(before) else None
        if len(ladder.grid_scores(calls, m, cut)):
            row["cross_m5"] = cross_m5(calls, m, cut, cut_e, rel_e)
        sc = headline_report.scored(calls, m, cut, heads, testset=f"{hd.NAME}/v2", release=rel)
        if len(sc):
            row["m18"] = headline_report.h1_row(headline_report.main_only(sc))
            ex = dating_horizon.excess_by_year(dating_horizon.frame(calls, m, cut, heads, testset=f"{hd.NAME}/v2",
                                                                    release=rel))
            row["news_through"] = dating_horizon.horizon(ex)
        pilot, main = fake_date._frames(calls, m, fd.NAME, fake_date.VERSION)
        if len(pilot) or len(main):
            gate = fake_date.gate_row(pilot, fake_date.PILOT)
            row["m19_gate"] = gate
            if gate["passes"]:
                row["m19"] = {**recall_perm(main, last), "leak": fake_date.leak_share(
                    recall_perm(main, last, B=1)["r"], row.get("recall", {}).get("r", float("nan")))}
        mf = mask_date.frame(calls, m, paths, cut)
        if len(mf):
            row["m20"] = mask_date.model_row(mf, cut)
        row["release_line"] = release_line_status(rel)
        gate = ladder_report.pilot_gate(calls, meta, model=m)
        row["controls"] = "passes" if readiness.controls_pass(gate["rows"]) else "fails"
        rows.append(row)
    _holm(rows, "cross_m15", lambda t: t["usable"])
    _holm(rows, "cross_m5", lambda t: t["usable"])
    guard = [r for r in rows if "cross_m5" in r and "p_holm" in r["cross_m5"] and pd.notna(r["cross_m5"]["answered_p"])]
    for r, p in zip(guard, stats.holm([r["cross_m5"]["answered_p"] for r in guard])):
        r["cross_m5"]["answered_p_holm"] = float(p)
    for key in ("recall", "m18", "m19", "m20"):
        _holm(rows, key)
    return {"rows": rows, "early": EARLY}


def _cell(t: dict | None, stat: str, yes: str, fmt: str = "{:+.2f}") -> str:
    if not t:
        return "–"
    v = t.get(stat)
    interval = f" [{_f(t.get('lo'), fmt)}, {_f(t.get('hi'), fmt)}]" if t.get("lo") is not None else ""
    return f"{_f(v, fmt)}{interval}, p {_f(t.get('p_holm', t.get('p')), '{:.3f}')}: {_reading(t, yes)}"


def render(res: dict) -> str:
    rows = res["rows"]
    L = ["# The new models (Gemma 4 31B, Gemma 4 26B A4B, Qwen3.6 27B)\n",
         "Registered in `docs/preregistration/2026-10-07-new-models.md` before their answers. Released in April 2026, "
         "they have about four months after release in our data, so the primary tests need no after side. Holm across "
         "the three models within each test.\n",
         _table(["model", "cutoff", "release", "months after release", "positive controls"],
                [[short(r["model"]), str(r["cutoff"]), str(r["release"]), r["clean_months"], r["controls"]]
                 for r in rows]),
         "\n## Primary: is it memory?\n",
         _table(["model", "M15, same months vs GPT-3.5 (D [95%], p Holm)", "months", "M5, same months vs GPT-3.5",
                 "recall inside the window (M15 r vs month shuffle)", "M18 vs text only (BEFORE)"],
                [[short(r["model"]), _cell(r.get("cross_m15"), "D", "memory, not a time effect"),
                  (r.get("cross_m15") or {}).get("between", "–"),
                  _cell(r.get("cross_m5"), "D", "memory, not a time effect", "{:+.3f}"),
                  _cell(r.get("recall"), "r", "recall shown"),
                  _cell(r.get("m18"), "d", "dates better than the text alone", "{:+.3f}")] for r in rows]),
         "\n## The usual defences\n",
         _table(["model", "M19 gate (answered / informative)", "M19: dated forecasts track the real moves",
                 "leak share", "M20 masking (beyond habits, BEFORE paths)"],
                [[short(r["model"]),
                  (f"{_f(r['m19_gate']['answered'], '{:.0%}')} / {_f(r['m19_gate']['informative'], '{:.0%}')}"
                   + ("" if r["m19_gate"]["passes"] else ": did not forecast")) if r.get("m19_gate") else "–",
                  _cell(r.get("m19"), "r", "hindsight leaks through the date"),
                  _f((r.get("m19") or {}).get("leak"), "{:.0%}"),
                  _cell(r.get("m20"), "x", "dates the unlabeled path", "{:+.3f}")] for r in rows]),
         "\n## Descriptive\n",
         _table(["model", "Nordic beyond the US move (partial r, BEFORE)", "news clearly dated through (v2)",
                 "release-line tests"],
                [[short(r["model"]),
                  f"{_f((r.get('beyond_us') or {}).get('r'), '{:+.2f}')} [{_f((r.get('beyond_us') or {}).get('lo'), '{:+.2f}')}, "
                  f"{_f((r.get('beyond_us') or {}).get('hi'), '{:+.2f}')}]" if r.get("beyond_us") else "–",
                  str(r.get("news_through") or "–"), r["release_line"] or "run (enough months after release)"]
                 for r in rows]),
         "\nDense vs mixture of experts: the two Gemma 4 models share a maker, a cutoff and a host (Novita, bf16), so "
         "their rows differ mainly by architecture (31B dense vs 26B with about 4B active).\n"]
    return "\n".join(L)


def main() -> None:
    text = render(compute())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
