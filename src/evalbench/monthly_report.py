"""M11 monthly return v2 report: every analysis registered in docs/preregistration/2026-10-05-monthly-return-v2.md,
from the stored answers only (no model calls, no cost). Pilot E's E1 rule on v2 (release line, Δκ, 20,000 draws,
Holm across the nine models, the refusal guard), pilot E's E4 (absolute direction against the market-direction
guesser), next to pilot E's registered v1 cells; and how jagged the rolling 6-month curve is on v1 and v2.
Writes results/monthly/report.md.

uv run python -X utf8 -m evalbench.monthly_report
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import analysis, ladder, pilot, stats, store
from .cutoffs import resolve_cutoff
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .pilot_e import guarded, refusal_affected, release_groups
from .pilot_report import _f, _table, short

OUT = ROOT / "results" / "monthly"
METHOD = "monthly_return"
MODELS = ladder.REFERENCES + ladder.NEW + ["openai/gpt-oss-20b", "qwen/qwen3.8-flash"]
BLIND = "baseline/blind@2024-06-30"
CONTROL_FAILS = {"openai/gpt-oss-20b"}       # readiness: fails the positive controls
B = 20000


def wobble(curve: pd.DataFrame) -> float:
    """How jagged a rolling curve is: the median absolute change between windows one month apart (steps across a
    gap left out)."""
    if len(curve) < 2:
        return float("nan")
    c = curve.assign(_p=pd.PeriodIndex(curve["month"], freq="M")).sort_values("_p")
    ords = np.array([p.ordinal for p in c["_p"]])
    steps = np.abs(np.diff(c["mean"].to_numpy(float)))[np.diff(ords) == 1]
    return float(np.median(steps)) if len(steps) else float("nan")


def directional_only(df: pd.DataFrame) -> pd.DataFrame:
    """Answered items that name a direction: refusals and the 0% hedge (scored 0.5 on both sides) left out."""
    if not len(df):
        return df
    nd = df["no_direction"].fillna(False).astype(bool) if "no_direction" in df else pd.Series(False, index=df.index)
    return df[~df["refused"].astype(bool) & ~nd]


def hedge_share(df: pd.DataFrame) -> float:
    """The share of answered items answered with exactly 0%."""
    a = df[~df["refused"].astype(bool)]
    if not len(a):
        return float("nan")
    if "no_direction" not in a:
        return 0.0
    return float(a["no_direction"].fillna(False).astype(bool).mean())


def hedge_by_half(df: pd.DataFrame) -> pd.Series:
    """The share of answers of exactly 0% per half-year (real, answered items): the 'I don't know' rate over time."""
    h = analysis.hedge_frame(df)
    if not len(h):
        return pd.Series(dtype=float)
    d = pd.to_datetime(h["date"])
    half = d.dt.year.astype(str) + np.where(d.dt.month <= 6, "-H1", "-H2")
    return h["score"].groupby(half.to_numpy()).mean()


def compute() -> dict:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    sets = {v: {it.item_id: it for it in load_testset(METHOD, v)} for v in ("v1", "v2")}

    def scored(model: str, version: str) -> pd.DataFrame:
        sub = calls[(calls["model"] == model) & (calls["testset"] == f"{METHOD}/{version}")]
        return analysis.item_scores(sub, sets[version], METHOD, resolve_cutoff(model, meta))

    rows, e4, curves, hedged, halves = [], [], [], [], {}
    for m in MODELS:
        cut, rel = resolve_cutoff(m, meta), ladder.RELEASE[m]
        df = scored(m, "v2")
        df = df.assign(group=release_groups(df["date"], cut, rel).to_numpy()) if len(df) else df
        reg = analysis.summarize(df, METHOD, B=B)
        ans = analysis.summarize(df[~df["refused"].astype(bool)], METHOD, B=B) if len(df) else reg
        rows.append({"model": m, **reg, "answered_dkappa": ans["dkappa"], "answered_p": ans["p_value"],
                     "refusal_affected": refusal_affected(reg["coverage_pre"], reg["coverage_post"]),
                     "mean_pre": float(df.loc[(df["group"] == 1) & ~df["fabricated"], "score"].mean()),
                     "mean_post": float(df.loc[(df["group"] == 0) & ~df["fabricated"], "score"].mean())})
        halves[m] = hedge_by_half(df) if len(df) else pd.Series(dtype=float)
        dirs = directional_only(df[~df["fabricated"].astype(bool)]) if len(df) else df
        d = analysis.summarize(directional_only(df), METHOD, B=B) if len(df) else reg
        pre = dirs[dirs["group"] == 1] if len(dirs) else dirs
        lo, hi = stats.bootstrap_mean(pre["score"], pre["cluster"], B=B) if len(pre) else (np.nan, np.nan)
        side = lambda g: hedge_share(df[~df["fabricated"].astype(bool) & (df["group"] == g)]) if len(df) else np.nan
        hedged.append({"model": m, "hedge_pre": side(1), "hedge_post": side(0),
                       "dir_pre": float(pre["score"].mean()) if len(pre) else np.nan, "dir_lo": lo, "dir_hi": hi,
                       **{k: d[k] for k in ("n_pre", "n_post", "dkappa", "ci_lo", "ci_hi", "p_value")},
                       "enough": bool(d["n_pre"] >= pilot.MIN_PRIMARY and d["n_post"] >= pilot.MIN_PRIMARY)})
        real = df[~df["fabricated"].astype(bool)].copy()
        truth_up = real["item_id"].map(lambda i: sets["v2"][i].truth["return_pct"] > 0)
        market_up = real["item_id"].map(lambda i: sets["v2"][i].truth["market_return_pct"] > 0)
        real["guesser"] = (truth_up == market_up).astype(float)
        row = {"model": m}
        for g, side in ((1, "pre"), (0, "post")):
            part = real[real["group"] == g]
            a = part[~part["refused"].astype(bool)]
            row.update({f"n_{side}": len(part),
                        f"abs_{side}": float(a["abs_direction_hit"].astype(float).mean()) if len(a) else np.nan,
                        f"guesser_{side}": float(part["guesser"].mean()) if len(part) else np.nan,
                        f"err_{side}": float(a["abs_err_pp"].median()) if len(a) else np.nan})
        e4.append(row)
    for key, out in (("p_value", "p_holm"), ("answered_p", "answered_p_holm")):
        for r, p in zip(rows, stats.holm([r[key] for r in rows])):
            r[out] = float(p)
    for r in rows:
        r["reading"] = guarded(pilot.primary_reading(r), r["answered_p_holm"], r["answered_dkappa"])
    tested = [r for r in hedged if r["enough"]]
    for r, p in zip(tested, stats.holm([r["p_value"] for r in tested])):
        r["p_holm"] = float(p)

    for m in [*MODELS, BLIND]:
        c = {"model": m}
        for v in ("v1", "v2"):
            df = scored(m, v)
            curve = analysis.rolling_curve(df, window=6, min_n=15, B=200) if len(df) else pd.DataFrame()
            c.update({f"wobble_{v}": wobble(curve) if len(curve) else np.nan,
                      f"windows_{v}": len(curve), f"n_{v}": int(curve["n"].median()) if len(curve) else 0})
        curves.append(c)

    v1 = {}
    path = ROOT / "results" / "pilot_e" / "results.json"
    if path.exists():
        v1 = {r["model"]: r["reading"] for r in json.loads(path.read_text(encoding="utf-8"))["e1"]
              if r["method"] == METHOD}
    return {"cutoffs": {m: str(resolve_cutoff(m, meta)) for m in MODELS}, "rows": rows, "e4": e4, "curves": curves, "v1": v1, "hedged": hedged, "halves": halves,
            "missing": {m: len(sets["v2"]) - len(scored(m, "v2")) for m in MODELS
                        if len(sets["v2"]) - len(scored(m, "v2"))}}


def render(res: dict) -> str:
    L = ["# M11 monthly return v2\n",
         "Registered in `docs/preregistration/2026-10-05-monthly-return-v2.md` before any answer. `monthly_return/v2`: "
         "1,340 items (1,206 real). Score: 1 if the answer is on the right side of that month's median Nordic share "
         "(chance 0.5). Release line: before = more than 91 days before the stated cutoff, after = after release. "
         "Δκ = (before − after) / 0.5, cluster bootstrap by company (20,000 draws), Holm across the nine models, with "
         "the refusal guard.\n",
         f"Unanswered items: {', '.join(f'{short(m)} {n}' for m, n in res['missing'].items()) or 'none'}.\n",
         "## Main table (registered)\n",
         _table(["model", "before / after items", "score before", "score after", "Δκ", "95% interval", "p (Holm)",
                 "answered before / after", "reading (v2)", "pilot E reading (v1)"],
                [[short(r["model"]) + (" †" if r["model"] in CONTROL_FAILS else ""), f"{r['n_pre']} / {r['n_post']}",
                  _f(r["mean_pre"], "{:.3f}"), _f(r["mean_post"], "{:.3f}"), _f(r["dkappa"], "{:+.2f}"),
                  f"[{_f(r['ci_lo'], '{:+.2f}')}, {_f(r['ci_hi'], '{:+.2f}')}]", _f(r["p_holm"], "{:.3f}"),
                  f"{_f(r['coverage_pre'], '{:.0%}')} / {_f(r['coverage_post'], '{:.0%}')}"
                  + (" ⚠" if r["refusal_affected"] else ""), r["reading"], res["v1"].get(r["model"], "not asked")]
                 for r in res["rows"]]),
         "\n† fails the positive controls: its \"none detected\" reads \"too hard for this model\".\n",
         "## Exploratory: directional answers only (not registered)\n",
         "An answer of exactly 0% scores 0.5 on both sides. Several models give it mostly for months after their "
         "cutoff, so their after side is nearly all 0.5 and the registered Δκ compares answers with non-answers. Two "
         "checks on the answers that name a direction (refusals and 0% left out): the score before the cutoff against "
         "chance (0.5 by construction: half the shares beat the month's median), and the before/after test where "
         "both sides keep at least 20 such answers (Holm across those).\n",
         _table(["model", "answers of 0% before / after", "before: score [95% interval]", "before / after items",
                 "Δκ", "95% interval", "p (Holm)"],
                [[short(r["model"]), f"{_f(r['hedge_pre'], '{:.0%}')} / {_f(r['hedge_post'], '{:.0%}')}",
                  f"{_f(r['dir_pre'], '{:.3f}')} [{_f(r['dir_lo'], '{:.3f}')}, {_f(r['dir_hi'], '{:.3f}')}]",
                  f"{r['n_pre']} / {r['n_post']}"]
                 + ([_f(r["dkappa"], "{:+.2f}"), f"[{_f(r['ci_lo'], '{:+.2f}')}, {_f(r['ci_hi'], '{:+.2f}')}]",
                     _f(r.get("p_holm"), "{:.3f}")] if r["enough"] else ["too few after", "–", "–"])
                 for r in res["hedged"]]),
         "\n## Exploratory: how often each model answers 0% (\"I don't know\"), per half-year\n",
         "Forced to give a number, several models answer 0% for months they do not know. A jump marks where a model stops "
         "claiming to know; a model that never answers 0% gives no signal. Since scoring spec §7 (2026-10-06) the app "
         "leaves these answers out of its scores and shows this rate on its own.\n",
         _table(["model", "stated cutoff", *sorted({h for s_ in res["halves"].values() for h in s_.index})],
                [[short(m), res["cutoffs"][m], *[_f(s_.get(h), "{:.0%}") for h in
                                                  sorted({h for s2 in res["halves"].values() for h in s2.index})]]
                 for m, s_ in res["halves"].items()]),
         "\n## Absolute direction (pilot E's E4)\n",
         "Up or down, against the guesser that names the market's direction that month (not 50%); answered items.\n",
         _table(["model", "before: right direction", "before: market guesser", "after: right direction",
                 "after: market guesser", "median error before / after (pp)"],
                [[short(r["model"]), _f(r["abs_pre"], "{:.0%}"), _f(r["guesser_pre"], "{:.0%}"),
                  _f(r["abs_post"], "{:.0%}"), _f(r["guesser_post"], "{:.0%}"),
                  f"{_f(r['err_pre'], '{:.1f}')} / {_f(r['err_post'], '{:.1f}')}"] for r in res["e4"]]),
         "\n## The rolling 6-month curve, v1 and v2 (descriptive)\n",
         "Wobble = the median change between windows one month apart; with no real change it is noise alone.\n",
         _table(["model", "v1: windows / answers per window / wobble", "v2: windows / answers per window / wobble"],
                [[short(c["model"]), f"{c['windows_v1']} / {c['n_v1']} / {_f(c['wobble_v1'], '{:.3f}')}",
                  f"{c['windows_v2']} / {c['n_v2']} / {_f(c['wobble_v2'], '{:.3f}')}"] for c in res["curves"]])]
    return "\n".join(L) + "\n"


def main() -> None:
    res = compute()
    OUT.mkdir(parents=True, exist_ok=True)
    text = render(res)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
