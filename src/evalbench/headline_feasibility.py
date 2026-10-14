"""Free feasibility check for M18 headline dating (docs/preregistration/2026-10-05-headline-dating.md), from the built
set and the public model list only: how the set looks, how well the blind baseline dates it, how precisely the knee fit
can locate a horizon, and the cost ceilings of the micro-pilot and the full run. No model is called.

uv run python -X utf8 -m evalbench.headline_feasibility
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import ladder, runner
from .items import ROOT, load_current_testsets, load_testset
from .llm.openrouter import fetch_models, load_pins, models_by_id
from .methods import headline_date as hd
from .pilot_report import _f, _table

OUT = ROOT / "results" / "headline"
PILOT_MODELS = ["openai/gpt-3.5-turbo", "meta-llama/llama-3.3-70b-instruct", "google/gemini-2.5-flash",
                "deepseek/deepseek-chat-v3.1"]
ALL_MODELS = ladder.NEW + ladder.REFERENCES + ["openai/gpt-oss-20b", "qwen/qwen3.8-flash"]


def knee_fit(month_ord: np.ndarray, err: np.ndarray, weights: np.ndarray, min_side: int = 6) -> dict | None:
    """Weighted least squares of err = c + b * max(0, t - H): flat before the knee H, sloping after it (b near -1 when
    answers stop moving forward). H runs over the rows' months with at least min_side rows on each side. Inputs: one
    row per month (mean signed error, its number of answers as the weight). None when no H qualifies."""
    order = np.argsort(month_ord, kind="stable")
    t, e, w = month_ord[order], err[order], weights[order]
    sw = np.sqrt(w)
    best = None
    for k in range(min_side - 1, len(t) - min_side):
        H = t[k]
        x = np.maximum(t - H, 0.0)
        A = np.column_stack([np.ones_like(x), x]) * sw[:, None]
        coef, *_ = np.linalg.lstsq(A, e * sw, rcond=None)
        sse = float(((A @ coef - e * sw) ** 2).sum())
        if best is None or sse < best[0] - 1e-12:
            best = (sse, H, coef)
    if best is None:
        return None
    return {"H": pd.Period(ordinal=int(round(best[1])), freq="M"), "c": float(best[2][0]), "slope": float(best[2][1])}


def monthly(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per month (answered items only): ordinal, mean signed error, count. df: month (Period), err_months."""
    g = df.groupby("month")["err_months"].agg(["mean", "size"])
    return np.array([m.ordinal for m in g.index], dtype=float), g["mean"].to_numpy(float), g["size"].to_numpy(float)


def boot_knee(df: pd.DataFrame, B: int = 2000, seed: int = 0) -> dict:
    """The knee and a month-cluster bootstrap 95% interval (months resampled with their answers)."""
    t, e, w = monthly(df)
    point = knee_fit(t, e, w)
    if point is None:
        return {"H": None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(B):
        idx = rng.integers(0, len(t), len(t))
        f = knee_fit(t[idx], e[idx], w[idx])
        if f is not None:
            draws.append(f["H"].ordinal)
    if not draws:
        return {**point, "lo": None, "hi": None}
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {**point, "lo": pd.Period(ordinal=int(round(lo)), freq="M"), "hi": pd.Period(ordinal=int(round(hi)), freq="M")}


def simulate_knee(h0: pd.Period, sd: float, per_month: int = 12, B: int = 2000, seed: int = 0) -> dict:
    """Synthetic answers: the truth plus noise before h0, h0 plus noise after; the knee fit's interval."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in hd.MONTHS:
        for _ in range(per_month):
            centre = m.ordinal if m.ordinal <= h0.ordinal else h0.ordinal
            rows.append({"month": m, "err_months": round(centre + rng.normal(0, sd)) - m.ordinal})
    return boot_knee(pd.DataFrame(rows), B=B, seed=seed)


def main() -> None:
    items = load_testset("headline_date", "v1")
    main_items = [it for it in items if not hd.is_pilot(it.item_id)]
    df = pd.DataFrame([{"year": it.truth["month"][:4], "lang": it.strata["language"], "type": it.strata["newstype"],
                        "err": hd.month_error(it.truth["blind_month"], it.truth["month"])} for it in main_items])
    df["score"] = np.maximum(0, 1 - df["err"].abs() / 12)
    by_year = df.groupby("year").agg(n=("err", "size"), blind_score=("score", "mean"),
                                      median_abs_err=("err", lambda s: s.abs().median()),
                                      within_3=("err", lambda s: (s.abs() <= 3).mean()))
    sims = [(h, sd, simulate_knee(pd.Period(h, "M"), sd, B=500)) for h in ("2021-09", "2023-12", "2025-03")
            for sd in (4.0, 12.0)]
    meta, ts, pins = models_by_id(fetch_models()), load_current_testsets(), load_pins()
    ts = {**ts, "headline_date": ("headline_date/v1", load_testset("headline_date", "v1"))}   # the v1 plan's set
    pilot = runner.plan(PILOT_MODELS, ["headline_date"], hd.PILOT_N, "forced", ts, pins=pins)
    full = runner.plan(ALL_MODELS, ["headline_date"], 10_000, "forced", ts, pins=pins)
    L = ["# M18 headline dating: feasibility (free)\n",
         f"Set: {len(items)} items ({hd.PILOT_N} pilot, {len(main_items)} main). Languages: "
         f"{df['lang'].value_counts().to_dict()}; types: {df['type'].value_counts().to_dict()}.\n",
         f"Blind baseline overall (main items): score {df['score'].mean():.3f}, median {df['err'].abs().median():.0f} "
         f"months off, within ±3 months {(df['err'].abs() <= 3).mean():.0%}, exact month {(df['err'] == 0).mean():.0%}.\n",
         "## The blind baseline by year (main items)\n",
         _table(["year", "n", "blind score", "median months off", "within ±3"],
                [[y, int(r.n), _f(r.blind_score, "{:.3f}"), _f(r.median_abs_err, "{:.0f}"), _f(r.within_3, "{:.0%}")]
                 for y, r in by_year.iterrows()]),
         "\n## Knee-fit precision on synthetic answers (12 per month)\n",
         _table(["true horizon", "answer noise (SD, months)", "estimate", "95% interval"],
                [[h, sd, str(s["H"]), f"{s['lo']}..{s['hi']}"] for h, sd, s in sims]),
         "\n## Cost ceilings\n",
         f"- Micro-pilot ({len(PILOT_MODELS)} models x {hd.PILOT_N} pilot items): {len(pilot)} calls, "
         f"ceiling ${runner.estimate_cost(pilot, meta):.3f}",
         f"- Full run (nine models, all {len(items)} items): {len(full)} calls, "
         f"ceiling ${runner.estimate_cost(full, meta):.2f}\n"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feasibility.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
