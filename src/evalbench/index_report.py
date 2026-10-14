"""Index report, from the stored answers only (no model calls, no cost). Writes results/index/.

- M14/M15 (Nordic): the revised confirmatory test registered in docs/preregistration/2026-10-01-index.md §6.3; the 24
  micro-pilot questions are left out.
- M16/M17 (US, PRIVATE: Yahoo data) and the US − Nordic comparison: docs/preregistration/2026-10-01-us-index.md §4,
  with E1 (answer style), E2 (per index) and E4 "Nordic beyond US" (§6.1).

uv run python -m evalbench.index_report
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import analysis, market, stats, store
from .cutoffs import resolve_cutoff
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .market import sides  # noqa: F401  (the registered windows, shared with the app's cells)
from .pilot_d_report import MODELS, RELEASE
from .pilot_report import _f, _jsonable, _table, short

OUT = ROOT / "results" / "index"
BLIND = "baseline/blind@2024-06-30"
PILOT = 24
B = 20000
NO_AFTER = {"qwen/qwen3.8-flash"}
MARKETS = {"nordic": ("index_level", "index_return", PILOT), "us": ("us_index_level", "us_index_return", 0)}


def _p_two_sided(draws: np.ndarray) -> float:
    return min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())) if len(draws) else float("nan")


def _interval(draws: np.ndarray) -> tuple[float, float, float]:
    """(lo, hi, two-sided p) of bootstrap draws; NaN when there are none (a side without months)."""
    draws = draws[~np.isnan(draws)]
    if not len(draws):
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi), _p_two_sided(draws)


def corr_drop(df: pd.DataFrame, before_last: pd.Period, after_first: pd.Period, B: int = B, seed: int = 0) -> dict:
    """r_before - r_after (answers that never vary: r = 0) with a cluster bootstrap over months (each month's
    questions resampled together); two-sided p = 2 x the smaller tail beyond 0. Vectorised: each draw is a vector
    of month counts, drawn with the same generator calls as the original loop, so the results are unchanged."""
    bef, aft = df[df["month"] <= before_last], df[df["month"] >= after_first]
    r_b, r_a = market.corr(bef["answer"], bef["truth"]), market.corr(aft["answer"], aft["truth"])
    sb, sa = market._month_sums(bef), market._month_sums(aft)
    kb, ka = len(sb), len(sa)
    if not kb or not ka:
        nan = float("nan")
        return {"r_before": r_b, "r_after": r_a, "diff": r_b - r_a, "lo": nan, "hi": nan, "p": nan,
                "r_before_lo": nan, "r_before_hi": nan, "n_before": len(bef), "n_after": len(aft)}
    rng = np.random.default_rng(seed)
    cb, ca = np.zeros((B, kb)), np.zeros((B, ka))
    for i in range(B):
        cb[i] = np.bincount(rng.integers(0, kb, kb), minlength=kb)
        ca[i] = np.bincount(rng.integers(0, ka, ka), minlength=ka)
    rb = market._corr_from_sums(cb @ sb)
    lo, hi, p = _interval(rb - market._corr_from_sums(ca @ sa))
    # recall before the cutoff on its own (same draws; descriptive): a model can recall the market inside its
    # window although its drop is not shown, when it gives few real answers after release
    rb = rb[np.isfinite(rb)]
    rb_lo, rb_hi = (np.percentile(rb, [2.5, 97.5]) if len(rb) else (float("nan"),) * 2)
    return {"r_before": r_b, "r_after": r_a, "diff": r_b - r_a, "lo": lo, "hi": hi, "p": p,
            "r_before_lo": float(rb_lo), "r_before_hi": float(rb_hi), "n_before": len(bef), "n_after": len(aft)}


def recall_interval(df: pd.DataFrame, before_last: pd.Period, B: int = B, seed: int = 0) -> tuple[float, float]:
    """The 95% month-bootstrap interval of r before the cutoff, for a model without an AFTER window (descriptive)."""
    sb = market._month_sums(df[df["month"] <= before_last])
    if not len(sb):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    counts = np.array([np.bincount(rng.integers(0, len(sb), len(sb)), minlength=len(sb)) for _ in range(B)])
    r = market._corr_from_sums(counts @ sb)
    r = r[np.isfinite(r)]
    return tuple(float(v) for v in np.percentile(r, [2.5, 97.5])) if len(r) else (float("nan"), float("nan"))


def _mean_test(values: pd.Series, clusters: pd.Series, B: int = B, seed: int = 0) -> tuple[float, float, float, float]:
    """(mean, lo, hi, two-sided p against 0), resampling clusters. The counts come from one vectorised draw that uses
    the same random stream as drawing k cluster indices B times in a row (stats._resample_counts), so the numbers are
    exactly those of the original one-draw-at-a-time loop."""
    g = values.astype(float).groupby(clusters)
    sums, ns = g.sum().to_numpy(), g.size().to_numpy(float)
    k = len(sums)
    if not k:
        return float("nan"), float("nan"), float("nan"), float("nan")
    counts = stats._resample_counts(k, B, seed)
    lo, hi, p = _interval((counts @ sums) / (counts @ ns))
    return float(values.mean()), lo, hi, p


def _month_counts(k: int, B: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(k, np.full(k, 1.0 / k), size=B).astype(float)


def paired_corr_diff(us: pd.DataFrame, nordic: pd.DataFrame, B: int = B, seed: int = 0) -> dict:
    """r(US) - r(Nordic) on the months both frames have, with a paired cluster bootstrap: each resampled month brings
    its US and its Nordic questions (US plan §4.2)."""
    months = sorted(set(us["month"]) & set(nordic["month"]))
    us, nordic = us[us["month"].isin(months)], nordic[nordic["month"].isin(months)]
    su, sn = market._month_sums(us), market._month_sums(nordic)
    r_u, r_n = market.corr(us["answer"], us["truth"]), market.corr(nordic["answer"], nordic["truth"])
    if months:
        counts = _month_counts(len(months), B, seed)
        lo, hi, p = _interval(market._corr_from_sums(counts @ su) - market._corr_from_sums(counts @ sn))
    else:
        lo = hi = p = float("nan")
    return {"r_us": r_u, "r_nordic": r_n, "diff": r_u - r_n, "lo": lo, "hi": hi, "p": p,
            "n_months": len(months), "n_us": len(us), "n_nordic": len(nordic)}


def _partial_sums(df: pd.DataFrame, us_move: pd.Series) -> np.ndarray:
    """Per month: n and the sums of a, t, u and their squares and cross products (answer, truth, US move)."""
    a, t = df["answer"].to_numpy(float), df["truth"].to_numpy(float)
    u = df["month"].map(us_move).to_numpy(float)
    parts = pd.DataFrame({"m": df["month"].to_numpy(), "n": 1.0, "a": a, "t": t, "u": u, "aa": a * a, "tt": t * t,
                          "uu": u * u, "at": a * t, "au": a * u, "tu": t * u})
    return parts.groupby("m", sort=True)[["n", "a", "t", "u", "aa", "tt", "uu", "at", "au", "tu"]].sum().to_numpy()


def _partial_from_sums(s: np.ndarray) -> np.ndarray:
    """Partial correlation of answer and truth given the US move, one per row of summed statistics. Answers that never
    vary, or that are exactly a function of the US move (nothing left after removing it), give 0."""
    n, a, t, u, aa, tt, uu, at, au, tu = s.T
    with np.errstate(invalid="ignore", divide="ignore"):
        va, vt, vu = aa - a * a / n, tt - t * t / n, uu - u * u / n
        cat, cau, ctu = at - a * t / n, au - a * u / n, tu - t * u / n
        a_flat, t_flat = va <= 1e-9 * np.maximum(aa, 1.0), vt <= 1e-9 * np.maximum(tt, 1.0)
        u_flat = vu <= 1e-9 * np.maximum(uu, 1.0)    # one US move for all rows: nothing to remove, r = r_at
        r_at = np.where(a_flat | t_flat, 0.0, cat / np.sqrt(va * vt))
        r_au = np.where(a_flat | u_flat, 0.0, cau / np.sqrt(va * vu))
        r_tu = np.where(t_flat | u_flat, 0.0, ctu / np.sqrt(vt * vu))
        left_a, left_t = 1 - r_au ** 2, 1 - r_tu ** 2
        flat = a_flat | t_flat | (left_a <= 1e-12) | (left_t <= 1e-12)
        r = np.where(flat, 0.0, (r_at - r_au * r_tu) / np.sqrt(np.where(flat, 1.0, left_a * left_t)))
    return np.where(n < 4, np.nan, np.clip(r, -1.0, 1.0))


def partial_corr(df: pd.DataFrame, us_move: pd.Series) -> float:
    """E4: the correlation between the model's Nordic answers and the true Nordic changes, both with the month's US
    move removed by least squares (US plan §6.1)."""
    if df.empty:
        return float("nan")
    return float(_partial_from_sums(_partial_sums(df, us_move).sum(axis=0, keepdims=True))[0])


def partial_test(df: pd.DataFrame, us_move: pd.Series, B: int = B, seed: int = 0) -> dict:
    """partial_corr with a cluster bootstrap by month, two-sided against 0."""
    if df.empty:
        return {"r": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "n": 0}
    sums = _partial_sums(df, us_move)
    lo, hi, p = _interval(_partial_from_sums(_month_counts(len(sums), B, seed) @ sums))
    return {"r": partial_corr(df, us_move), "lo": lo, "hi": hi, "p": p, "n": len(df)}


def answer_style(df: pd.DataFrame, top: int = 3) -> dict:
    """E1: how many different answers a model gives, and its most repeated values with their shares."""
    if df.empty:
        return {"n": 0, "distinct": 0, "top": []}
    vc = df["answer"].round(2).value_counts()
    return {"n": len(df), "distinct": int(len(vc)),
            "top": [(float(v), round(float(c) / len(df), 4)) for v, c in vc.head(top).items()]}


def _market(calls, meta, models: list[str], market_key: str, release: dict = RELEASE) -> dict:
    """The registered tests of one market: the change drop (M15/M17), model - blind on levels (M14/M16), answer rates,
    and the frames the comparisons need."""
    level_m, return_m, skip = MARKETS[market_key]
    items = {m: load_testset(m, "v1")[skip:] for m in (level_m, return_m)}
    lvl_items = {it.item_id: it for it in items[level_m]}
    ret_items = {it.item_id: it for it in items[return_m]}
    ret_calls = calls[calls["testset"] == f"{return_m}/v1"]
    lvl_calls = calls[calls["testset"] == f"{level_m}/v1"]
    blind = analysis.item_scores(lvl_calls[lvl_calls["model"] == BLIND], lvl_items, level_m, resolve_cutoff(BLIND, {}))
    blind = dict(zip(blind["item_id"], blind["score"]))
    res = {"ret": [], "lvl": [], "answer_rates": [], "frames": {}, "levels": {}}
    for m in models:
        cut, rel = resolve_cutoff(m, meta), release[m]
        last, first = sides(cut, rel)
        df = market.answers_frame(ret_calls, items[return_m], m, return_m)
        res["frames"][m] = (df, last, first)
        asked = ret_calls[(ret_calls["model"] == m) & ret_calls["item_id"].isin(ret_items)]
        months = asked["item_id"].map({i: pd.Period(it.payload["month"], "M") for i, it in ret_items.items()})
        rate = lambda mask: float(asked[mask]["parsed"].map(lambda p: json.loads(p).get("value") is not None).mean()) \
            if mask.any() else float("nan")
        res["answer_rates"].append({"model": m, "before": rate(months <= last), "after": rate(months >= first)})
        sc = analysis.item_scores(ret_calls[ret_calls["model"] == m], ret_items, return_m, cut)
        sc["month"] = pd.PeriodIndex(pd.to_datetime(sc["date"]), freq="M")
        hit_b, hit_a = sc[sc["month"] <= last]["score"].mean(), sc[sc["month"] >= first]["score"].mean()
        if m in NO_AFTER:
            bef = df[df["month"] <= last]
            r_lo, r_hi = recall_interval(df, last)
            res["ret"].append({"model": m, "r_before": market.corr(bef["answer"], bef["truth"]), "n_before": len(bef),
                               "r_before_lo": r_lo, "r_before_hi": r_hi, "hit_before": hit_b,
                               "note": "no AFTER window"})
        else:
            r = corr_drop(df, last, first)
            bef, aft = df[df["month"] <= last], df[df["month"] >= first]
            res["ret"].append({"model": m, **r, "hit_before": hit_b, "hit_after": hit_a,
                               "spearman_before": bef["answer"].corr(bef["truth"], method="spearman"),
                               "spearman_after": aft["answer"].corr(aft["truth"], method="spearman")})
        lv = analysis.item_scores(lvl_calls[lvl_calls["model"] == m], lvl_items, level_m, cut)
        lv["month"] = pd.PeriodIndex(pd.to_datetime(lv["date"]), freq="M")
        lv["symbol"] = lv["item_id"].map(lambda i: lvl_items[i].payload["symbol"])
        lv["d"] = lv["score"] - lv["item_id"].map(blind)
        res["levels"][m] = (lv, last)
        bef, aft = lv[lv["month"] <= last].dropna(subset=["d"]), lv[lv["month"] >= first].dropna(subset=["d"])
        mean, lo, hi, p = _mean_test(bef["d"], bef["month"].astype(str))
        res["lvl"].append({"model": m, "n_before": len(bef), "model_before": bef["score"].mean(),
                           "blind_before": bef["score"].mean() - bef["d"].mean(), "d_before": mean, "lo": lo, "hi": hi,
                           "p": p, "d_after": aft["d"].mean() if len(aft) else float("nan"),
                           "within5_before": bef["within_5pct"].mean(), "within5_after": aft["within_5pct"].mean()
                           if len(aft) else float("nan"),
                           "within2_before": bef["within_2pct"].mean(), "no_after": m in NO_AFTER})
    for key in ("ret", "lvl"):
        tested = [r for r in res[key] if "p" in r and not (key == "lvl" and r["no_after"])]
        for r, p in zip(tested, stats.holm([r["p"] for r in tested])):
            r["p_holm"] = float(p)
    for r in res["ret"]:
        if "p_holm" in r:
            r["reading"] = ("memory detected" if r["p_holm"] < 0.05 and r["diff"] > 0 else "none detected")
    for r in res["lvl"]:
        if "p_holm" in r:
            r["reading"] = ("beats the blind guesser" if r["p_holm"] < 0.05 and r["d_before"] > 0
                            else "below the blind guesser" if r["p_holm"] < 0.05 else "no difference")
    return res


def us_move_by_month() -> pd.Series:
    """The month's US move: the mean true monthly change of the four US indices (US plan §6.1)."""
    rows = [(pd.Period(it.payload["month"], "M"), it.truth["return_pct"]) for it in load_testset("us_index_return", "v1")]
    return pd.DataFrame(rows, columns=["month", "r"]).groupby("month")["r"].mean()


def compute(models: list[str] = MODELS, release: dict = RELEASE) -> dict:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    nordic = _market(calls, meta, models, "nordic", release)
    res = {"m15": nordic["ret"], "m14": nordic["lvl"], "answer_rates": nordic["answer_rates"]}
    if not (calls["testset"] == "us_index_return/v1").any():
        return res
    us = _market(calls, meta, models, "us", release)
    res.update({"m17": us["ret"], "m16": us["lvl"], "us_answer_rates": us["answer_rates"]})
    move = us_move_by_month()
    nordic_move = pd.DataFrame([(pd.Period(it.payload["month"], "M"), it.truth["return_pct"])
                                for it in load_testset("index_return", "v1")[PILOT:]], columns=["month", "r"]
                               ).groupby("month")["r"].mean()
    shared = nordic_move.index.intersection(move.index)
    res["truth_corr"] = {"r": market.corr(nordic_move[shared], move[shared]), "n_months": len(shared)}
    res["us_vs_nordic"], res["e4"], res["e1"], res["e2"] = [], [], [], []
    for m in models:
        dn, last, first = nordic["frames"][m]
        du = us["frames"][m][0]
        nb, ub = dn[dn["month"] <= last], du[du["month"] <= last]
        res["us_vs_nordic"].append({"model": m, **paired_corr_diff(ub, nb)})
        na = dn[dn["month"] >= first]
        res["e4"].append({"model": m, "r_nordic_before": market.corr(nb["answer"], nb["truth"]),
                          **partial_test(nb, move),
                          "r_after": partial_corr(na, move) if m not in NO_AFTER and len(na) else float("nan")})
        for label, d in (("Nordic", dn), ("US", du)):
            res["e1"].append({"model": m, "market": label, "before": answer_style(d[d["month"] <= last]),
                              "after": answer_style(d[d["month"] >= first]) if m not in NO_AFTER else None})
        for label, (lv, lv_last) in (("Nordic", nordic["levels"][m]), ("US", us["levels"][m])):
            bef = lv[lv["month"] <= lv_last]
            for sym, g in bef.groupby("symbol"):
                res["e2"].append({"model": m, "market": label, "symbol": sym, "n": len(g),
                                  "closeness": g["score"].mean(), "within5": g["within_5pct"].mean(),
                                  "answered": 1 - g["refused"].mean()})
    for key in ("us_vs_nordic", "e4"):
        for r, p in zip(res[key], stats.holm([r["p"] for r in res[key]])):
            r["p_holm"] = float(p)
    for r in res["us_vs_nordic"]:
        r["reading"] = ("knows the US market better" if r["p_holm"] < 0.05 and r["diff"] > 0
                        else "knows the Nordic market better" if r["p_holm"] < 0.05 else "no clear difference")
    for r in res["e4"]:
        r["reading"] = ("Nordic-specific memory" if r["p_holm"] < 0.05 and r["r"] > 0
                        else "no Nordic-specific signal")
    return res


def _change_table(rows: list[dict]) -> str:
    return _table(["model", "r before", "r after", "difference [95% CI]", "p (Holm)", "direction right b/a",
                   "Spearman b/a", "n b/a", "reading"],
                  [[short(r["model"]), _f(r["r_before"], "{:+.2f}"), _f(r.get("r_after"), "{:+.2f}"),
                    f"{_f(r.get('diff'), '{:+.2f}')} [{_f(r.get('lo'), '{:+.2f}')}, {_f(r.get('hi'), '{:+.2f}')}]",
                    _f(r.get("p_holm"), "{:.4f}"),
                    f"{_f(r['hit_before'], '{:.0%}')} / {_f(r.get('hit_after'), '{:.0%}')}",
                    f"{_f(r.get('spearman_before'), '{:+.2f}')} / {_f(r.get('spearman_after'), '{:+.2f}')}",
                    f"{r['n_before']} / {r.get('n_after', '–')}", r.get("reading", r.get("note", ""))]
                   for r in rows])


def _level_table(rows: list[dict]) -> str:
    return _table(["model", "closeness model / blind (before)", "difference [95% CI]", "p (Holm)",
                   "within ±5% before / after", "within ±2% before", "difference after (secondary)", "reading"],
                  [[short(r["model"]), f"{r['model_before']:.3f} / {r['blind_before']:.3f}",
                    f"{r['d_before']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]", _f(r.get("p_holm"), "{:.4f}"),
                    f"{_f(r['within5_before'], '{:.0%}')} / {_f(r['within5_after'], '{:.0%}')}",
                    _f(r["within2_before"], "{:.0%}"), _f(r["d_after"], "{:+.3f}"),
                    r.get("reading", "descriptive (no AFTER window)")] for r in rows])


def _rates(rows: list[dict]) -> str:
    return ", ".join(f"{short(r['model'])} {_f(r['before'], '{:.0%}')} / {_f(r['after'], '{:.0%}')}" for r in rows)


def _style(s: dict | None) -> str:
    if not s or not s["n"]:
        return "–"
    tops = ", ".join(f"{v:+g} ({share:.0%})" for v, share in s["top"])
    return f"{s['distinct']} distinct of {s['n']}; most repeated: {tops}"


def render(res: dict) -> str:
    L = ["# M14/M15 market-level memory (plan §6.3, revised confirmatory test)\n",
         "Four Nordic indices × months 2016-01..2026-08, the 24 micro-pilot questions left out. BEFORE: every month up to "
         "the last whole month before cutoff − 91 days; AFTER: every month after the release month. Holm across the "
         "five models with an AFTER window, separately per method.\n",
         "## M15: do the monthly-change answers track the market? (correlation before − after)\n"]
    L.append(_change_table(res["m15"]))
    L.append("\nAnswer rates (M15), before / after: " + _rates(res["answer_rates"]))
    L.append("\n## M14: on months it may have seen, does the model beat the blind guesser? (closeness, model − blind)\n")
    L.append(_level_table(res["m14"]))
    if "m17" not in res:
        return "\n".join(L) + "\n"
    L += ["\n# M16/M17 US indices and the US − Nordic comparison (US plan §4–§6; PRIVATE, Yahoo data)\n",
          "S&P 500, Dow Jones, Nasdaq Composite, Nasdaq-100 × months 2016-01..2026-08; the same questions, scoring and "
          "windows as M14/M15. Holm across the five models with an AFTER window, separately per method.\n",
          "## M17: do the US monthly-change answers track the market?\n", _change_table(res["m17"]),
          "\nAnswer rates (M17), before / after: " + _rates(res["us_answer_rates"]),
          "\n## M16: on months it may have seen, does the model beat the blind guesser? (US levels)\n",
          _level_table(res["m16"]),
          "\n## US vs Nordic: r before (US) − r before (Nordic), paired by month (Holm across six)\n",
          _table(["model", "r US", "r Nordic", "difference [95% CI]", "p (Holm)", "months", "reading"],
                 [[short(r["model"]), f"{r['r_us']:+.2f}", f"{r['r_nordic']:+.2f}",
                   f"{r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]", f"{r['p_holm']:.4f}", r["n_months"],
                   r["reading"]] for r in res["us_vs_nordic"]]),
          f"\nThe true monthly changes of the two markets (mean of each market's indices) correlate "
          f"{res['truth_corr']['r']:+.2f} over {res['truth_corr']['n_months']} months.\n",
          "## E4 (exploratory, §6.1): Nordic beyond US, the partial correlation controlling for the month's US move\n",
          _table(["model", "Nordic r before", "beyond US [95% CI]", "p (Holm)", "after (descriptive)", "n", "reading"],
                 [[short(r["model"]), f"{r['r_nordic_before']:+.2f}", f"{r['r']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]",
                   f"{r['p_holm']:.4f}", _f(r["r_after"], "{:+.2f}"), r["n"], r["reading"]] for r in res["e4"]]),
          "\n## E1 (exploratory): answer style on the monthly-change questions\n",
          _table(["model", "market", "before the cutoff", "after the release"],
                 [[short(r["model"]), r["market"], _style(r["before"]), _style(r["after"])] for r in res["e1"]]),
          "\n## E2 (exploratory): level closeness per index, BEFORE months (model − blind is not comparable across "
          "markets)\n",
          _table(["model", "market", "index", "closeness", "within ±5%", "answered", "n"],
                 [[short(r["model"]), r["market"], r["symbol"], f"{r['closeness']:.2f}", f"{r['within5']:.0%}",
                   f"{r['answered']:.0%}", r["n"]] for r in res["e2"]])]
    return "\n".join(L) + "\n"


def main() -> None:
    res = _jsonable(compute())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT / "report.md").write_text(render(res), encoding="utf-8")
    print(f"wrote {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
