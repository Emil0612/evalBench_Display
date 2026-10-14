"""M12/M13 report: every analysis registered in docs/preregistration/2026-10-01-price-choice.md, from the stored
answers only (no model calls, no cost). Stage 1 (§3): is the multiple-choice format sensitive (M13), next to the same
model's recall on the same M9 questions. Stage 2 (§4, §5): the confirmatory before/after test on M12 and the
exploratory E2-E5. Writes results/choice/stage1.md and stage2.md.

uv run python -m evalbench.choice_report --stage 1
uv run python -m evalbench.choice_report --stage 2
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from . import analysis, pilot, stats, store
from .cutoffs import resolve_cutoff
from .grid_report import windows
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods.price_choice import P0
from .pilot_d_report import MODELS, RELEASE
from .pilot_e import refusal_affected, with_score
from .pilot_report import _f, _jsonable, _table, short

OUT = ROOT / "results" / "choice"
BLIND = "baseline/blind@2024-06-30"
B = 20000


def sensitive(lo: float, blind: float) -> bool:
    """§3: the lower 95% bound of the mean M13 score is above both chance (0.25) and the blind guesser's mean."""
    return bool(lo > max(P0, blind))


def _scores(calls: pd.DataFrame, model: str, method: str, version: str, meta: dict) -> pd.DataFrame:
    items = {it.item_id: it for it in load_testset(method, version)}
    sub = calls[(calls["model"] == model) & (calls["testset"] == f"{method}/{version}")]
    return analysis.item_scores(sub, items, method, resolve_cutoff(model, meta))


def stage1(models: list[str] = MODELS) -> dict:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    source = {it.item_id: it.truth["source_item"] for it in load_testset("known_choice", "v1")}
    blind = _scores(calls, BLIND, "known_choice", "v1", meta)
    blind_by = dict(zip(blind["item_id"], blind["score"]))
    rows = []
    for m in models:
        mc = _scores(calls, m, "known_choice", "v1", meta)
        if mc.empty:
            continue
        lo, hi = stats.bootstrap_mean(mc["score"], mc["cluster"], B=B)
        b = float(np.mean([blind_by[i] for i in mc["item_id"] if i in blind_by]))
        recall = _scores(calls, m, "known_level", "v1", meta)
        recall = recall[recall["item_id"].isin({source[i] for i in mc["item_id"]})]
        rows.append({
            "model": m, "n": len(mc), "mean": float(mc["score"].mean()), "lo": lo, "hi": hi, "blind": b,
            "sensitive": sensitive(lo, b), "known_3of4": float(mc["known"].mean()),
            "one_letter": float(mc["one_letter"].mean()), "consistent": float(mc["consistent"].mean()),
            "refused": float(mc["refused"].mean()),
            "recall_n": len(recall),
            "recall_closeness": float(pd.to_numeric(recall.get("closeness"), errors="coerce").fillna(0).mean())
            if len(recall) else float("nan"),
            "recall_in_range": float(recall["in_range"].fillna(False).astype(bool).mean())
            if "in_range" in recall and len(recall) else float("nan"),
            "recall_within_10": float(recall["score"].mean()) if len(recall) else float("nan")})
    return {"set": "known_choice/v1", "blind_mean": float(blind["score"].mean()) if len(blind) else None, "rows": rows}


def render_stage1(res: dict) -> str:
    L = ["# M13 stage 1: is the multiple-choice format sensitive? (plan §3)\n",
         f"Set: {res['set']} (the 15 largest Nordic companies, 2016–2022, before every cutoff). Score: the share of the "
         "four orderings answered right (chance 0.25). Sensitive = the lower 95% bound (cluster bootstrap by company) "
         "above both chance and the blind guesser on the same questions. Recall = the same model on the same "
         "company-months in M9 (known_level/v1).\n"]
    L.append(_table(["model", "questions", "mean [95% CI]", "blind", "sensitive", "recognised (3+ of 4)",
                     "same letter ×4", "same price ×4", "recall: closeness", "recall: within ±10%",
                     "recall: price that occurred"],
                    [[short(r["model"]), r["n"], f"{r['mean']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]",
                      _f(r["blind"], "{:.3f}"), "yes" if r["sensitive"] else "no",
                      _f(r["known_3of4"], "{:.0%}"), _f(r["one_letter"], "{:.0%}"), _f(r["consistent"], "{:.0%}"),
                      _f(r["recall_closeness"], "{:.3f}"), _f(r["recall_within_10"], "{:.0%}"),
                      _f(r["recall_in_range"], "{:.0%}")] for r in res["rows"]]))
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- stage 2 (§4, §5)
STAGE2_MODELS = [m for m in MODELS if m != "qwen/qwen3.8-flash"]   # deviation 1: no AFTER window


def clean_after(item, release) -> bool:
    """E4: every wrong option is from a month after the model's release month (nothing to eliminate by memory)."""
    first = pd.Period(release, "M") + 1
    return all(pd.Period(m, "M") >= first for m in item.truth["decoy_months"])


def _grouped(df: pd.DataFrame, before, after) -> pd.DataFrame:
    month = pd.PeriodIndex(pd.to_datetime(df["date"]), freq="M")
    return df.assign(month=month, group=np.where(month.isin(before), 1, np.where(month.isin(after), 0, -1)))


def stage2(models: list[str] = STAGE2_MODELS) -> dict:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    items = {it.item_id: it for it in load_testset("price_choice", "v1")}
    s1 = {r["model"]: r for r in stage1(models)["rows"]}
    res = {"set": "price_choice/v1", "h": [], "e2": [], "e3": [], "e4": [], "e5": []}
    for m in models:
        cut, rel = resolve_cutoff(m, meta), RELEASE[m]
        before, after = windows(cut, rel)
        df = _scores(calls, m, "price_choice", "v1", meta)
        if df.empty or after is None:
            continue
        df = _grouped(df, before, after)
        reg = analysis.summarize(df, "price_choice", B=B)
        ans = analysis.summarize(df[~df["refused"].astype(bool)], "price_choice", B=B)
        row = {"model": m, "before": f"{before[0]}..{before[-1]}", "after": f"{after[0]}..{after[-1]}",
               **{k: reg[k] for k in ("n_pre", "n_post", "mean_pre", "mean_post", "dkappa", "ci_lo", "ci_hi",
                                      "p_value", "coverage_pre", "coverage_post", "malformed_rate")},
               "answered_dkappa": ans["dkappa"], "answered_p": ans["p_value"],
               "refusal_affected": refusal_affected(reg["coverage_pre"], reg["coverage_post"]),
               "sensitive": bool(s1.get(m, {}).get("sensitive"))}
        res["h"].append(row)
        # E2: recall on the same company-months (M5 v3), same windows
        recall = with_score(_scores(calls, m, "price_level", "v3", meta), "price_level", "primary")
        src = {it.truth["source_item"]: iid for iid, it in items.items()}
        recall = _grouped(recall[recall["item_id"].isin(src)], before, after)
        hit = _scores(calls, m, "price_level", "v3", meta)
        hit = dict(zip(hit["item_id"], hit["score"]))
        recall["hit10"] = recall["item_id"].map(hit)
        for side, g in ((1, "before"), (0, "after")):
            mc, rc = df[df["group"] == side], recall[recall["group"] == side]
            res["e2"].append({"model": m, "side": g, "recognised_3of4": float(mc["known"].mean()),
                              "mc_score": float(mc["score"].mean()), "recall_closeness": float(rc["score"].mean()),
                              "recall_within_10": float(rc["hit10"].mean())})
        # E3: by company size
        for size in ("large", "mid", "small"):
            part = df[df.get("s_size") == size]
            res["e3"].append({"model": m, "size": size,
                              "before": float(part[part["group"] == 1]["score"].mean()),
                              "after": float(part[part["group"] == 0]["score"].mean()),
                              "n_before": int((part["group"] == 1).sum()), "n_after": int((part["group"] == 0).sum())})
        # E4: AFTER on questions whose wrong options are also all after the release
        clean = df[(df["group"] == 0) & df["item_id"].map(lambda i: clean_after(items[i], rel))]
        res["e4"].append({"model": m, "n_clean": len(clean), "after_clean": float(clean["score"].mean())
                          if len(clean) >= 20 else float("nan"), "after_all": float(df[df["group"] == 0]["score"].mean())})
        # E5: habits
        res["e5"].append({"model": m, "one_letter": float(df["one_letter"].mean()),
                          "consistent": float(df["consistent"].mean()), "refused": float(df["refused"].mean())})
    tested = res["h"]
    for key, out in (("p_value", "p_holm"), ("answered_p", "answered_p_holm")):
        for r, p in zip(tested, stats.holm([r[key] for r in tested])):
            r[out] = float(p)
    for r in tested:
        reading = pilot.primary_reading(r)
        if reading.startswith("memorization detected") and r["refusal_affected"] and \
                (r["answered_p_holm"] >= 0.05 or r["answered_dkappa"] <= 0):
            reading = "not confirmed: refusal-driven (" + reading + ")"
        if reading.startswith("none detected") and not r["sensitive"]:
            reading += " · too hard for this model (not sensitive in stage 1)"
        r["reading"] = reading
    return res


def render_stage2(res: dict) -> str:
    L = ["# M12 stage 2: before vs after release (plan §4) and exploratory E2–E5 (§5)\n",
         f"Set: {res['set']}. Score: the share of the four orderings answered right (chance 0.25); Δκ = (before − after) "
         "/ 0.75. BEFORE: the 12 months ending 3 months before the stated cutoff; AFTER: the first 12 months after "
         "release. Holm across the five models; refusal guard when answer rates differ by more than 10 points.\n",
         "## Confirmatory test\n"]
    L.append(_table(["model", "BEFORE", "AFTER", "score before → after", "Δκ", "95% CI", "p (Holm)", "answered b/a",
                     "stage 1", "reading"],
                    [[short(r["model"]), r["before"], r["after"], f"{r['mean_pre']:.3f} → {r['mean_post']:.3f}",
                      _f(r["dkappa"]), f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}]", _f(r["p_holm"], "{:.4f}"),
                      f"{r['coverage_pre']:.0%} / {r['coverage_post']:.0%}" + (" ⚠" if r["refusal_affected"] else ""),
                      "sensitive" if r["sensitive"] else "not sensitive", r["reading"]] for r in res["h"]]))
    L.append("\n## E2: recognition (M12) vs recall (M5 v3) on the same company-months\n")
    L.append(_table(["model", "side", "M12 score", "recognised (3+ of 4)", "M5 closeness", "M5 within ±10%"],
                    [[short(r["model"]), r["side"], f"{r['mc_score']:.3f}", f"{r['recognised_3of4']:.0%}",
                      f"{r['recall_closeness']:.3f}", f"{r['recall_within_10']:.0%}"] for r in res["e2"]]))
    L.append("\n## E3: by company size (M12 score before → after; n)\n")
    L.append(_table(["model", "size", "before → after", "n"],
                    [[short(r["model"]), r["size"], f"{_f(r['before'], '{:.3f}')} → {_f(r['after'], '{:.3f}')}",
                      f"{r['n_before']} / {r['n_after']}"] for r in res["e3"]]))
    L.append("\n## E4: elimination check (AFTER, all four options from after the release)\n")
    L.append(_table(["model", "AFTER, all questions", "AFTER, clean questions", "n clean"],
                    [[short(r["model"]), f"{r['after_all']:.3f}", _f(r["after_clean"], "{:.3f}"), r["n_clean"]]
                     for r in res["e4"]]))
    L.append("\n## E5: habits\n")
    L.append(_table(["model", "same letter ×4", "same price ×4", "refused"],
                    [[short(r["model"]), f"{r['one_letter']:.0%}", f"{r['consistent']:.0%}", f"{r['refused']:.0%}"]
                     for r in res["e5"]]))
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 1:
        res, text, name = _jsonable(stage1()), None, "stage1"
        text = render_stage1(res)
    else:
        res = _jsonable(stage2())
        text, name = render_stage2(res), "stage2"
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    (OUT / f"{name}.md").write_text(text, encoding="utf-8")
    print(f"wrote {OUT / (name + '.md')}")


if __name__ == "__main__":
    main()
