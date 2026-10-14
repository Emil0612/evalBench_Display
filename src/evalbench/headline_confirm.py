"""M18 confirmation: is news-dating memory about the world rather than companies?
Registered in docs/preregistration/2026-10-06-headline-macro.md before any answer.

The set (headline_date/v2): fresh headlines, none used in v1, 6 about a listed company and 6 about none in every month
2016-01..2026-07, with the same filters, prompt, scoring and text-only baseline as v1. The test, per model: on the
months before its cutoff (the registered M18 window), the no-company headlines minus the company headlines, on
(C1) model minus text-only score and (C2) exact-month hits minus the model's own share of answers naming that month,
the months next to its habitual answer left out. Months are resampled; Holm across the five models, separately for
C1 and C2. No model calls here.

uv run python -X utf8 -m evalbench.headline_confirm build    # builds testsets/headline_date/v2 (free)
uv run python -X utf8 -m evalbench.headline_confirm          # the registered report -> results/headline/confirm.md
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import dating_horizon, headline_report, stats, store
from .cutoffs import resolve_cutoff
from .data import DATA_DIR, company_size_map
from .items import ROOT, Item, load_testset, save_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods import headline_date as hd
from .pilot_report import _f, _table, short

NAME, VERSION = hd.NAME, "v2"
TESTSET = f"{NAME}/{VERSION}"
SEED = 20261006
PER_GROUP = 6
MODELS = ["openai/gpt-3.5-turbo", "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.3-70b-instruct",
          "google/gemini-2.5-flash", "deepseek/deepseek-chat-v3.1"]
GROUPS = ("none", "listed")
OUT = ROOT / "results" / "headline"


def load_pool(path=hd.NEWS_FILE) -> pd.DataFrame:
    """The eligible headlines (v1's filters) with the listed companies each names."""
    news = pd.read_parquet(path, columns=["newsid", "date", "language", "newstype", "distributor", "headline",
                                          "same_text_count", "insrefs"])
    news["date"] = pd.to_datetime(news["date"])
    return hd.eligible(news)


def _group(refs) -> str:
    return "none" if refs is None or len(refs) == 0 else "listed"


def build_items(pool: pd.DataFrame, seed: int, exclude: set, months=hd.MONTHS, per_group: int = PER_GROUP) -> list[Item]:
    """Per month, `per_group` headlines about a listed company and `per_group` about none, none of them in `exclude`
    (v1's headlines); each with the text-only baseline's month, trained as in v1 on headlines not picked."""
    rng = np.random.default_rng(seed)
    pool = pool[~pool["newsid"].isin(exclude)].assign(group=lambda d: d["insrefs"].map(_group))
    picked = {}
    for m in months:
        parts = []
        for g in GROUPS:
            cand = pool[(pool["month"] == m) & (pool["group"] == g)]
            if len(cand) < per_group:
                parts = []
                break
            parts.append(cand.iloc[rng.permutation(len(cand))[:per_group]])
        if parts:
            picked[m] = pd.concat(parts)
    used = set(pd.concat(picked.values())["newsid"]) if picked else set()
    rest = pool[~pool["newsid"].isin(used)]
    train = pd.concat([g.iloc[rng.permutation(len(g))[:hd.TRAIN_PER_MONTH]] for _, g in rest.groupby("month")])
    items = []
    for m, df in picked.items():
        blind = hd.blind_predict(train, m, df["headline"].tolist())
        for (_, r), b in zip(df.iterrows(), blind):
            items.append(Item(
                item_id=f"hd2-{len(items):04d}", method=NAME, date=r["date"].date().isoformat(), cluster=f"m{m}",
                payload={"headline": r["headline"], "language": r["language"]},
                truth={"month": str(m), "blind_month": str(b), "newsid": r["newsid"], "newstype": int(r["newstype"]),
                       "distributor": r["distributor"], "company": r["group"]},
                strata={"language": r["language"], "newstype": hd.NEWSTYPES[int(r["newstype"])], "company": r["group"]}))
    return [items[i] for i in rng.permutation(len(items))]


def build() -> None:
    v1 = {it.truth["newsid"] for it in load_testset(NAME, "v1")}
    its = build_items(load_pool(), SEED, v1)
    path = save_testset(its, NAME, VERSION, {
        "seed": SEED, "prompt_version": hd.PROMPT_VERSION, "filters": hd.FILTERS + " Confirmation set: no v1 headline; "
        f"{PER_GROUP} about a listed company and {PER_GROUP} about none per month (news_index insrefs).",
        **hd.manifest_extra()})
    print(f"{len(its)} items -> {path}")


def contrast(df: pd.DataFrame, col: str, B: int = 20000, seed: int = 0) -> dict:
    """mean(col | no company) - mean(col | listed), resampling months (df: month, group, col)."""
    d = df.dropna(subset=[col])
    counts_by = d["group"].value_counts()
    if not all(counts_by.get(g, 0) for g in GROUPS):        # a group with no rows: no estimate, not a crash
        nan = float("nan")
        return {"d": nan, "lo": nan, "hi": nan, "p": nan, "n_none": int(counts_by.get("none", 0)),
                "n_listed": int(counts_by.get("listed", 0))}
    piv = d.groupby(["month", "group"])[col].agg(["sum", "size"]).unstack(fill_value=0)
    s_none, n_none = piv[("sum", "none")].to_numpy(float), piv[("size", "none")].to_numpy(float)
    s_lst, n_lst = piv[("sum", "listed")].to_numpy(float), piv[("size", "listed")].to_numpy(float)
    point = s_none.sum() / n_none.sum() - s_lst.sum() / n_lst.sum()
    counts = stats._resample_counts(len(piv), B, seed)
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = counts @ s_none / (counts @ n_none) - counts @ s_lst / (counts @ n_lst)
    draws = draws[~np.isnan(draws)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"d": float(point), "lo": float(lo), "hi": float(hi), "p": stats.boot_p(draws),
            "n_none": int(n_none.sum()), "n_listed": int(n_lst.sum())}


def exact_excess(answer, true: pd.Period, refused: bool, share: dict, habit) -> float:
    """C2 for one headline: 1 if the answer names the true month, minus the model's own share of answers naming that
    month. Left out (NaN) for a refused or unanswered headline (no month was named: plan §6, 2026-10-06) and for
    the months next to the model's habitual answer."""
    if refused or answer is None or pd.isna(answer):
        return float("nan")
    t = true.ordinal
    if habit is not None and abs(t - habit) <= dating_horizon.HABIT_GAP:
        return float("nan")
    return float(answer.ordinal == t) - share.get(t, 0.0)


def model_frame(calls, model: str, meta: dict, items: dict) -> pd.DataFrame:
    """One row per item on the months before the cutoff: month, group, C1 (score - text-only), C2 (exact-month hit
    minus the model's own answer share, NaN near its habitual month), refused."""
    cut = resolve_cutoff(model, meta)
    df = headline_report.scored(calls, model, cut, items, testset=TESTSET)
    if df.empty:
        return df
    f = dating_horizon.frame(calls, model, cut, items, testset=TESTSET).set_index("item_id")
    answered = f[~f["refused"].astype(bool)].dropna(subset=["answer"])
    share = pd.Series([p.ordinal for p in answered["answer"]]).value_counts(normalize=True).to_dict()
    habit = pd.Series([p.ordinal for p in answered["answer"]]).mode().iloc[0] if len(answered) else None
    c2 = {i: exact_excess(r["answer"], r["true"], bool(r["refused"]), share, habit) for i, r in f.iterrows()}
    df = df[df["side"] == 1]
    return pd.DataFrame({"item_id": df["item_id"], "month": df["month"].astype(str),
                         "group": df["item_id"].map(lambda i: items[i].truth["company"]),
                         "c1": (df["score"] - df["blind_score"]).to_numpy(), "c2": df["item_id"].map(c2),
                         "refused": df["refused"].astype(bool).to_numpy()})


def read(rows: list[dict]) -> list[dict]:
    """Holm across the five models the plan names (a model without answers counts with p = 1 and is not confirmed),
    separately for C1, C2 and the answered-only C1; then each model's reading."""
    for key in ("c1_p", "c2_p", "c1_answered_p"):
        ps = [r.get(key, 1.0) if r.get("n") else 1.0 for r in rows]
        ps = [1.0 if p is None or pd.isna(p) else p for p in ps]
        for r, p in zip(rows, stats.holm(ps)):
            r[key + "_holm"] = float(p)
    for r in rows:
        r["confirmed"] = bool(r.get("n") and r["c1_d"] > 0 and r["c1_p_holm"] < 0.05 and r["c2_d"] > 0
                              and r["c2_p_holm"] < 0.05 and r["c1_answered"] > 0 and r["c1_answered_p_holm"] < 0.05)
    return rows


def verdict(rows: list[dict]) -> str:
    k = sum(r["confirmed"] for r in rows)
    return f"{'confirmed' if k >= 3 else 'not confirmed'} ({k} of {len(rows)} models; the plan needs 3)"


def compute(B: int = 20000) -> list[dict]:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    its = load_testset(NAME, VERSION)
    items = {it.item_id: it for it in its}
    news = pd.read_parquet(DATA_DIR / "news_index.parquet", columns=["newsid", "insrefs"],
                           filters=[("newsid", "in", sorted({it.truth["newsid"] for it in its}))])
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    sizes = hd.company_sizes(its, dict(zip(news["newsid"], news["insrefs"])),
                             dict(zip(uni["insref"], uni["company_key"])), company_size_map(uni))
    rows = []
    for m in MODELS:
        df = model_frame(calls, m, meta, items)
        if df.empty:
            rows.append({"model": m, "n": 0})
            continue
        ans = df[~df["refused"]]
        by_size = {s: float(df.loc[(df["group"] == "listed") & (df["item_id"].map(sizes) == s), "c1"].mean())
                   for s in ("large", "mid", "small")}
        rows.append({"model": m, "n": len(df), "answer_rate": float(1 - df["refused"].mean()),
                     "answer_none": float(1 - df.loc[df["group"] == "none", "refused"].mean()),
                     "answer_listed": float(1 - df.loc[df["group"] == "listed", "refused"].mean()),
                     **{f"size_{k}": v for k, v in by_size.items()},
                     **{f"c1_{k}": v for k, v in contrast(df, "c1", B).items()},
                     **{f"c2_{k}": v for k, v in contrast(df, "c2", B).items()},
                     "c1_answered": contrast(ans, "c1", B)["d"], "c1_answered_p": contrast(ans, "c1", B)["p"],
                     "none_d": float(df.loc[df["group"] == "none", "c1"].mean()),
                     "listed_d": float(df.loc[df["group"] == "listed", "c1"].mean())})
    for r in rows:
        r["B"] = B
    return read(rows)


def render(rows: list[dict]) -> str:
    done = [r for r in rows if r["n"]]
    B = rows[0].get("B", 20000) if rows else 20000
    L = ["# M18 confirmation: memory of world news, not company news\n",
         "Registered in `docs/preregistration/2026-10-06-headline-macro.md` before any answer. Set "
         f"`{TESTSET}`: fresh headlines, {PER_GROUP} about a listed company and {PER_GROUP} about none per month. "
         "Per model, on the months before its cutoff: no-company minus company headlines on C1 (model minus "
         "text-only score) and C2 (exact-month hits minus the model's own answer share; months next to its habit "
         f"left out; refused headlines left out of C2). Months resampled ({B:,} draws); Holm across the five models. "
         "Confirmed: C1 and C2 both positive and significant, and C1 also on the answered headlines alone.\n",
         f"**Overall: {verdict(rows)}.**\n",
         _table(["model", "headlines (none / listed)", "answered", "no company: model − text", "company: model − text",
                 "C1 contrast [95%]", "p (Holm)", "C2 contrast [95%]", "p (Holm)", "C1 answered only", "confirmed"],
                [[short(r["model"]), f"{r['c1_n_none']} / {r['c1_n_listed']}", _f(r["answer_rate"], "{:.0%}"),
                  _f(r["none_d"], "{:+.3f}"), _f(r["listed_d"], "{:+.3f}"),
                  f"{_f(r['c1_d'], '{:+.3f}')} [{_f(r['c1_lo'], '{:+.3f}')}, {_f(r['c1_hi'], '{:+.3f}')}]",
                  _f(r["c1_p_holm"], "{:.3f}"),
                  f"{_f(r['c2_d'], '{:+.3f}')} [{_f(r['c2_lo'], '{:+.3f}')}, {_f(r['c2_hi'], '{:+.3f}')}]",
                  _f(r["c2_p_holm"], "{:.3f}"),
                  f"{_f(r['c1_answered'], '{:+.3f}')} (p {_f(r['c1_answered_p_holm'], '{:.3f}')})",
                  "yes" if r["confirmed"] else "no"] for r in done]),
         "\n## Reported alongside (plan §4)\n",
         _table(["model", "answered: no company / listed", "company headlines by size: model − text (large / mid / small)"],
                [[short(r["model"]), f"{_f(r['answer_none'], '{:.0%}')} / {_f(r['answer_listed'], '{:.0%}')}",
                  " / ".join(_f(r[f"size_{s}"], "{:+.3f}") for s in ("large", "mid", "small"))] for r in done])]
    missing = [short(r["model"]) for r in rows if not r["n"]]
    if missing:
        L.append(f"\nNo answers yet: {', '.join(missing)}.")
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["build"]:
        build()
        return
    text = render(compute())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "confirm.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
