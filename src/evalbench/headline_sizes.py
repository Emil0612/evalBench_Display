"""M18 by company size (exploratory, 2026-10-06): the registered H1 comparison (model minus the text-only baseline on
the months before the cutoff) within each company-size group, a headline taking the size of the listed company it is
about; headlines about no listed company are their own group. Holm across every model x group with at least 20
headlines. Writes results/headline/sizes.md; no model calls, no cost.

uv run python -X utf8 -m evalbench.headline_sizes
"""
from __future__ import annotations

import pandas as pd

from . import headline_report as hr
from . import stats, store
from .cutoffs import resolve_cutoff
from .data import DATA_DIR, company_size_map
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods import headline_date as hd
from .pilot_report import _f, _table, short

OUT = ROOT / "results" / "headline"
GROUPS = ("large", "mid", "small", "none")
LABEL = {"large": "large caps", "mid": "mid caps", "small": "small caps", "none": "no listed company"}


def compute() -> list[dict]:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    its = load_testset(hd.NAME, "v1")
    items = {it.item_id: it for it in its}
    news = pd.read_parquet(DATA_DIR / "news_index.parquet", columns=["newsid", "insrefs"],
                           filters=[("newsid", "in", sorted({it.truth["newsid"] for it in its}))])
    uni = pd.read_parquet(DATA_DIR / "universe.parquet")
    sizes = hd.company_sizes(its, dict(zip(news["newsid"], news["insrefs"])),
                             dict(zip(uni["insref"], uni["company_key"])), company_size_map(uni))
    rows = []
    for m in hr.MODELS:
        df = hr.main_only(hr.scored(calls, m, resolve_cutoff(m, meta), items))
        if df.empty:
            continue
        df = df.assign(size=df["item_id"].map(sizes).fillna("unknown"))
        rows += [{"model": m, **r} for r in hr.size_rows(df[df["size"].isin(GROUPS)])]
    tested = [r for r in rows if r["n"] >= 20]
    for r, p in zip(tested, stats.holm([r["p"] for r in tested])):
        r["p_holm"] = float(p)
    return rows


def render(rows: list[dict]) -> str:
    L = ["# M18 by company size (exploratory)\n",
         "The registered H1 comparison within each group: on the months before the cutoff, the model's score minus the "
         "text-only baseline's on the same headlines (a refusal scores 0), resampling months. A headline takes the "
         "market-cap tercile of the listed company it is about (the largest when it names several); headlines about "
         "no listed company (macro, sector, editorial news) are their own group. Holm across every model x group "
         "with at least 20 headlines. Not registered.\n"]
    for g in GROUPS:
        part = [r for r in rows if r["size"] == g]
        L += [f"## {LABEL[g].capitalize()}\n",
              _table(["model", "headlines before the cutoff", "model / text-only score", "model − text only",
                      "95% interval", "p (Holm)"],
                     [[short(r["model"]), r["n"], f"{_f(r['model_score'], '{:.3f}')} / {_f(r['blind_score'], '{:.3f}')}",
                       _f(r["d"], "{:+.3f}"), f"[{_f(r['lo'], '{:+.3f}')}, {_f(r['hi'], '{:+.3f}')}]",
                       _f(r.get("p_holm"), "{:.3f}") if r["n"] >= 20 else "too few"] for r in part]), ""]
    return "\n".join(L)


def main() -> None:
    text = render(compute())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sizes.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
