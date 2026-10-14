"""Effective cutoffs (exploratory): where each model's knowledge actually ends, per kind of data, next to its stated
cutoff (Cheng et al. 2024, "Dated Data": effective cutoffs differ from stated ones, and by resource).

- Market moves (M15): the first half-year from which most index questions get no informative answer (a refusal, 0%,
  or the model's stock value) and stay so. A switch after the model's release is no horizon (the model just keeps
  guessing) and reads "no clear switch". A yearly correlation is too noisy to time it with 4 indices: 12 months a
  year, and calm years correlate weakly even for a model that knows them.
- News (M18): the last year dated clearly from dating_horizon (exact-month hits beyond the model's own answer
  habits). Where news knowledge stops is not measured: models almost never name later years.
- Monthly stock returns (M11): the first half-year from which most answers are exactly 0% ("I don't know").
Price levels are left out: they persist from month to month, so a model that knows earlier prices also does well on
later years without any hindsight, and cannot time where its memory ends. No model calls, no cost.

uv run python -X utf8 -m evalbench.effective_cutoffs
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis, cutoff_estimate, dating_horizon, headline_report, ladder, ladder_extend, store
from .cutoffs import resolve_cutoff, resolve_release
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .pilot_report import _table, short

OUT = ROOT / "results" / "readiness"
MODELS = headline_report.MODELS
EXTRA = ladder_extend.CANDIDATES          # answered only the index questions
NEW = cutoff_estimate.NEW                # added 2026-10-07, read like MODELS (plan 2026-10-14-zero-answer-horizon)


def half_of(month) -> str:
    m = str(month)
    return f"{m[:4]}-H{1 if int(m[5:7]) <= 6 else 2}"


def zero_switch(halves: pd.Series, threshold: float = 0.5) -> str | None:
    """The first half-year from which the share is at least `threshold`, and stays so."""
    h = halves.sort_index()
    for i, k in enumerate(h.index):
        if (h.iloc[i:] >= threshold).all():
            return str(k)
    return None


def switch_certainty(flags: pd.DataFrame, B: int = 1000, seed: int = 0) -> dict:
    """How firm a switch is (flags: one row per question asked, h = half-year, u = 1 if no informative answer): the
    share of resamples (questions redrawn within each half-year) that land on the same half-year, their 95% range,
    and the switch at thresholds of 40% and 60%."""
    rng = np.random.default_rng(seed)
    groups = {h: g["u"].to_numpy(float) for h, g in flags.groupby("h")}
    share = pd.Series({h: v.mean() for h, v in groups.items()})
    switch = zero_switch(share)
    draws = [zero_switch(pd.Series({h: v[rng.integers(0, len(v), len(v))].mean() for h, v in groups.items()}))
             for _ in range(B)]
    found = sorted(d for d in draws if d)
    return {"switch": switch, "stable": sum(d == switch for d in draws) / B,
            "range": (found[int(0.025 * len(found))], found[max(int(0.975 * len(found)) - 1, 0)]) if found else None,
            "at_40": zero_switch(share, 0.4), "at_60": zero_switch(share, 0.6)}


def uninformative_share(frame: pd.DataFrame, asked: pd.Series) -> pd.Series:
    """Per half-year: the share of the questions asked that got no informative answer: refused (not in frame), 0%,
    or the model's stock value (its most frequent answer)."""
    stock = frame["answer"].round(2).mode().iloc[0] if len(frame) else None
    f = frame.assign(h=frame["month"].map(half_of),
                     stock=(frame["answer"].round(2) == stock) | (frame["answer"].astype(float) == 0))
    answered = f.groupby("h").size().reindex(asked.index, fill_value=0)
    return ((asked - answered) + f.groupby("h")["stock"].sum().reindex(asked.index, fill_value=0)) / asked


def uninformative_switch(frame: pd.DataFrame, asked: pd.Series) -> str | None:
    return zero_switch(uninformative_share(frame, asked))


def uninformative_flags(frame: pd.DataFrame, items) -> pd.DataFrame:
    """One row per question asked: its half-year and 1 if it got no informative answer (refused, 0%, stock value)."""
    stock = frame["answer"].round(2).mode().iloc[0] if len(frame) else None
    informative = dict(zip(frame["item_id"], ~((frame["answer"].round(2) == stock) | (frame["answer"] == 0))))
    return pd.DataFrame({"h": [half_of(i.payload["month"]) for i in items],
                         "u": [0.0 if informative.get(i.item_id, False) else 1.0 for i in items]})


def market_reading(switch: str | None, release) -> str:
    """The switch as a horizon, or 'no clear switch' when there is none or it comes after the release half-year."""
    if switch is None or (release is not None and switch > half_of(f"{release:%Y-%m}")):
        return "no clear switch"
    return switch


def compute() -> list[dict]:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    heads = dating_horizon.load_sets()
    m11 = {it.item_id: it for it in load_testset("monthly_return", "v2")}
    index_items = ladder._index_items("index_return")
    asked = pd.Series([half_of(i.payload["month"]) for i in index_items]).value_counts()
    rows = []
    for m in MODELS + EXTRA + NEW:
        cut, rel = resolve_cutoff(m, meta), resolve_release(m, meta)
        row = {"model": m, "cutoff": cut, "release": rel}
        idx = ladder.index_frame(calls, m)
        if len(idx):
            cert = switch_certainty(uninformative_flags(idx, index_items))
            row.update(market=market_reading(uninformative_switch(idx, asked), rel),
                       market_share=uninformative_share(idx, asked), certainty=cert)
        if m not in EXTRA:
            ex = dating_horizon.excess_by_year(dating_horizon.model_frame(
                calls, m, cut, heads, release=None if m in MODELS else rel))   # MODELS: the registered releases
            row["news"] = str(dating_horizon.horizon(ex)) if len(ex) and dating_horizon.horizon(ex) else None
            sub = calls[(calls["model"] == m) & (calls["testset"] == "monthly_return/v2")]
            mr = analysis.item_scores(sub, m11, "monthly_return", cut)
            if len(mr):
                a = mr[~mr["refused"].astype(bool) & ~mr["fabricated"].astype(bool)]
                hedge = a["no_direction"].fillna(False).astype(bool) if "no_direction" in a else False
                months = a["item_id"].map(lambda i: m11[i].payload["month"])
                row["zero_m11"] = zero_switch(pd.Series(np.asarray(hedge, float), index=months.map(half_of).to_numpy())
                                              .groupby(level=0).mean())
        rows.append(row)
    return rows


def _firm(r: dict) -> str:
    c = r.get("certainty")
    if not c or r.get("market") == "no clear switch":
        return "–"
    rng = f"{c['range'][0]}..{c['range'][1]}" if c["range"] else "–"
    return f"{c['stable']:.0%}; {rng}; {c['at_40'] or 'none'} / {c['at_60'] or 'none'}"


def _shaky(r: dict) -> bool:
    """A switch that resampling or a 40/60% threshold moves is reported as uncertain."""
    c = r.get("certainty")
    return bool(c and r.get("market") not in (None, "no clear switch")
                and (c["stable"] < 0.6 or c["at_40"] != c["switch"] or c["at_60"] != c["switch"]))


def render(rows: list[dict]) -> str:
    L = ["# Effective cutoffs: where each model's knowledge actually ends (exploratory)\n",
         "Next to each model's stated cutoff, from stored answers. **Market:** the first half-year from which most "
         "Nordic index questions (M15) get no informative answer (a refusal, 0%, or the model's stock value) and stay "
         "so; a switch after the release reads \"no clear switch\" (the model keeps guessing). **News:** the span in "
         "which its exact-month dating of headlines ends (M18, beyond its own answer habits). **Stock returns:** the "
         "half-year from which most of its monthly-return answers are 0% (M11, \"I don't know\"). **How firm:** the share "
         "of 1,000 resamples (questions redrawn within each half-year) that give the same half-year, their 95% range, "
         "and the switch with a 40% or 60% threshold instead of 50%; a switch that moves is marked uncertain. "
         "Checked on the "
         "models with known cutoffs: GPT-3.5 Turbo, GPT-4o-mini and Llama 3.3 stop answering the market within a few "
         "months after their cutoffs. Models that answer from a small menu of values throughout (the Qwen models) "
         "give a less sharp market switch. A registered out-of-sample check on the three models added on 2026-10-07 "
         "(docs/preregistration/2026-10-14-zero-answer-horizon.md): every switch counted (not uncertain) falls between "
         "one half-year before the cutoff's half-year and two after it, as on the older models. Price levels cannot "
         "time it: they persist, so earlier prices help with later years without any hindsight.\n",
         _table(["model", "stated cutoff", "release", "market: stops answering", "how firm (resampled; 95% range; "
                 "at 40% / 60%)", "news: clearly dated through", "stock returns: switch to 0%"],
                [[short(r["model"]), str(r["cutoff"]), str(r["release"] or "–"),
                  r.get("market", "–") if not _shaky(r) else f"{r.get('market')} (uncertain)", _firm(r),
                  r.get("news") or "–", r.get("zero_m11") or "–"] for r in rows]),
         "\n## Market: share of index questions without an informative answer, per half-year\n"]
    halves = sorted({h for r in rows if "market_share" in r for h in r["market_share"].index if h >= "2021-H1"})
    L.append(_table(["model", *halves],
                    [[short(r["model"]), *[f"{r['market_share'].get(h, float('nan')):.0%}" for h in halves]]
                     for r in rows if "market_share" in r]))
    return "\n".join(L) + "\n"


def main() -> None:
    text = render(compute())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "effective_cutoffs.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
