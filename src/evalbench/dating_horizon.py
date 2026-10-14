"""An anchoring-free knowledge horizon from the stored M18 answers (exploratory; not registered). The registered knee
fit failed because models answer a habitual month when they do not know. Here, per publication year: the share of
headlines a model dates to the exact month, minus the share its own answer habits would hit anyway (how often it names
that month across all headlines). The months next to its habitual answer are left out, where a fallback answer is
right by luck. Memory shows as excess above 0; the horizon is the last year whose excess is clearly above 0 (the lower
end of its 95% interval, resampling headlines). A model's answers are pooled over every headline set it was asked: v1
(all nine models) and v2, the fresh headlines of the M18 confirmation (five models); the report also shows each set
alone, so the replication on fresh headlines is visible. No model calls, no cost.

uv run python -X utf8 -m evalbench.dating_horizon
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import headline_report, ladder, store
from .cutoffs import resolve_cutoff
from .items import ROOT, load_testset
from .llm.openrouter import fetch_models, models_by_id
from .methods import headline_date as hd
from .pilot_report import _f, _table, short

OUT = ROOT / "results" / "headline"
HABIT_GAP = 1        # months either side of the habitual month left out
SETS = ("v1", "v2")  # v2: fresh headlines from the M18 confirmation run (five of the nine models)
COLUMNS = ["item_id", "true", "answer", "refused"]


def expected_hit(answers: pd.Series, true: pd.Period, window: int = 0) -> float:
    """The share of the model's answers (over all headlines) within `window` months of `true`: what its habits alone
    would score on a headline from that month."""
    ords = np.array([a.ordinal for a in answers])
    return float((np.abs(ords - true.ordinal) <= window).mean()) if len(ords) else float("nan")


def excess_by_year(df: pd.DataFrame, B: int = 2000, seed: int = 0) -> pd.DataFrame:
    """df: item_id, true (Period), answer (Period or None), refused. Answered headlines only; the months within
    HABIT_GAP of the habitual answer are left out. Per year: n, observed exact share, expected share, excess and its
    95% interval."""
    a = df[~df["refused"].astype(bool)].dropna(subset=["answer"]).copy()
    if a.empty:
        return pd.DataFrame(columns=["year", "n", "observed", "expected", "excess", "lo", "hi"])
    ans = np.array([p.ordinal for p in a["answer"]])
    counts = pd.Series(ans).value_counts()
    share = (counts / len(ans)).to_dict()
    habit = int(counts.idxmax())
    true = np.array([p.ordinal for p in a["true"]])
    keep = np.abs(true - habit) > HABIT_GAP
    a, ans, true = a[keep], ans[keep], true[keep]
    hit = (ans == true).astype(float)
    exp = np.array([share.get(t, 0.0) for t in true])
    d = pd.DataFrame({"year": [p.year for p in a["true"]], "hit": hit, "exp": exp, "x": hit - exp})
    rng = np.random.default_rng(seed)
    rows = []
    for year, g in d.groupby("year"):
        x = g["x"].to_numpy()
        draws = x[rng.integers(0, len(x), (B, len(x)))].mean(axis=1)
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rows.append({"year": int(year), "n": len(g), "observed": float(g["hit"].mean()),
                     "expected": float(g["exp"].mean()), "excess": float(x.mean()), "lo": float(lo), "hi": float(hi)})
    return pd.DataFrame(rows)


def horizon(ex: pd.DataFrame) -> int | None:
    """The last year whose excess is clearly above 0 (lower end of its interval above 0); None if no year is."""
    above = ex[ex["lo"] > 0]
    return int(above["year"].max()) if len(above) else None


def silent_from(ex: pd.DataFrame) -> int | None:
    """The first year from which the model dates no headline to the exact month, that year and every later one: the
    upper end of the horizon (None if it still hits in the last year)."""
    years = ex.sort_values("year")
    silent = None
    for r in years.itertuples():
        silent = (silent or r.year) if r.observed == 0 else None
    return int(silent) if silent is not None else None


def dating_span(clear: int | None, silent: int | None) -> str | None:
    """Where a model's knowledge of news ends: from the last year it clearly dates headlines to the exact month above
    its own answer habits, to the year before it dates none exactly (exploratory)."""
    if clear is None:
        return None
    if silent is None:
        return f"{clear} or later"
    last = max(clear, silent - 1)
    return str(clear) if last == clear else f"{clear}–{last}"


def frame(calls: pd.DataFrame, model: str, cutoff, items: dict, testset: str = headline_report.TESTSET,
          release=None) -> pd.DataFrame:
    sc = headline_report.scored(calls, model, cutoff, items, testset=testset, release=release)
    if sc.empty:
        return pd.DataFrame(columns=COLUMNS)
    sc = headline_report.main_only(sc)
    answer = sc["answer"].map(lambda v: pd.Period(v, "M") if isinstance(v, str) else None) if "answer" in sc else None
    return pd.DataFrame({"item_id": sc["item_id"], "true": sc["month"], "answer": answer, "refused": sc["refused"]})


def pooled(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One model's answers over several headline sets (their items are distinct); sets it was not asked are skipped."""
    kept = [f for f in frames if len(f)]
    return pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=COLUMNS)


def load_sets() -> dict[str, dict]:
    return {v: {it.item_id: it for it in load_testset(hd.NAME, v)} for v in SETS}


def model_frame(calls: pd.DataFrame, model: str, cutoff, sets: dict[str, dict], release=None) -> pd.DataFrame:
    """The model's answers pooled over every headline set: the frame the horizon is read from."""
    return pooled([frame(calls, model, cutoff, items, testset=f"{hd.NAME}/{v}", release=release)
                   for v, items in sets.items()])


def _band(ex: pd.DataFrame) -> str:
    if ex.empty:
        return "not asked"
    h, s = horizon(ex), silent_from(ex)
    return f"{h if h else '–'} / {s if s else '–'}"


def main() -> None:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    sets = load_sets()
    summary, detail = [], []
    for m in headline_report.MODELS:
        cut = resolve_cutoff(m, meta)
        alone = {v: excess_by_year(frame(calls, m, cut, items, testset=f"{hd.NAME}/{v}")) for v, items in sets.items()}
        both = model_frame(calls, m, cut, sets)
        ex = excess_by_year(both)
        h, s = horizon(ex), silent_from(ex)
        summary.append([short(m), str(cut), str(ladder.RELEASE[m]), len(both), str(h) if h else "–",
                        str(s) if s else "–", *[_band(alone[v]) for v in sets]])
        detail.append((m, ex))
    L = ["# M18: an anchoring-free horizon (exploratory)\n",
         "Per publication year: the share of headlines a model dates to the exact month, minus the share its own "
         "answer habits would hit anyway (how often it names that month over all headlines), leaving out the months "
         "next to its habitual answer. The horizon lies between the last year whose excess has a 95% interval above 0 (a "
         "conservative lower end: exact hits are rare, so a year of headlines has wide intervals) and the first year "
         "from which the model dates no headline to the exact month at all. Answers are pooled over both headline "
         "sets: v1 (all models) and v2, the fresh headlines of the M18 confirmation (five models); the last two "
         "columns show each set alone (last year clearly dated / no exact dates from), so the replication on fresh "
         "headlines can be read directly. **Read the last clearly dated year as the result.** The \"no exact dates "
         "from\" year is mostly mechanical: models almost never name years after the one they treat as now (Gemini "
         "and DeepSeek never name 2025, the Llamas almost never 2024), and a model cannot hit a year it never names; "
         "so where news knowledge stops is not measured. The text-only guesser, which has no memory, has no clearly "
         "dated year. Not registered: a lead for a future plan, not a result.\n",
         _table(["model", "stated cutoff", "release", "headlines", "last year clearly dated", "no exact dates from",
                 "v1 alone", "v2 alone (fresh)"], summary), "\n## Per year (both sets pooled)\n"]
    for m, ex in detail:
        L.append(f"**{short(m)}**\n")
        L.append(_table(["year", "headlines", "exact", "expected by habit", "excess", "95% interval"],
                        [[r.year, r.n, _f(r.observed, "{:.1%}"), _f(r.expected, "{:.1%}"), _f(r.excess, "{:+.1%}"),
                          f"[{_f(r.lo, '{:+.1%}')}, {_f(r.hi, '{:+.1%}')}]"] for r in ex.itertuples()]))
        L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "horizon.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:4]))


if __name__ == "__main__":
    main()
