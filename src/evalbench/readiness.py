"""Backtest readiness per model: one row per model that gathers what the registered analyses found (no new statistics).
Each reading comes from its own registered report, recomputed with exactly the models that report registered:
M5 price levels from the monthly-grid replication (v4) and the cutoff ladder's family R; M15 market moves from the
Nordic index plan and family R; M18 news dating from the headline-dating plan; the positive controls by the rule that
gated GPT-3.5 (pilot E's E2 rule). Where a model's knowledge of news ends is exploratory (`dating_horizon`), as is
where its market answers stop being informative (`effective_cutoffs`). For using a model in a backtest it adds the
fake-date test (M19: does a date in the prompt stop hindsight?) and the masking test (M20: can it date market data
with the names and dates removed?), both registered.
Writes results/readiness/report.md (local); no model calls, no cost.

uv run python -X utf8 -m evalbench.readiness
"""
from __future__ import annotations

import pandas as pd

from . import (dating_horizon, effective_cutoffs, fake_date, grid_report, headline_report, index_report, ladder,
               mask_date, store)
from .cutoffs import resolve_cutoff
from .items import ROOT
from .ladder_report import pilot_gate
from .llm.openrouter import fetch_models, load_pins, models_by_id
from .pilot_d_report import MODELS as PILOT_MODELS
from .pilot_report import _f, _table, short

OUT = ROOT / "results" / "readiness"
MODELS = headline_report.MODELS
# models whose weights are published: a backtest with them stays reproducible after the API version is retired
OPEN_WEIGHTS = {"meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.3-70b-instruct", "openai/gpt-oss-20b",
                "deepseek/deepseek-chat-v3.1"}


def reading(text: str | None) -> str:
    """A registered report's reading in three words: detected, not detected, refusal-driven (or no data)."""
    if not text:
        return "no data"
    t = text.lower()
    if "refusal" in t:
        return "refusal-driven"
    if t.startswith("none") or t.startswith("not") or "no after" in t:
        return "not detected"
    if "detected" in t or "dates better" in t or "beats" in t:
        return "detected"
    return "not detected"


def controls_pass(rows: list[dict]) -> bool:
    """The positive controls by their registered rules: M9 above the blind guesser (pilot E's E2: the interval's lower
    end above the blind rate), M7 above both 0.5 and the blind guesser (pilot D's E2). One passing control is enough."""
    for r in rows:
        lo, blind = r.get("lo"), r.get("blind")
        if lo is None or blind is None or pd.isna(lo) or pd.isna(blind):
            continue
        if r["label"].startswith("M7") and lo > 0.5 and lo > blind:
            return True
        if r["label"].startswith("M9") and lo > blind:
            return True
    return False


def prompt_date_reading(gate: dict | None, row: dict | None) -> str:
    """M19 for a backtest: does telling the model the date stop its hindsight? 'leaks', 'no leak shown', 'did not
    forecast' (it failed the gate: no real forecasts to test) or 'not tested'."""
    if gate is None:
        return "not tested"
    if not gate["passes"]:
        return "did not forecast"
    if row is None:
        return "not tested"
    return "leaks" if row["reading"] == "hindsight leaks through the date" else "no leak shown"


def masking_reading(row: dict | None, coarse: int | None = None) -> str:
    """M20 for a backtest: with names and dates removed, can the model still place market data in time? Only dating
    beyond its own answer habits counts ('below its habits' shows no dating either). The registered test asks for the
    month (within one); `coarse` is the narrowest wider window at which an exploratory re-score places the paths."""
    if not row or not row.get("n"):
        return "not tested"
    if row["reading"] == "dates the unlabeled path":
        return "dates masked paths"
    return f"within ±{coarse} months (exploratory)" if coarse else "not dated to the month"


COARSE_WINDOWS = (3, 6, 12)


def coarse_windows(by_window: dict[int, list[dict]]) -> dict[str, int]:
    """{model: the narrowest window whose re-score is positive with Holm p < 0.05 across the models}."""
    out: dict[str, int] = {}
    for w in sorted(by_window):
        for r in by_window[w]:
            if r.get("x", 0) > 0 and r.get("p_holm", 1) < 0.05 and r["model"] not in out:
                out[r["model"]] = w
    return out


def market_recall(registered: str, r_before_lo: float | None) -> str:
    """M15 inside the window: the registered drop test's reading, or 'recall shown' when the drop is not shown but
    the model's recall before the cutoff clearly is (the 95% month-bootstrap interval of r above 0). A model that
    gives few real answers after release cannot show a drop, yet it still recalls the market inside its window."""
    if registered == "detected":
        return "detected"
    if r_before_lo is not None and not pd.isna(r_before_lo) and r_before_lo > 0:
        return "recall shown"
    return registered


def market_fade(row: dict | None) -> str:
    """Where the model's market answers stop being informative (effective cutoffs) and how firm that switch is."""
    if not row or not row.get("market"):
        return "–"
    if row["market"] == "no clear switch":
        return "no clear switch"
    firmness = "uncertain" if effective_cutoffs._shaky(row) else "firm"
    return f"{row['market']} ({firmness}, {row['certainty']['stable']:.0%})"


def _leak_note(row: dict) -> str:
    leak = row.get("leak")
    return "" if leak is None or pd.isna(leak) else f" (keeps {leak:.0%} of recall)"


def verdict(row: dict) -> str:
    """One plain sentence or three per model for a reader choosing a model for a backtest."""
    parts = []
    if row["controls"] == "fails":
        parts.append("Fails the positive controls, so its blanks mean too hard, not clean.")
    if row["clean_months"] > 0:
        parts.append(f"Clean for backtests from {row['clean_from']}: {row['clean_months']} months of our data "
                     "(to 2026-08), after its release.")
    else:
        parts.append(f"No clean months in our data (released {row['release']}).")
    kinds = [name for key, name in (("m5", "price levels"), ("m15", "market moves"), ("m18", "news dating"))
             if row[key] in ("detected", "recall shown")]
    parts.append("Inside its training window it carries hindsight on " + ", ".join(kinds) + "." if kinds else
                 "No hindsight detected inside its window by our tests, which does not make that window clean.")
    if row["m15"] == "recall shown":
        why = ("it has no months after release in our data" if row["clean_months"] == 0 else
               "it gives few real answers after release, or keeps tracking the market somewhat")
        parts.append(f"It recalls the market's monthly moves inside its window (r {row['m15_before']:+.2f}), though "
                     f"the registered drop after release is not shown: {why}.")
    if row.get("dating_ends"):
        parts.append(f"It clearly dates Nordic news through {row['dating_ends']} (exploratory; it seldom names later "
                     f"years, so where its news knowledge stops is not measured); when unsure it names {row['habit']}.")
    elif row.get("habit"):
        parts.append(f"When unsure about a date it names {row['habit']}.")
    prompt = row.get("prompt_date")
    if prompt == "leaks":
        leak = row.get("leak")
        kept = "" if leak is None or pd.isna(leak) else \
            f" (its dated forecasts keep about {leak:.0%} of its plain recall)"
        parts.append(f"A date in the prompt does not stop its hindsight{kept}.")
    elif prompt == "no leak shown":
        parts.append("Told the date, no leak shown in our test (not proof that a date protects).")
    elif prompt == "did not forecast":
        parts.append("Told the date, it gave stock forecasts rather than real ones, so a leak could not be tested.")
    if row.get("masking") == "dates masked paths":
        parts.append("Masked market data (no names, dates or levels) still lets it place some periods in time, "
                     "most of all those through the March 2020 crash.")
    elif row.get("masking") == "not dated to the month":
        parts.append("With names, dates and levels removed it could not date market paths to the month in our test.")
    elif str(row.get("masking", "")).startswith("within"):
        parts.append("With names, dates and levels removed it could not date market paths to the month, but it "
                     f"places them {row['masking'].replace(' (exploratory)', '')} beyond its answer habits (an "
                     "exploratory re-score), so masking is not a complete defence.")
    return " ".join(parts)


def compute() -> list[dict]:
    calls = store.load_calls(store.connect(), mode="forced", temperature=0.0, reasoning="auto", seed=-1)
    meta = models_by_id(fetch_models())
    pins = load_pins()
    m5 = {r["model"]: r for r in grid_report.compute(PILOT_MODELS, ("v4",))["h"]
          + grid_report.compute(ladder.NEW, ("v4",), release=ladder.RELEASE)["h"]}
    m15 = {r["model"]: r for r in index_report.compute(PILOT_MODELS)["m15"]
           + index_report.compute(ladder.NEW, release=ladder.RELEASE)["m15"]}
    head = headline_report.compute(calls, meta)
    m18 = {r["model"]: r for r in head["h1"]}
    habit = {r["model"]: r for r in head["e3"]}
    heads = dating_horizon.load_sets()
    fades = {r["model"]: r for r in effective_cutoffs.compute()}
    fd = fake_date.compute()
    fd_gate = {g["model"]: g for g in fd["gate"]}
    fd_rows = {r["model"]: r for r in fd["rows"]}
    masked = {r["model"]: r for r in mask_date.compute()["rows"]}
    coarse = coarse_windows({w: mask_date.compute(window=w)["rows"] for w in COARSE_WINDOWS})
    rows = []
    for m in MODELS:
        cut, rel = resolve_cutoff(m, meta), ladder.RELEASE[m]
        gate = pilot_gate(calls, meta, model=m)
        ex = dating_horizon.excess_by_year(dating_horizon.model_frame(calls, m, cut, heads))
        rows.append({
            "model": m, "cutoff": str(cut), "release": rel, "clean_from": str(ladder.first_after(rel)),
            "clean_months": ladder.clean_months(rel),
            "weights": "open weights" if m in OPEN_WEIGHTS else "API only",
            "pin": pins.get(m, "–"),
            "controls": "passes" if controls_pass(gate["rows"]) else "fails",
            "controls_detail": "; ".join(f"{r['label']} {_f(r.get('rate'), '{:.0%}')} vs {_f(r.get('blind'), '{:.0%}')}"
                                         for r in gate["rows"]),
            "m5": reading(m5.get(m, {}).get("reading") or m5.get(m, {}).get("note")),
            "m5_value": m5.get(m, {}).get("dkappa"),
            "m15": market_recall(reading(m15.get(m, {}).get("reading") or m15.get(m, {}).get("note")),
                                 m15.get(m, {}).get("r_before_lo")),
            "m15_before": m15.get(m, {}).get("r_before"), "m15_after": m15.get(m, {}).get("r_after"),
            "m18": reading(m18.get(m, {}).get("reading")), "m18_value": m18.get(m, {}).get("d"),
            "habit": habit.get(m, {}).get("mode"), "habit_share": habit.get(m, {}).get("share"),
            "dating_ends": str(dating_horizon.horizon(ex)) if dating_horizon.horizon(ex) else None,
            "market_fade": market_fade(fades.get(m)),
            "prompt_date": prompt_date_reading(fd_gate.get(m), fd_rows.get(m)),
            "leak": fd_rows.get(m, {}).get("leak"),
            "masking": masking_reading(masked.get(m), coarse.get(m))})
        rows[-1]["verdict"] = verdict(rows[-1])
    return rows


def render(rows: list[dict]) -> str:
    L = ["# Backtest readiness per model\n",
         "Gathered from the registered analyses (each recomputed with the models its plan registered); no new "
         "statistics. A model is clean for a backtest only after its release; inside its training window it carries "
         "the hindsight listed. Diagnostic evidence, not a certificate: \"not detected\" never means a period is "
         "clean.\n",
         "## The models\n",
         _table(["model", "stated cutoff", "release", "clean from", "clean months to 2026-08", "weights",
                 "pinned endpoint", "positive controls"],
                [[short(r["model"]), r["cutoff"], str(r["release"]), r["clean_from"], r["clean_months"], r["weights"],
                  r["pin"], r["controls"]] for r in rows]),
         "\n## Hindsight inside each model's training window\n",
         _table(["model", "M5 price levels", "M15 market moves (r before → after)", "M18 news dating (vs text only)",
                 "habitual month (share of answers naming it)", "news clearly dated through (exploratory)"],
                [[short(r["model"]), f"{r['m5']} ({_f(r['m5_value'], '{:+.2f}')})",
                  f"{r['m15']} ({_f(r['m15_before'], '{:+.2f}')} → {_f(r['m15_after'], '{:+.2f}')})",
                  f"{r['m18']} ({_f(r['m18_value'], '{:+.3f}')})",
                  f"{r['habit'] or '–'} ({_f(r['habit_share'], '{:.0%}')})", r["dating_ends"] or "–"] for r in rows]),
         "\n## Using a model for a backtest\n",
         "Only months after a model's release are clean. Inside its window, two common defences were tested: telling "
         "the model the date (\"Today is ...\", M19) and hiding names, dates and levels from the data (M20). "
         "\"Market answers stop varying\": the half-year from which most of its index answers are refusals, 0% or a "
         "stock value (it marks when the answers stop varying, not whether they tracked the market); \"firm\" means "
         "resampling the questions lands on the same half-year (the share shown). \"News clearly dated through\": "
         "the last year whose headlines it dates to the month beyond its habits; models seldom name later years, so "
         "where news knowledge stops is not measured.\n",
         _table(["model", "clean from", "market answers stop varying", "news clearly dated through",
                 "date in the prompt (M19)", "masked market data (M20)"],
                [[short(r["model"]), r["clean_from"], r["market_fade"], r["dating_ends"] or "–",
                  r["prompt_date"] + (_leak_note(r) if r["prompt_date"] == "leaks" else ""), r["masking"]]
                 for r in rows]),
         "\n**Reading the last two columns:** \"leaks\": told the date, its forecasts still track the real market "
         "before its release; \"no leak shown\" is not proof of safety; \"did not forecast\": it answered stock "
         "values, so there was nothing to test. \"dates masked paths\": it dates unlabeled market data to the month "
         "beyond its answer habits (registered); \"within ±N months\": not to the month, but an exploratory re-score "
         "at that coarser precision places the paths; \"not dated to the month\": neither shown, which is not proof "
         "that masking protects. Not every model was in these two tests.\n",
         "\n## Per model\n"]
    L += [f"- **{short(r['model'])}:** {r['verdict']}" for r in rows]
    L += ["\n## Sources and rules\n",
          "- **Clean window:** whole months after the release month (`ladder.RELEASE`; GPT-3.5 Turbo's release is "
          "its served snapshot's, 2024-01-25), to the end of our data (2026-08).",
          "- **M5:** monthly-grid replication `price_level/v4` (grid plan §6; Holm across its five scorable models) "
          "and the cutoff ladder's family R for GPT-3.5 Turbo, GPT-4o-mini and Llama 3.1 (Holm across those three).",
          "- **M15:** Nordic index plan §6.3 (correlation drop, Holm across its models) and family R for the three "
          "ladder models. \"Recall shown\" (descriptive): the drop is not shown, but the 95% month-bootstrap interval "
          "of the model's r before the cutoff lies above 0.",
          "- **M18:** headline-dating plan §3 (model minus text-only baseline on BEFORE months, Holm across nine). "
          "DeepSeek V3.1's detection rests on its habitual month (summary sensitivity check).",
          "- **Positive controls:** M9 price levels and M7 annual returns on items before the cutoff; passes when "
          "at least one control meets its registered rule: M9's interval above the blind guesser (pilot E's E2), M7's "
          "above both 0.5 and the blind guesser (pilot D's E2).",
          "- **Habitual month and news clearly dated through:** exploratory, from the M18 answers (both headline "
          "sets pooled). The habitual month is the month a model names most often. News is clearly dated through the "
          "last publication year in which the model dates headlines to the exact month clearly more often than its "
          "own answer habits would (95% interval above 0; the months next to its habitual answer left out; "
          "`results/headline/horizon.md`). The year it stops is not measured: models almost never name later years.",
          "- **Market answers stop varying:** exploratory (`results/readiness/effective_cutoffs.md`): the half-year "
          "from which most of the model's index answers are refusals, 0% or a stock value; uncertain when resampling "
          "or a 40/60% threshold moves it.",
          "- **Date in the prompt:** fake-date plan (M19; gate on the micro-pilot questions, then the correlation "
          "drop before vs after release, Holm across five models); the share of recall kept is the forecast r over "
          "the plain M15 r.",
          "- **Masked market data:** masking plan (M20; the month within one, beyond the model's answer habits, Holm "
          "across five models). The coarser windows (±3, ±6, ±12 months; same rule and Holm) are an exploratory "
          "re-score made after the registered result.",
          "- **Weights:** open weights keep a backtest reproducible after an API version is retired."]
    return "\n".join(L) + "\n"


def main() -> None:
    rows = compute()
    OUT.mkdir(parents=True, exist_ok=True)
    text = render(rows)
    (OUT / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
