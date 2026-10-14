# The "I don't know" horizon on the three new models (an out-of-sample check)

**Date:** 2026-10-14. Written and committed before the half-year shares below were computed for Gemma 4 31B, Gemma 4
26B A4B and Qwen3.6 27B. No model calls, no cost: the answers were collected on 2026-10-07
(`docs/preregistration/2026-10-07-new-models.md`).

## 1. Background

Forced to give a number, several models answer "0%" (or give no informative answer) for months they do not know, and
the switch lines up with their cutoff (findings doc, Finding 7 and next step 4). Since 2026-10-06 the exploratory
report `evalbench.effective_cutoffs` reads two such switches per model:

- **Market (M15, `index_return/v1`):** the first half-year from which at least 50% of the index questions get no
  informative answer (a refusal, 0%, or the model's most frequent value) and stay so. A switch after the half-year of
  the model's release is no horizon and reads "no clear switch".
- **Stock returns (M11, `monthly_return/v2`):** the first half-year from which at least 50% of the answered questions
  get exactly 0% ("I don't know") and stay so.

The findings doc proposed registering "the half-year in which 0% answers pass 50%" as a horizon estimate. On the 12
models already in that report this is no longer possible: their switches have been seen. They stay descriptive. The
three models added on 2026-10-07 are not in the report, and their shares per half-year have not been computed or
looked at. Known beforehand: their share of 0% answers on M11 over all months before their cutoffs (Gemma 4 31B 24%,
Gemma 4 26B A4B 7%, Qwen3.6 27B 8%; `docs/findings.md`).

## 2. The rule (frozen)

The code of `src/evalbench/effective_cutoffs.py` at commit 5343878: `zero_switch` (threshold 0.5),
`uninformative_share`, `switch_certainty` (1,000 resamples, seed 0) and `market_reading`. A switch is **uncertain**
when the report already marks it so: fewer than 60% of resamples land on it, or a 40% or 60% threshold moves it. The
only change before computing: the three models are added to the report's model list, read like the main models
(market, news, stock returns).

## 3. Agreement window

A switch **agrees** with a cutoff when it falls between one half-year before the cutoff's half-year and two after it.
Calibrated on the existing rows with a listed or model-card cutoff and a switch:

| Model | Cutoff (half-year) | Market switch | Offset | Stock-return switch | Offset |
|---|---|---|---|---|---|
| GPT-3.5 Turbo | 2021-09 (2021-H2) | 2022-H2 | +2 | – | |
| GPT-4o-mini | 2023-10 (2023-H2) | 2024-H1 | +1 | 2024-H1 | +1 |
| Llama 3.3 70B | 2023-12 (2023-H2) | 2024-H1 | +1 | 2023-H1 | −1 |
| GPT-4.1 Mini | 2024-06 (2024-H1) | 2025-H1 | +2 | 2025-H1 | +2 |
| Gemini 2.5 Flash | 2025-01 (2025-H1) | 2024-H2 (uncertain) | −1 | – | |
| DeepSeek V3.2 | 2025-07 (2025-H2) | 2025-H1 | −1 | – | |
| Qwen3 235B 2507 | 2025-06 (2025-H1) | 2025-H1 | 0 | – | |
| Qwen3-Next 80B | 2025-09 (2025-H2) | 2023-H1 (uncertain) | −5 | – | |

All but Qwen3-Next (a model that answers from a small menu of values, marked uncertain) fall within −1 to +2. Left
out: rows with no switch (Llama 3.1, DeepSeek V3.1 and gpt-oss on the market: "no clear switch") and Qwen3.8 Flash,
whose cutoff is our own estimate.

## 4. Predictions

| Model | Cutoff and source | Release | Window for both switches |
|---|---|---|---|
| Gemma 4 31B | 2025-01 (model card) | 2026-04-02 | 2024-H2 .. 2026-H1 |
| Gemma 4 26B A4B | 2025-01 (model card) | 2026-04-03 | 2024-H2 .. 2026-H1 |
| Qwen3.6 27B | 2024-09 (our estimate) | 2026-04-27 | 2024-H1 .. 2025-H2 |

Qwen3.6's cutoff is our own estimate from the US recall changepoint, so its row also checks that estimate with a
different reading on different (Nordic) questions.

## 5. Outcomes and decision

Each of the six cells (3 models × 2 switches) reads one of:
- **agrees** or **disagrees** with the window, for a switch that is not uncertain;
- **uncertain:** reported, not counted;
- **no signal:** no clear switch, or the share never stays at 50% or more.

**The rule holds** when every counted switch agrees and at least two cells are counted. Then the three models join the
effective-cutoffs report and its snapshot, and the findings doc may call the switch a supported horizon reading
(still exploratory, never a cutoff). **If it fails,** the models still join the report, with the failure stated in
the report text, the findings doc and the result below. "No signal" is not a failure: models that never say "I don't
know" give no reading (as GPT-3.5 and Gemini on M11).

The news column (M18, `dating_horizon`) is filled for the three models by its existing rule, exploratory, with no
prediction. The backtest-readiness report is not extended: it gathers the registered analyses of its own models.

## 6. Result

To be added below after computing, dated, without changing anything above.

**Result (2026-10-14, computed after commit e097f58; `uv run python -X utf8 -m evalbench.effective_cutoffs`).**
Before any number was seen, one fix outside the rule (the first run stopped with an error): the news column needed each new model's release date (the registered
models' releases are built in), so it is passed in from the listing. The 12 existing rows of the report are unchanged.

| Model | Window | Market switch (how firm) | Reading | Stock-return switch | Reading |
|---|---|---|---|---|---|
| Gemma 4 31B | 2024-H2 .. 2026-H1 | 2025-H1 (96% of resamples; same at 40% and 60%) | agrees | 2025-H1 | agrees |
| Gemma 4 26B A4B | 2024-H2 .. 2026-H1 | 2025-H1 (99%; same at 40% and 60%) | agrees | 2025-H1 | agrees |
| Qwen3.6 27B | 2024-H1 .. 2025-H2 | 2025-H1 (72%; 2025-H2 at 60%) | uncertain, not counted | 2025-H1 | agrees |

**The rule holds:** five cells are counted and all five agree. Market share without an informative answer, per
half-year: Gemma 4 31B 28% (2024-H2) → 100% (2025-H1); Gemma 4 26B A4B 22% → 89%; Qwen3.6 27B moves up unevenly
(50% in 2023-H2, 28% in 2024-H2, 56% in 2025-H1, 83% in 2025-H2), the menu-answer pattern of the older Qwen models.
Both Gemma models switch in the half-year of their model-card cutoff (offset 0). Qwen3.6's switches land one half-year
after its estimated cutoff (2024-09), inside the window: the estimate is not contradicted. The switch cannot say
whether it is early or late (the Gemma models switch in their cutoff's half-year, the older models between one
half-year before theirs and two after).

**Read with care:** the report resamples only the market switch, so the stock-return switches are counted without a
firmness check. Three models are a small test; the window was calibrated on eight models, seven of them inside it. The news column (exploratory, no
prediction): Gemma 4 31B and Qwen3.6 clearly dated through 2024, Gemma 4 26B A4B through 2023.
