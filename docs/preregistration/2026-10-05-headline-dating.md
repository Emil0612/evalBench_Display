# Analysis plan: M18 headline dating

**Date:** 2026-10-05. Sections are committed in order, each before the answers it uses are collected.

## 0. Micro-pilot and gate (committed before the pilot run)
- **Why:** models may refuse to date headlines, or answer in a form we cannot read, and the blind baseline may be too
  strong to beat. The pilot checks both before about 15,000 answers are bought.
- **Run:** the pilot block of `headline_date/v1` (the first 127 items, one per month 2016-01..2026-07), forced,
  temperature 0, four models: GPT-3.5 Turbo, Llama 3.3 70B, Gemini 2.5 Flash, DeepSeek V3.1. These items are
  excluded from every analysis below.
- **Gate:** GO if both hold:
  1. at least 3 of the 4 models answer (a readable year and month) at least 70% of the pilot items;
  2. at least 2 of the 4 models score higher than the blind baseline on their own BEFORE months (on or before the
     last whole month before cutoff − 91 days), mean over the same items, point estimates.

  Otherwise NO-GO: §1–§7 are not written and the pilot is reported.
- **Feasibility, from the built set only** (`python -m evalbench.headline_feasibility`, before any answer):
  - **The set:** 1,651 items (127 pilot, 1,524 main). Main items: Swedish 1,086, English 438; press releases 680,
    editorial 662, macro 182.
  - **The blind baseline on the main items:**
    - score 0.134;
    - median 24 months off;
    - 7% within ±3 months, 1% the exact month;
    - best on 2019–2021 (score 0.19–0.23), worst at the ends (2016: 0.095, 2026: 0.044), as a regression to the
      middle would be.
    - The date-cue filters left no strong cues.
  - **The knee fit on synthetic answers** (12 per month): the 95% interval spans about ±2–4 months with answer noise
    of 4–12 months, for horizons in 2021, 2023 and 2025.
  - **Cost ceilings:** micro-pilot $0.047 (508 calls); full run $1.37 (14,859 calls).

### Result (2026-10-05, after the pilot run, before any other answer)
- Runs `4a452526b5` and the retry `48f8f7b2e5`: 508 of 508 answers, $0.036.
- **Gate: GO.** Four of four models answer at least 70%; three of four score above the blind baseline on their
  BEFORE pilot items:

| Model | Answer rate | BEFORE items | Model score | Blind score (same items) |
|---|---|---|---|---|
| GPT-3.5 Turbo | 100% | 66 | 0.167 | 0.144 |
| Llama 3.3 70B | 80% | 93 | 0.134 | 0.167 |
| Gemini 2.5 Flash | 100% | 106 | 0.300 | 0.160 |
| DeepSeek V3.1 | 100% | 107 | 0.227 | 0.159 |

## 1. What is run (committed before the full run)
- `headline_date/v1`, all 1,651 items, forced, temperature 0, nine models with their registered pins:
  - GPT-3.5 Turbo, GPT-4o-mini, Llama 3.1 70B, Llama 3.3 70B, GPT-4.1 Mini, gpt-oss-20b, Gemini 2.5 Flash,
    DeepSeek V3.1, Qwen3.8 Flash.
  - The four pilot models' pilot answers are already stored and are not asked again.
- **Analyses use the 1,524 main items only;** the pilot block is excluded.

## 2. Lines and scores
- **Score per item:** max(0, 1 − |months off| / 12); refused or unreadable = 0.
- **Blind baseline:** the stored `blind_month` per item (a hashed bag-of-words ridge regression, refit leaving out
  the test month ± 1).
- **BEFORE:** months on or before the last whole month before cutoff − 91 days.
- **AFTER:** months after the release month (GPT-3.5's release 2024-01-25, cutoff-ladder plan deviation 1).

## 3. H1 (confirmatory, purpose A): inside its window, does the model date headlines better than the blind baseline?
- Per model: d = score(model) − score(blind) per BEFORE item.
- Mean d with a cluster bootstrap by month (20,000 draws), two-sided p, Holm across the nine models.
- **Refusal guard:** also on answered items only; a detection needs both, and a positive answered-only d.
- **Reading:** "dates better than the text alone" when d > 0, Holm p < 0.05 and the guard holds.
- **What would change our reading:** d ≤ 0 for every model means dating is cue reading for these models, and M18
  cannot measure memory.

## 4. The horizon (primary estimate, purpose B)
- **Per model, on answered main items:** the per-month mean signed error (answer − truth, months), weighted by the
  number of answers.
- **Fit:** error = c + b·max(0, t − H), least squares over H on the monthly grid, at least 6 months on each side.
- **Interval:** a month-cluster bootstrap 95% interval (2,000 draws).
- **Reported per model:** H, its interval, and where it falls against the stated cutoff and the release.
  1. **"at or before the cutoff":** the interval ends on or before cutoff − 91 days;
  2. **"in the gap":** the interval lies within (cutoff − 91 days, release];
  3. **"after release":** the interval starts after release;
  4. **"undetermined":** otherwise.
- **No significance test:** an estimate.
- **A model with fewer than 6 answered months after a candidate knee:** "no estimate".

## 5. Exploratory (reported whatever they show)
- **E1:** exact month, within ±3 months and same-year rates, BEFORE / gap / AFTER, model vs blind.
- **E2:** H1's d by language (Swedish, English) and by type (macro, editorial, press release).
- **E3:** each model's most frequent answer month (its "self-perceived now") next to H.
- **E4:** H next to the M5/M15 fade of the cutoff-ladder addendum, per model.

## 6. Cost and stopping
- **Cost:** ceiling $1.37 (14,859 calls before the pilot; the pilot's 508 answers are stored); spend cap $1.50.
- **Stopping:** no further draws after the answers are seen. Any extension is a registered addendum.

## 7. Deviations
1. **2026-10-05, at the full run (budget only):** the spend cap was set to $1.75, the Run page's ceiling at the most expensive provider, instead of $1.50, which could have stopped the run early. Nothing in the analysis changes. The full run `580a0bbfaa` and its retry `465dcded69` stored all 14,351 answers for $0.75.
