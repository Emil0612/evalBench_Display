# Analysis plan: the monthly price-level grid

**Date:** 2026-09-30. Committed before any model answered this set. Changes after that go in §5, dated.

**Why:** the rolling-window view (Evidence → Over time) showed a possible drop around each model's cutoff (clearest for
DeepSeek V3.1, pilot E's H1), but with 3-5 questions per month it is too noisy to read. This set asks the same question
as M5 for exactly 12 companies in every month, so the curve has even precision over time.

## 1. What is run

| | |
|---|---|
| Models | the six pilot D/E models, same pins (Llama 3.3 70B, GPT-4.1 Mini, gpt-oss-20b, Gemini 2.5 Flash, DeepSeek V3.1, Qwen3.8 Flash) |
| Set | `price_level/v3` (d5865e89): 804 questions, 12 per month for every month 2021-01..2026-07 (4 large, 4 mid, 4 small companies), 612 companies, no fabricated companies, no (company, month) pair from `price_level/v1` or `v2` |
| Calls | 804 per model, 4,824 in total; listed-price upper bound $0.53 (expect about $0.25) |
| Settings | as pilot E: forced answers, temperature 0, reasoning auto, 1 sample, no seed, standard privacy, pinned |
| Controls | oracle, random and blind, run before this plan: oracle detected (Δκ +0.76, p = 0.002), random not (+0.03). The blind guesser does better on recent months (−0.07): later periods are easier, which works against finding memory |

## 2. Scores and windows
- **Score per question:** closeness (scoring spec, 2026-09-30): 1 − |log(answer / truth)| / log 2, floored at 0, against
  the closer of the traded and the split-adjusted monthly average. A refusal or null answer scores 0; the refusal
  guard of pilot E (deviation 1) applies to every test below.
- **Near-line windows** (per model): BEFORE = the 12 months ending 3 months before the stated cutoff (the last whole
  month before cutoff − 91 days and the 11 before it). AFTER = the first 12 months after the release month, or as many
  as the data has (at least 6).

## 3. Confirmatory test
- **H (per model):** mean closeness in BEFORE is higher than in AFTER.
- **Test:** cluster bootstrap by company, 20,000 draws, two-sided; **Holm across the five models with an AFTER
  window** (Qwen3.8 Flash has none: released 2026-08, after the last month).
- **Confirmed** for a model if its Holm-adjusted p < 0.05, the difference is positive, and the refusal guard holds.
- **Why a near-line comparison:** both windows are close in time, so drift in how hard prices are to guess matters
  less than when 2021 is compared with 2026 (Meeus et al.'s regression-discontinuity advice).

## 4. Exploratory (reported whatever they show)
- **E1, the rolling curves:** 6-month windows moved one month at a time, per model, with the blind guesser.
- **E2, difficulty-adjusted:** the same BEFORE/AFTER comparison on (model closeness − blind-guesser closeness) per
  question, which cancels how hard each period is. Holm across the five models.
- **E3, where the drop is:** the single best break month in each model's monthly mean closeness (least squares, one
  change), with a cluster-bootstrap 95% interval, next to the stated cutoff and the release. Does the interval fall
  inside cutoff − 6 months .. release + 3 months?
- **E4, headline rates:** within the month's own range and within ±10%, BEFORE vs AFTER.
- **E5, Qwen3.8 Flash:** descriptive only (its rolling curve up to the end of the data).

## 5. Deviations

### 2026-10-05: parser fix, Gemini refusals stored as answers (a reading error, not a change of rule)
- **The error:** replies such as "I cannot answer questions about future stock prices. My knowledge cutoff is 2024."
  were read for the number in their text and stored as a price of 2024 (or 2025, 2023). "Cannot answer" was not in
  the refusal pattern. The pattern gained it on 2026-10-01 for the index methods (Nordic index plan §6 item 4); the
  stored price replies were re-read now.
- **Re-read:** `python -m evalbench.reparse price_level us_level split_clock monthly_return annual_return
  price_choice --apply`; 113 readings changed, all Gemini 2.5 Flash: 73 stored answers and 40 malformed replies became
  refusals (price_level 84 across v1–v4, monthly_return 16, us_level 5, price_choice 4, annual_return 3,
  split_clock 1). Each change is logged in the database's `reparse_log`.
- **Effect on this report:** no verdict changes; every Δ, interval and p is identical (a misread year and a refusal
  both score 0). Only Gemini's answer rates and its answered-only Δ move:

| Set | Answered before / after | Δ answered only | Reading |
|---|---|---|---|
| v3 | 95% / 80% → 92% / 74% | +0.10 → +0.09 | refusal-driven (unchanged) |
| v4 | 91% / 85% → 90% / 76% | +0.12 → +0.11 | detected (unchanged); now refusal-affected (gap 14 points) |

- **Published numbers stay as published.** The summary's statement that Gemini passes the guard on v4 because its
  refusal gap is 6 points no longer holds: the gap is 14 points, and Gemini stays detected because its answered-only
  Δ is still significant.

## 6. Addendum: a second grid draw (price_level/v4)

**Date:** 2026-09-30. Written after the v3 results (summarised in `docs/findings.md`) and before any
model answered v4. Deciding to add data after seeing results is a researcher choice, so this section fixes in advance
what the new data is used for and when drawing stops.

- **Why:** the v3 break-month intervals (E3) were 2-4 years wide for every model except DeepSeek V3.1.
- **Set:** `price_level/v4` (8ae73185): 1,608 questions, 24 per month for every month 2021-01..2026-07 (8 large,
  8 mid, 8 small), seed 20261001, 983 companies, no fabricated companies, no (company, month) pair from v1, v2 or v3.
  Pooled with v3: 36 per month, 1,189 companies.
- **Run:** the same six models and settings as §1; 9,648 calls, expected about $0.36 (v3 cost $0.18 for half as many).
- **R, replication (confirmatory, v4 alone):** the §3 test repeated unchanged on v4 alone: the same windows, score,
  refusal guard and Holm correction across the five models. v4 is a fresh sample, so this is an independent replication of the v3
  result (DeepSeek V3.1 confirmed; Llama 3.3 70B and GPT-4.1 Mini borderline).
- **E3b, the break month on v3 + v4 pooled (the purpose of the draw):** the §4 E3 method with 1,000 bootstrap draws,
  reported next to the v3 estimate, with the same check (interval inside cutoff − 6 months .. release + 3 months).
- **Pooled near-line test, E2 and E4 on v3 + v4:** descriptive only; the pooled p-values are not a new confirmation
  because half of the data was already analysed.
- **Stopping rule:** no further grid draws in this cycle, whatever the intervals show.
- **Commands:** `python -m evalbench.grid_report --versions v4` (R) and `--versions v3 v4` (pooled).

## 7. Addendum: a staggered check on the grid (exploratory)

**Date:** 2026-10-01. Written and committed before this analysis was computed (the v3 and v4 answers exist; this
comparison has not been run).

- **Why:** the replication (§6) compares each model's months before its cutoff with its months after release, about two
  years later. A drop could be a time effect (recent months harder for any model), not memory.
- **Idea:** the same questions answered by two models on different sides of their lines. Qwen3.8 Flash (stated cutoff
  2026-03, our provisional estimate; release 2026-08) is before its cutoff for all of 2025, while every other model is
  after its release for part of 2025.
- **Months, per other model E:**
  - *between:* after E's release month, up to Qwen's last whole month before its cutoff − 91 days (2025-11). E cannot
    know them; Qwen may.
  - *baseline:* E's registered BEFORE window (§2). Both models may know them.
- **Estimate:** per question, d = closeness(Qwen) − closeness(E). D = mean d (between) − mean d (baseline).
  - Memory predicts D > 0: Qwen keeps its level where E cannot know.
  - A pure time effect predicts D ≈ 0, because both models answer the same questions.
- **Test:** cluster bootstrap by company, 20,000 draws, two-sided, Holm across the five models. Also computed on
  questions both models answered (the refusal check). Data: v3 + v4 pooled.
- **Caveats:**
  - Qwen's cutoff is our estimate. If its knowledge ends earlier, D shrinks towards 0, so the check is conservative.
  - It is one reference model, so a positive D means "relative to Qwen".
- **Status:** exploratory; no confirmatory claim.
