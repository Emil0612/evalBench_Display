# Analysis plan: the cutoff ladder

**Date:** 2026-10-05. Sections are committed in order, each before the answers it uses are collected.

## 0. Micro-pilot and gate (committed before the pilot run)
- **Why:** an older model can fail the positive controls; then its "after release" answers say nothing (as with
  gpt-oss-20b). The ladder's confirmatory pairs rest on GPT-3.5 Turbo.
- **Run:** GPT-3.5 Turbo (`openai/gpt-3.5-turbo`, pinned to OpenAI), forced answers, temperature 0, on the two
  positive-control sets: M7 `known_return/v1` (98) and M9 `known_level/v1` (150). Neither set is used by any
  confirmatory test below.
- **Items that count:** dated on or before 2021-06 (the last whole month before GPT-3.5's cutoff 2021-09-30 − 91
  days): 70 of M7 (years 2016–2020), 118 of M9.
- **Gate (pilot E's E2 rule, against the blind guesser on the same items):** GO if GPT-3.5 is *sensitive* (the lower
  end of its cluster-bootstrap 95% interval, 20,000 draws, above the blind guesser's rate) on at least one of:
  M9 within the month's range, M9 within ±10%, M7 direction. Otherwise NO-GO: GPT-3.5 reads "too weak for these
  questions", family S below is not run, and only purpose B results for GPT-4o-mini and Llama 3.1 are reported.
- **Snapshot check:** every answer's `model_returned` must be `gpt-3.5-turbo-0125`. Then GPT-3.5's release for every
  analysis is 2024-01-25 (the snapshot), recorded in `config/model_releases.yaml`. Any other snapshot: stop, and decide
  the release date in a dated deviation before §1–§7 are written.
- **Feasibility, from stored answers only** (`python -m evalbench.ladder_feasibility`, before any new answer):
  - Two pairs have enough months: GPT-3.5 → Gemini 2.5 Flash (2024-02..2024-10) and GPT-3.5 → DeepSeek V3.1
    (2024-02..2024-11).
  - Projected power of the M15 correlation test, GPT-3.5 → Gemini: 94% if GPT-3.5's own correlation on the months
    it knows is 0.6, 33% if it is 0.3. GPT-3.5 → DeepSeek V3.1: 7% or less (DeepSeek's correlation on those months
    is +0.54 with a wide interval).
  - The price-level (M5) version has 1–13% power for both pairs: the references' own closeness also falls between
    the baseline and the between months.
  - Cost ceilings: micro-pilot $0.036 (248 calls); full run $0.98 (11,266 calls).

### Result (2026-10-05, after the pilot run, before any other answer of the new models)
- Run `5c0e93b7ed`: 248 of 248 answers, $0.016, every answer from OpenAI's endpoint.
- **Gate: GO.** GPT-3.5 Turbo is sensitive on all three controls (items on or before 2021-06):

| Control | n | GPT-3.5 [95% CI] | Blind guesser |
|---|---|---|---|
| M9 within the month's range | 118 | 29% [23%, 34%] | 14% |
| M9 within ±10% | 118 | 58% [45%, 71%] | 28% |
| M7 direction | 70 | 64% [55%, 73%] | 39% |

### Deviation 1 (2026-10-05, after the pilot, before §1–§7): the served-model check
- **What was seen:** `model_returned` is OpenRouter's own id, `openai/gpt-3.5-turbo`, on all 248 answers. OpenRouter
  does not pass on OpenAI's dated snapshot name, so the registered check ("every answer is `gpt-3.5-turbo-0125`")
  cannot be done from the stored answers.
- **Decision:** the check becomes "every answer came from the one model asked for" (met). GPT-3.5's release stays
  2024-01-25, the release of `gpt-3.5-turbo-0125`, the last GPT-3.5 Turbo snapshot and the one OpenAI's
  `gpt-3.5-turbo` alias has pointed to since 2024. Whichever snapshot served the answers, its release is on or
  before that date, so the date is conservative: it can only move months out of GPT-3.5's AFTER window, never
  into it. The listing date (2023-05-28) stays the sensitivity analysis E5.

## 1. What is run (committed before any answer of these models exists)
- **New models**, forced, temperature 0, one attempt, pinned:
  - GPT-3.5 Turbo (`openai/gpt-3.5-turbo`; cutoff 2021-09-30; release 2024-01-25, deviation 1);
  - GPT-4o-mini (`openai/gpt-4o-mini-2024-07-18`; cutoff 2023-10-31; release 2024-07-18);
  - Llama 3.1 70B (`meta-llama/llama-3.1-70b-instruct`, DeepInfra fp8; cutoff 2023-12-31; release 2024-07-23).
- **Sets** (the newest versions; the six current models answered them all):
  - M5 `price_level/v4` (1,608: 24 per month, 2021-01..2026-07);
  - M14/M15 `index_level/v1`, `index_return/v1` (479 each, 2016–2026; the first 24 items, the 2026-10-01 pilot, are
    excluded from every analysis);
  - M16/M17 `us_index_level/v1`, `us_index_return/v1` (512 each; PRIVATE, exploratory only);
  - M7/M9 for GPT-4o-mini and Llama 3.1 (GPT-3.5 answered them in §0).
- **Late references** (stored answers, no new calls): Gemini 2.5 Flash and DeepSeek V3.1, each detected on M5 v4
  and M15 in its registered report.

## 2. Lines and windows
- **BEFORE edge:** the last whole month before cutoff − 91 days. **AFTER edge:** the month after the release month.
- **A pair** (early model E, late reference L):
  - between months = from E's first AFTER month to L's last BEFORE month (L may know them, E cannot);
  - baseline months = months both may know (on or before both BEFORE edges).
- **A pair enters the confirmatory family** only with at least 6 between and 6 baseline months, and only if L was
  detected on that measure in its registered report. From the coverage check: GPT-3.5 → Gemini (2024-02..2024-10)
  and GPT-3.5 → DeepSeek V3.1 (2024-02..2024-11). Pairs that fail the rule are exploratory (§5).

## 3. Family S (confirmatory): is the gap memory, not a time effect?
- **S-M15:** D = (r_L − r_E) on between months − (r_L − r_E) on baseline months.
  - r is the Pearson correlation of answers with the true monthly change of the four Nordic indices.
  - It uses the questions both models answered.
  - Paired cluster bootstrap by month (both models' answers of a month resampled together), 20,000 draws,
    two-sided.
- **Holm** across the two pairs.
- **Reading:** "memory, not a time effect" when D > 0 and Holm p < 0.05.
- **The price-level version (M5) is exploratory (§5, E7), not confirmatory:** the feasibility check (§0) projects
  1–13% power for it, because the references' own closeness also falls between the baseline and the between months.
  Keeping it in the family would only lower the bar for S-M15.
- **What would change our reading:**
  - D ≈ 0 with a tight interval: the before/after gaps of the earlier reports are at least partly a time effect.
  - D < 0: the early model does *better* on months it cannot know. That points to a problem with the windows, not
    to memory.

## 4. Family R (confirmatory): do the new models remember?
- **Per new model:**
  - M15 correlation drop r_before − r_after (index plan §6.3 test: month-cluster bootstrap, 20,000 draws);
  - M5 closeness before vs after on `price_level/v4` (grid plan §2 windows, refusal guard).
- **Holm** across the new models within each measure, as in the earlier reports.
- **These give each model's backtest reading (purpose B):**
  - clean months in our data (GPT-3.5 31, GPT-4o-mini 25, Llama 3.1 25);
  - how strong its hindsight is inside its window.

## 5. Exploratory (reported whatever they show)
- **E1, US (M17):** S-M15 on `us_index_return/v1`. PRIVATE.
- **E2, short pairs:** GPT-4o-mini and Llama 3.1 against Gemini and DeepSeek V3.1 (3–4 between months), same
  estimators, no Holm.
- **E3, release effect:** Llama 3.1 vs Llama 3.3.
  - The same cutoff (2023-12), released 2024-07 and 2024-12.
  - Between = 2024-08..2024-11, after 3.1's release and before 3.3's.
  - Baseline = both models' BEFORE months.
  - D > 0 means knowledge arrives between cutoff and release (post-training), which supports the release line.
  - Precision differs (fp8 vs bf16), so this is a caveat on the result.
- **E4, GPT-3.5's grey zone** (2021-07..2024-01, in 6-month bins): its M15 correlation and M5 closeness per bin, next
  to its BEFORE and AFTER values. Can the grey zone be reclaimed for backtests?
- **E5, listing-date sensitivity:** S-M15 with GPT-3.5's release at its listing (2023-05-28). Every pair with
  ≥ 6 between months, separate Holm.
- **E6, clean windows:** per model, cutoff, release, clean months to 2026-08, the family R results.
- **E7, the price-level staggered test:** D = mean(closeness_L − closeness_E) on between months − the same on
  baseline months, per question on `price_level/v4` (refused = 0), cluster bootstrap by company, 20,000 draws, also
  on questions both answered. The two family S pairs, no Holm.

## 6. Cost and stopping
- **Cost:** ceiling $0.98 (11,266 calls) from the feasibility check; spend cap $1.00 per run.
- **Stopping:** no further draws or models are added after the answers are seen. Any extension is a new addendum,
  registered first, with its own stopping rule.

## 7. Deviations
Deviation 1 is in §0. No others yet.
