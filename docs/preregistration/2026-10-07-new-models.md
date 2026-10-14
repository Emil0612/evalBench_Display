# New models: Gemma 4 31B, Gemma 4 26B A4B, Qwen3.6 27B (registered before any of their answers)

Registered 2026-10-07, before these models answered any question. The analysis code is committed with this plan:
`src/evalbench/cutoff_estimate.py` (part 1) and `src/evalbench/new_models.py` (part 2). The earlier registered
reports and their test families are not changed: the new models form their own family.

## 1. Why these models

Three current open-weight models, so the same models can later be self-hosted:

- **Gemma 4 31B** (dense) and **Gemma 4 26B A4B** (mixture of experts, about 4B parameters active): the same maker,
  training data cutoff and host, so the pair compares a dense model with a mixture of experts. Gemma 4 also shares its
  cutoff with Gemini 2.5 Flash: a small open model against a large closed one from the same maker.
- **Qwen3.6 27B** (dense): a second maker at the same size.

## 2. Models, endpoints and dates

| Model | Pinned endpoint | Released | Knowledge cutoff |
|---|---|---|---|
| `google/gemma-4-31b-it` | Novita, bf16 | 2026-04-02 | 2025-01-31: the Gemma 4 model card ("a cutoff date of January 2025") |
| `google/gemma-4-26b-a4b-it` | Novita, bf16 | 2026-04-03 | 2025-01-31: the same statement, for the whole family |
| `qwen/qwen3.6-27b` | Alibaba | 2026-04-27 | none published: estimated in part 1 |

Release dates are OpenRouter's listing dates. All three run as the other models do: forced answers, temperature 0,
the least reasoning the model allows (reasoning is switched off; Qwen3.6 has it on by default).

## 3. Part 1: estimating Qwen3.6's cutoff (fixed before any answer)

- **Calibration set:** the US monthly-change questions (M17, `us_index_return/v1`, 512 questions), which are not part
  of the new models' Nordic tests.
- **Rule:** for each month t in the 36 months before the model's release, recall = the correlation of the answers
  with the true monthly moves over the 6 months ending at t, minus the same over the 6 months after t (answers that
  never vary count as r = 0). The changepoint is the t with the largest drop. Recall fades before a stated cutoff, so
  the estimate is the changepoint plus the median gap (changepoint − stated cutoff) of the reference models, rounded
  to whole months, as the end of that month, never after the release month. If the largest drop is below 0.3, the
  estimate falls back to the release month − 12 months.
- **Reference models:** those whose maker states a cutoff and that answered the US questions: GPT-4o-mini, Llama 3.1
  70B, Llama 3.3 70B, GPT-4.1 Mini, Gemini 2.5 Flash, gpt-oss-20b. Left out: GPT-3.5 Turbo (its served snapshot
  postdates its stated cutoff; cutoff-ladder plan §0, deviation 1), DeepSeek V3.1 (no official cutoff), Qwen3.8 Flash
  (our own estimate).
- **Fixed from the stored answers today:** median gap −7 months, so the offset is **+7 months**; the leave-one-out
  error over the six reference models is 2.3 months on average (`results/new_models/cutoff_estimate.md`).
- **Out-of-sample check:** the same rule applied to the two Gemma 4 models, whose cutoff is published. Reported, not
  used: their tests use the model card's cutoff.
- **Freezing:** the Qwen3.6 estimate is written into `config/model_cutoffs.yaml` (source "estimate") with a dated
  addendum below before any of its Nordic answers is analysed. Until then that file holds a provisional value
  (release − 12 months) so the app works.
- **A deviation already made:** the choice of 6-month windows and the 36-month search span was made on the stored
  answers of the reference models (US answers with 12-month windows, Nordic answers and pooled answers were also
  tried; the 6-month US version had the smallest error). The Gemma check is the honest test of that choice.

## 4. The runs

1. **Smoke test:** 3 questions of each method in the battery, all three models. Checks the answer format, that
   temperature 0 is accepted and that reasoning is off (answer lengths).
2. **Calibration:** M17, 512 questions per model.
3. **Battery:** M5 (`price_level/v4`, 1,608), M14 and M15 (`index_level/v1`, `index_return/v1`, 479 each), M18
   (`headline_date/v2`, 1,524), M11 (`monthly_return/v2`, 1,340), M19 (`index_forecast/v1`, 479), M20
   (`path_date/v1`, 424), the positive controls M7 (98), M9 (150) and M13 (131 × 4 orders), and the US checks M16
   (512) and M10 (150). Left out: M12 (at chance for every model so far; 2,980 calls per model) and the secondary
   methods that showed nothing (M1–M4, M6, M8). Runs use each method's newest set, so M18 uses v2 (v1 is not run).

## 5. Part 2: the tests (`python -m evalbench.new_models`)

Released in April 2026, the models have about four months after release in our data (to 2026-08). The registered
before/after-release tests need at least six; below that they read **insufficient**. The primary tests need no after
side. BEFORE = up to the last whole month before cutoff − 91 days, as everywhere.

- **P1, same months against a model that cannot know them** (the cutoff ladder's design): M15 correlation, new model
  minus GPT-3.5 Turbo on the months after GPT-3.5's release (from 2024-02) up to the new model's BEFORE edge, minus
  the same difference on months both may know; months resampled together (20,000 draws). Needs at least 6 months on
  each side. Memory predicts D > 0; a time effect shared by both cancels.
- **P2, the same on M5 closeness** (paired questions, companies resampled), with the refusal guard (the result must
  also hold on questions both answered). A price of 0 counts as not answered in that guard (scoring spec §10); it
  scores 0 in D, as a refusal does. Added after the smoke test (240 answers, before the battery): both Gemma models
  answered 0 to 3 of their 5 smoke-test price questions.
- **P3, recall inside the window:** the M15 correlation on BEFORE months against a month-shuffle null: each index keeps
  its own true moves, reassigned across months (20,000 shuffles), so knowing which index moves more earns nothing.
  One-sided p.
- **P4, M18:** model minus the text-only baseline on BEFORE months (`headline_date/v2`), as the headline-dating plan.
- Each of P1–P4: Holm across the three models; **detected** when Holm p < 0.05 and the effect is positive.

**The defences** (same families, Holm across the three):

- **M19 fake date:** the gate of the fake-date plan on its 24 pilot questions (answers at least 80%, and at least 50%
  get an informative forecast: not 0%, not the model's single most frequent value); then do the dated forecasts on BEFORE months track the real moves, against the month shuffle of P3?
  Leak share = forecast r ÷ the P3 recall r. This differs from the fake-date plan's test (a drop from before to after
  release), which needs an after side; on the stored answers the two can differ (GPT-4.1 Mini: drop not shown, but
  forecasts before release track the real moves, +0.21).
- **M20 masking:** the masking plan's measure (within one month, beyond the model's own answer habits) on paths
  ending before the BEFORE edge.

**Descriptive** (no test): the positive controls (the rule that gated GPT-3.5), Nordic recall beyond the US move
(partial correlation, BEFORE), news clearly dated through (the anchoring-free horizon on v2), M11, M14, M16/M10 as in
their reports, and the Gemma 31B vs 26B A4B comparison (side by side; no test of the difference).

**Not concluded from these runs:** that a model is clean anywhere inside its window. With few months after release, the
release-line tests wait until our data reaches further.

## 6. Deviations and addenda

- (§3) The window and span choice, as stated there.
- **Addendum 2026-10-07, after the calibration run (1fec81b6e2 + retry 038b13bead; 1,536 US answers), before the
  battery:** the rule gives Qwen3.6 27B a changepoint at 2024-02 (a drop above the 0.3 floor), so its cutoff is
  **2024-09-30**, now frozen in `config/model_cutoffs.yaml`. (The US recall values behind it are a private check and
  are not published here.) The out-of-sample check: both Gemma 4 models change at 2024-11 (drops above the floor),
  estimated 2025-06 against the model card's 2025-01, an error of +5 months each (the reference models' leave-one-out
  errors ranged from −1 to +8). Gemma's recall stays strong closer to its stated cutoff than the reference models'
  does, so the +7 offset overshoots it. The Gemma tests use the model card's date, as registered.
