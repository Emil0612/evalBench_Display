# Analysis plan: M12/M13, which price fits this month (multiple choice)

**Date:** 2026-10-01. Committed before any model answered these sets. Changes after that go in §6, dated.

**Why:** the recall tests ask a model to *produce* a price. That needs memory, retrieval in that format and willingness to
commit. gpt-oss-20b fails the recall positive controls (M9 6%, M7 57%) and refuses half the questions after its
release, so "none detected" there means "too hard", not "no memory". Recognition is easier than recall: models often
pick a fact from options that they cannot produce (evolveQA 2025; "Inside-Out: Hidden Factual Knowledge in LLMs",
2025). This set asks M5's question as a choice between four prices.

## 1. Design (and the flaws it answers)

| Flaw | Answer in the design |
|---|---|
| Knowing a share's usual level would solve a choice between arbitrary prices | All four options are the same share's real average close (as traded) in different months: only knowing *when* each price happened helps |
| Letter preferences (13–85% swings with option order; Llama and DeepSeek chose "A" 73–86% on M3) | Each question asked four times, the correct option in each position once (permutation balancing); the score is the share of the four answered right; chance 0.25. `one_letter` flags the same letter all four times |
| Elimination: a model that cannot know the month rules out options from months it knows | Wrong options come only from months 4–24 away (36 for calm shares); every question records their months; E4 below re-tests on questions whose options all lie after the model's release |
| Persistence: a share's price is usually close to last year's | Measured by the blind guesser (the option closest to the same month a year earlier) |
| Option tells (highest, lowest, middle, rounding) | The answer's rank among the four is drawn uniformly, and whether it is the option closest to the four's geometric mean is drawn with probability 1/4 (wrong options come from months around the answer, so in a trending share they would centre on its price); every pair of options at least 15% apart; all printed with the same number of decimals |
| Splits put an option on another price scale | No wrong option from the other side of a split |

**Blind-baseline audit** (no model; pass if every rule ≤ 0.35):

| Set | Highest | Lowest | Middle | Inner pair | Persistence |
|---|---|---|---|---|---|
| M12 `price_choice/v1` | 0.26 | 0.23 | 0.19 | 0.25 | 0.28 |
| M13 `known_choice/v1` | 0.21 | 0.18 | 0.22 | 0.30 | 0.22 |

**Sets:**
- **M12 `price_choice/v1`** (342a0edd): 745 questions from the 804 company-months of `price_level/v3` (the monthly grid;
  59 dropped for lack of distinct prices), large 237 / mid 253 / small 255. The same company-months as M5 v3, so recall
  and recognition can be compared question by question.
- **M13 `known_choice/v1`** (eaed089d): 131 of M9's 150 questions (the 15 largest Nordic companies, 2016–2022, before
  every cutoff).

**Controls** (free, run before this plan):
- **Oracle:** 1.00 before its cutoff, 0.26 after; Δκ +0.98.
- **Blind guesser:** 0.28 / 0.30, Δκ −0.03.
- **Random:** Δκ +0.05 (p 0.03) on the stored seed, but −0.03 to +0.03 (p 0.21–0.99) on four other seeds: chance.

## 2. Run
- **Models:** the six grid models, same pins and settings (forced answers, temperature 0, reasoning auto, standard privacy).
- **Stage 1, "MC validation (M13)":** 131 × 4 orderings × 6 = 3,144 calls, about $0.15.
- **Stage 2, "MC grid (M12)":** 745 × 4 × 6 = 17,880 calls, about $0.85. Listed-price bound for both: $2.45.
- Stage 2 runs whatever stage 1 shows; stage 1 decides how a model's stage 2 result is *read*.

## 3. Stage 1: is the format sensitive? (per model, no multiplicity correction: a validity check)
- **Sensitive** if the lower 95% bound (cluster bootstrap by company, 20,000 draws) of the mean M13 score is above
  both 0.25 and the blind guesser's mean on the same questions.
- **Recognition vs recall, reported next to it:** the same model's M9 scores on the same company-months (closeness,
  and "a price that occurred that month").
- **gpt-oss-20b:** sensitive on M13 while failing M9 recall → it holds price knowledge it does not produce, and M12 is
  the instrument for it. Not sensitive → it does not hold it in a usable form, and its M12 "none detected" means "too
  hard".

## 4. Stage 2: confirmatory test (M12)
- **H (per model):** the mean M12 score is higher in BEFORE than in AFTER.
- **Windows (as the monthly-grid plan, §2 there):** BEFORE = the 12 months ending with the last whole month before
  cutoff − 91 days; AFTER = the first 12 months after the release month (at least 6). Qwen3.8 Flash has no AFTER.
- **Test:** Δκ = (before − after) / 0.75; cluster bootstrap by company, 20,000 draws, two-sided; Holm across the five
  models with an AFTER window. Confirmed if the Holm p < 0.05, Δκ > 0, and the refusal guard holds (when answer rates
  before and after differ by more than 10 points, the effect must also be significant on answered questions only).
- **Reading:** a model that is not sensitive in stage 1 has a "none detected" read as "too hard for this model".

## 5. Exploratory (reported whatever they show)
- **E1, rolling curves:** M12 over time per model (6-month windows), next to M5 on the same months.
- **E2, recognition vs recall:** M12 vs M5 v3 per model and side. Per question: is "recognised" (3+ of 4) more common
  than "recalled" (M5 within ±10%)?
- **E3, company size:** M12 by size, before and after.
- **E4, elimination check:** the AFTER result again on questions whose three wrong options are also from months after
  the model's release (reported when at least 20 such questions exist).
- **E5, habits:** the share of `one_letter` and `consistent` answers per model.

## 6. Deviations
1. **2026-10-01, after stage 1 and before stage 2 was run: Qwen3.8 Flash is left out of stage 2.**
   - **Why:** it has no AFTER window (released 2026-08, after the data ends), so it cannot enter the confirmatory test,
     and it was not sensitive in stage 1 (0.284 [0.235, 0.341]).
   - **The other five models run as planned (14,900 calls).** gpt-oss-20b stays in by decision, although it
     was not sensitive in stage 1 (0.271 [0.206, 0.327]); its result is read as §4 says ("none detected" = too hard
     for this model).
   - **The Holm family is unchanged:** the five models with an AFTER window.

## 7. Addendum: a probability probe for gpt-oss-20b (exploratory; 2026-10-01, before any probe answer)
- **Why:** gpt-oss-20b was not sensitive in stage 1 by its chosen letters (0.271). Its letter habits (the same letter
  all four times on 18% of questions) may hide a weak preference. Its pinned endpoint (CoreWeave, fp4) returns token
  log-probabilities.
- **Sets:**
  - `known_choice_prob/v1` (4f6d7548): exactly the M13 questions;
  - `price_choice_prob/v1` (49033c11): exactly the M12 questions.
- **Score:** the probability the model gives the correct letter (the top-5 alternatives at the answer token,
  renormalised over A–D), averaged over the four orderings; chance 0.25. An ordering without probabilities counts its
  chosen letter fully, and the share scored from probabilities is reported.
- **Runs, gpt-oss-20b only, same pin:**
  1. **Smoke:** 1 question (4 calls) to confirm probabilities come back from a reasoning model.
  2. **M13p at the usual effort:** 524 calls.
  3. **M13p at high effort:** 524 calls.
- **Reading, the stage-1 rule:**
  - **Leans toward the right price:** the lower 95% bound (cluster bootstrap by company) above both 0.25 and the
    blind guesser.
  - **If so:** M12p for gpt-oss (2,980 calls) as a separate, newly registered test.
  - **If not:** gpt-oss does not hold this price knowledge in a form these tests can reach.
