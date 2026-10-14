# M18 confirmation: is news-dating memory about the world, not companies? (registered before any answer)

Registered 2026-10-06, after the set was built and before any model answered it. The analysis code
(`src/evalbench/headline_confirm.py`) is committed with this plan.

## 1. Why

In the headline-dating test (M18, `docs/preregistration/2026-10-05-headline-dating.md`), an exploratory split by the
listed company a headline is about (`results/headline/sizes.md`) found that models beat the text-only baseline only
on headlines about no listed company (macro, sector and editorial news): Gemini 2.5 Flash +0.158, Llama 3.1 70B
+0.101, DeepSeek V3.1 +0.099, Llama 3.3 70B +0.076, and nothing on company headlines of any size. It was found after
the registered test, so it is confirmed here on fresh headlines.

Checks on the v1 answers before this plan (exploratory): the no-company minus company contrast is positive for all
five models below (+0.07 to +0.12); it holds with the months within 6 or 12 of each model's habitual answer left out
(+0.09 to +0.14) and on exact-month hits beyond each model's own answer habits (+0.04 to +0.08), so it is not the
habitual month. A staggered check on the same answers (GPT-3.5 Turbo against Gemini and DeepSeek on 2024 headlines)
looked strong on the score but vanished on exact-month hits: there the habitual month did the work. That is why C2
below is required.

## 2. The set

- **`headline_date/v2`** (md5 `08f14c6b31ee4df97ee364697268e24f`): 1,524 headlines, none in v1; in each of the 127
  months 2016-01..2026-07, 6 headlines that name a listed company (news index `insrefs`) and 6 that name none; seed
  20261006. Same filters, prompt (p1), scoring (1 − months off / 12) and text-only baseline (a bag-of-words ridge
  trained on other headlines, without the test month and its neighbours) as v1. 1,069 Swedish, 455 English.
- **Models:** GPT-3.5 Turbo, Llama 3.1 70B, Llama 3.3 70B, Gemini 2.5 Flash, DeepSeek V3.1 (the four with the
  effect in v1, and GPT-3.5 Turbo, which showed it too); forced answers, temperature 0, pinned endpoints as in v1.
- **Windows:** as the registered M18 test: the months on or before the last whole month before the stated cutoff
  minus 91 days.

## 3. Hypothesis and test (per model)

- **C1:** on the window's headlines, mean(model score − text-only score) on headlines naming no listed company minus
  the same on headlines naming one. A refusal scores 0, as in the registered M18 test.
- **C2:** the same contrast on exact-month hits minus the model's own share of answers naming that month (over all
  its v2 answers), leaving out headlines from the months next to its habitual answer (±1). This removes credit a
  model earns by answering its favourite month.
- **Statistics:** months resampled with replacement (20,000 draws, both groups together); two-sided bootstrap p
  (`stats.boot_p`); Holm across the five models, separately for C1 and C2.
- **Confirmed for a model** if C1 > 0 and C2 > 0, both Holm p < 0.05, and C1 also holds on the answered headlines
  alone (Holm p < 0.05, positive): the refusal guard.
- **Overall:** the finding is confirmed if at least 3 of the 5 models confirm. With fewer, it is reported as "not
  confirmed" with each model's numbers, and the v1 split stays exploratory.

## 4. Reported alongside (fixed now, not tests)

- Each group's model − text-only on its own, per model.
- Answer rates; the company headlines split by market-cap tercile (exploratory).

## 5. What would change our reading

- C1 confirmed but not C2: the contrast rides on the habitual month; reported as not confirmed.
- A model with a low answer rate on one group: read with the answered-only figure.

## 6. Deviations

### 2026-10-06, before any model answered v2: three clarifications after a code review

- **Refusals in C2:** a refused or unanswered headline is left out of C2, since it names no month to hit (counting
  it as a miss would push C2 up when one group is refused more, without any memory). C1 keeps the registered M18 rule
  (a refusal scores 0), with the answered-only check of §3.
- **Holm over the five named models:** a model without answers enters Holm with p = 1 and is not confirmed, so the
  family is always the five models of §2 and "3 of 5" keeps its meaning.
- **§4 in the report:** answer rates are shown per group (no company / listed), and the company headlines' model −
  text-only per market-cap tercile.

Checked on simulated answers before the run: with a planted effect all five models confirm; with none, or with
refusals concentrated on company headlines, none does.
