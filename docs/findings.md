# What do language models remember about Nordic markets? Findings so far

**Date:** 2026-10-14. Covers every run from 2026-09-29 to 2026-10-14.
**Status:** method pilots on hosted models through OpenRouter. These are results on hosted models whose weights and
serving cannot be verified: diagnostic, not certified, and nothing here says that any model or period is "safe".
**Detailed numbers:** the study plans in `docs/preregistration/`, each with its results appended (listed at the end).

---

## In short

- **Models remember how the market moved.** Asked how a Nordic index changed in a given month, four of five models
  track the real moves closely for months they trained on (correlation up to +0.8), and not at all after their
  release. US indices, checked privately, are remembered even better.
- **They remember single Nordic share prices only roughly.** Every model that answers is a little closer to the true
  price in months it trained on (4 of 5 detected), but the answers are far off in absolute terms.
- **The memory is coarse.** Models know roughly where a share traded over a period, not which month a price belongs to.
- **Famous beats obscure.** US shares and indices (a private check) are remembered far better than Nordic ones; large
  Nordic companies better than small ones; the main Stockholm index better than the others.
- **The stated training cutoff is not a safe line.** Knowledge often thins in the months before it and can reach past
  it, and about half of the before-to-after drop happens between the stated cutoff and the release (exploratory). The
  model's release date is the earliest defensible boundary for a backtest.
- **It is memory, not "recent months are harder".** On the same 2024 months, a model that may have trained on them
  (Gemini) still tracks the market while one that cannot (GPT-3.5) gives a stock answer; on months both may know,
  both track it (+0.79 and +0.55).
- **Models also remember when Nordic news happened.** Asked to date a headline with its dates removed, Gemini and
  Llama 3.1 beat a text-only baseline inside their training windows. It is the first text-based test that finds
  memory. It is mostly memory of world and macro news, not company news. Models clearly date news through about
  their cutoff year (GPT-3.5 into 2022, a little past its 2021-09 cutoff; the Llamas 2023; Gemini and DeepSeek 2024).
- **They do not know which shares beat the market.** Asked how a single share moved in a given month, scored against
  that month's median share, every model is at chance, even inside its training window (at most about 3–6 points
  above chance, on 1,340 questions).
- **Models show where their knowledge ends.** Forced to give a number, several models answer "0%" for months they
  do not know, and the switch lines up with their cutoff (GPT-4o-mini: 16% → 97% of answers around its 2023-10
  cutoff). A check registered before it was computed agrees on the three newest models.
- **Telling a model the date does not stop its hindsight.** Told "today is the end of last month", Llama 3.3 and
  Gemini still forecast the next month's index move with a correlation of +0.40 on months they trained on: about
  half of what they recall when asked directly.
- **Hiding the dates helps, but is not a complete defence.** Shown an unlabeled stretch of index changes, four of
  five models cannot name the month; Gemini partly can, most of all around the 2020 crash. At a coarser precision
  (an after-the-fact re-score), GPT-4.1 Mini places the paths within about 3 months and DeepSeek within a year.
- **Three new open-weight models show the same pattern.** Gemma 4 31B, Gemma 4 26B A4B and Qwen3.6 27B all recall the
  Nordic market's moves inside their window; on 2024 months both Gemma models track the market while GPT-3.5 cannot
  (memory, not a time effect), and Gemma 4 31B and Qwen3.6 keep two thirds of their recall when told the date.

**For backtesting:** a model asked about dates inside its training window carries real hindsight, strongest for the
market as a whole and for years well inside the window. Any backtest that lets a model see such dates inherits it, and
telling the model the date does not remove it. Company-specific monthly moves, the signal a stock-picking strategy
trades on, show no measurable leak.
Older models give longer clean windows in our data: GPT-3.5 Turbo 31 months, against 12–20 for current models.

---

## How we measure memory

A model scoring well on a question proves nothing on its own: it may simply guess well. So every test compares the
same model on two sets of matched questions:

- **Before:** months well before the model's stated training cutoff (more than 91 days), where it may have read the
  answer.
- **After:** months after the model's release, where it cannot have.

If it does clearly better *before* than *after*, it remembered. The months between cutoff and release are left out.

Three more safeguards run through everything:

- **A blind guesser** that uses no memory (for prices: "the same as a year earlier"). Where it matters, a model has to
  beat it, not just chance.
- **Positive controls:** questions every model should know (the largest companies, 2016–2022). A model that fails
  these can't show memory anywhere, so its blanks elsewhere mean "too hard", not "no memory".
- **A refusal guard:** a refused question scores 0. If a model refuses mostly recent questions, that alone makes
  "before" look better. A result only counts if it also holds on the questions the model actually answered.

Every confirmatory test was written down and committed before its answers were collected; later changes are recorded
as dated deviations in the plans.

---

## The models

| Model | Stated cutoff | Release | Notes |
|---|---|---|---|
| Llama 3.3 70B | 2023-12 | 2024-12 | |
| GPT-4.1 Mini | 2024-06 | 2025-04 | |
| gpt-oss-20b | 2024-06 | 2025-08 | fails the positive controls (see below) |
| Gemini 2.5 Flash | 2025-01 | 2025-06 | often refuses recent dates |
| DeepSeek V3.1 | 2025-03 | 2025-08 | cutoff not officially published |
| Qwen3.8 Flash | 2026-03 (our estimate) | 2026-08 | too recent: no "after" months in our data |
| GPT-3.5 Turbo | 2021-09 | 2024-01 | added 2026-10-05; release = the snapshot it serves (listed 2023-05) |
| GPT-4o-mini | 2023-10 | 2024-07 | added 2026-10-05 |
| Llama 3.1 70B | 2023-12 | 2024-07 | added 2026-10-05; fp8 |

Each model was pinned to a single provider endpoint, at temperature 0, with forced answers.

---

## Finding 1: models remember the market's path (the strongest result)

**The question:** "By how much did the OMXS30 index change in March 2023?" Asked for every month from 2021 to 2026, on
four Nordic indices.

**The test:** how strongly the model's answers move with the real monthly changes, before vs after release.

| Model | Correlation with the real moves: before → after | Reading |
|---|---|---|
| Gemini 2.5 Flash | +0.80 → −0.08 | **detected** |
| DeepSeek V3.1 | +0.79 → −0.02 | **detected** |
| Llama 3.3 70B | +0.67 → −0.05 | **detected** |
| GPT-4.1 Mini | +0.46 → +0.10 | **detected** |
| gpt-oss-20b | +0.18 → +0.03 | none detected |

**Is it Nordic, or just the world market?** Nordic and US markets move together, so a model that only knows the US
path would look right on Nordic months too. Removing each month's US move leaves:

- **Gemini (+0.41), DeepSeek (+0.37) and Llama (+0.28):** genuinely Nordic memory.
- **GPT-4.1 Mini (+0.04):** nothing left. Its Nordic result is the global market's path.

**The three older models** (cutoff ladder, family R): GPT-4o-mini +0.44 → 0.00, detected. GPT-3.5 Turbo and Llama 3.1
70B also track the market before their cutoff (+0.55 each), but their drop after release is not shown: they give few
real answers after release, and Llama 3.1 keeps +0.29. The backtest report marks them "recall shown".

**US for comparison** (a private check whose data cannot be published): US index moves and levels are recalled more
closely than Nordic ones.

**Index levels:** almost every model knows the OMXS30's level well, but several confuse the Stockholm all-share index
with the OMXS30 or get the Helsinki index's scale wrong. Only Llama and Gemini know all four Nordic indices.

**What it means:** in the months before its cutoff, a model's guess about the market correlates up to 0.8 with what
actually happened. After release this disappears and models fall back to a stock answer (0%, or
+2 to +2.5%). Any strategy with market exposure is at risk of this hindsight.

---

## Finding 2: single share prices are remembered, but only roughly

**The question:** "What was the average price of [company] in [month]?" Scored by closeness: 1 = exact, 0 = off by a
factor of 2 or more.

**The test:** a monthly grid, 12 questions in every month from 2021 to 2026, drawn twice (v3, then a fresh v4 as the
registered replication).

| Model (v4 replication) | Closeness before → after | Difference [95% CI] | Reading |
|---|---|---|---|
| Gemini 2.5 Flash | 0.280 → 0.157 | +0.12 [+0.07, +0.18] | **detected** ¹ |
| Llama 3.3 70B | 0.250 → 0.156 | +0.09 [+0.04, +0.15] | **detected** |
| DeepSeek V3.1 | 0.287 → 0.196 | +0.09 [+0.03, +0.15] | **detected** |
| GPT-4.1 Mini | 0.233 → 0.164 | +0.07 [+0.02, +0.12] | **detected** |
| gpt-oss-20b | 0.099 → 0.039 | +0.06 [+0.03, +0.09] | driven by refusals |

¹ Gemini refuses more often after its release (answer rate 90% → 76%) but stays detected on the questions it answered
(+0.11). This uses the re-read of 2026-10-05, which turned 73 stored Gemini refusals into refusals; no verdict changed.

- **In the app:** the Overview corrects across every model shown (nine), stricter than the replication's five; GPT-4.1
  Mini's +0.07 then reads Holm p 0.054. With the five registered models selected on Evidence it reads detected.
- **Consistent:** same direction and similar size on both draws and across four model families. DeepSeek's memory
  was detected in three separate runs (pilots D and E, and the grid).
- **Not about harder months:** measured against the blind guesser on the same questions, the before–after gap stays
  for all five models (significant for all five on v4). The blind guesser actually finds the *later* months easier.
- **But weak in absolute terms:** the blind guess "same month last year" is closer than every model. Only 4–6% of
  answers name a price that actually occurred that month.
- **Stale levels, not monthly prices (checked 2026-10-06, free):** after release, models repeat prices from around
  their knowledge edge, and even before the cutoff an answer is about as close to the price a year earlier as to the
  month asked. Looser scoring (within 10% or 30%) makes the test weaker, not stronger; no scoring of the month's own
  move finds anything.
- **A price of 0 is "I don't know" (scoring spec §10, 2026-10-07):** gpt-oss-20b answers 0 to 41% of its questions
  after release; its gap came from those answers (+0.042 → −0.007 without them).

**Size matters** (closeness before the cutoff, four main models):

| Company size | Closeness |
|---|---|
| Large caps | 0.34–0.51 |
| Mid caps | 0.23–0.29 |
| Small caps | 0.09–0.17 |

**Large vs ordinary Nordic shares** (same question format, before the cutoff; share of answers that name a price that
actually occurred that month):

| | Share of answers |
|---|---|
| The 15 largest Nordic companies | 16–29% |
| Ordinary Nordic listings | 4–11% |

US large caps, asked the same question in a private check, are recalled far better than both: prominent securities
are recalled well, a Nordic segment much less.

---

## Finding 3: the memory is coarse, not month-specific

**The question:** "Which of these four prices was [company]'s average in [month]?" All four are real prices of the
same share, from other months, so knowing its usual level doesn't help; only knowing *when* does.

- **On the ordinary grid: nothing, for any model.** Everyone scores at chance (0.23–0.29 against 0.25), before and
  after.
- **On the 15 largest companies:** four models do better than chance (Gemini 0.47, DeepSeek 0.35, Llama 0.33,
  GPT-4.1 Mini 0.31).

**What it means:** models know roughly where a share traded over a period (its level or regime), not monthly price
paths. For ordinary companies their recalled price is typically off by a factor of about 3, far too coarse to tell
apart prices 15–40% apart. For numbers, choosing between options is harder than naming one.

---

## Finding 4: memory fades gradually, and can reach past the stated cutoff

- **DeepSeek's** memory drops at its **release** (2025-08), not at its stated cutoff (2025-03).
- **Llama, GPT-4.1 Mini and gpt-oss** scored as well or better on months between their cutoff and release as before
  the cutoff in pilot E (22–38 questions each); the larger grid below shows a partial fade instead.
- **What the over-time charts show (checked 2026-10-07, exploratory):** part of the shape comes from the charts, part
  from the market, and part is real.
  - *The charts:* rolling windows make a drop start 2 (M5) to 5 (M15) months early; M15 bumps after release rest on a
    handful of answers.
  - *The market:* the slow M5 fade over the years before the cutoff is mostly the calendar (every model, including
    those that trained on them, answers 2022–23 prices worse than 2021 prices; with the calendar held fixed, M5 is
    flat before the cutoff); the slow slide after the cutoff is remembered prices going stale; M15's wiggles are
    market turbulence.
  - *Real, with a known cause:* knowledge thins before the stated cutoff, because recent months are thinly covered
    when training data is collected; an independent benchmark finds the same early drops for the same models.
  - *Open:* market knowledge ends 4–6 months before the stated cutoff for Gemini, DeepSeek and Llama 3.3, beyond the
    chart effect; some M5 drops between cutoff and release are steeper than stale prices predict; one model's edges
    differ by kind of data; Llama 3.3 answers "0%" early; mid caps fade while large and small caps do not.
- A single break month can't be pinned down: tripling the data did not narrow the estimates.
- **A first staggered check** (the same months answered by a later model) pointed the right way for DeepSeek and
  Gemini but was inconclusive, because the later reference model (Qwen) itself barely knows 2025. Finding 5 repeats it
  with an earlier model instead.
- **Between cutoff and release** (all nine models, exploratory): 40–65% of the before → after drop in price closeness
  happens inside this gap for most models. Its first half still looks mostly like "before", its second half like
  "after".
- **Market-move memory is strongest for older years.** Over the last 12 months before the cutoff, only Gemini, DeepSeek
  and GPT-3.5 still track the market clearly.

**What it means:** use the **release date** as the line for backtests, not the stated cutoff.

---

## Finding 5: it is memory, not a time effect (the cutoff ladder)

**The worry:** every before/after comparison sets months about two years apart. If recent months were simply harder,
that alone would produce a gap.

**The test (registered before the answers were collected):** an *early* model that cannot know some months (GPT-3.5
Turbo, released 2024-01) against a *late* model that may (Gemini 2.5 Flash), on the same questions about Nordic index
moves. A time effect would hit both models; memory only the one without the data.

| Months | Gemini tracks the real moves | GPT-3.5 tracks the real moves |
|---|---|---|
| Both may know (2016–2021) | +0.79 | +0.55 |
| Only Gemini may know (2024-02..2024-10) | +0.80 | −0.25 (a stock answer, −2.5%, on 26 of 27 questions) |

- **Difference of differences: +0.81 [+0.24, +1.10], Holm p 0.016: memory, not a time effect.**
- **Caveats:** it rests on 20 paired answers (Gemini answers 74% of those questions). One GPT-3.5 answer carries about a
  quarter of the size; without it the result is +0.55, p 0.028.
- **A news version cannot back it up (checked 2026-10-06, exploratory):** on the same 2024 months, Gemini's and
  DeepSeek's lead over GPT-3.5 Turbo in dating headlines looked large (+0.20 and +0.27), but it vanished on exact-month
  hits beyond each model's own answer habits (−0.02 and −0.01): Gemini's habitual month (2023-11) lies near 2024, and
  its knowledge of news mostly ends around 2023–24. So the index result stays the only staggered evidence.
- **On more months (registered extension, 2026-10-06): not shown.** Of three later-cutoff candidates only DeepSeek
  V3.2 passed the gate, and on its 15 months D was −0.31 [−1.08, +0.62]: its market memory ends around October 2024,
  not at its stated cutoff (July 2025), so the extra months were months it does not know. On the original 9 months it
  behaves like Gemini (+0.68, p 0.056; exploratory). The Gemini result stands as it was
  (`docs/preregistration/2026-10-06-ladder-extension.md`).
- **Against DeepSeek V3.1:** the same direction (+0.52), not significant. The free feasibility check had predicted low
  power for this pair.
- **The new models on their own:** GPT-3.5 remembers price levels (+0.09 closeness); GPT-4o-mini remembers market
  moves (+0.44 correlation); Llama 3.1 leans the same way without significance.

**Clean backtest windows in our data** (whole months after release, to 2026-08):

| GPT-3.5 Turbo | GPT-4o-mini | Llama 3.1 70B | Llama 3.3 70B | GPT-4.1 Mini | Gemini 2.5 Flash | DeepSeek V3.1 |
|---|---|---|---|---|---|---|
| 31 | 25 | 25 | 20 | 16 | 14 | 12 |

---

## Finding 6: models remember when Nordic news happened (headline dating, M18)

**The question:** "In which year and month was this headline published?" Millistream headlines, Swedish and English,
with every date cue removed. 12 per month from 2016 to 2026. Compared with a model-free baseline that dates headlines
from their words alone (median 24 months off).

**The registered test:** inside its training window, is the model closer to the true month than the baseline?

| Model | Better than the text alone, d (score 0–1) | Reading |
|---|---|---|
| Gemini 2.5 Flash | +0.107 | **detected** (holds without its default-month headlines: +0.077) |
| Llama 3.1 70B | +0.053 | **detected** (holds: +0.053) |
| DeepSeek V3.1 | +0.068 | detected, but it rests on its default month (+0.009 without it) |
| the other six | −0.025 to +0.038 | not shown |

- **Exact months:** inside their windows, models name the exact month of 1–8% of headlines (Gemini 8%) against the
  baseline's 1–2%; after release, 0% for every model.
- **When unsure, models fall back on a default month:**
  - GPT-3.5 Turbo and GPT-4o-mini use **exactly their stated cutoff month** (2021-09, 2023-10).
  - The newer models use 2023, one to two years before their stated cutoffs.
- **News clearly dated through** (exact months beyond the model's own answer habits, both headline sets pooled):
  GPT-3.5 2022, GPT-4o-mini 2022, both Llamas 2023, GPT-4.1 Mini 2022, Gemini and DeepSeek 2024, Qwen 2025. Where the
  knowledge stops is not measured: models almost never name a year after the one they treat as "now", and a model
  cannot hit a year it never names.
- **The memory is about the world, not companies (exploratory, 2026-10-06, `python -m evalbench.headline_sizes`):**
  split by the listed company a headline is about, models beat the text-only baseline only on headlines about no
  listed company (macro, sector and editorial news, about half the set): Gemini +0.158, Llama 3.1 +0.101, DeepSeek
  +0.099, Llama 3.3 +0.076 (Holm p ≤ 0.007). On company headlines, large, mid or small, no model does; large-cap
  news is even slightly below the baseline for most. The opposite of prices, where large caps are remembered best.
  Found after the registered test, so a lead to confirm, not a detection. It survives the habit checks (months within
  12 of each model's habitual answer left out: +0.09 to +0.14; exact-month hits beyond its own answer habits: +0.04 to
  +0.08 for the no-company minus company contrast).
- **The registered confirmation on 1,524 fresh headlines (2026-10-06): not confirmed overall (2 of 5 models; the plan
  needed 3).** Gemini 2.5 Flash (+0.111) and DeepSeek V3.1 (+0.082) confirm on both measures (p ≤ 0.001); GPT-3.5
  Turbo, Llama 3.1 and Llama 3.3 point the same way (+0.04–0.05) but not significantly on the score contrast. So: the
  models with the strongest news memory remember world events far better than company news; for the others the
  pattern is weaker (`docs/preregistration/2026-10-06-headline-macro.md`).
- **The registered horizon estimate failed:** it assumed answers follow the true date, and they mostly sit near the
  default. It is reported as uninformative, not replaced.

**What it means:**
- Memory reaches news events, not only prices and market moves, though coarsely.
- For backtests, a model clearly knows news through about its stated cutoff year, sometimes a little past it
  (GPT-3.5); the release date stays the safe line. Its default answer month is a cheap hint of where it thinks "now" is.

---

## Finding 7: no memory of which shares beat the market (M11 v2)

**The question:** "By how many percent did [company]'s share price change in March 2023?" Scored 1 if the answer puts
the share on the right side of that month's median Nordic share, so knowing only the market's direction earns nothing
and chance is 0.5. This is the question a stock-picking strategy trades on.

**The test (registered before the answers were collected):** a larger set, 1,340 questions (1,206 real companies),
9–28 per month from 2021 to 2026; before vs after release for nine models.

| Model | Score before the cutoff | Reading | Upper bound (v2) | Pilot E (v1, 300 questions) |
|---|---|---|---|---|
| GPT-3.5 Turbo | 0.457 | none detected | +0.15 | not asked |
| GPT-4o-mini | 0.489 | none detected | +0.05 ¹ | not asked |
| Llama 3.1 70B | 0.494 | none detected | +0.13 | not asked |
| Llama 3.3 70B | 0.498 | none detected | +0.05 ¹ | at most +0.38 |
| GPT-4.1 Mini | 0.493 | none detected | +0.07 ¹ | at most +0.39 |
| Gemini 2.5 Flash | 0.477 | refusal-driven (answers 94% before, 38% after release) | – | refusal-driven |
| DeepSeek V3.1 | 0.497 | none detected | +0.21 | at most +0.63 |
| gpt-oss-20b | 0.504 | none detected (fails the positive controls) | +0.25 | at most +0.71 |
| Qwen3.8 Flash | 0.515 | no questions after its release | – | same |

¹ Too narrow: see "0% answers" below.

**"0%" answers:** forced to give a number, Llama 3.3, GPT-4o-mini and GPT-4.1 Mini answer exactly 0% to 74%, 51% and
28% of the questions, almost only for months after their cutoff (after release: 100%, 99%, 89%). An answer of 0%
scored 0.5 (chance), so their after side was nearly all 0.5 and their bounds looked tighter than the evidence. On the
answers that name a direction, every model scores 0.46–0.52 before its cutoff, every 95% interval includes 0.5, and
the upper ends are 0.53–0.56: at most about 3–6 points above chance. A no-memory rule of thumb ("large companies beat
the median") already reaches about 0.55 on this set (55% of large caps beat the median against 43% of small ones).

**A rule change followed (scoring spec §7, 2026-10-06):** on M6 and M11 an answer of exactly 0% now counts as "I don't
know" and is left out of the app's scores, as refused questions are; the registered reports keep their own rule.
Scoring it 0 instead would turn "I don't know" into a false detection, as refusals did before the refusal guard. Two
of the old M6 detections (Llama 3.3, GPT-4.1 Mini) turned out to compare answers before the cutoff with mostly 0%
answers after it.

**The 0% answers as a horizon signal (exploratory):** how often a model answers 0% jumps where it stops claiming to
know (Evidence → Over time → Answers of 0%):

| Model | Stated cutoff | Share of answers that are 0%, per half-year |
|---|---|---|
| GPT-4o-mini | 2023-10 | 16% (2023-H1) → 50% (2023-H2) → 97% (2024-H1) |
| GPT-4.1 Mini | 2024-06 | 3% (2024-H1) → 28% (2024-H2) → 91% (2025-H1) |
| Qwen3.8 Flash | 2026-03 | 37% (2025-H2) → 95% (2026-H1) |
| Llama 3.3 70B | 2023-12 | 23% (2022-H2) → 82% (2023-H1) → 98% (2023-H2) |

Gemini 2.5 Flash, GPT-3.5 Turbo and gpt-oss-20b never answer 0%, so they give no signal here.

**Registered on three new models (2026-10-14):** the same switch, written down before it was computed for Gemma 4 31B,
Gemma 4 26B A4B and Qwen3.6 27B, lands in the half-year of each model's cutoff or the next one, inside the window fixed
from the older models (Finding 10; `docs/preregistration/2026-10-14-zero-answer-horizon.md`).

**What it means:**
- Market-wide memory (Finding 1) does not extend to which shares beat the market in a month.
- For backtests, a month-by-month stock-picking strategy gets very little from leaked returns on this measure.
  Leaks through the market's direction, price levels and news remain, and "none detected" never makes a period clean.

---

## Finding 8: telling a model the date does not stop its hindsight (M19, fake-date test)

**The question:** M15's index questions asked as forecasts made at the end of the previous month ("Today is 31 March
2023. You only know information that was public on or before today [...] how will the OMX Stockholm 30 change during
April 2023?"). An honest forecaster cannot track single months.

**The test (registered before the answers):** the forecasts' correlation with what happened, before the cutoff vs
after the release, for the models that actually give varied forecasts.

| Model | Forecast correlation before → after | Reading | Share of its plain recall kept |
|---|---|---|---|
| Llama 3.3 70B | +0.40 → −0.03 | **leaks** (Holm p 0.017) | 60% |
| Gemini 2.5 Flash | +0.40 → +0.05 | **leaks** (Holm p 0.021) | 50% |
| GPT-4.1 Mini | +0.21 → +0.01 | not shown | 46% |
| GPT-4o-mini, DeepSeek V3.1 | – | did not forecast (mostly +1.2 to +2.5%) | – |

**What it means:** a date in the prompt removes about half of a model's market hindsight, not all; models that stop
forecasting give no signal at all. A date instruction is no substitute for using months after the model's release
(`docs/preregistration/2026-10-06-fake-date.md`).

---

## Finding 9: hiding the dates helps, but is not a complete defence (M20, masking test)

**The question:** the 12 monthly changes of an unnamed Nordic index, with no dates, no name and no levels: "in which
month does this period end?" Scored as hits within a month beyond the model's own answer habits (about 0 without
memory; the oracle control scores +0.97, the random and blind controls 0).

| Model | Within a month (raw) | Beyond its habits | Reading |
|---|---|---|---|
| Gemini 2.5 Flash | 10% | +0.035 (Holm p 0.016) | **dates the unlabeled path**, most on paths through March 2020 |
| GPT-4.1 Mini, Llama 3.3, DeepSeek V3.1 | 1–4% | about 0 | not shown |
| GPT-4o-mini | 0% (names only 3 months) | 0 | does not try |

- **At a coarser precision (an after-the-fact re-score, exploratory):** GPT-4.1 Mini places the paths within ±3 to
  ±6 months beyond its habits (Holm p 0.026 and 0.042) and DeepSeek within ±12 months (p 0.034); Gemini gets stronger
  the coarser the window.

**What it means:** unlike a date in the prompt (Finding 8), masking removes much of the hindsight at month precision:
four of five models cannot name the month of an unlabeled market path. But it is not a complete defence: the strongest
market memory (Gemini) recognises some periods, most of all the 2020 crash, and two more models place paths within a
few months to a year. Whether a model *uses* masked data in a forecast is not tested yet
(`docs/preregistration/2026-10-06-masking.md`).

---

## Finding 10: three new open-weight models (Gemma 4 31B, Gemma 4 26B A4B, Qwen3.6 27B)

Added 2026-10-07 and registered before their answers (`docs/preregistration/2026-10-07-new-models.md`). Released in
April 2026, they have only four months after release in our data, so their primary tests need no after side.

| | Gemma 4 31B (dense) | Gemma 4 26B A4B (mixture of experts) | Qwen3.6 27B (dense) |
|---|---|---|---|
| Same 2024 months vs GPT-3.5 (M15) | **+0.82**, p 0.015: memory | **+0.91**, p 0.004: memory | too few months |
| Recall inside the window (vs month shuffle) | **+0.52** | **+0.44** | **+0.47** |
| Told the date (M19) | **leaks**, keeps 66% | did not forecast | **leaks**, keeps 70% |
| Masking (M20) | not shown | not shown | not shown |
| Prices (M5), news (M18) | not shown | not shown (M5 refusal-driven) | not shown |

- **Qwen3.6's cutoff** is not published; estimated from its US-market answers at 2024-09 by a rule fixed in advance.
  On the two Gemma models, whose cutoff is published (January 2025), the same rule lands 5 months late.
- **Dense vs mixture of experts** (descriptive): the dense 31B recalls a little more, has more Nordic-specific memory
  beyond the US move (+0.16 vs +0.08) and dates news a year later; on the same-months test the 26B is as strong. A
  lean towards the dense model, not a clear difference.
- **Same maker, same cutoff:** Gemma 4 31B recalls less than Gemini 2.5 Flash overall (+0.52 vs +0.80 on Nordic
  moves), but both pass the same-months test on 2024 with similar size.
- **They know how the market moved, not where things stood** (descriptive): Nordic index levels and share prices are
  largely unknown to them (index-level closeness 0.11–0.17 against 0.63–0.70 for Gemini and Llama 3.3; the 15 largest
  Nordic shares within 10% for 13–25%; US shares, in a private check, far more often), and the Gemma models answer 0
  ("I don't know") to about 30% of share-price questions. Stock picking (M11) is at chance, as for every model.
- **Calibration lesson:** the cutoff estimator overshot both Gemma models by 5 months, because they keep their
  knowledge almost to their stated cutoff; the usual 7-month fade is typical, not universal.
- **Where they stop answering (registered 2026-10-14, `docs/preregistration/2026-10-14-zero-answer-horizon.md`):** on
  the market (M15) and on single-share returns (M11), all three switch to "0%" or uninformative answers in 2025-H1:
  the half-year of the Gemma models' cutoff, one after Qwen3.6's estimated cutoff. Every switch the plan counted falls
  inside the window fixed in advance from the older models (−1 to +2 half-years); Qwen3.6's market switch is uncertain
  and was not counted. The check holds; it does not tell whether Qwen3.6's estimate is early or late.

**What it means:** the market-memory result is not specific to the first nine models; current open-weight models of
about 30B carry it too, and a date in the prompt protects even less with them
(`docs/preregistration/2026-10-07-new-models.md`).

---

## Backtest readiness per model

One table per model from the registered results (`uv run python -m evalbench.readiness` regenerates the full
report). A model is clean for a backtest only after its release.

| Model | Clean from | Clean months to 2026-08 | Weights | Hindsight inside its window |
|---|---|---|---|---|
| GPT-3.5 Turbo | 2024-02 | 31 | API only | price levels, market moves (recall shown) |
| GPT-4o-mini | 2024-08 | 25 | API only | market moves |
| Llama 3.1 70B | 2024-08 | 25 | open | market moves (recall shown), news dating |
| Llama 3.3 70B | 2025-01 | 20 | open | price levels, market moves |
| GPT-4.1 Mini | 2025-05 | 16 | API only | price levels, market moves |
| Gemini 2.5 Flash | 2025-07 | 14 | API only | price levels, market moves, news dating |
| DeepSeek V3.1 | 2025-09 | 12 | open | price levels, market moves, news dating (habit-driven) |
| gpt-oss-20b | 2025-09 | 12 | open | none detected, but it fails the positive controls |
| Qwen3.8 Flash | – | 0 | API only | market moves (recall shown); no months after release to test a drop |

- **Longest clean windows:** GPT-3.5 Turbo, then GPT-4o-mini and Llama 3.1 70B; Llama 3.1 has open weights, so a
  backtest with it stays reproducible.
- **No model is clean inside its window;** "none detected" never means clean.
- **Using a model inside its window (the report's "Using a model for a backtest" table):** for each model, where its
  market and news memory fade, and whether the two common defences work. Telling the model the date does not stop
  hindsight for Llama 3.3 70B and Gemini 2.5 Flash (their dated forecasts keep 60% and 50% of their recall; Finding
  8); GPT-4.1 Mini shows no leak, and GPT-4o-mini and DeepSeek V3.1 give stock forecasts when told a date, so it could
  not be tested. With names, dates and levels hidden, four of the five models cannot name the month; Gemini 2.5 Flash
  still places some periods (most of all the March 2020 crash), and at a coarser precision GPT-4.1 Mini and DeepSeek
  V3.1 place paths within months (Finding 9). The practical rule stays: only months after a model's release are
  clean; masking is the better defence inside the window, but not a complete one.
- **Where knowledge of Nordic news ends (exploratory, `python -m evalbench.dating_horizon`):** per year, how often a
  model names the exact month, minus how often its usual answers would hit that month by luck. **It replicates on
  fresh headlines:** the five models of the M18 confirmation also answered 1,524 new headlines, and on those alone the
  year after which they date nothing came out the same for all five, with the last clearly dated year the same or
  later (one year for Gemini, two for GPT-3.5, whose 2021 is borderline in both sets). Pooled (about 290 headlines a
  year), news is clearly dated through: GPT-3.5 2022 (stated cutoff 2021-09, so a little past it), both Llamas 2023
  (cutoff 2023-12), Gemini 2024 (2025-01) and DeepSeek V3.1 2024 (2025-03); from the first set only, GPT-4o-mini 2022
  (2023-10), GPT-4.1 Mini 2022 although its stated cutoff is mid-2024, Qwen 2025; gpt-oss shows no clear dating at
  all. Where news knowledge *stops* is not measured: models almost never name later years (the text-only guesser,
  which has no memory, comes out "silent" from 2024 too), so only the "clearly dated through" year is reported. An
  earlier reading (dated within 3 months) put five models at the year of their favourite answer and is replaced.
- **A second, independent hint of the same edge:** the switch to "0%" answers on monthly-return questions (Finding 7)
  lands near the stated cutoff for GPT-4o-mini and GPT-4.1 Mini, and a little before it for Llama 3.3. The index
  questions show the same switch within a year (see Suggested next steps).
- **Effective cutoffs per kind of data (exploratory, `python -m evalbench.effective_cutoffs`):** where each model's
  market answers stop varying (refusals, 0% or a stock value from then on; this marks when the answers stop varying,
  not whether they tracked the market), next to the year its news is clearly dated through. Checked on known cutoffs: GPT-3.5 Turbo stops in 2022-H2 (cutoff 2021-09), GPT-4o-mini and Llama 3.3
  in 2024-H1 (cutoffs 2023-10 and 2023-12). Gemini stops in 2024-H2, DeepSeek V3.2 in 2025-H1 (stated cutoff
  2025-07), Qwen3.8 Flash already in 2024-H1 (stated 2026-03; it answers from a small menu, so less sharp). News
  knowledge often ends earlier than market knowledge: GPT-4.1 Mini dates news to 2022–2023 but answers the market
  into 2025. Price levels cannot time it: they persist, so earlier prices help with later years without hindsight.
- **News dating on the Profile page** compares each year with the text-only guess on the same headlines. The year of a
  model's favourite answer is marked "◇ habit": GPT-3.5's only year above the baseline (2021) and GPT-4o-mini's 2023
  are such years, so they are partly luck.

---

## What did not work, and why

| Method | What it asked | Result |
|---|---|---|
| M1 Exact price | the closing price on a given day | too hard: 0–6% within 1%, for every model |
| M2 Hidden number | fill a masked number in a news article | nothing; many numbers could be worked out from the text |
| M3 Spot the original (DE-COP) | pick the real passage among three rewrites | models spot machine rewrites by style, even after their cutoff (Gemini 44–74% across versions, chance 25%) |
| M4 Guided completion | continue an article, with and without a hint | the hint changed nothing (within ±0.03) |
| M6 Annual return | a company's return over a year | gaps driven by one year (2022, a market-wide fall); unreliable. Under the "0%" rule most of its old detections have too few answers after release to test |
| M11 Monthly return | a company's move relative to the market | no company-specific memory (Finding 7): a firm null, not a failed method |

The common thread: **text memory and company-specific returns show nothing measurable** with these models. What
leaks is market-level movement and rough price levels.

---

## Notes per model

- **Gemini 2.5 Flash:** the strongest market memory, and it refuses many recent questions ("my knowledge cutoff is
  …"). Always read its results next to the answered-only figure.
- **DeepSeek V3.1:** the most consistent price-level memory (detected three times); recalls the US market even more
  closely (private check).
- **Llama 3.3 70B:** detected on prices and on the market (Nordic, and US in the private check); Nordic-specific
  market memory.
- **GPT-4.1 Mini:** detected on prices; its market memory is global, not Nordic.
- **gpt-oss-20b:** fails the positive controls (6% on the largest companies' price levels), so its blanks mean "too
  hard for this model". It reasons before answering, so its answer probabilities are all-or-nothing; testing it
  further needs a self-hosted copy.
- **Qwen3.8 Flash:** too recent to test before/after with data ending in 2026-08; it answers Nordic market questions
  from a small set of stock values.
- **GPT-3.5 Turbo:** passes every positive control; remembers price levels and market moves (+0.55); dates news into
  2022, past its stated cutoff; the longest clean window (31 months).
- **GPT-4o-mini:** remembers the market's moves; 25 clean months.
- **Llama 3.1 70B:** recalls market moves (+0.55) though its drop after release is not shown; dates news; open
  weights, so it stays reproducible; 25 clean months.

---

## Lessons about method

1. **Measure the baseline; never assume chance.** Gemini picks the original DE-COP passage 78% of the time even after
   its cutoff. A fixed chance level would have called every period "exposed".
2. **Guard against refusals.** Without the guard, three pilot E cells would have looked like memory.
3. **Check market-memory claims against the world market.** Without that check, GPT-4.1 Mini would have counted as
   Nordic memory.
4. **Simulate before paying.** The null multiple-choice result could have been predicted for free from answers already
   stored.
5. **Correlation, not just direction.** Most months go up, so "always up" already gets direction right; the memory
   showed in the correlation.
6. **A free feasibility check pays off.** Before the cutoff-ladder run, a projection from stored answers predicted
   which comparisons could succeed (Gemini) and which could not (DeepSeek, price levels). Both predictions held.
7. **A simulation is only as good as its assumed answers.** The headline-dating feasibility check simulated answers
   centred on the truth; real models mostly answer a default month, which broke the registered horizon estimate.
8. **"I don't know" is neither right nor wrong.** Scoring a hedge as chance makes results look more precise than they
   are; scoring it as wrong makes "I don't know" look like memory. Leave it out and report how often it happens.
9. **Check how a score is made up before trusting a tight interval.** M11's narrowest bounds came from models that
   mostly did not answer.
10. **Charts mislead in predictable ways.** Rolling windows smear a sudden drop into an early slope, and market
    periods (the calm 2017, the 2022 fall) look like model effects. Simulating a model with known behaviour through
    the same chart separates the two.
11. **Don't trust stated dates.** Stated cutoffs are labels: knowledge often thins months before them and can reach
    past them, and some are not published at all. Measure them, and use the release date as the safe line.

---

## Method decisions

1. **The scoring rules** (`docs/scoring.md`): closeness as the main score for prices, "within the month's own range"
   as the headline, returns scored against the market, (§7) an answer of 0% to a return question as "I don't know",
   (§8) refusals left out and reported, (§9) market questions as their own fact type, and (§10) a share price of 0 as
   "I don't know". Every report and the app use them; further changes would be a new, dated version.
2. **Refusals (scoring rules §8, 2026-10-06).** A refused question is left out of the app's own numbers and reported
   as a rate; the registered tests keep their rule (a refusal scores 0, with the refusal guard). Counting refusals as
   wrong had made Gemini's monthly-return grid show four "memory" years that came only from its refusing 62% of the
   questions after its release.
3. **DE-COP (M3): frozen at v7 (2026-10-06).** Seven versions (about $1.6 with the gate runs) failed the same check:
   models pick the original among machine rewrites by style, even after their release (Gemini 44–74%, chance 25%). M3
   stays secondary. Text memory would need likelihood tests on self-hosted models instead.
4. **Market-level questions: a fact type of their own (scoring rules §9, 2026-10-06).** M14–M17 now form
   "market", and their Profile cells use the registered statistics (the correlation with the real moves; levels
   against the blind guesser on the same months). Read as up/down calls, the old cells could not show the strongest
   result we have.
5. **US data is a private check only:** its results are not published, and only number-free comparisons appear here.

---

## Suggested next steps

Done since the first version of this overview: models with earlier cutoffs and a staggered test of the time effect
(Finding 5); headline dating (Finding 6); the backtest-readiness table; a larger monthly-return test (Finding 7); the
fake-date and masking tests (Findings 8 and 9); three new open-weight models (Gemma 4 31B and 26B A4B, Qwen3.6 27B;
`docs/preregistration/2026-10-07-new-models.md`).

1. **Forecasts from masked data,** with fake paths (real monthly changes in shuffled order) as a negative control: does
   masking protect a backtest, not only hide the date? About 4,300 calls.
2. **A two-choice news horizon** ("was this headline published in month A or month B?", the true month against one 12
   months off): a news horizon that does not depend on which years a model names. About 3,000 calls.
3. **The fake-date test for GPT-3.5 Turbo and Llama 3.1,** the models with the longest clean windows. About 960 calls.
4. **The horizon signal holds on independent data (checked 2026-10-06, free):** on the index questions, answered five
   days before the monthly-return set, the switch to "0%" answers lands within a year of the monthly-return switch for
   all four models that give them, and is sharper (GPT-4.1 Mini: the same half-year, 2025-H1; GPT-4o-mini 2024-H1, half
   a year later; Llama 3.3 2024-H1, right at its cutoff and a year later than on single shares; Qwen3.8 Flash 2025-H2,
   half a year earlier). Done 2026-10-14: registered as an out-of-sample check on the three new models before it was
   computed for them, and it held (Finding 10). A fresh paid run adds little, since two independent sets already agree.
5. **Robustness:** a second registered wording and a repeat of a quarter of the questions for M15 and M18 (each
   question has been asked once, in one wording), then matched pairs, event questions, and company size measured at
   the time.
6. **Self-hosted models:** needed for gpt-oss, for any probability-based (likelihood) test, and for models with
   published training data or a known cutoff, which would validate the methods themselves.

Considered and set aside for now:
- **Re-identifying masked news** (which company, when): models do not know company-specific monthly moves even when
  told the name and month (Finding 7), and date headlines only weakly (Finding 6), so masking would protect little that
  we can measure. Published work points the same way: anonymising company names in headlines did not reduce in-sample
  performance in one study, where general knowledge of the companies distorted results more than look-ahead did.
- **Likelihood probes through hosted APIs:** such tests barely beat chance on large models in published comparisons,
  and before/after splits like ours are exactly where simple text classifiers that ignore the model do as well. Our
  own text tests (M2–M4) found nothing. Worth revisiting only on self-hosted models with a matched reference model.

---

## Cost

About **$9.9 in total** for every run so far, including test-set rewrites and retries (the cutoff ladder: $0.49;
headline dating: $0.79; monthly return v2: $0.58; the M18 confirmation: $0.51; fake date: $0.13; masking: $0.17; the
three new models, every method: $1.09). A full run of one method costs roughly $0.10–0.80.

---

## Where the details are

Each study plan was committed before its answers were collected; its results are appended at the end of the plan.
Every report is regenerated from the stored answers, at no cost, by the command next to it (run from the repository
root; all commands are in the [README's command reference](../README.md#command-reference)). The six main findings in
short: [Results at a glance](at-a-glance.md).

| Topic | Plan, with results | Report command |
|---|---|---|
| Pilot C: first four methods | [pilot C](preregistration/2026-09-29-pilot-c.md) | `uv run python -m evalbench.pilot_report` |
| Pilot D: eight methods, six models | [pilot D](preregistration/2026-09-29-pilot-d.md) | `uv run python -m evalbench.pilot_d_report` |
| Pilot E: price levels confirmed | [pilot E](preregistration/2026-09-30-pilot-e.md) | `uv run python -m evalbench.pilot_e_report` |
| Monthly price grid (v3, v4, staggered check) | [monthly grid](preregistration/2026-09-30-monthly-grid.md) | `uv run python -m evalbench.grid_report --versions v3 v4 --staggered` |
| Multiple choice (M12/M13) | [price choice](preregistration/2026-10-01-price-choice.md) | `uv run python -m evalbench.choice_report --stage 1\|2` |
| Nordic indices (M14/M15) | [index](preregistration/2026-10-01-index.md) | `uv run python -m evalbench.index_report` |
| Cutoff ladder (memory vs time; older models) | [cutoff ladder](preregistration/2026-10-05-cutoff-ladder.md) | `uv run python -m evalbench.ladder_report` |
| Headline dating (M18) | [headline dating](preregistration/2026-10-05-headline-dating.md) | `uv run python -m evalbench.headline_report`, `uv run python -m evalbench.dating_horizon` |
| Monthly return v2 (M11) | [monthly return v2](preregistration/2026-10-05-monthly-return-v2.md) | `uv run python -m evalbench.monthly_report` |
| M18 confirmation (world vs company news) | [headline macro](preregistration/2026-10-06-headline-macro.md) | `uv run python -m evalbench.headline_confirm` |
| Cutoff-ladder extension | [ladder extension](preregistration/2026-10-06-ladder-extension.md) | `uv run python -m evalbench.ladder_extend report` |
| Fake date (M19) | [fake date](preregistration/2026-10-06-fake-date.md) | `uv run python -m evalbench.fake_date` |
| Masking (M20) | [masking](preregistration/2026-10-06-masking.md) | `uv run python -m evalbench.mask_date` |
| New models (Gemma 4, Qwen3.6) | [new models](preregistration/2026-10-07-new-models.md) | `uv run python -m evalbench.new_models` |
| The "I don't know" horizon on the new models | [zero-answer horizon](preregistration/2026-10-14-zero-answer-horizon.md) | `uv run python -m evalbench.effective_cutoffs` |
| Backtest readiness per model | | `uv run python -m evalbench.readiness` |
| Scoring rules (§1–§10) | [scoring.md](scoring.md) | |

---

## Glossary

- **Cutoff:** the date a model's training data is said to end.
- **Release:** the date the model became available (its OpenRouter listing; for GPT-3.5 Turbo, the release of the
  snapshot it serves now).
- **Closeness:** how near a price answer is to the truth, from 1 (exact) to 0 (off by a factor of 2 or more; for
  indices, 25% or more).
- **Correlation:** from −1 to +1; how closely the model's answers move with the real values. 0 = no relation.
- **Blind guesser:** a rule with no memory, such as "the same as a year earlier".
- **Positive control:** a question every model should be able to answer; it checks that the question format works.
- **Detected:** significantly better before than after, after correcting for testing several models at once (Holm),
  and also on answered questions alone.
- **95% CI:** the range the true difference most likely lies in.
