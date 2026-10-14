# Scoring price and return answers: a proposal

**Date:** 2026-09-30. **Status:** draft proposal. Its rules are written into the pilot E
pre-registration before any pilot E answer exists.

**Why this document:** how close does a model's price have to be to count as correct? Every
rule we have used so far (1% for a day, 10% for a month) was a judgement call, and the choice changes what "correct"
means. This proposes one rule per question type, fixed in advance, with the data behind each choice. All numbers
below come from market data only (Millistream, 2021-01 to 2026-07, the M5 universe), never from model answers, so the
rules cannot have been tuned to make a model look better or worse.

## 1. What we do today

| Question | Correct when | Also reported |
|---|---|---|
| M1 exact close on a day | within 1% of the close (as traded or split-adjusted) | hit rates at 1/3/5/10%, log error |
| M5 average price in a month | within 10% of the month's average (as traded or adjusted) | hit rates at 5/10/25%, raw-only check |
| M6 / M7 annual return | right direction (up or down) | right side of the median share; within 10 pp (M7) |
| M8 Split Clock | closer to the rewritten than to the traded price (log scale) | rewritten / traded / neither counts |

## 2. What the market data says

**A month is wide.** The distance between a month's lowest and highest close, as a share of its average:

| | Large caps | Mid caps | Small caps | All |
|---|---|---|---|---|
| Median monthly range | 11.8% | 14.0% | 18.6% | 14.3% |
| Median yearly range | 49% | 68% | 91% | 64% |

**How often a guesser without memory counts as correct.** The guesser answers the same month a year earlier (the
app's blind baseline), which is what a model that knows the rough level but not the month would say:

| Rule for a monthly level | Large | Mid | Small | All |
|---|---|---|---|---|
| within ±3% | 7% | 5% | 4% | 6% |
| within ±5% | 11% | 9% | 7% | 9% |
| within ±10% (today's rule) | 22% | 18% | 14% | 18% |
| within ±25% | 52% | 40% | 32% | 41% |
| **within the month's own low–high range** | **13%** | **12%** | **12%** | **12%** |

- **A fixed band is easier for big companies than for small ones.** At ±10% the guesser scores 22% on large caps but
  only 14% on small caps, so the same rule means different things across the size strata the benchmark compares.
- **The month's own range is equally strict for every size:** 12–13% everywhere, because a volatile share gets a wider
  range automatically.
- **The positive-control run agrees.** On the new M9 set (the 15 largest Nordic companies) the blind guesser already
  counts as correct 24% of the time at ±10%. The M10 US figure is a private check and is not published here.

**Returns are not 50/50.**

| | Monthly returns | Yearly returns |
|---|---|---|
| Share of "up" periods | 44% | 41% (small caps 30%) |
| Median size of a return | 7 pp | 31 pp |
| A guesser answering the market's direction is right | 60% | 64% (small caps 70%) |

Knowing only which way the market went already gives 60–64% on direction. Finance benchmark guidance warns against
using 50% as chance; the data shows it would be badly wrong here.

## 3. Proposal

### Principles
1. **One primary rule per question type, fixed before any answer exists.** Everything else is reported as a named
   sensitivity check, never swapped in afterwards.
2. **Always report the full band table** (3 / 5 / 10 / 25% for levels), so anyone can apply their own definition.
3. **Use a continuous score as the primary measure where possible.** A threshold throws information away (11% off
   and 300% off count the same). The continuous score uses all of it and needs no arbitrary cut.
4. **Compare with the model's own measured baseline**, with release as the primary line (pilot E plan). A generous
   rule raises both sides of the comparison, so the before/after difference stays meaningful; the rule mainly
   decides how much statistical power we have.

### Monthly price level (M5, and the M9 / M10 controls)
- **Primary: the closeness score**, from 0 to 1. 1 = exactly right; 0 = off by a factor of 2 or more (twice or half
  the true average); in between it falls smoothly on a log scale (off by √2 ≈ 41% scores 0.5). The comparison is the
  average closeness before the line vs after it.
  - It fits the existing Δκ machinery unchanged (a score between 0 and 1, chance 0), so no new statistics are needed.
  - The factor-2 cap stops a single absurd answer (e.g. 100× off) from dominating the average.
- **Headline "correct" (Profile cells, plain-language reporting): the answer lies within the month's own low–high
  range**, as traded or split-adjusted. In words: "the model named a price that actually occurred that month". It is
  equally strict for large and small companies (see §2).
- **Reported alongside:** ±3 / 5 / 10 / 25% hit rates and the median log error.

### Yearly price level (new question type)
- **Primary:** the closeness score as above.
- **Headline:** within ±10% of the year's average. The year's own range (median 64%) is too wide to mean "correct";
  at ±10% the blind guesser (the previous year's average) scores 20%.
- **Reported alongside:** ±5 / 10 / 25%.

### Exact close on a day (M1, secondary)
- **Primary:** the closeness score.
- **Headline:** within the day's own low–high range (the price actually traded that day), instead of a fixed 1%.

### Returns (monthly: new question type; yearly: M6, M7)
- **Primary: direction relative to the market.** Correct when the answer puts the share on the right side of that
  period's median Nordic share (did it beat the market or not?). Chance is 50% by construction, and knowing only the
  market's direction ("2022 was a crash") no longer earns points. That kind of market-wide knowledge is reported
  separately, because it matters for backtests too.
- **Also reported:**
  - absolute direction, against its measured baseline (the market-direction guesser: 60% monthly, 64% yearly), not
    against 50%;
  - the size of the error in percentage points (continuous), and a band: within ±5 pp for a month, ±10 pp for a year
    (the blind guesser "same as last year" is within 10 pp for 17% of company-years).

### Raw vs adjusted prices
- **Headline rules accept either the price as traded or the split-adjusted price,** and record which one the answer
  matched. A model that learned from today's websites sees adjusted history; one that learned from contemporary news
  saw traded prices. Both are memory.
- **The benchmark's convention (price as traded only) is always reported as a named check.**
- **Returns use the split-adjusted series,** so a split does not look like a crash. Dividends are not included
  (Millistream's adjusted series covers capital events only); the question asks for the share price change, which
  matches.
- **Split Clock** is the one method that uses the difference between the two on purpose.

### Positive controls (validity of the price format)
- **M9 known price levels:** the 15 largest Nordic companies, 2016–2022, every month before every cutoff. A model is
  "sensitive" when its headline hit rate's lower 95% bound is above the blind guesser's rate on the same items (as M7).
- **M10 US price levels (private):** famous US shares 2021–2026 (Apple, Microsoft, …). A model should score well
  before its cutoff and clearly worse after its release. This checks that the method detects memory where it
  certainly exists (the positive-control requirement). Built from Yahoo Finance prices: kept in `data/private/`, never
  published, never in the default views.
- **Reading:** a model that fails the positive controls has its "none detected" results on the same question type read
  as "too hard for this model", not as "no memory".

## 4. Choices to settle
1. **Monthly levels:** closeness score as primary, "within the month's own range" as the headline rule? (Alternative:
   a fixed band, e.g. ±5%.)
2. **Yearly levels:** closeness primary, ±10% headline?
3. **Returns:** direction relative to the market as primary, with market-wide knowledge reported separately?
4. **Raw vs adjusted:** accept either for the headline, with raw-only as a named check? Or the benchmark's raw-only
   convention as primary?
5. **Positive controls:** M9 within the licensed data; M10 as a private check only?

## 5. What changes in the code (small)
- New M5 items already record the month's low–high range, and every price-level answer now stores whether it falls
  in that range and its closeness score (the score itself stays at ±10% until this is adopted).
- To do once adopted: switch the primary score per question type, add the yearly-level and monthly-return question
  types, and show the headline rule in the Profile cells.

## 6. Change of 2026-10-01: an answer of exactly 0 to a return question scores 0.5
- **What:** on M6 (annual return), M7 (known returns) and M11 (monthly return), an answer of exactly 0 now scores 0.5,
  the chance level, recorded as "no direction". Before, the code read 0 as "down" (M6, M7) or as a claim about the
  market (M11), so a hedge scored 1 or 0 by luck. A share that really did not move still rewards a 0 (M6, M7).
- **Why:** forced to answer, Llama 3.3 70B gives 0 on 41% of M6 and 73% of M11 questions, and GPT-4.1 Mini on 21% and
  27%. These are "no idea" answers, not directions; scoring them as directions adds noise and, on M6, gives free hits
  in falling years.
- **Found:** while checking an Items-page example (Bio-Works, 2025, answers of 0 marked wrong against +172%); decided
  before any analysis was rerun with it.
- **Effect** (release line, stored answers of pilots D and E; Δκ and the cluster-bootstrap p, before Holm):

| Method | Model | Answers of 0 | Δκ old → new | p old → new |
|---|---|---|---|---|
| M6 | Llama 3.3 70B | 61 | +0.41 → +0.48 | 0.014 → 0.002 |
| M6 | GPT-4.1 Mini | 32 | +0.35 → +0.38 | 0.116 → 0.001 |
| M6 | DeepSeek V3.1 | 3 | +0.60 → +0.51 | 0.003 → 0.012 |
| M11 | Llama 3.3 70B | 219 | +0.11 → +0.12 | 0.44 → 0.038 |
| M11 | GPT-4.1 Mini | 81 | +0.11 → +0.07 | 0.47 → 0.46 |
| M11 | DeepSeek V3.1 | 17 | +0.33 → +0.28 | 0.035 → 0.051 |

  gpt-oss-20b and Gemini 2.5 Flash never answer 0 and are unchanged. Effect sizes barely move; uncertainty drops,
  because coin-flip scores on hedges are gone.
- **What stays as published:** the pilot D and E reports and summaries were computed under the old rule and are not
  rewritten. The app and any new report use the new rule. M6 gaps remain mostly market-wide (2022); read them with
  the "vs median" check, not as company memory.

## 7. Change of 2026-10-01: the app scores price levels by closeness everywhere
- **What:** M5, M9 and M10 are scored by closeness on every page (Overview heatmap, Evidence tiles and charts, Profile,
  Compare, Items), as the reports have done since pilot E. The ±10% hit is kept per question (`hit_10pct`) for the
  correct/wrong labels on Items, the disagreement list and McNemar test on Compare.
- **Why:** the app showed ±10% hit rates next to closeness charts, so the same method had two scores on one page.
- **Status:** decided for the app. Profile cells for price
  levels now show the mean closeness (e.g. 0.23), not a percentage.
- **Refusal guard in the app (same day):** an amber "refusal-driven" cell now needs answer rates that differ by more
  than 10 points before and after (pilot E's "refusal-affected"). Before, a cell whose model answered nearly
  everything could turn amber because its answered-only p-value moved across 0.05 by its Holm rank among the other
  cells alone (GPT-4.1 Mini and DeepSeek on M5, Llama on M6: identical Δκ, adjusted p 0.046 vs 0.052). The registered
  reports keep their own rule.

## 7. Change of 2026-10-06: an answer of exactly 0 to M6 or M11 is an abstention

- **What:** in the app, an answer of exactly 0% to M6 (annual return) or M11 (monthly return) is left out of every
  score, as the refusal guard leaves out refusals; §6 scored it 0.5. How often a model answers 0% is shown on its own
  (Evidence → Over time → Answers of 0%). The Items page still lists these answers, labelled "no direction". M7 (known
  returns, a registered positive control; 0–3% such answers) keeps §6. M15/M17 keep §6 too: their registered test
  correlates the answers themselves, where 0% is a prediction ("the index did not move").
- **Why:** forced to answer, several models give 0% for months they do not know, almost only after their cutoff (M11
  v2: Llama 3.3 70B 100% of its answers after release vs 50% before, GPT-4o-mini 99% vs 6%, GPT-4.1 Mini 89% vs 2%).
  Scored 0.5, these answers pin the after side at chance, so the before/after test compares answers with non-answers:
  it can show a gap that is no memory (M6, below) and an interval narrower than the evidence (M11 v2: "at most +0.05").
  Scored 0 they would turn "I don't know" into a false detection, as refusals did before the guard.
- **Checks before applying:** with the 0% answers left out, chance stays at 0.5 (49.4% of the real M11 v2 items beat
  their month's median; the random control scores 0.512 on directional answers). The share of 0% answers jumps near
  the cutoff for the models that give them (GPT-4o-mini 0.16 → 0.50 in 2023-H2 → 0.97, cutoff 2023-10; GPT-4.1 Mini
  0.03 → 0.28 in 2024-H2 → 0.91, cutoff 2024-06; Qwen3.8 Flash 0.37 → 0.95 in 2026-H1, cutoff 2026-03; Llama 3.3 0.23 →
  0.82 in 2023-H1, cutoff 2023-12). Gemini 2.5 Flash, GPT-3.5 Turbo and gpt-oss-20b never answer 0% and give no signal.
- **Effect on the app's readings** (release line, Holm within the method, the app's refusal guard):

| Method | Model | Before / after items | Reading under §6 → §7 |
|---|---|---|---|
| M11 v2 | Llama 3.3 70B | 619/347 → 312/2 | none detected → too few items |
| M11 v2 | GPT-4o-mini | 577/438 → 543/3 | none detected → too few items |
| M11 v2 | GPT-4.1 Mini | 723/258 → 712/28 | none detected (at most +0.07) → none detected (at most +0.41) |
| M11 v2 | the other six | almost unchanged | unchanged |
| M6 | Llama 3.3 70B | 54/54 → 46/6 | memorization detected (+0.48) → too few items |
| M6 | GPT-4.1 Mini | 81/27 → 81/0 | memorization detected (+0.38) → not scorable |
| M6 | DeepSeek V3.1 | 82/27 → 81/25 | memorization detected (+0.51) → none detected (+0.53; Holm p 0.054) |
| M6 | Gemini 2.5 Flash | 81/27 → 81/27 | memorization detected → refusal-driven (its answered-only Holm rank moves) |
| M6 | gpt-oss-20b | 81/27 | memorization detected (+0.86), unchanged |

  Llama 3.3's and GPT-4.1 Mini's M6 detections compared answers before the cutoff with mostly 0% answers after (Llama
  3.3: 48 of its 54 answers after release). DeepSeek's turns on 3 answers of a 25-question after side. M6 remains a
  secondary method whose gaps are mostly market-wide (§6).
- **What stays as published:** every registered report keeps its own rule, including the M11 v2 table
  (`docs/preregistration/2026-10-05-monthly-return-v2.md`), which reports the directional-only check next to it.

## 8. Change of 2026-10-06: refused questions are left out and reported as a rate

- **What:** a refused question ("I can't know 2026 prices") is left out of the app's own numbers: the Profile cells,
  the Evidence and Overview cells of methods without a registered test, the breakdown and the over-time charts. Its
  share is reported instead: on hover in each Profile cell, in the Profile card's baseline line (where at least 20
  answers remain), and in the Evidence data table (answer and refusal rates). The registered tests (the M5 grid,
  M14–M18) keep their registered rule: a refusal scores 0, with the refusal guard.
- **Why:** scored 0, a refusal looks like a wrong answer, so a model that refuses recent dates looks as if it
  remembers older ones. Scored as chance, it pins scores and narrows intervals, as the 0% answers did (§7). The
  benchmark's reporting format already keeps outcomes apart (answer, abstain, refusal, malformed).
- **Checks before applying** (release line, stored answers): of the 75 Evidence cells without a registered test, 2
  change, both from amber "refusal-driven" to "none detected" (Gemini 2.5 Flash on M6 and M11): the refusal guard
  already reached the same verdict. The Profile had no refusal guard: Gemini's M11 grid showed four "above
  baseline" years (2021–2024) only because it refuses 62% of the questions after its release; with refusals left out
  all four read "none detected". On M15 (then read as up/down calls), six of Gemini's "above" years had the same
  cause.
- **What stays as published:** every registered report and the pilot reports (pilots C and D build their profile
  tables with the old rule).

## 9. Change of 2026-10-06: market questions are their own fact type

- **What:** M14–M17 (index levels and monthly changes, Nordic and US) form the fact type "market",
  next to price, return, event and document; before they were filed under price (levels) and return (changes).
  Their Profile cells use their registered statistics: M15/M17 cells are the correlation of the answers with the
  real moves that year, against the same correlation after the cutoff or release, resampling months; M14/M16 cells
  are the model minus the blind guesser (last year's level) on the same months, as M18 is compared with its
  text-only baseline. The micro-pilot block is left out, as in the registered tests.
- **Why:** market-level memory is the strongest and broadest result (M15: correlation +0.46 to +0.80 before release,
  about 0 after), and it is about an index, not a company: company-size strata do not apply. Read as up/down calls,
  as before, the Profile could not show it: most months go up, so "always up" already scores 0.62, and with
  refusals left out (§8) no Gemini or Llama year was above its baseline.
- **Effect** (the Profile's six default models): M15 shows "above baseline" years for Llama 3.3 70B (2018, 2019,
  2020, 2023) and GPT-4o-mini (2018); before-cutoff correlations of +0.6 to +0.85 in many other years do not pass Holm
  across the page with about 48 answers per year. M14 shows years above the blind guesser for five of the six (not
  gpt-oss-20b), mostly 2017 and 2021: years after a large market move, where "last year's level" is a poor guess and
  any model that roughly knows the index beats it. Read single M14 years with care; the registered M14 test pools
  all months before the cutoff.

## 10. Change of 2026-10-07: a share price of 0 is an abstention

- **What:** in the app, an answer of 0 (or below) to M5 (price level) or M10 (US price levels) is left out of every
  score, like a 0% return under §7; before, it was scored as an answer with closeness 0. How often a model answers 0
  is shown on its own (Evidence → Over time → Answers of 0). M9 (known levels, a registered positive control) keeps
  its registered score, and every registered report keeps its own rule.
- **Why:** no share trades at 0, so the answer is "I don't know", not a price. Scored 0, it turns not knowing into a
  wrong answer, and a model that says 0 mostly after its cutoff looks as if it remembers the earlier months. The
  over-time check of 2026-10-07 found it behind two curves that collapse at the end (Qwen3.8 Flash: 9% → 91% of its
  answers; gpt-oss-20b in 2026: 34–78%).
- **Effect** (M5 v4, release line, answered questions, closeness before minus after):

| Model | Answers of 0 before / after | Gap with them → without |
|---|---|---|
| gpt-oss-20b | 0.1% / 40.6% | +0.042 → −0.007 |
| DeepSeek V3.1 | 1.1% / 6.5% | +0.066 → +0.055 |
| Llama 3.1 70B | 11.0% / 13.0% | +0.072 → +0.076 |
| the other five | 0–2% | unchanged within 0.002 |

  gpt-oss's M5 gap came from its answers of 0 after its release. On M10 only gpt-oss changes (US figures are a private
  check and are not published here).
  Qwen3.8 Flash answers 0 to 11% of its questions before its cutoff; it has no months after release.
