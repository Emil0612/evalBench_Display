# Results at a glance

The six main findings so far, in plain language, each with links to its numbers and its study plan. The full write-up
is [the findings overview](findings.md). These are method pilots on hosted models: diagnostic results, not rankings,
and "not detected" never means that a model or a period is safe. Every model, with its cutoff and release, is listed
in the [README's model table](../README.md#the-models).

**How to read them:** each test compares a model on months **before** its stated training cutoff (where it may have
read the answer) with months **after** its release (where it cannot have). Doing clearly better before than after is
memory.

## 1. Models remember how the Nordic market moved

Asked how a Nordic index changed in a given month, most models follow the real moves for months they trained on
(correlation +0.44 to +0.80) and not at all after their release (about 0). It is the strongest result in the project.

Details: [index memory, Finding 1](findings.md#finding-1-models-remember-the-markets-path-the-strongest-result)
(the older models: GPT-4o-mini detected; GPT-3.5 Turbo and Llama 3.1 recall the market, +0.55, but their drop is not
significant) · Plans: [Nordic indices](preregistration/2026-10-01-index.md) ·
[the cutoff ladder](preregistration/2026-10-05-cutoff-ladder.md)

## 2. It is memory, not "recent months are harder"

On the same 2024 months, a model that may have trained on them (Gemini 2.5 Flash, and the two Gemma 4 models) still
follows the market, while one released earlier (GPT-3.5 Turbo) cannot and gives a stock answer. On months both may
know, both follow it.

Details: [the cutoff ladder, Finding 5](findings.md#finding-5-it-is-memory-not-a-time-effect-the-cutoff-ladder) ·
[the models added on 2026-10-07, Finding 10](findings.md#finding-10-three-new-open-weight-models-gemma-4-31b-gemma-4-26b-a4b-qwen36-27b)
· Plans: [the cutoff ladder](preregistration/2026-10-05-cutoff-ladder.md) ·
[the new models](preregistration/2026-10-07-new-models.md)

## 3. Single shares: rough price memory, no stock-picking edge

Models are a little closer to a Nordic share's true price in months they trained on (detected for 4 of 5 models, and
replicated), but the answers are far off in absolute terms, and they know roughly where a share traded, not which month.
Asked whether a single share beat the market in a given month, every model is at chance, even inside its training
window.

Details: [M5 price grid, Finding 2](findings.md#finding-2-single-share-prices-are-remembered-but-only-roughly) ·
[which month, Finding 3](findings.md#finding-3-the-memory-is-coarse-not-month-specific) ·
[M11 monthly returns, Finding 7](findings.md#finding-7-no-memory-of-which-shares-beat-the-market-m11-v2) ·
Plans: [the monthly grid](preregistration/2026-09-30-monthly-grid.md) ·
[monthly returns](preregistration/2026-10-05-monthly-return-v2.md)

## 4. Models remember when news happened, mostly world news

Asked to date a Nordic headline with its dates removed, Gemini 2.5 Flash and Llama 3.1 beat a baseline that reads the
words alone, inside their training windows. The memory is of world and macro news far more than of company news.

Details: [M18 headline dating, Finding 6](findings.md#finding-6-models-remember-when-nordic-news-happened-headline-dating-m18)
· Plans: [headline dating](preregistration/2026-10-05-headline-dating.md) ·
[confirmation on fresh headlines](preregistration/2026-10-06-headline-macro.md)

## 5. The stated cutoff is not a safe line; the release is

Knowledge often thins before the stated cutoff and can reach past it. Forced to answer, several models switch to
"0%" or uninformative answers where their knowledge ends, close to their cutoff. A check registered before it was
computed (2026-10-14) found the same on the three newest models: every switch landed in the half-year of the model's
cutoff or the next one. For a backtest, only months after a model's release are clean: in our data 31 months for
GPT-3.5 Turbo, 12–25 for the other models released in 2024–25, and 0–4 for those released in 2026.

Details: [backtest readiness per model](findings.md#backtest-readiness-per-model) ·
[memory fades before the cutoff, Finding 4](findings.md#finding-4-memory-fades-gradually-and-can-reach-past-the-stated-cutoff)
· Plan: [the "I don't know" horizon](preregistration/2026-10-14-zero-answer-horizon.md)

## 6. Telling a model the date does not stop its hindsight; hiding the dates helps, partly

Told "today is the end of last month", Llama 3.3 and Gemini 2.5 Flash still forecast the next month's index move with a
correlation of +0.40 on months they trained on, about half of what they recall when asked directly. Shown an
unlabeled stretch of index changes, most models cannot name the month; Gemini partly can.

Details: [M19, a date in the prompt, Finding 8](findings.md#finding-8-telling-a-model-the-date-does-not-stop-its-hindsight-m19-fake-date-test)
· [M20, unlabeled market paths, Finding 9](findings.md#finding-9-hiding-the-dates-helps-but-is-not-a-complete-defence-m20-masking-test)
· Plans: [fake date](preregistration/2026-10-06-fake-date.md) · [masking](preregistration/2026-10-06-masking.md)

---

Every report is regenerated from the stored answers by the commands in the
[README's command reference](../README.md#command-reference).
