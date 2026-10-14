# evalBench: does a model remember Nordic market data?

evalBench measures what large language models have **memorized** about Nordic (mostly Swedish) stock prices,
market indices and financial news. Every test asks a model about facts from **before** its training cutoff and
about matched facts from **after** its release, which it cannot know. A model that memorized its training data does
clearly better on the first group. The difference, corrected for guessing, is the memorization signal.

**Why it matters:** anyone who backtests a trading strategy with a language model needs to know whether the model
already "saw" the period it is tested on. If it memorized how prices and news turned out, the backtest is
contaminated by hindsight.

The project has 20 test methods, a scoring and statistics layer (bootstrap intervals, multiple-testing correction,
refusal guards, blind baselines), a run engine for OpenRouter and self-hosted models, frozen test sets, and a
Streamlit app to run tests and read the results. Every confirmatory study was written down before its answers were
collected (`docs/preregistration/`). It is a **diagnostic**: it never ranks models, and "none detected" never means
a period is safe or clean.

**About this repository.** This is a portfolio version of a research prototype. The market data behind the real
results is licensed (Millistream) and is not included, nor are the stored model answers. What you can do here:

- **Run the demo** (below): synthetic data and free control models, the full pipeline and app, no key, no cost.
- **Read what was found** on real data with 15 hosted models: [`docs/at-a-glance.md`](docs/at-a-glance.md)
  (six findings, plain language) and [`docs/findings.md`](docs/findings.md) (the full write-up).

- [Quick start](#quick-start)
- [What we found](#what-we-found)
- [Data](#data)
- [The app, page by page](#the-app-page-by-page)
- [Tutorials](#tutorials)
- [How scoring works](#how-scoring-works)
- [The methods](#the-methods)
- [The models](#the-models)
- [Command reference](#command-reference)
- [Repository layout](#repository-layout)
- [Data rules](#data-rules)
- [Development](#development)

---

## Quick start

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                              # install everything, dev tools included
uv run python -m evalbench.demo      # synthetic data, test sets and free control answers (about a minute)
uv run evalbench-ui                  # the app on http://localhost:8501
```

**The demo** writes synthetic data to `data/` (120 fictional companies with simulated prices, invented news,
simulated index paths), builds the test sets from it and answers them with three free control models: an **oracle**
that knows every answer before its cutoff (2024-06-30), a **random** guesser and a **blind** baseline that uses no
memory. The Overview then shows what a detection looks like (the oracle) and what no signal looks like. A banner
marks demo mode; nothing in it is real market data. It refuses to run over real data in `data/`.

**With real data:**

1. **Put the data in place and build the test sets** (see [Data](#data)). Without them the app starts but has
   nothing to ask.
2. **Optional, only for paid runs:** copy `env.example` to the dotenv file in this folder (the file python-dotenv
   reads) and fill in `OPENROUTER_API_KEY`. Analysis, the control models and self-hosted models need no key.
3. **Start the app:**

   ```powershell
   uv run evalbench-ui          # http://localhost:8501   (another port: uv run evalbench-ui 8601)
   ```

## Data

The repository holds the code and the documentation, but **no data**: the market data is licensed from Millistream,
so `data/`, `results/` and `testsets/` are gitignored. To run anything on real data, the files below have to be
placed in `data/`. The scripts that extract them from the licensed data are not part of this repository; the demo
(above) writes synthetic files with the same columns.

| File in `data/` | What it is |
|---|---|
| `universe.parquet` | 2,167 Nordic share listings: company key, primary listing, currency, market cap, coverage |
| `prices_daily.parquet` | Daily raw and split-adjusted close, market cap and turnover per listing, 2015–2026 |
| `news_sample.parquet` | 16k news articles with body text, stratified by year, source and language |
| `news_index.parquet` | Every article's metadata and headline, no body text (the headline methods only) |
| `index_daily.parquet` | Daily closes of OMXS30, OMXSPI, OMXN40 and OMXHPI |

The adjusted prices are adjusted for capital events (splits, rights issues, spin-offs), **not** for cash dividends.

**US reference data (private, optional).** Only the private checks M10, M16 and M17 need it. It comes from Yahoo
Finance, free and without a key:

```powershell
uv run --with yfinance python -m evalbench.fetch_us_prices      # -> data/private/us_prices.parquet
uv run --with yfinance python -m evalbench.fetch_us_indices     # -> data/private/us_index_daily.parquet
```

**Test sets.** Built from `data/`. Each set is frozen: items are drawn once with a fixed seed, written to
`testsets/<method>/<version>.jsonl`, and never changed. The committed `<version>.manifest.json` records the seed, the
filters, the item count per year and an `items_md5`; a manifest without its `.jsonl` does not activate that
version. New runs always use the newest version of each method. To build the first versions:

```powershell
uv run python -m evalbench.build_testsets --mock-paraphrases         # v1 of every method it can build
uv run python -m evalbench.build_testsets --methods price_level --version v5 --seed 123
```

The command builds every method from `data/` (the index methods from `index_daily.parquet`, one question per index
and month) and skips, with the reason, what it cannot build: the US sets when the US data has not been fetched,
and the multiple-choice sets M12, M13 and their probes, which take the company-months of another test set
(`methods/price_choice.build_items`). DE-COP gets fake distractors with `--mock-paraphrases`. Later versions with
their own designs (the monthly price grid `price_level/v3`, `v4`, DE-COP v3+) were built with the builder
functions in each method module (`methods/price_level.build_monthly_grid`, `build_testsets.run_decop_v3`, ...).
The manifest's `filters` text describes each design. **DE-COP rewrites are paid** (an OpenRouter model writes them): the command prints an upper-bound cost and
asks before spending anything.

**Results.** Every model answer is stored in `results/evalbench.sqlite`, created on the first run. The numbers in
`docs/` come from the project's own database and frozen sets, which are not in the repository.

## The app, page by page

The sidebar shows the status: test sets ready, real models run, stored answers, OpenRouter spend, and whether a
key is set.

**Method colours.** Wherever a method is named, its colour says its role: **control** green (a level every model
should reach), **probe** blue (scored from token probabilities), **private** orange (built from data that may
never be published), **secondary** grey (kept, but not part of the main results). Main methods are white. Pickers
that cannot colour text (drop-downs) show a coloured dot instead.

### Overview
The front page. It shows the headline numbers (test items, models, stored answers, spend), the **latest results**
heatmap (Δκ for every model and main method, forced answers, release line; red cells are significant detections),
a **model timeline** (each model's cutoff, grey zone and release on one axis), how the score works, a card for every
method with its question, formula, chance level and source, the controls, and a three-step guide.

### Test sets
Every frozen set: item count, years covered, seed, build date, fabricated companies; a chart of items per year (each
year needs items on both sides of the cutoffs you test) and the selection rules per method. The DE-COP part shows
the paraphrase model, the **blind-baseline audit** (can the original passage be found without any model?), and a
side-by-side review of originals and their rewrites.

### Run
Where model calls are made. Three steps:
1. **Choose:** models (search all OpenRouter models, or browse by maker and price; the free controls oracle,
   random and blind are always listed), methods, items per method, answer mode (*forced*: the model must give a
   number; *abstain*: "I don't know" is allowed), temperature, samples per item, reasoning effort, seed, data
   privacy routing. Selections can be saved and loaded. Self-hosted models are added here too. Cutoffs that
   OpenRouter does not list can be entered and saved (`config/model_cutoffs.yaml`).
2. **Review:** exactly what will be asked (calls per model and method, answers already stored and reused) and the
   most it can cost. Real cost is usually 40–55% of this ceiling.
3. **Confirm:** set a **spend cap** and start. The run goes to a background worker, so the page can be closed. The
   run panel shows progress, spend and failures and has a Stop button; failed calls are not billed and are retried
   by running the same selection again. **Run history** lists every run.

### Profile
The **knowledge profile**: per model, one cell per year (and per company size, optionally) with what it recalls,
next to its baseline on facts it cannot know. Pick a fact type and a method; colour by measured recall
(closeness for price levels) or by excess over the baseline; baseline = after the cutoff or after the release.
Cells that are above baseline are outlined in red, cells with too few items are hatched, years after the cutoff
are dashed.

### Evidence
The statistics behind the Profile, for the models and methods you pick (up to 8 models; p-values are
Holm-corrected across every cell shown):
- **At a glance:** per model, which methods detect memorization.
- **Model timeline** and the **Δκ heatmap** (every model × method).
- **Per method:** a forest plot per method, estimate and 95% interval per model.
- **Over time:** each model's score month by month (rolling 6-month window with a 95% band, or quarters), its
  cutoff (dashed) and release (solid), the blind guesser for comparison, and a size split for company questions.
  Opens on M12. A link opens the questions behind any point on the Items page.
- **Where does knowledge actually end?** The date where each model's score drops most, with an interval.
- **Where does it come from?** The same index within subgroups (company size, currency, language, source).
- **Robustness checks:** a placebo cutoff (2022-07-01, among items every model had seen) and staggered cutoffs.
- **Data:** the full table with coverage, refusal, malformed and hallucination rates and every p-value; downloads.

### Compare
Two models on the same frozen items for one method: the difference in Δκ with a paired bootstrap interval, a
McNemar test, their scores by quarter, and every item where they disagree, with both raw answers.

### Market
Market-level memory (M14–M17): does a model know where the index went? Each model's monthly answers next to the
real index path, the rolling 12-month correlation between answers and the real monthly change, how often a model
repeats one "stock answer" (a sign it no longer knows), and the Nordic-vs-US comparison with the E4 test (Nordic
memory beyond what the US market explains). The US market is a private check.

### Items
Every test item: the ground truth, the exact prompt each model saw, and each model's raw reply, parsed answer,
outcome (correct, wrong, no direction, abstained, refused, malformed) and score. Filter by method, version, model,
side of the cutoff, outcome, company size and month range.

## Tutorials

### 1. A free dry run with the control models
Checks the whole pipeline without a key or any cost.
1. On **Run**, choose the models `mock/oracle`, `mock/random` and `baseline/blind`, a few methods, and 20 items.
2. Review and confirm (the cost is $0).
3. On **Evidence**, the oracle (knows everything before its cutoff) must be flagged on every method, and random
   and blind must stay near zero. If they do not, something in the scoring is wrong.

### 2. A paid run
1. Make sure the key is set (the sidebar says so).
2. On **Run**, pick the models and methods. For open-weight models, pin a provider in `config/provider_pins.yaml`
   first, so every answer comes from the same weights and precision.
3. Read the **Review** step: calls, answers reused, and the cost ceiling.
4. Set the spend cap a little above the ceiling and confirm. Watch progress in the run panel.
5. If some calls failed (provider timeouts), run the same selection again: only the missing answers are asked.

Before a larger run, do the **feasibility check**: estimate the expected result from answers already stored (for
example, turn existing recall answers into the multiple-choice pick and see whether it beats chance). It costs
nothing. Write the plan down first (`docs/preregistration/`), with its
stopping rule.

### 3. Reading a result
1. **Overview** heatmap: which model × method cells are red.
2. **Evidence → Per method:** is the interval clearly right of zero? Is the cell marked refusal-driven or too few
   items?
3. **Evidence → Over time:** does the score drop at the cutoff or the release, or does it fade gradually?
4. **Robustness checks:** is the placebo cutoff quiet?
5. **Items:** read the raw answers behind a surprising number.
6. **Profile:** the year-by-year picture to share.

### 4. Adding a self-hosted model
Any OpenAI-compatible server works (vLLM, llama.cpp server, Ollama). On **Run → Self-hosted models**, enter the
server URL, the model name, its cutoff and release date, quantization, revision and hardware; or copy
`config/self_hosted.example.yaml` to `config/self_hosted.yaml` and edit it. The model appears as `self/<name>`,
costs nothing and needs no OpenRouter key. Self-hosting is also the way to get real token probabilities (the M12p
and M13p probes).

### 5. Re-running a report
The reports read stored answers only: no model calls, no cost. For example
`uv run python -m evalbench.grid_report --versions v3 v4` or `uv run python -m evalbench.index_report`. They write
to `results/<report>/`. See the [command reference](#command-reference).

### 6. After a parser fix
When a method's answer parser changes, re-read the stored replies instead of asking again:

```powershell
uv run python -m evalbench.reparse price_level              # dry run: lists every reading that would change
uv run python -m evalbench.reparse price_level --apply      # writes them; logged in reparse_log, refused during a run
```

A fix made after results were seen is a deviation: measure its effect on every report first, and record it in the
plan with a before/after table. Published numbers stay as published.

### 7. Adding a new method
1. Write `src/evalbench/methods/<name>.py` with `NAME`, `PROMPT_VERSION`, `FILTERS`, `P0` (chance level),
   `build_items`, the prompt and a parser + scorer; add tests.
2. Register it in `methods/__init__.py`: `LABELS`, and a role set if it is not a main method (`SECONDARY`,
   `POSITIVE_CONTROLS`, `PRIVATE_CHECKS`, `PROBES`).
3. Add it to `pilot.FACT_TYPES`, to the controls in `llm/mock.py`, to `ui.py` (`STRATA`, `SHORT_NAMES`,
   `truth_summary`) and give it an Overview card (the Overview breaks without one).
4. Build its test set, run the free controls, then a small pilot.

## How scoring works

**One score per item**, from 0 to 1 (exact match, closeness, direction, ...). Each method has a known chance level
`p₀`. Then, per model and method:

```text
Δ  = mean score before − mean score on facts the model cannot know
Δκ = Δ / (1 − p₀)          0 = no signal, 1 = everything before the line is known
```

- **Before** = more than 3 months before the model's stated training cutoff.
- **Cannot know** = after the model's **release** (the default, "release line"; the months between cutoff and release
  are left out), or more than 3 months after its cutoff ("cutoff line", a secondary view). The release line is
  primary because stated cutoffs understate knowledge: models often score as well in the grey zone as before it.
- The baseline is **measured**, not assumed: the model's own score on facts it cannot know.
- **Intervals and p-values** come from one cluster bootstrap (2,000 draws, resampling whole companies or months),
  so a significance star never disagrees with its interval. p-values are Holm-corrected across the cells shown.
- **Price levels** are scored by closeness, `max(0, 1 − |ln(answer/truth)| / ln 2)`: 1 = exact, 0 = off by a factor of
  2. Index levels use ln 1.25 (a 25% miss scores 0). A return answer of exactly 0 is "no direction": on M6 and
  M11 it means "I don't know" and is left out of the scores (Evidence → Over time → Answers of 0% shows how often); on
  M7 and the index changes (M15/M17) it scores 0.5. A share price of 0 on M5 and M10 is also "I don't know" and is
  left out the same way (scoring spec §10).
- **Refusals** are left out of the app's own numbers and reported as a rate (spec §8); the registered tests keep their
  registered rule (a refusal scores 0, with the refusal guard below).

**When a cell is not a detection** even with a positive Δκ:
- **too few items:** fewer than 20 items on either side;
- **unresolved:** more than 20% of the answers could not be read;
- **refusal-driven:** answer rates differ by more than 10 points before vs after, and the difference is not also
  significant on answered items alone (a model that refuses recent questions looks like it "knows" the old ones).

**Controls:** the oracle must be flagged everywhere; random and the blind guesser (model-free heuristics from the
item itself) must stay near zero; 10% of some price items are fabricated companies, where the right answer is "I
don't know" (hallucination check).

The market-level methods M14–M17 (the fact type "market", spec §9) use their registered tests instead of Δκ (see
their rows below). The full scoring rules, with every change and its effect, are in `docs/scoring.md`.

## The methods

| # | Method (code name) | Role | The question | Score | Chance |
|---|---|---|---|---|---|
| M1 | Exact price (`price_recall`) | secondary | "What was the closing price of Volvo B on 2023-03-01, in SEK?" | within 1% | ≈ 0 |
| M2 | Hidden number (`number_cloze`) | secondary | A real news excerpt with one figure replaced by [MASK]; the exact number must come back. Numbers that can be computed from the text are excluded. | exact | ≈ 0 |
| M3 | Spot the original (`decop`) | secondary | The real passage among three paraphrases, asked in four rotations (DE-COP). | share of rotations right | 0.25 |
| M4 | Guided completion (`completion`) | secondary | An article's start, continued once told its source and date, once not. | ROUGE-L guided − general | 0 |
| M5 | Price level (`price_level`) | main | "What was the average closing price of Volvo B during March 2023?" | closeness (ln 2) | ≈ 0 |
| M6 | Annual return (`annual_return`) | secondary | "By how many percent did Volvo B change during 2022?" | direction | ≈ 0.5 |
| M7 | Known returns (`known_return`) | control | M6 for the 15 largest Nordic companies, 2016–2022, before every cutoff. | direction | ≈ 0.5 |
| M8 | Split Clock (`split_clock`) | secondary | The average price in a month before a big split. Price sites rewrite old prices after a split; an answer near the rewritten price means the model saw later data. | nearer the adjusted price | ≈ 0 |
| M9 | Known price levels (`known_level`) | control | M5 for the 15 largest Nordic companies, 2016–2022. | within 10% | ≈ 0 |
| M10 | US price levels (`us_level`) | private | M5 for famous US shares (Apple, Microsoft, ...), 2021–2026. Yahoo data. | within 10% | ≈ 0 |
| M11 | Monthly return (`monthly_return`) | main | "By how many percent did Volvo B change during March 2023?" Right if the answer is on the right side of that month's median share, so knowing the market earns nothing. | direction vs median | 0.5 |
| M12 | Which price fits (`price_choice`) | main | "Which was Volvo B's average price in March 2023? A) 180 B) 250 C) 215 D) 150" (illustrative). Wrong options are the company's own prices in other months; asked in four orderings. | share of orderings right | 0.25 |
| M13 | Known price choice (`known_choice`) | control | M12 for M9's questions. | share of orderings right | 0.25 |
| M14 | Index level (`index_level`) | main | "What was the closing level of the OMX Stockholm 30 on the last trading day of March 2023?" | closeness (ln 1.25) | ≈ 0 |
| M15 | Index monthly change (`index_return`) | main | "By how many percent did the OMXS30 change during March 2023?" | see below | ≈ 0.5 |
| M16 | US index level (`us_index_level`) | private | M14 for the S&P 500, Dow Jones, Nasdaq Composite and Nasdaq-100. | closeness (ln 1.25) | ≈ 0 |
| M17 | US index monthly change (`us_index_return`) | private | M15 for the four US indices, same months as the Nordic set. | see below | ≈ 0.5 |
| M18 | Headline date (`headline_date`) | main | A real Millistream headline with every date cue removed: "In which year and month was this published?" | 1 − months off / 12, against a text-only baseline that dates headlines from their words | baseline |
| M19 | Fake date (`index_forecast`) | secondary | M15 asked as a forecast: "Today is 28 February 2023. By how much will the OMXS30 change in March?" | correlation with what happened, before vs after release | ≈ 0 |
| M20 | Masked path (`path_date`) | secondary | The 12 monthly changes of an unnamed Nordic index, no dates, name or levels: "In which month does this period end?" | within one month, minus the model's own answer habits | 0 |
| M12p | Which price fits, probabilities (`price_choice_prob`) | probe | M12 scored by the probability of the right letter, for models that will not name or pick a price. Needs real token probabilities. | P(right letter) | 0.25 |
| M13p | Known price choice, probabilities (`known_choice_prob`) | probe | M13 the same way. | P(right letter) | 0.25 |

**Roles.** *Main* methods make up the results. *Controls* are a level to beat, not a before/after difference: every
item is before every cutoff, so a model that fails them makes "none detected" on the main methods uninformative.
*Private* methods use Yahoo Finance data, which may not be published: they are off the default views. *Secondary*
methods are kept but off the default views: M3's original can be spotted without any memory (the blind audit finds
it from quote marks and from being the "hub" all rewrites came from; frozen at v7), M4's hint changed nothing in
pilot C, M1 is too hard for every model, M2's numbers can often be worked out from the text, M6's gaps are mostly
market-wide and M8 has too few items. M19 and M20 are registered tests of two common backtest defences (telling the
model the date; hiding names, dates and levels), filed as secondary because they ask about a defence, not about a
period. *Probes* need token probabilities, which hosted reasoning models do not give in a usable form.

**Market-level tests (M14–M17).** M15 and M17 compare the Pearson correlation between a model's monthly-change
answers and the real change, before vs after (answers that never vary give r = 0); the bootstrap resamples months.
M14 and M16 compare the model's closeness on months before its cutoff with a blind guesser (last year's level),
because a before/after comparison would mostly measure the 2025–26 rally. A Nordic market-memory claim also needs the
**E4** check: the Nordic correlation that remains after removing the month's US move, since a model can follow the
Nordic market only because it follows the global one.

## What we found

On real data, with 15 hosted models (GPT-3.5 Turbo to Gemma 4) and about 200,000 stored answers. Start with
[`docs/at-a-glance.md`](docs/at-a-glance.md); the full write-up is [`docs/findings.md`](docs/findings.md), and each
study plan in `docs/preregistration/` has its results appended. In short:
- **Market moves (M15):** models recall how the Nordic market moved in months they trained on (correlation +0.44 to
  +0.80 before the cutoff, about 0 after release); for three, this goes beyond what the US market explains.
- **Share prices (M5):** a rough, stale memory of price levels, replicated; not of each month's move.
- **News (M18):** some models date headlines better than the text alone, mostly world and macro news.
- **Stock picking (M11):** no model knows which shares beat the market in a month.
- **Defences:** a date in the prompt does not stop hindsight (M19); masking hides the dates at month precision for
  most models, not completely (M20).
- **For backtests:** only months after a model's release are clean (`uv run python -m evalbench.readiness` writes a
  guide per model).

## The models

Every model the reports cover. **Before** a model's cutoff means more than 3 months before the stated cutoff; it
**cannot know** a month after its release. Where a date is not set in `config/`, it is OpenRouter's (the cutoff
it states, the date it listed the model). Regenerate the table with `uv run python -m evalbench.model_table`.

<!-- model-table:start -->
| Model (OpenRouter id) | Group | Stated cutoff | Cutoff from | Release | Release from |
|---|---|---|---|---|---|
| `openai/gpt-3.5-turbo` | main | 2021-09-30 | OpenRouter listing, frozen in config | 2024-01-25 | set in config (OpenRouter lists 2023-05-28) |
| `openai/gpt-4o-mini-2024-07-18` | main | 2023-10-31 | OpenRouter listing, frozen in config | 2024-07-18 | OpenRouter listing |
| `meta-llama/llama-3.1-70b-instruct` | main | 2023-12-31 | OpenRouter listing, frozen in config | 2024-07-23 | OpenRouter listing |
| `meta-llama/llama-3.3-70b-instruct` | main | 2023-12-31 | OpenRouter listing | 2024-12-06 | OpenRouter listing |
| `openai/gpt-4.1-mini` | main | 2024-06-30 | OpenRouter listing | 2025-04-14 | OpenRouter listing |
| `google/gemini-2.5-flash` | main | 2025-01-31 | OpenRouter listing | 2025-06-17 | OpenRouter listing |
| `deepseek/deepseek-chat-v3.1` | main | 2025-03-31 | OpenRouter listing | 2025-08-21 | OpenRouter listing |
| `openai/gpt-oss-20b` | main | 2024-06-30 | OpenRouter listing | 2025-08-05 | OpenRouter listing |
| `qwen/qwen3.8-flash` | main | 2026-03-31 | our estimate | 2026-08-26 | OpenRouter listing |
| `deepseek/deepseek-v3.2-exp` | ladder extension (index questions only) | 2025-07-31 | OpenRouter listing, frozen in config | 2025-09-29 | OpenRouter listing |
| `qwen/qwen3-235b-a22b-2507` | ladder extension (index questions only) | 2025-06-30 | OpenRouter listing, frozen in config | 2025-07-21 | OpenRouter listing |
| `qwen/qwen3-next-80b-a3b-instruct` | ladder extension (index questions only) | 2025-09-30 | OpenRouter listing, frozen in config | 2025-09-11 | OpenRouter listing |
| `google/gemma-4-31b-it` | added 2026-10-07 | 2025-01-31 | model card | 2026-04-02 | OpenRouter listing |
| `google/gemma-4-26b-a4b-it` | added 2026-10-07 | 2025-01-31 | model card | 2026-04-03 | OpenRouter listing |
| `qwen/qwen3.6-27b` | added 2026-10-07 | 2024-09-30 | our estimate | 2026-04-27 | OpenRouter listing |
<!-- model-table:end -->

## Command reference

All commands run from this folder. Reports read stored answers only and cost nothing.

| Command | What it does |
|---|---|
| `uv run python -m evalbench.demo` | Demo mode: synthetic data, test sets and free control answers (no data, no key, no cost) |
| `uv run evalbench-ui [port]` | Start the app (default port 8501) |
| `uv run python -m evalbench.build_testsets ...` | Build test sets from `data/` (DE-COP rewrites are paid and ask first) |
| `uv run python -m evalbench.decop_audit [versions]` | DE-COP blind-baseline audit; `--balance` / `--reselect VERSION` |
| `uv run python -m evalbench.pilot_report` | Pilot C report (`--controls` = dry run on the control models) |
| `uv run python -m evalbench.pilot_d_report` | Pilot D report |
| `uv run python -m evalbench.pilot_e_report` | Pilot E report |
| `uv run python -m evalbench.grid_report [--versions v3 v4] [--staggered]` | Monthly price grid report |
| `uv run python -m evalbench.choice_report --stage 1\|2` | M12/M13 multiple-choice report |
| `uv run python -m evalbench.index_report` | M14–M17 index report, US vs Nordic, E4 |
| `uv run python -m evalbench.ladder_feasibility` | Cutoff ladder: free coverage, power and cost check before a run |
| `uv run python -m evalbench.ladder_report [--pilot]` | Cutoff ladder report (`--pilot` = the GPT-3.5 control gate only) |
| `uv run python -m evalbench.headline_feasibility` | M18 headline dating: free set, baseline, knee-precision and cost check |
| `uv run python -m evalbench.headline_report [--pilot]` | M18 headline dating report (`--pilot` = the micro-pilot gate only) |
| `uv run python -m evalbench.readiness` | Backtest readiness per model and the backtest guide (clean window, hindsight, date in the prompt, masking) |
| `uv run python -m evalbench.effective_cutoffs` | Where each model's answers stop being informative, with how firm that is |
| `uv run python -m evalbench.dating_horizon` | The anchoring-free M18 horizon: news clearly dated through which year |
| `uv run python -m evalbench.monthly_report` | M11 v2 report (monthly returns against the month's median share) |
| `uv run python -m evalbench.headline_confirm [build]` | M18 confirmation on fresh headlines (world vs company news) |
| `uv run python -m evalbench.ladder_extend gate\|report` | Cutoff-ladder extension with newer models |
| `uv run python -m evalbench.fake_date [build]` | M19 fake-date test (`build` makes its set from M15's) |
| `uv run python -m evalbench.mask_date [build]` | M20 masking test (`build` makes the unlabeled paths) |
| `uv run python -m evalbench.cutoff_estimate` | Estimates an unpublished knowledge cutoff from the US answers (validated on published ones) |
| `uv run python -m evalbench.new_models` | The tests for the models added on 2026-10-07 (Gemma 4 31B and 26B A4B, Qwen3.6 27B) |
| `uv run python -m evalbench.model_table` | Regenerate the model table in this README (cutoffs, releases and where they come from) |
| `uv run python -m evalbench.reparse <methods> [--apply]` | Re-read stored replies after a parser fix |
| `uv run python -m evalbench.worker` | The background run worker (started by the Run page) |
| `uv run --with yfinance python -m evalbench.fetch_us_prices` / `fetch_us_indices` | Private US data into `data/private/` |
| `uv run pytest` | The test suite (no network, no cost) |

## Repository layout

```text
app/                    Streamlit app: streamlit_app.py (navigation) and pages/
src/evalbench/
  methods/              one module per method: items, prompt, parser, score
  llm/                  OpenRouter client, self-hosted servers, mock/control models
  runner.py, worker.py  planning, running and resuming calls; the background worker
  store.py              SQLite: every call, reply, cost and run (results/evalbench.sqlite)
  analysis.py, stats.py item scores, Δκ, bootstraps, Holm, verdicts, registered tests
  market.py             market-level statistics (correlations, stock answers)
  ui.py                 shared app layer: loaders, charts, colours, components
  *_report.py           the pre-registered reports
  demo.py               demo mode: synthetic data and free control runs
config/                 model cutoff overrides, provider pins, self-hosted example
testsets/               frozen test sets, built locally (not in the repository)
docs/at-a-glance.md     the six main findings in plain language
docs/findings.md        the full write-up of the results
docs/preregistration/   study plans, written before the answers were seen, with their results appended
docs/scoring.md         the scoring rules (§1-§10) and every change to them
docs/research/          the literature survey behind the methods
tests/                  unit and pipeline tests
```

## Data rules

- **Millistream data** (prices, articles, anything derived from them, including the test sets) is licensed and is
  not in this repository. The documents report model-performance aggregates only (Δκ, correlations, intervals,
  item counts).
- **Yahoo Finance data** (M10, M16, M17) is for private use only: `data/private/`, off the default views, and its
  results are not published. It is a prototyping reference, not a source of record.
- Sending article text to model providers (OpenRouter and its hosts) is allowed. The default privacy routing is
  "standard" (any provider); no-training and zero-data-retention routing are options on the Run page.
- **Keys** live only in the dotenv file, which is gitignored. Self-hosted entries name an environment variable,
  never a key.

## Development

```powershell
uv run pytest                # about 780 tests, no network, no cost
```

- Pre-register before every paid study; changes after seeing results go in dated deviations, and published numbers
  stay as published.
- Every model is pinned to one provider endpoint; replaced answers are archived, never deleted.
