# M11 monthly return v2: a larger set (registered before any answer)

Registered 2026-10-05, after the set was built and before any model answered it.

## 1. Why

`monthly_return/v1` (300 items, 270 real) has 1–12 real items per month. A 6-month window in the over-time views then
holds 15–45 answers of a 0/1 score at 50% chance, so the rolling line moves by ±0.1–0.2 from one month to the next
without any real change, and windows near the 15-answer floor drop out. v2 asks the same question with about four
times as many items, so the over-time views can show change and the before/after comparison has more power.

## 2. The set

- **`monthly_return/v2`** (md5 `d324470a797bae196070fac896ea47a8`): 1,340 items, 1,206 real and 134 fabricated
  companies, seed 20261005, built with the v1 sampler unchanged (`build_testsets --methods monthly_return --n 1340
  --seed 20261005 --version v2`).
- Same question, prompt (p1), filters and scoring as v1. Months 2021-01..2026-07, 9–28 real items per month (median
  18), every 6-month window 84–110 real items; 402 per size tercile. 2 company-months repeat a v1 item.
- **Models:** GPT-3.5 Turbo, GPT-4o-mini (2024-07-18), Llama 3.1 70B, Llama 3.3 70B, GPT-4.1 Mini, Gemini 2.5 Flash,
  DeepSeek V3.1, gpt-oss-20b, Qwen3.8 Flash; forced answers, temperature 0, pinned endpoints as in
  `config/provider_pins.yaml`. The local blind guesser (previous month's return), oracle and random controls answer
  the same items.

## 3. Analysis (fixed now)

- **Main table:** pilot E's E1 rule on v2 for every model: score 1 if the answer is on the right side of that month's
  median Nordic share (0.5 for an answer of exactly 0), Δκ = (mean PRE − mean POST) / (1 − 0.5) on the release line
  (pilot E §2 windows), cluster bootstrap by company (20,000 draws), Holm across the nine models. Fake companies are
  left out. Coverage rules as in pilot E (at least 20 real items per side, at most 20% malformed).
- **Refusal guard:** as pilot E §6: a detection must also hold on the answered items only, or it reads
  "refusal-driven".
- **Absolute direction:** pilot E's E4 (against the market-direction guesser, not 50%), reported per model.
- **Over-time view:** the Evidence page's rolling 6-month curve per model on v2, with the blind guesser as reference;
  descriptive only.
- **v1 stays as registered.** Pilot E's M11 cells are not replaced. v2 is reported next to them; where v1 and v2
  disagree, both are shown and the larger set is read as the better estimate.

## 4. Positive controls

The models' positive-control readings are taken from their registered reports (pilot D/E, cutoff ladder §0); a model
that fails them (gpt-oss-20b) has its M11 "none detected" read as "too hard for this model".

## 5. Deviations

None yet.
