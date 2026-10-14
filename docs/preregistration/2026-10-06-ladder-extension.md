# Cutoff ladder on more months: memory or a time effect? (registered before any answer)

Registered 2026-10-06, before any candidate answered. The analysis code (`src/evalbench/ladder_extend.py`) is
committed with this plan.

## 1. Why

Every before/after comparison sets older months against recent ones. If recent months were simply harder, that alone
would produce a gap. The registered cutoff ladder (`docs/preregistration/2026-10-05-cutoff-ladder.md`, family S) held
the months fixed and changed the model: on the Nordic index questions (M15) for 2024-02..2024-10, Gemini 2.5 Flash
(which may know them) tracked the real monthly moves and GPT-3.5 Turbo (which cannot) gave a stock answer:
D = +0.81 [+0.24, +1.10], Holm p 0.016. It rests on 9 months and 20 paired answers; without one GPT-3.5 answer it is
+0.55. Later-cutoff models stretch the test window to 14–17 months.

## 2. Models and questions

- **Early model:** GPT-3.5 Turbo, unchanged (cutoff 2021-09-30, release 2024-01-25; its answers are stored).
- **Candidates (late models)**, chosen for temperature 0 without forced reasoning, a fixed endpoint and a stated
  cutoff well after 2024 (the GPT-5.x models were left out: they do not accept temperature 0):

  | Candidate | Stated cutoff (frozen) | Pinned endpoint | Test months |
  |---|---|---|---|
  | DeepSeek V3.2 (exp) | 2025-07-31 | siliconflow/fp8 | 2024-02..2025-04 (15) |
  | Qwen3 235B A22B (2507) | 2025-06-30 | nebius/fp8 | 2024-02..2025-03 (14) |
  | Qwen3 Next 80B A3B Instruct | 2025-09-30 | deepinfra/fp8 | 2024-02..2025-06 (17) |

- **Questions:** the Nordic index monthly-change set `index_return/v1` (M15), all questions, forced answers,
  temperature 0; the 24 micro-pilot items are left out of every analysis, as in the index plan.

## 3. Step 1: the gate (months outside every test window only)

A candidate enters the test only if it:
- answers at least 80% of the questions;
- tracks the real monthly index moves with r ≥ 0.60 on the months both models may know (to 2021-06);
- and with r ≥ 0.60 on 2022-01..2024-01 (after GPT-3.5's window, before every test window).

Calibration on the nine models already run (answer rate; r on the two gate periods): Gemini 2.5 Flash 90%, +0.79,
+0.83 (the original late model); DeepSeek V3.1 99%, +0.82, +0.83; Llama 3.3 100%, +0.70, +0.61 pass; Qwen3.8 Flash
100%, +0.58, +0.54 (the weak reference of the earlier inconclusive staggered check) and gpt-oss-20b 100%, +0.23, +0.03
fail. `ladder_extend gate` computes it and reads no test-window month. Gating on the months both models may know can
only make the test harder (that correlation is subtracted in D).

## 4. Step 2: the test (for the candidates that passed)

- **Statistic:** the registered family S statistic (`ladder.staggered_corr`): D = (r_late − r_GPT-3.5) on the test
  months − (r_late − r_GPT-3.5) on the months both may know, over the questions both models answered; months resampled
  together for both models (20,000 draws).
- **Family:** Holm across the candidates that passed the gate.
- **Readings per candidate:** D > 0 and Holm p < 0.05: memory, not a time effect; D < 0 and Holm p < 0.05: the opposite
  of memory; else not shown.
- **Overall:** the ladder is **replicated on more months** if at least one candidate reads "memory, not a time effect"
  and none reads the opposite. If no candidate passes the gate, the extension is not run and that is reported.

## 5. Reading notes (fixed now)

- Memory is weakest in the last year before a cutoff (2026-10-05 exploratory analysis), so the last test months may
  dilute D: the test is conservative about them.
- The candidates' stated cutoffs are listings; if a model's knowledge ends earlier, D shrinks, never grows.

## 6. Deviations

None yet.
