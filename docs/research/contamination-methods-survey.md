# Contamination and memorization detection for black-box LLMs: a survey for a Nordic stock and news benchmark

*Compiled 2026-09-29. Scope: methods from 2023 to 2026 that can be applied through OpenRouter chat models. Some open-weight providers return `logprobs`; Claude and Gemini return no logprobs.*

**Cost basis used below:** an item is about 300–700 input tokens. Prices assume a frontier model at about **$3 per 1M input tokens and $15 per 1M output tokens**. Open-weight models are 5–30× cheaper. Reasoning models can multiply output cost 5–50× through hidden reasoning tokens, so set reasoning to off or minimal. "Chance" means the accuracy a model gets with no memorization.

---

## 1. Black-box methods (text only, work on Claude and Gemini)

### 1.1 Direct numeric recall with a pre/post-cutoff contrast ("The Memorization Problem")
- **Paper:** Lopez-Lira, Tang, Zhu (2025, v2 Dec 2025), [arXiv:2504.14765](https://arxiv.org/abs/2504.14765).
- **What it does:** asks for exact historical values at temperature 0, for example *"What was the closing price of {ticker} on {date}? … JSON {answer, confidence}"*. There are two variants: without context, and with context (the two previous closes are given).
- **Metrics:**
  - MPE = mean((ŷ−y)/y)
  - MAPE = mean(|ŷ−y|/y)
  - directional accuracy = mean(1[sign(ŷ_t−y_{t−1}) = sign(y_t−y_{t−1})])
  - threshold accuracy
  - confidence calibration = corr(confidence, APE)
  - refusal count
- **Results (GPT-4o, cutoff Oct 2023):**

  | Condition | Pre-cutoff | Post-cutoff |
  |---|---|---|
  | S&P 500 daily level, no context: MAPE | 0.61% | 16.9% |
  | Directional accuracy | 80.6% | 45.7% (near chance) |
  | Refusals | 0 | 62/292 |

  - **Individual stocks:** the pattern is uneven. META had 0.37% MAPE, while AAPL had 36% MAPE without context.
  - **Headline dating:** given WSJ front-page headlines, the model gets the year right 98.5% of the time pre-cutoff and 28.8% post-cutoff.
  - **Masking fails:** with names removed, GPT-4o identifies Apple, Meta and Microsoft from anonymized earnings calls 100% of the time.
  - **Instructions fail:** telling the model to "ignore info after 2010" gave 97.6% pre / 98.0% post accuracy, against a true post-cutoff accuracy of 40%.
- **Access and cost:** black-box, 1 call per item, about $0.10 per 50 items.
- **Weaknesses:**
  - (i) **With context**, MAPE is small even post-cutoff (0.5–0.9%), because a random walk gets that result for free. Always compare against the naive baseline ŷ = y_{t−1}.
  - (ii) Levels can be partly guessed from general knowledge ("Volvo trades around 200 SEK").
  - (iii) Split and dividend adjustment and currency make "the true price" ambiguous.
  - (iv) The paper proves that evidence of memorization is only a *lower bound*, and that small post-cutoff samples cannot establish skill.
- **(a) Stock prices:** this is the core test. Ask for the SEK close of `ERIC-B`/`VOLV-B` on date d, without context.
  - Score hit@±1% (unguessable except by memory), log-APE, and direction versus the previous day (chance about 50%, or use the base rate of up-days).
  - Add a "with-context" variant and score **skill over the random walk**: S = 1 − Σ|ŷ−y| / Σ|y_{t−1}−y|.
  - Expect memorization mostly for large caps and famous event days (SBB 2023, SAS, Klarna IPO), not small caps.
- **(b) News:** the headline-to-date variant. Give a headline or lead with dates stripped and ask for the publication year, month and day. Score year and month accuracy and |Δdays|.

### 1.2 Date-only directional recall / Lookahead Propensity (LAP)
- **Paper:** Gao, Jiang, Yan (Dec 2025), [arXiv:2512.23847](https://arxiv.org/abs/2512.23847).
- **Prompt:** *"On {t+1}, did the closing stock price of {company} ({ticker}) go up or down compared to the previous trading day? Answer based only on what you recall … If you do not recall, answer 'unknown'. Respond with exactly one word: up, down, or unknown."*
- **Formulas:** LAP_{i,t} = P(up) + P(down) ∈ [0,1], and the direction signal is (U−D) = P(up) − P(down).
- **Finding:** in Llama-3.3-70B, LAP is clearly positive in the training period and "collapses essentially to zero right after the training-data cutoff". Forecast accuracy loads on LAP × signal only before the cutoff.
- **Access:** logprobs for 1 call per item. Without logprobs, the authors say to estimate the probabilities from **answer frequencies over repeated samples at temperature > 0**. With k = 10 samples at T = 1 that is about $0.25 per 50 items.
- **Weaknesses:**
  - RLHF'd models vary a lot in willingness to say "unknown", so LAP partly measures abstention style. Compare the same model pre versus post, never raw levels across models.
  - Directional accuracy on up/down has only a 50% chance baseline, so it needs many items.
- **(a) Stock prices:** use the prompt directly. Pick event days (|return| > 5%) plus random days as a matched control.
- **(b) News:** "Given only company X and date d, did X publish a profit warning, report or acquisition?"

### 1.3 Masked slot / number guessing (TS-Guessing)
- **Paper:** Deng, Zhao, Tang, Gerstein, Cohan (NAACL 2024), [arXiv:2311.09783](https://arxiv.org/abs/2311.09783).
- **How it works:** mask a pivotal keyword in a question, or mask an *incorrect* MCQ option, then ask the model to fill `[MASK]`.
- **Metrics:** exact match EM = mean(1[norm(ŷ) = norm(y)]) and ROUGE-L F1.
- **Results:** GPT-4 reached 57% EM on masked wrong options in MMLU, and ChatGPT 52%. A model deliberately fine-tuned on MMLU reached about 100%.
- **Filtering:** the authors drop yes/no items and near-duplicate options (ROUGE-L > 0.65) to remove guessable items.
- **Access and cost:** black-box, 1 call per item, about $0.15 per 50 items.
- **Weaknesses:** guessable slots (common words, round numbers, template phrases) give false positives. Mask only low-prior tokens.
- **(b) News (best fit):**
  - Mask a **specific number or entity** in a Millistream release or article, for example *"Net sales increased by [MASK]% to SEK [MASK] million"* or the CEO's name.
  - Numbers have near-zero chance EM. Score EM and |relative error|, and compare pre versus post.
  - Avoid numbers that can be derived from other numbers in the same text.
- **(a) Stock prices:** give a 5-day close series with one day masked. The random walk makes this partly inferable, so score skill over linear interpolation.
- **Related:** the name cloze in "Speak, Memory" (Chang et al. 2023, [arXiv:2305.00118](https://arxiv.org/abs/2305.00118)).
- **2026 note:** translation hides contamination from English-only probes, and slot guessing with choice reordering still detects it (Abbas et al. 2026, [arXiv:2601.14994](https://arxiv.org/abs/2601.14994)). This is relevant because the corpus mixes Swedish and English.

### 1.4 DE-COP: verbatim versus paraphrase multiple choice
- **Paper:** Duarte, Zhao, Oliveira, Li (ICML 2024), [arXiv:2402.09910](https://arxiv.org/abs/2402.09910).
- **How it works:**
  - Build a 4-option MCQ: 1 verbatim passage plus 3 paraphrases written by another LLM (Claude 2 in the paper).
  - Ask all **24 orderings** to cancel position bias.
- **Scoring:**
  - Score = accuracy against a chance of 0.25.
  - At item level, use AUC when logits are available.
  - For groups, a t-test of suspect versus clean.
- **Results:**
  - AUC 0.92 on BookTection, 9.6 points above Min-K%.
  - Black-box ChatGPT and Claude reached 72% on suspect books against a best baseline of 35%.
  - Humans scored 34%, so paraphrases are not trivially distinguishable.
- **Access and cost:** black-box. Full DE-COP is 24 calls per item, about $2.5 per 50 items. A 4-rotation Latin square is 4 calls per item, about $0.4.
- **Weaknesses:**
  - Paraphrase artifacts: the original is more "natural" or specific, or the paraphrase is in the tested model's own style. Use a different paraphraser and check humans or a blind model for bias.
  - Popular text is quoted widely online, which inflates accuracy.
  - Swedish paraphrase quality matters.
- **(b) News:** a verbatim sentence or paragraph versus 3 paraphrases. Always include post-cutoff articles, whose accuracy estimates the "stylistic tell" baseline. Use that as the empirical chance instead of 0.25.
- **(a) Stock prices:** the true close versus 3 distractors. Use **real closes from nearby days** or ±2–5% shifts so every option is plausible. Chance is 0.25.

### 1.5 Data Contamination Quiz (DCQ)
- **Paper:** Golchin & Surdeanu (2023/24), [arXiv:2311.06233](https://arxiv.org/abs/2311.06233).
- **How it works:** a 5-option quiz with the original plus 3 **word-level** perturbations (synonym swaps that keep structure) plus "none of the above". Accuracy above chance estimates contamination, with a correction for position bias.
- **Access and cost:** black-box, 1–5 calls per item.
- **Compared with DE-COP:** the distractors are much harder because they are near-verbatim, so fewer stylistic tells. The trade-off is that synonym errors can be spotted by grammar alone.
- **(b) News:** strong. **(a) Stock prices:** this reduces to the numeric MCQ in 1.4.

### 1.6 Guided versus general instruction completion ("Time Travel in LLMs")
- **Paper:** Golchin & Surdeanu (ICLR 2024), [arXiv:2308.08493](https://arxiv.org/abs/2308.08493).
- **How it works:**
  - The guided prompt names the source (dataset name and split) and gives the first part of an instance, then asks for the rest.
  - The general prompt gives the same prefix without the source.
  - Overlap with the true continuation is measured by ROUGE-L and BLEURT.
- **Contamination is flagged if either:**
  - mean(guided) − mean(general) is significant by **bootstrap resampling (p ≤ .05)**, or
  - a GPT-4 few-shot judge finds ≥1 exact or ≥2 near-exact matches in 10 instances.
- **Access and cost:** black-box, 2 calls per item, about $0.35 per 50 items.
- **Weaknesses:**
  - **Templated text** gives high ROUGE without memorization. Examples: "Kallelse till årsstämma", flagging notices, buyback reports ("Återköp av aktier …"), MAR boilerplate.
  - Safety-tuned models refuse "reproduce copyrighted text".
- **(b) News:**
  - Guided prompt: "Millistream/Cision press release, {company}, {date}. Headline + first sentence → continue."
  - Score ROUGE-L plus **overlap of numbers and named entities** in the continuation, which is hard to guess.
  - Use the same template types post-cutoff as the control.
- **(a) Stock prices:** weak fit. It becomes "continue this price series", which is the same as 1.1.

### 1.7 CDD: contamination detection via output distribution
- **Paper:** Dong, Jiang, Liu, Jin, Gu, Yang, Li (ACL Findings 2024), [arXiv:2402.15938](https://arxiv.org/abs/2402.15938).
- **How it works:** sample 1 greedy output s₀ and n = 50 samples at T = 0.8. Then compute:
  - ρ*(d) = (1/n) Σᵢ 1[ED(sᵢ, s₀) = d]
  - Peak = Σ_{d ≤ α·l} ρ*(d), with α = 0.05 and l = the maximum length
  - The item is "leaked" if Peak > ξ = 0.01.
- **Intuition:** memorization makes the output distribution **peaked**.
- **Access and cost:** black-box, but needs temperature sampling. 51 calls per item is about $4 per 50 items. With n = 20 it drops to about $1.5.
- **Weaknesses:**
  - Short or deterministic answers, such as one number or a formulaic sentence, are peaked for any confident model, so false positives are high.
  - Confidently wrong hallucinations are also peaked.
  - It only makes sense as a pre-versus-post contrast within one model.
- **(a) Stock prices:** the fraction of samples within ±0.5% of the greedy answer, compared pre versus post.
- **(b) News:** the continuation task from 1.6.

### 1.8 Sampling-based n-gram coverage MIAs
- **Papers:**
  - SaMIA: Kaneko et al. 2024, [arXiv:2404.11262](https://arxiv.org/abs/2404.11262).
  - N-Gram Coverage Attack: Hallinan et al., COLM 2025, [arXiv:2508.09603](https://arxiv.org/abs/2508.09603).
  - SimMIA and WikiMIA-25: Yi & Li 2026, [arXiv:2601.11314](https://arxiv.org/abs/2601.11314).
  - WPMIA: 2026, [arXiv:2609.10611](https://arxiv.org/abs/2609.10611).
  - VeilProbe: 2025, [arXiv:2506.19399](https://arxiv.org/abs/2506.19399).
- **How it works:** give a prefix, sample k continuations, and score each against the true suffix. Take the max or mean over samples of the n-gram coverage (the fraction of suffix tokens inside n-grams shared with the sample) or ROUGE-N. Verbatim memorization in the sense of Carlini et al. ([arXiv:2202.07646](https://arxiv.org/abs/2202.07646)) is the special case of an exact greedy suffix match.
- **Access and cost:** black-box, k = 10–50 calls per item, about $1–4 per 50 items.
- **Weaknesses:** this has the same templating issue as 1.6. Hallinan et al. report that newer models such as GPT-4o are more robust (weaker signal).

### 1.9 ConStat and perturbation or paraphrase consistency
- **ConStat:** Dekoninck, Müller, Vechev (NeurIPS 2024), [arXiv:2405.16281](https://arxiv.org/abs/2405.16281).
  - Contamination is defined as *non-generalizing performance*. Using reference models, fit f mapping reference-benchmark accuracy to original-benchmark accuracy. Then δ = acc_orig − f(acc_ref), with a bootstrap p-value.
  - It is black-box but needs a rephrased or synthetic twin benchmark and several reference models.
- **Related:**
  - PaCoST (paired confidence on original versus rephrased, [arXiv:2406.18326](https://arxiv.org/abs/2406.18326)).
  - Cross-Context Verification (2026, [arXiv:2603.21454](https://arxiv.org/abs/2603.21454)).
- **Adaptation:** treat the pre-cutoff items as the "original" and post-cutoff items as the "reference". Several models with different cutoffs make a natural ConStat panel (see §3).

### 1.10 Date manipulation and counterfactual controls (HindsightBench, 2026)
- **Paper:** Jia (Jul/Aug 2026), [arXiv:2607.18867](https://arxiv.org/abs/2607.18867). A black-box audit with **no logprobs**.
- **The 2×2 date matrix:** the same item is shown with the date *revealed / date-only / masked / transplanted* (a fake date).
- **Metrics:** trigger strength, transplant effect, post-cutoff placebo, recoverability and "behaviorally effective cutoff".
- **Findings:**
  - Date-triggered hindsight was absent in 2024 open-weight models but present in all 2026-generation models tested.
  - Effective cutoffs came up to **8 months earlier** than vendor claims.
  - Quantization and serving regime changed the results, which matters on OpenRouter, where providers differ.
- **Adaptation:**
  - Ask the same price question with the real date and with a transplanted date (the same ticker a year off, or a fabricated ticker "NORDX").
  - **Fabricated entities or articles** measure the false-positive floor, for example "confident recall" of prices that never existed.
- **Related finance work (2023–26):**
  - Glasserman & Lin ([arXiv:2309.17322](https://arxiv.org/abs/2309.17322)) found anonymized headlines *beat* named headlines in-sample, which they call the "distraction effect".
  - Sarkar & Vafa (ICML 2025, [SSRN 4754678](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4754678)) test lookahead using events that are unpredictable by construction.
  - Look-Ahead-Bench ([arXiv:2601.13770](https://arxiv.org/abs/2601.13770)) measures alpha decay across periods.
  - Kong et al. 2026 ([arXiv:2602.14233](https://arxiv.org/abs/2602.14233)) propose "temporal sanitation" and entity-substitution tests.
  - Point-in-time models such as ChronoGPT ([arXiv:2502.21206](https://arxiv.org/abs/2502.21206)) and DatedGPT ([arXiv:2603.11838](https://arxiv.org/abs/2603.11838)) are clean-by-construction baselines.
  - Merchant & Levy ([arXiv:2512.06607](https://arxiv.org/abs/2512.06607)) remove the bias at inference time.

### 1.11 Cutoff probing ("Dated Data", LLMLagBench)
- **Dated Data:** Cheng et al. (COLM 2024), [arXiv:2403.12958](https://arxiv.org/abs/2403.12958).
  - The *effective* cutoff is the argmin over time of perplexity on time-versioned documents (Wikipedia revisions and CommonCrawl dumps). This needs full-sequence logprobs, so open weights.
  - Finding: effective cutoffs often differ from reported ones. Causes are old data inside new CommonCrawl dumps and deduplication failures.
- **LLMLagBench:** Pęzik et al. 2025, [arXiv:2511.12116](https://arxiv.org/abs/2511.12116). A black-box version based on recall of recent events.
- **Adaptation:** plot recall accuracy against month (1.1 and the headline-date task) and fit a change point. This gives **your own estimate of each model's cutoff**. Do not trust vendor dates.

---

## 2. Logprob and white-box methods (open-weight models only; flag them as optional)

| Method | Formula | Access | Notes |
|---|---|---|---|
| **Min-K% Prob** (Shi et al., ICLR 2024, [arXiv:2310.16789](https://arxiv.org/abs/2310.16789)) | (1/\|S\|) Σ_{xᵢ∈MinK%(x)} log p(xᵢ\|x<ᵢ), k≈20% | Logprobs of the **prompt tokens** (echo). Most chat APIs return only output-token logprobs, so in practice this needs weights. | Its benchmark WikiMIA splits members and non-members by date, the exact confound criticised in §2.1. |
| **Min-K%++** (Zhang et al., ICLR 2025, [arXiv:2404.02936](https://arxiv.org/abs/2404.02936)) | per token (log p(x_t\|x<t) − μ)/σ, with μ = E_{z∼p}[log p(z\|x<t)] and σ² = Var_{z∼p}[log p(z\|x<t)]; mean over the lowest k% | Full vocabulary distribution (weights). A top-20 approximation is crude. | |
| **zlib / reference ratios** (Carlini et al. 2021, [arXiv:2012.07805](https://arxiv.org/abs/2012.07805)) | log PPL(x) / zlib_bits(x); log PPL_target / log PPL_ref; lowercase ratio | Token logprobs | Very sensitive to text type. Templated press releases are low-PPL no matter what. |
| **PAC** (Ye et al. 2024, [arXiv:2405.11930](https://arxiv.org/abs/2405.11930)) | polarized distance = mean log-p(top 5% tokens) − mean log-p(bottom 30% tokens); flag if L(z) − mean_N L(swap-augmented z) > ε | Token probabilities. "Black-box" only through a logit_bias probing trick that OpenRouter does not reliably support. | |
| **Oren et al. exchangeability** ([arXiv:2310.17623](https://arxiv.org/abs/2310.17623)) | permutation p = (1 + #{π: log p(x_π) ≥ log p(x_canon)})/(m+1) | Logprobs | Needs an ordered dataset. Not a natural fit here. |
| **Kernel Divergence Score** (Choi et al., ICML 2025, [arXiv:2502.00678](https://arxiv.org/abs/2502.00678)) | divergence of the embedding kernel matrix before and after fine-tuning on the set | Weights plus fine-tuning | Out of scope. |
| **Dataset inference** (Maini et al. 2024, [arXiv:2406.06443](https://arxiv.org/abs/2406.06443)); DC-PDD ([arXiv:2409.14781](https://arxiv.org/abs/2409.14781)); LogitTrace ([arXiv:2509.20909](https://arxiv.org/abs/2509.20909)) | aggregate MIA features, then a t-test at set level | Logprobs or weights | |

### 2.1 Critiques you must design around
- **Duan et al. (COLM 2024)**, "Do MIAs work on LLMs?", [arXiv:2402.07841](https://arxiv.org/abs/2402.07841). On i.i.d. member/non-member splits of the Pile (MIMIR), MIAs are **near chance (AUC ≈ 0.5–0.6)**. Earlier high AUCs came from **temporal distribution shift** between members and non-members.
- **Das, Zhang, Tramèr (2024)**, "Blind baselines beat MIAs", [arXiv:2406.16201](https://arxiv.org/abs/2406.16201). On 8 MI datasets, **model-free** classifiers beat every published MIA. These were date thresholds and bag-of-words or n-gram classifiers.
- **Meeus et al. SoK** ([arXiv:2406.17975](https://arxiv.org/abs/2406.17975)) reaches the same conclusion and recommends randomized, controlled member/non-member designs.
- **Janicki et al. 2026** ([arXiv:2604.19561](https://arxiv.org/abs/2604.19561)) found all black-box MIAs at about 0.5 AUC on modern LLMs.
- **MemGuard-Alpha 2026** ([arXiv:2603.26797](https://arxiv.org/abs/2603.26797)), in finance:
  - Temporal features give AUC 1.0, which is "arithmetic rather than detection".
  - After normalizing for model scale, the MIA scores fell to 0.49–0.54.
- **What this means for this benchmark:**
  - Your pre/post split *is* a temporal shift. A difference in raw score can come from the items: 2020–21 COVID vocabulary, price regimes, or the article mix.
  - Prefer tests whose chance level does not depend on the distribution (EM on unguessable numbers, MCQ with 1/k chance).
  - Match pre and post items.
  - Run a blind baseline: can a bag-of-words or date-free classifier, or a model *with no access* to the answer, reach the same score?
  - Above all, use the **cross-model staggered-cutoff design** below.

---

## 3. Comparing models in a statistically sound way

1. **Primary estimand (per model and method):** Δ = score_pre − score_post.
   - Get the 95% CI from a **cluster bootstrap**: resample tickers or articles, not rows, with 10,000 replicates and BCa or percentile intervals.
   - Use a **permutation test** that shuffles pre/post labels within strata (ticker, year, article type) for the p-value.
2. **Chance-corrected scores:** report κ = (acc − p₀)/(1 − p₀), where p₀ is:
   - 1/k for MCQ, or the empirical post-cutoff accuracy when stylistic tells exist;
   - the base rate of up-days for direction;
   - about 0 for EM on numbers.
   - For numeric recall, report **skill over the random walk**, S = 1 − MAE_model / MAE_RW, and hit@±1%.
3. **Tests against chance:** a one-sided exact binomial test, k ~ Bin(n, p₀).
   - Rotations or permutations of one item are *not* independent. Aggregate per item first (majority or mean), then test across items.
   - For comparing two models on the *same* items, use McNemar's test.
4. **Model comparison as difference-in-differences:** Δ_A − Δ_B, with a paired bootstrap over shared items.
   - Alternatively fit a mixed-effects logit: correct ~ pre × model + (1|item) + (1|ticker).
   - **Staggered cutoffs are the strongest design.** Items dated between cutoff_A and cutoff_B are post-cutoff for A but pre-cutoff for B. Item difficulty then cancels.
5. **Cutoff uncertainty:**
   - Exclude a ±3-month buffer around the claimed cutoff, or estimate the cutoff by change-point or segmented logistic regression of accuracy on date.
   - Effective cutoffs can come 8 months *before* the claimed ones (HindsightBench), and post-training can add later data.
6. **Power:** with 50 items per group, the power to detect 25% → 45% at α = 0.05 is only about 55%. Detecting that effect with 80% power needs about 90 per group.
   - Items are cheap, so use **≥100 pre + ≥100 post** for the headline tests.
   - Lopez-Lira et al. warn explicitly that small post-cutoff samples are uninformative.
7. **Multiple testing:** control models × methods × subsets.
   - Use Holm–Bonferroni for confirmatory claims and Benjamini–Hochberg FDR for exploratory heatmaps.
   - Pre-register one primary endpoint per method.
8. **Refusals and "I don't know":**
   - Log them separately; they are a signal in their own right. LAP *is* essentially 1 − P(unknown), and GPT-4o's refusals rose from 0 to about 20% post-cutoff.
   - Report (i) the **intention-to-treat** score, counting a refusal as wrong or as the maximum error;
   - (ii) **coverage** (the answer rate) pre versus post;
   - (iii) **conditional accuracy** among answered items.
   - For numbers, run a "best guess, must answer" prompt variant as well.
   - Never drop refusals silently, since that creates survivorship bias.
9. **Controls to include:**
   - Matched pre/post items (the same tickers, similar volatility and article type).
   - **Fabricated tickers and articles** for the false-positive floor.
   - Transplanted dates.
   - Post-cutoff items using the same templates.
   - Disable web search: no `:online` suffix and no plugins.
   - Pin providers (`provider.order`, `allow_fallbacks: false`, and `require_parameters: true` when requesting logprobs).
   - Record the model version string and provider.

---

## 4. Recommendation: a one-day, black-box, cheap demo (< $5 per model; about $1–3 at 50 pre + 50 post items per method)

| Rank | Method | Why |
|---|---|---|
| **1** | **Direct close-price recall** (1.1), no context, 1 call per item | This is the published headline result in exactly your domain. A number is unguessable, so hit@±1% has near-zero chance. It is the most intuitive plot: error against date, with the cutoff line. Add a with-context variant scored against the random walk. It also gives a behavioral cutoff estimate. About $0.2. |
| **2** | **Masked-number cloze in news** (1.3, TS-Guessing style), 1 call per item | This is the lowest-false-positive text test: exact figures (revenue, % change, deal value, CEO names) almost never match by chance. It works in Swedish and English. Filter templated items and derivable numbers. About $0.2. |
| **3** | **DE-COP / DCQ multiple choice** (1.4/1.5) for news (verbatim vs 3 paraphrases) and prices (true close vs 3 nearby real closes), 4 rotations | The chance level is clean (25%), so it supports a binomial test and κ. It works on refusal-prone models because they only pick a letter. Use the post-cutoff accuracy as the empirical chance to absorb paraphrase tells. About $0.5–1. |
| **4** | **Date-only up/down/unknown (LAP)** (1.2) | This tests look-ahead bias directly and makes abstention measurable. Use logprobs where the provider supports them, otherwise 5–10 samples at T = 1. It is weaker per item (50% chance), so use event days and pool items. About $0.3. |
| **5** | **Headline to publication date** (1.1b) | Cheap, and it gives a smooth "knowledge versus month" curve for estimating the cutoff. Batching about 5–10 headlines per call cuts cost. Mask explicit date cues such as "Q3 2021". About $0.1. |
| **6** | **Guided vs general completion** (1.6), with ROUGE-L plus overlap of numbers and entities | This is the canonical contamination test and is easy to explain. It is ranked last because press-release templates inflate ROUGE and models often refuse verbatim reproduction. About $0.4. |

**Held back:**
- **CDD** is costly (about $4 per 50 items) and ambiguous on short answers.
- **Min-K% and PAC** need prompt-token or full-vocabulary logprobs.
- **KDS** needs fine-tuning.
- **N-gram coverage MIAs** need many samples, and the 2026 evidence is that black-box MIAs sit near 0.5 AUC on modern models.

**Controls to add to every method:** a fabricated-entity control, a date-transplant control, and the staggered-cutoff DiD across at least two models with different cutoffs. Together these three pre-empt the "it's just distribution shift" critique (Duan 2024; Das et al. 2024), and they cost almost nothing extra.

**What to expect:** for Nordic data, memorization should be much weaker than for the S&P 500. It will concentrate on OMXS30 large caps and on high-coverage events. Present a *null or weak result with tight confidence intervals* as a valid finding. Per Lopez-Lira et al., any positive result is a lower bound on memorization.
