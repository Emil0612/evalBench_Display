import pandas as pd
import streamlit as st

from evalbench import ui
from evalbench.llm.mock import is_local
from evalbench.methods import LABELS, MAIN_METHODS, METHODS, SECONDARY

calls = ui.load_calls()
ts = ui.testsets()
api = calls[~calls["model"].map(is_local)] if len(calls) else calls
last = pd.to_datetime(calls["created"]).max().strftime("%b %d, %Y %H:%M") if len(calls) else "no runs yet"
ui.page_header(
    "Does a model remember Nordic market data?",
    "Contamination tests for large language models on Swedish and Nordic stock prices and financial news. Every "
    "test asks the <b>same model</b> about facts from <b>before</b> its training cutoff and matched facts from "
    "<b>after</b> it. A model that memorized its training data does clearly better on the first group.",
    meta=f"{len(METHODS)} methods ({len(MAIN_METHODS)} main, {len(SECONDARY)} secondary, the rest controls, "
         f"probes and private checks) · "
         f"{sum(len(i) for _, i in ts.values()):,} frozen test items · last answer stored {last}")

if api.empty:
    ui.callout("<b>Controls only.</b> So far only the free local control models have been run; they show what "
               "detection looks like when memorization is certain (oracle) and when it is absent (random, blind). "
               "Real models are run from the Run page after you approve the cost.", "warn")

ui.cards([
    {"label": "Test items", "value": f"{sum(len(i) for _, i in ts.values()):,}", "sub": "frozen, shared by every model"},
    {"label": "Models evaluated", "value": f"{calls['model'].nunique() if len(calls) else 0}",
     "sub": f"{api['model'].nunique() if len(api) else 0} real · the rest are controls"},
    {"label": "Stored answers", "value": f"{len(calls):,}", "sub": "every raw response is kept"},
    {"label": "OpenRouter spend", "value": ui.money(float(api['cost'].sum()) if len(api) else 0.0),
     "sub": "hard cap on every run"},
])

forced = ui.load_calls(ui.DEFAULT_SETTING)
if len(forced):
    answered = sorted(forced["model"].unique())
    real = [m for m in answered if not is_local(m)]
    pool = real or answered
    models = ui.by_cutoff(ui.default_models(pool, ui.cutoffs(pool, ui.meta_by_id())), ui.meta_by_id())
    ats = ui.analysis_testsets(ui.DEFAULT_SETTING)
    answered_sets = set(forced["testset"].unique())          # once, not per method
    methods = [m for m in ats if ats[m][0] in answered_sets and m in MAIN_METHODS]
    fp = ui.fingerprint(forced, ui.cutoffs(models, ui.meta_by_id()))
    table = ui.summary(tuple(models), tuple(methods), ui.DEFAULT_SETTING, fp, "release").copy()
    table["verdict"] = table.apply(ui.verdict, axis=1)
    ui.section("Latest results", "Memorization index Δκ per model and method (forced answers, temperature 0), models "
                                 "in cutoff order. Red cells are significant detections: the model does better on facts from "
                                 "before its cutoff than on facts after its release, which it cannot know (M14, M16 and M18: it beats a "
                                 "blind baseline on the months before its cutoff). Grey cells are not significant. The cutoff line is one click away on Evidence.")
    with st.container(border=True):
        ui.eyebrow(f"{len(models)} models × {len(methods)} methods · Δκ (M14-M18: their registered statistic) with "
                   "95% interval")
        ui.show_chart(ui.heatmap_chart(table, models, methods))
        if ui.heat_note(table):
            st.caption(ui.heat_note(table))
        if {"index_level", "index_return"} & set(methods):
            st.caption("The index columns show their registered tests instead of Δκ. M15: how much the correlation "
                       "between the answers and the real monthly changes drops from before the cutoff to after the "
                       "release. M14: model − blind guesser closeness on the months before the cutoff. Details on the "
                       "Market page.")
        if "headline_date" in methods:
            st.caption("M18 shows its registered test instead of Δκ: on the months before the cutoff, the model's "
                       "dating score minus a text-only baseline's on the same headlines (negative = below it). Its "
                       "plain before/after comparison would always look like memory: no model can name a month after "
                       "its knowledge ends.")
        c1, c2, _ = st.columns([1, 1, 1])
        c1.page_link("pages/profile.py", label="Profile (year by year)", icon=":material/calendar_view_month:")
        c2.page_link("pages/results.py", label="Evidence (statistics)", icon=":material/insights:")
    ui.section("Model timeline", "When each model's knowledge could end, next to the questions we can ask.")
    with st.container(border=True):
        ui.show_chart(ui.timeline_chart(models, ui.meta_by_id()))
        st.caption("Each row is one model on a shared time axis. Dark: before its stated training cutoff (▾),"
                   " facts it may have seen. Light grey: the grey zone between cutoff and release (●): "
                   "probably unseen, but only the maker says so. Palest: after its release, facts it cannot "
                   "know. Questions after the dashed line do not exist yet, which is why late models can only"
                   " be judged on the cutoff line, or not at all.")

ui.section("How the score works", "One number per model and method, with an interval that says how sure it is.")
with st.container(border=True):
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    with c1:
        st.latex(r"\Delta = \bar s_{\text{before cutoff}} - \bar s_{\text{cannot know}}"
                 r"\qquad\qquad \Delta\kappa = \frac{\Delta}{1 - p_0}")
    c2.markdown("**s** — score of one item, from 0 to 1  \n**p₀** — score expected from pure guessing  \n"
                "**Δκ = 0** — no difference, **Δκ = 1** — everything before the cutoff is known  \n"
                "**cannot know** — after the model's release (the default), or more than 3 months after its cutoff")
    st.caption("Before = more than 3 months before the stated cutoff; the months between cutoff and release are left "
               "out on the release line. Intervals and p-values come from one "
               "cluster bootstrap (2,000 draws, resampling whole companies), so a significance star never disagrees "
               "with its interval; p-values are Holm-corrected within each method across the models shown.")

ui.section("Ways to ask", "Each method probes memory differently; agreement between them is the strongest evidence. "
                          "M1–M4, M6 and M8 are secondary: M1 is too hard for every model; M2's numbers can mostly be "
                          "worked out from the article; M3's original can be spotted without memory (frozen at v7); "
                          "M4's hint changed nothing in pilot C; M6's gaps come from one market-wide year (2022, "
                          "pilot D); M8 has too few questions after the line.")
cards = {
    "price_recall": ("Can it recall a closing price?",
                     "“What was the closing price of Volvo B on 2023-03-01, in SEK?” No price context is given.",
                     r"s = \mathbb{1}\left[\,|\hat y - y| \le 1\%\cdot y\,\right]", "chance ≈ 0", "Lopez-Lira et al. 2025"),
    "number_cloze": ("Can it fill in a hidden number?",
                     "A real news excerpt with one figure replaced by [MASK]; the exact number must come back.",
                     r"s = \mathbb{1}[\hat x = x]", "chance ≈ 0", "a 'cloze' test · Deng et al. 2024 (TS-Guessing)"),
    "decop": ("Can it spot the original text?",
              "The real passage hidden among three paraphrases, asked in four rotations so position cannot help.",
              r"s = \tfrac14 \textstyle\sum_{r=1}^{4} \mathbb{1}[\text{correct}_r]", "chance = 0.25",
              "Duarte et al. 2024 (DE-COP)"),
    "completion": ("Does naming the source unlock the text?",
                   "The start of an article is continued twice: once told its source and date (guided), once not.",
                   r"s = \text{ROUGE-L}_{\text{guided}} - \text{ROUGE-L}_{\text{general}}", "chance = 0",
                   "Golchin & Surdeanu 2024 · secondary"),
    "price_level": ("Does it know roughly where a price was?",
                    "“What was the average closing price of Volvo B during March 2023?” Scored by closeness: 1 = "
                    "exact, 0 = off by a factor of 2 or more.",
                    r"s = \max\left(0,\; 1 - \frac{|\ln(\hat y / \bar y)|}{\ln 2}\right)", "chance ≈ 0",
                    "Didisheim et al. 2025; approximate-recall family"),
    "annual_return": ("Does it know how a year went?",
                      "“By how many percent did Volvo B's share price change during 2022?” The direction is scored.",
                      r"s = \mathbb{1}[\operatorname{sign}\hat r = \operatorname{sign} r]", "chance ≈ 0.5",
                      "Lopez-Lira et al. 2025; return family"),
    "known_return": ("Does it know the famous years? (positive control)",
                     "The same return question for the 15 largest Nordic companies in 2016–2022, before every "
                     "cutoff. If a model fails these, the question is too hard and “none detected” means little.",
                     r"s = \mathbb{1}[\operatorname{sign}\hat r = \operatorname{sign} r]", "chance ≈ 0.5",
                     "positive control: questions every model should know"),
    "known_level": ("Does it know the famous prices? (positive control)",
                    "The same price-level question for the 15 largest Nordic companies in 2016–2022, before every "
                    "cutoff. If a model can't name even these, “none detected” on M5 means little.",
                    r"s = \mathbb{1}\left[\,|\hat y - \bar y| \le 10\%\cdot \bar y\,\right]", "chance ≈ 0",
                    "positive control: questions every model should know"),
    "price_choice": ("Does it recognise a price it cannot recall?",
                     "“Which was Volvo B's average price in March 2023? A) 180 B) 250 C) 215 D) 150 SEK” (illustrative) The wrong "
                     "options are Volvo's own prices in other months, so only knowing when each happened helps. Asked "
                     "four times, the right option in each position once.",
                     r"s = \tfrac{1}{4}\sum_{k=1}^{4} \mathbb{1}[\text{right in ordering } k]", "chance = 0.25",
                     "recognition vs recall (evolveQA 2025); permutation balancing (PriDe)"),
    "known_choice": ("Does it recognise the famous prices? (positive control)",
                     "M12's multiple choice for M9's questions: the 15 largest Nordic companies, 2016–2022.",
                     r"s = \tfrac{1}{4}\sum_{k=1}^{4} \mathbb{1}[\text{right in ordering } k]", "chance = 0.25",
                     "positive control: questions every model should know"),
    "index_level": ("Does it know where the market was?",
                    "“What was the closing level of the OMX Stockholm 30 index on the last trading day of March 2023?” "
                    "Scored by closeness, zero once 25% off (an index moves a few percent a month).",
                    r"s = \max\left(0,\; 1 - \frac{|\ln(\hat y / y)|}{\ln 1.25}\right)", "chance ≈ 0",
                    "Lopez-Lira et al. 2025 (index-level look-ahead)"),
    "index_return": ("Does it know which way the market went?",
                     "“By how many percent did the OMX Stockholm 30 index change during March 2023?” The direction is "
                     "scored; knowing the market's path is market-timing hindsight in any backtest.",
                     r"s = \mathbb{1}[\operatorname{sign}\hat r = \operatorname{sign} r]", "chance ≈ 0.5",
                     "Lopez-Lira et al. 2025 (98% direction recall for the S&P 500)"),
    "index_forecast": ("Does telling it the date stop its hindsight?",
                       "M15's question as a forecast: “Today is 31 March 2023 [...] how will the OMX Stockholm 30 "
                       "change during April 2023?” An honest forecaster cannot track single months; answers that "
                       "follow what really happened, on months the model trained on, are hindsight leaking through "
                       "the instruction.",
                       r"r = \operatorname{corr}(\hat r, r)", "honest forecaster ≈ 0",
                       "Eliseev & Seleznev 2026 (fake date tests)"),
    "path_date": ("Can it date a market path with the dates removed?",
                  "12 monthly changes of an unnamed Nordic index, no dates and no levels: “In which month does this "
                  "period end?” A model that can place the path in time recovers the dates a backtest tried to hide.",
                  r"s = \max\left(0,\; 1 - \frac{|\hat m - m|}{12}\right)", "no memory ≈ its own answer habits",
                  "Lopez-Lira, Tang & Zhu 2025 (masking fails)"),
    "us_index_level": ("Does it know where the US market was? (private)",
                       "M14's question for the S&P 500, Dow Jones, Nasdaq Composite and Nasdaq-100: the reference the "
                       "Nordic result is compared with. Private: Yahoo data, never published.",
                       r"s = \max\left(0,\; 1 - \frac{|\ln(\hat y / y)|}{\ln 1.25}\right)", "chance ≈ 0",
                       "Lopez-Lira et al. 2025 (S&P 500, DJIA, Nasdaq recall)"),
    "us_index_return": ("Does it know which way the US market went? (private)",
                        "M15's question for the four US indices; the same months as the Nordic set, so the two can "
                        "be compared month by month.",
                        r"s = \mathbb{1}[\operatorname{sign}\hat r = \operatorname{sign} r]", "chance ≈ 0.5",
                        "Lopez-Lira et al. 2025 (98% direction recall for the S&P 500)"),
    "price_choice_prob": ("Does it lean toward the right price? (probabilities)",
                          "M12's questions scored by the probability the model gives the right letter, for models that "
                          "will not name or pick a price. Needs a provider that returns token probabilities.",
                          r"s = \tfrac{1}{4}\sum_{k=1}^{4} P_k(\text{right letter})", "chance = 0.25",
                          "probability-based multiple choice (PriDe; Min-K% family)"),
    "known_choice_prob": ("Does it lean toward the famous prices? (probabilities)",
                          "M13's positive-control questions scored by the probability of the right letter.",
                          r"s = \tfrac{1}{4}\sum_{k=1}^{4} P_k(\text{right letter})", "chance = 0.25",
                          "positive control: questions every model should know"),
    "headline_date": ("Does it know when it happened?",
                      "“In which year and month was this headline published?” Millistream headlines with every date "
                      "cue removed. Inside its training window a model with memory beats a model-free baseline that "
                      "dates headlines from their words; after its horizon its answers stop moving forward.",
                      r"s = \max\left(0,\; 1 - \frac{|\hat m - m|}{12}\right)", "blind baseline ≈ measured",
                      "Lopez-Lira et al. 2025 (headline dating); LLMLagBench 2025 (change-point horizon)"),
    "us_level": ("Can the test find memory we know exists? (private)",
                 "The same price-level question for famous US shares (Apple, Microsoft …), 2021–2026. Models should "
                 "know these before their cutoff and not after. Private: US prices are not ours to publish.",
                 r"s = \mathbb{1}\left[\,|\hat y - \bar y| \le 10\%\cdot \bar y\,\right]", "chance ≈ 0",
                 "Lopez-Lira et al. 2025; positive control"),
    "monthly_return": ("Does it know how a month went, beyond the market?",
                       "“By how many percent did Volvo B's share change during March 2023?” Correct when the answer puts "
                       "the share on the right side of that month's median share, so knowing the market alone earns "
                       "nothing.",
                       r"s = \mathbb{1}[\operatorname{sign}(\hat r - m) = \operatorname{sign}(r - m)]", "chance = 0.5",
                       "scoring rules (docs/scoring.md); return family"),
    "split_clock": ("Has it seen data from after a split?",
                    "The average price in a month before a big share split. Price sites rewrite old prices after a "
                    "split (400 kr → 40 kr): an answer near the rewritten price means the model saw later data.",
                    r"s = \mathbb{1}\left[\,|\log \hat y - \log y_{\text{adj}}| < \tfrac12 |\log k|\,\right]",
                    "chance ≈ 0", "our own method (Split Clock)"),
}
cols = st.columns(2, gap="medium")
for i, method in enumerate(METHODS):
    title, how, formula, chance, ref = cards[method]
    with cols[i % 2].container(border=True, height="stretch"):
        ui.method_eyebrow(method)
        st.markdown(f"**{title}**")
        st.markdown(how)
        st.latex(formula)
        # the set the analysis pages read (M18 stays on its registered v1), else the newest one
        name, items = ui.analysis_testsets(ui.DEFAULT_SETTING).get(method) or ts.get(method, ("not built", []))
        st.html(" ".join([ui.badge(chance, "muted"), ui.badge(f"{len(items)} items · {name}", "muted"),
                          ui.badge(ref, "info")]))

ui.section("Controls that keep it honest")
ui.cards([
    {"title": "Oracle (control)", "sub": "Knows every item before its cutoff. Every method must flag it.",
     "badges": [ui.badge("positive control", "good")]},
    {"title": "Random · Blind baseline", "sub": "Know nothing about the items. Every method must stay near zero.",
     "badges": [ui.badge("negative controls", "muted")]},
    {"title": "Fabricated companies", "sub": "10% of the price items are invented. The right answer is “I don't know”.",
     "badges": [ui.badge("hallucination check", "warn")]},
    {"title": "Same items, own cutoff", "sub": "All models answer identical frozen items; only each model's cutoff "
                                               "decides which side an item is on.",
     "badges": [ui.badge("fair comparison", "info")]},
])

ui.section("How to use it")
c1, c2, c3 = st.columns(3, gap="medium")
steps = [(c1, "1 · Test sets", "Check the frozen items and review the rewrites used by M3 (generated once in the "
                                "terminal).", "pages/testsets.py", ":material/dataset:"),
         (c2, "2 · Run", "Pick models and methods, review the plan and its cost, then confirm.",
          "pages/run.py", ":material/play_arrow:"),
         (c3, "3 · Profile and Evidence", "Profile: what each model can recall, year by year. Evidence: the "
                                          "statistics and checks behind it, and every raw answer on Items.",
          "pages/profile.py", ":material/calendar_view_month:")]
for col, title, text, page, icon in steps:
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{title}**")
        st.caption(text)
        st.page_link(page, label="Open", icon=icon)
