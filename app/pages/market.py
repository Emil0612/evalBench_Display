import pandas as pd
import streamlit as st

from evalbench import market, ui
from evalbench.llm.mock import is_local
from evalbench.methods.index_level import NAMES as NORDIC
from evalbench.methods.us_index_level import NAMES as US

MARKETS = {"nordic": ("Nordic", NORDIC, "index_level", "index_return"),
           "us": ("US (private check)", US, "us_index_level", "us_index_return")}

ui.page_header("Market", "Does a model know where the stock market went? Its answers next to the real index, month by "
                         "month: Nordic (M14 level, M15 monthly change) and, as a private reference, US (M16, M17).",
               meta="Before its cutoff the answers can follow the real path; after its release they cannot.")
meta = ui.meta_by_id()
with st.container(border=True):
    c0, c1 = st.columns([1, 3], vertical_alignment="bottom")
    with c0:
        setting = ui.setting_picker("market")
    calls = ui.load_calls(setting)
    ts = ui.analysis_testsets(setting)
    if calls.empty or "index_return" not in ts or "index_level" not in ts:
        ui.empty_state("No index questions answered yet.", "pages/run.py", "Run")
        st.stop()
    answered = calls[calls["method"].isin(["index_level", "index_return", "us_index_level", "us_index_return"])]
    options = ui.by_cutoff([m for m in answered["model"].unique() if not is_local(m)], meta)
    if not options:
        ui.empty_state("No real model has answered the index questions yet.", "pages/run.py", "Run")
        st.stop()
    st.session_state["mkt_models"] = [m for m in st.session_state.get("mkt_models", ui.default_visible(options))
                                      if m in options] or ui.default_visible(options)
    models = c1.multiselect("Models", options, key="mkt_models", format_func=ui.display_name)
    markets = [k for k in MARKETS if MARKETS[k][3] in ts]
    ui.keep_choice(st.session_state, "mkt_market", markets, "nordic")
    mkt = st.segmented_control("Market", markets, key="mkt_market", format_func=lambda k: f":orange[{MARKETS[k][0]}]" if k == "us" else MARKETS[k][0]) or "nordic"
    _, NAMES, level_method, return_method = MARKETS[mkt]
    ui.remember_page_choices(st.session_state)
if not models:
    st.stop()
cut_rows = [{"model": m, "month": str(pd.Period(ui.cutoffs([m], meta)[m], "M"))}
            for m in models if ui.cutoffs([m], meta)[m] is not None]
rel_rows = [{"model": m, "month": str(pd.Period(ui.release_date(m, meta), "M"))}
            for m in models if ui.release_date(m, meta) is not None]

# ---------------------------------------------------------------- answers next to the real path
ui.section("Answers next to the real index", "What really happened, and what each model answered for each month. "
                                             "A model that remembers the market puts its dots on the line before its "
                                             "cutoff.")


@st.fragment
def answers_vs_path(calls: pd.DataFrame, ts: dict, models: list[str], cut_rows: list[dict], rel_rows: list[dict],
                    names: dict, level_method: str, return_method: str) -> None:
    """The index and question pickers redraw only this chart (a fragment), not the whole page."""
    with st.container(border=True):
        d1, d2 = st.columns([1, 1.6], vertical_alignment="bottom")
        ui.keep_choice(st.session_state, "mkt_index", list(names), list(names)[0])
        index = d1.selectbox("Index", list(names), key="mkt_index", format_func=lambda s: f"{names[s]} ({s})")
        ui.keep_choice(st.session_state, "mkt_q", ["index_return", "index_level"], "index_return")
        question = d2.segmented_control("Question", ["index_return", "index_level"], key="mkt_q",
                                        format_func={"index_return": "Monthly change (M15/M17)",
                                                     "index_level": "Month-end level (M14/M16)"}.get) or "index_return"
        ui.remember_page_choices(st.session_state)
        method = {"index_return": return_method, "index_level": level_method}[question]
        name, items = ts[method]
        items_ix = [it for it in items if it.payload["symbol"] == index]
        key = "level" if question == "index_level" else "return_pct"
        truth = pd.DataFrame({"month": [it.payload["month"] for it in items_ix],
                              "value": [it.truth[key] for it in items_ix]}).sort_values("month")
        answered_here = calls[calls["testset"] == name]
        frames = []
        for m in models:
            df = market.answers_frame(answered_here, items_ix, m, method)
            if len(df):
                frames.append(df.assign(model=m, month=df["month"].astype(str)))
        if frames:
            level = question == "index_level"
            st.html(ui.path_chart_key())
            ui.show_cached_chart("market_path_chart", truth, pd.concat(frames), models, cut_rows, rel_rows,
                                 "index level" if level else "% change", "level" if level else "change")
            st.caption("Each dot is one question (one month of this index). A model with memory follows the line "
                       "before its cutoff; after its release it can only guess, so its answers flatten out. A model "
                       "that does not know the index repeats a few stock numbers, which show as flat rows of dots. The scale "
                       "follows the real path: an answer far off it (" +
                       ("more than 25% above or below, which scores 0" if level else "more than 10 points beyond") +
                       ") is drawn at the edge as a triangle ▲▼; hover it for the real answer.")
        else:
            st.caption("This model has not answered this index yet.")


answers_vs_path(calls, ts, models, cut_rows, rel_rows, NAMES, level_method, return_method)

# ---------------------------------------------------------------- rolling correlation
ui.section("How closely the answers track the market", "The correlation between a model's monthly-change answers "
                                                       "and the real changes, in a 12-month window moved one month "
                                                       "at a time (all four indices together).")
fp = ui.fingerprint(calls, ui.cutoffs(models, meta))
curves = []
for m in [*models, ui.BLIND_ID]:
    c = ui.market_rolling(m, setting, fp, method=return_method)
    if len(c):
        curves.append(c.rename(columns={"r": "mean"}).assign(model=m))
with st.container(border=True):
    if curves:
        st.html(ui.rolling_chart_key("correlation"))
        ui.show_cached_chart("rolling_chart", pd.concat(curves), models, cut_rows, rel_rows, "correlation")
        st.caption("1 = the answers move exactly with the market; 0 = no relation (an answer that never changes counts "
                   "as 0). The band is a 95% interval from resampling months; windows with fewer than 20 answers are "
                   "left out. Grey dashed: the blind guesser, which repeats last month's change. Memory shows as a "
                   "high curve before the cutoff that falls to about 0 after the release. A 12-month window holds few answers "
                   "(often 20-40), so its band is wide: swings of up to about ±0.5 are noise, even the blind guesser "
                   "moves that much. Read the registered before/after test above, not single wiggles.")
    else:
        st.caption("Not enough answered monthly-change questions for a 12-month window yet.")

# ---------------------------------------------------------------- stock answers
ui.section("Stock answers: how often a model repeats one number", "In a 12-month window, the share of a model's "
           "monthly-change answers that are one and the same value (its most repeated one). A model that knows the "
           "months gives varied answers; one that does not repeats a default.")
repeats = []
for m in models:
    r = ui.market_repeats(m, setting, fp, method=return_method)
    if len(r):
        repeats.append(r.assign(model=m))
with st.container(border=True):
    if repeats:
        st.html(ui.stock_answer_key())
        ui.show_cached_chart("stock_answer_chart", pd.concat(repeats), models, cut_rows, rel_rows)
        st.caption("Low (under ~20%): the answers change with the month, as memory would. High: one value dominates, "
                   "e.g. 0% for every month after Llama's release, or Qwen's −1.5% on the Nordic indices. Hover a "
                   "line for the repeated value and how many different answers the window had. All four indices of "
                   "the market are pooled.")
    else:
        st.caption("Not enough monthly-change answers for a 12-month window yet.")

# ---------------------------------------------------------------- Nordic vs US
if "us_index_return" in ts:
    ui.section("Nordic vs US", "The same rolling correlation for both markets, one panel per model: how much better the "
                               "models know the US market's monthly moves than the Nordic one's.")
    both = []
    for m in models:
        for label, meth in (("Nordic", "index_return"), ("US", "us_index_return")):
            c = ui.market_rolling(m, setting, fp, method=meth)
            if len(c):
                both.append(c.rename(columns={"r": "mean"}).assign(model=m, series=label))
    with st.container(border=True):
        if both:
            st.html(ui.market_compare_key())
            ui.show_cached_chart("rolling_market_chart", pd.concat(both), models, cut_rows, rel_rows)
            st.caption("Same questions, same months; the US indices are the S&P 500, Dow Jones, Nasdaq Composite and "
                       "Nasdaq-100. Hover a line for its 95% interval. US data is from Yahoo Finance: a private check, "
                       "never published.")
            comp = ui.market_comparison(tuple(models), setting, fp)
            if len(comp):
                st.markdown("**On the months before each model's cutoff**")
                st.dataframe(pd.DataFrame({
                    "Model": [ui.display_name(m) for m in comp["model"]],
                    "Nordic r": comp["r_nordic"].map("{:+.2f}".format),
                    "US r": comp["r_us"].map("{:+.2f}".format),
                    "US − Nordic": [f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]" for d, lo, hi in
                                    zip(comp["diff"], comp["lo"], comp["hi"])],
                    "Nordic beyond the US move": [f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]" for d, lo, hi in
                                                  zip(comp["beyond"], comp["beyond_lo"], comp["beyond_hi"])],
                    "Reading": [("Nordic-specific memory" if pb < 0.05 and b > 0 else
                                 "little Nordic memory" if r_n < 0.3 else
                                 "mostly the world market" if lo <= 0 else
                                 "some Nordic signal, not significant after correction")
                                for pb, b, lo, r_n in zip(comp["beyond_p_holm"], comp["beyond"], comp["beyond_lo"],
                                                          comp["r_nordic"])],
                }), hide_index=True, width="stretch")
                st.caption("r: how closely the monthly-change answers follow the real changes. US − Nordic: the "
                           "difference on the same months (bracket: 95% interval). Nordic beyond the US move: the "
                           "Nordic correlation left after removing what the US market did that month. Nordic and US "
                           "move together, so a model that only knows the world market still looks right on Nordic "
                           "months; this column shows whether it knows the Nordic market itself. Significance: Holm "
                           "across the models shown (the report uses all six).")
        else:
            st.caption("No model has answered both markets' monthly-change questions yet.")
