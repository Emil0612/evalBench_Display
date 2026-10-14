import pandas as pd
import streamlit as st

from evalbench import analysis, ui
from evalbench.llm.mock import is_local
from evalbench.methods import LABELS, MAIN_METHODS
from evalbench.pilot import FACT_TYPES

meta = ui.meta_by_id()
ui.page_header("Evidence", "The statistics behind the Profile: how much better each model does on facts from "
                           "<b>before</b> its training cutoff than on facts it <b>cannot know</b> (after its release, or "
                           "after its cutoff), corrected for guessing, plus the checks that make that difference credible. "
                           "Models are always shown in cutoff order: a diagnostic ranks nobody.",
)

FACT_ORDER = ("price", "return", "market", "event", "document")

with st.container(border=True):
    setting = ui.setting_picker_value("results")     # the answer-mode switch is drawn in the last row
    mode = setting[0]
    calls = ui.load_calls(setting)
    ts = ui.analysis_testsets(setting)
    answered_sets = set(calls["testset"].unique()) if len(calls) else set()   # once: ~160,000 rows
    available = [m for m in ts if ts[m][0] in answered_sets]
    # two short choices, as on the Profile: fact types, then the methods within them
    types = [t for t in FACT_ORDER if any(FACT_TYPES[m] == t for m in available)]
    counts = {t: sum(FACT_TYPES[m] == t for m in available) for t in types}
    ui.keep_choice(st.session_state, "ev_types", types, types, multi=True)
    chosen_types = st.pills("Fact types", types, selection_mode="multi", key="ev_types",
                            format_func=lambda t: t.capitalize() + (f" ({counts[t]})" if counts[t] > 1 else ""))
    in_types = ui.methods_in_order([m for m in available if FACT_TYPES[m] in (chosen_types or [])])
    methods_key = "ev_methods_" + "-".join(chosen_types or [])
    ui.keep_choice(st.session_state, methods_key, in_types, [m for m in in_types if m in MAIN_METHODS] or in_types,
                   multi=True)
    methods = st.pills("Methods", in_types, selection_mode="multi", key=methods_key, format_func=ui.method_md,
                       help="Controls (M7, M9: every item is before every cutoff, so they have no before/after "
                            "number), secondary (M1–M4, M6, M8) and private (M10, M16, M17) methods are not "
                            "selected by default.")
    ui.method_legend(in_types)
    methods = ui.methods_in_order(methods or [])
    all_models = sorted(calls["model"].unique()) if len(calls) else []
    real_first = sorted(all_models, key=lambda m: (is_local(m), ui.display_name(m)))
    real = [m for m in real_first if not is_local(m)]
    default = ui.default_models(real or real_first, ui.cutoffs(real or real_first, meta))
    ui.keep_choice(st.session_state, "ev_models", real_first, default, multi=True)
    models = st.multiselect("Models (up to 8)", real_first, key="ev_models", max_selections=8,
                            format_func=ui.display_name,
                            help="Controls are left out by default: p-values are Holm-corrected within each "
                                 "method across the models shown, so adding models makes the correction stricter.")
    models = ui.by_cutoff(models, meta)
    r1, r2 = st.columns([1.3, 2], vertical_alignment="bottom")
    ui.keep_choice(st.session_state, "ev_line", list(ui.LINES), "release")
    line = r1.segmented_control(
        "Compare with", list(ui.LINES), key="ev_line", format_func=ui.LINES.get,
        help="What 'cannot know' means. After release (default, pilot E): facts after the model became available; the "
             "months between cutoff and release are left out. After cutoff: facts more than 3 months after the stated "
             "training cutoff (more items, but relies on the maker's date). Controls have no release date and always "
             "use the cutoff.") or "release"
    with r2:
        ui.setting_picker("results")
if calls.empty:
    ui.empty_state(f"No answers stored for **{ui.setting_label(setting)}** yet.", "pages/run.py", "Start a run")
    st.stop()
ui.remember_page_choices(st.session_state)
if not models or not methods:
    st.info("Choose at least one model and one method.")
    st.stop()

after_word = "release" if line == "release" else "cutoff"
st.caption(f"Line: before the cutoff vs after the **{after_word}**. Test sets shown: "
           + " · ".join(f"{LABELS[m]}: {ts[m][0]}" for m in methods)
           + f" · p-values Holm-corrected within each method across the {len(models)} models shown")
fp = ui.fingerprint(calls, ui.cutoffs(models, meta))
table = ui.summary(tuple(models), tuple(methods), setting, fp, line).copy()
table["verdict"] = table.apply(ui.verdict, axis=1)
# the registered rules win over the bare statistic: too few items, refusal-driven and unresolved cells say so
table["reading"] = [analysis.reading(r) if ui.cell_status(r) in ("memorization detected", "none detected",
                                                                  "better after the cutoff", "not scorable")
                    else r["verdict"] for _, r in table.iterrows()]

# ---------------------------------------------------------------- at a glance
ui.section("At a glance", "Which methods find significant memorization for each model.")
KIND = {"memorization detected": "bad", "beats the blind baseline before the cutoff": "bad",
        "no significant difference": "muted",
        "better after the cutoff (check the setup)": "warn", "not confirmed: refusal-driven": "warn",
        "unresolved (too many unreadable answers)": "warn"}


def badge_text(method: str, row) -> str:
    short = LABELS[method].split()[0]
    if pd.isna(row["dkappa"]):
        return f"{short} n/a"
    if row["verdict"] == "no significant difference":
        return f"{short} ≤ {row['ci_hi']:+.2f}"
    if row["verdict"].startswith("too few items"):
        return f"{short} too few"
    if row["verdict"] == "not confirmed: refusal-driven":
        return f"{short} {row['dkappa']:+.2f} refusals"
    return f"{short} {row['dkappa']:+.2f}"
tiles = []
for model in models:
    rows = table[table["model"] == model].set_index("method")
    hits = int(rows["verdict"].map(ui.is_detection).sum())
    scorable = int(rows["dkappa"].notna().sum())
    cutoff = ui.cutoffs([model], meta)[model]
    found = [LABELS[m].split()[0] for m in methods if m in rows.index and ui.is_detection(rows.loc[m, "verdict"])]
    tiles.append({"title": ui.display_name(model), "value": (", ".join(found) + " detected") if found else "none detected",
                  "dot": ui.model_color(model), "color": ui.model_color(model), "tone": "bad" if found else "muted",
                  "sub": f"{ui.vendor(model)} · cutoff {cutoff or 'unknown'}"
                         + (f" · released {ui.release_date(model, meta)}" if ui.release_date(model, meta) else ""),
                  "badges": [ui.badge(badge_text(m, rows.loc[m]), KIND.get(rows.loc[m, "verdict"], "muted"))
                             for m in methods if m in rows.index]
                            + ([ui.badge(ui.CONTROL_FAILS[model], "warn")] if model in ui.CONTROL_FAILS else [])})
ui.cards(tiles)
st.caption("Red badge = memorization detected, showing Δκ (p < .05 after Holm correction). Grey badge = nothing "
           "detected, showing the most memorization the data still allows (upper end of the 95% interval). Refused "
           "questions are left out and their share is in the data table; only the registered tests (M5, M14–M18) "
           "count a refusal as wrong, and there an amber “refusals” badge marks a result that does not hold on the "
           "answered questions alone. “too few”: fewer than 20 answered questions on one side, so no verdict.")

if (table["n_samples"] > 1).any():
    k = int(table["n_samples"].max())
    ui.callout(f"<b>{k} samples per item.</b> Every number on this page is the <b>first attempt</b>. Best of {k} "
               f"(an item counts as known if any attempt got it) is in the full table under dkappa_best, computed only on the n_best items that have all {k} attempts — reported "
               f"next to the first attempt, never instead of it.", "info")

# ---------------------------------------------------------------- timeline
ui.section("Model timeline", "When each model's knowledge could end, next to the questions we can ask.")
with st.container(border=True):
    ui.show_chart(ui.timeline_chart(models, meta))
    st.caption("Each row is one model on a shared time axis. Dark: before its stated training cutoff (▾), "
               "facts it may have seen. Light grey: the grey zone between cutoff and release (●): probably "
               "unseen, but only the maker says so. Palest: after its release, facts it cannot know. "
               "Questions after the dashed line do not exist yet, which is why late models can only be judged"
               " on the cutoff line, or not at all.")

# ---------------------------------------------------------------- heatmap
ui.section("Memorization index Δκ", "Every model against every method. Red cells are significant detections; "
                                      "grey cells show the estimate but no significant difference.")
with st.container(border=True):
    ui.eyebrow(f"{len(models)} models × {len(methods)} methods · Δκ and 95% interval")
    ui.show_chart(ui.heatmap_chart(table, models, methods))
    if ui.heat_note(table):
        st.caption(ui.heat_note(table))
    st.caption(r"Stars: \* p < .05, \*\* p < .01 (cluster bootstrap, Holm-corrected within each method). n/a: no cutoff "
               "or no items on one side (positive controls such as M7 and M9 have none after any cutoff). Δκ can "
               "exceed 1 when the after-side score falls below chance.")

# ---------------------------------------------------------------- per method
ui.section("Per method", "Dot = estimate, line = 95% interval, models in cutoff order. An interval entirely right "
                         "of the dashed zero line is significant evidence of memorization.")
cols = st.columns(2, gap="medium")
for i, method in enumerate(methods):
    part = table[table["method"] == method]
    with cols[i % 2].container(border=True, height="stretch"):
        ui.method_eyebrow(method, f" · {int(part['dkappa'].notna().sum())} models")
        if part["dkappa"].notna().any():
            ui.show_chart(ui.forest_chart(part, all_models, order=models))
        else:
            st.caption("No scorable models for this method (missing cutoffs or no items on one side).")
        n_pre, n_post = int(part["n_pre"].max() or 0), int(part["n_post"].max() or 0)
        st.caption(f"Up to {n_pre} items before the cutoff and {n_post} after the {after_word} per model.")

# ---------------------------------------------------------------- over time
ui.section("Over time", "How each model scores on facts from each period. The memorization signature is a drop "
                        "after a model's cutoff (dashed line), at the latest by its release (solid line).")
DATING_VIEWS = {"years": "Within 3 months, by year", "dots": "Every answer"}
LEVELS = ("price_level", "known_level", "us_level")
SIZED = ("price_recall", "price_level", "annual_return", "monthly_return", "price_choice", "price_choice_prob")   # questions about companies of every size
SIZE_VIEWS = {"all": "All sizes", "large": "Large", "mid": "Mid", "small": "Small", "compare": "Compare sizes"}


def window_link(q_method: str, frames: list[pd.DataFrame], size: str = "all") -> None:
    """Open the questions behind one point of the rolling chart on the Items page (all answered versions)."""
    curves = {f["model"].iloc[0]: f.drop_duplicates("month") for f in frames if f["model"].iloc[0] in models}
    if not curves:
        return
    c1, c2, c3 = st.columns([1.4, 1, 1.4], vertical_alignment="bottom")
    who = c1.selectbox("Questions behind a point · model", list(curves), format_func=ui.display_name,
                       key="ev_q_link_model")
    months = list(curves[who]["month"])
    anchor = c2.selectbox("Window around", months, index=len(months) // 2, key="ev_q_link_month")
    m = pd.Period(anchor, "M")
    with c3:
        st.page_link("pages/items.py", label=f"Open {m - 2} … {m + 3} in Items", icon=":material/list_alt:",
                     query_params={"method": q_method, "model": who, "version": "all", "from": str(m - 2),
                                   "to": str(m + 3), "size": size if size in ui.SIZES else "all"})


def hedge_view(q_method: str) -> None:
    """The share of answers of exactly 0% per model over time: where a model stops claiming to know a month."""
    frames = [ui.hedge_rolling(m, q_method, setting, fp).assign(model=m) for m in models]
    frames = [f for f in frames if len(f)]
    cut_rows = [{"model": m, "month": str(pd.Period(ui.cutoffs([m], meta)[m], "M"))}
                for m in models if ui.cutoffs([m], meta)[m] is not None]
    rel_rows = [{"model": m, "month": str(pd.Period(ui.release_date(m, meta), "M"))}
                for m in models if ui.release_date(m, meta) is not None]
    if not frames:
        st.caption("Not enough answered questions for a rolling window.")
        return
    st.html(ui.rolling_chart_key("share of answers of 0%", blind=False))
    ui.show_chart(ui.rolling_chart(pd.concat(frames), models, cut_rows, rel_rows,
                                   value_label="share answering 0%"))
    st.caption("Each point: the share of a model's answers in a 6-month window that are exactly 0%, its way of saying "
               "'I don't know' when it must give a number. These answers are left out of every score on this page "
               "(scoring spec §7). A jump in the line marks the months a model stops claiming to know, often near its "
               "cutoff (dashed rule). A model that never answers 0% gives no signal here. Exploratory.")


@st.fragment
def over_time() -> None:
    """Switching the method or the view reruns only this section, not the whole page."""
    ui.keep_choice(st.session_state, "ev_q_method", methods, "price_choice" if "price_choice" in methods else methods[0])
    c1, c2 = st.columns([2, 1.3], vertical_alignment="bottom")
    q_method = c1.selectbox("Method", methods, key="ev_q_method", format_func=ui.method_option)
    views = ui.over_time_views(q_method)
    ui.keep_choice(st.session_state, "ev_q_view", list(views), "rolling")
    view = None if q_method == "headline_date" else c2.segmented_control(
        "View", list(views), key="ev_q_view", format_func=views.get,
                                help="Rolling: each point is a 6-month window, moved one month at a time, with a 95% "
                                     "band; every answered version of the method is used. Quarters: one dot per "
                                     "quarter, current version only."
                                     + (f" {views['hedge']}: how often the model answers "
                                        f"{views['hedge'].removeprefix('Answers of ')}, its way of saying 'I don't "
                                        "know'." if "hedge" in views else "")) or "rolling"
    size = "all"
    if q_method in SIZED and view != "hedge":
        ui.keep_choice(st.session_state, "ev_q_size", list(SIZE_VIEWS), "all")
        size = st.segmented_control("Company size", list(SIZE_VIEWS), key="ev_q_size", format_func=SIZE_VIEWS.get,
                                    help="Market-cap terciles of the Nordic listings. Large companies are written about "
                                         "far more, so memory should show there first. 'Compare sizes' draws one line "
                                         "per size in each model's panel.") or "all"
        if view == "quarters" and size == "compare":
            st.caption("Compare sizes is drawn in the rolling view; the quarters show all sizes.")
            size = "all"
    ui.remember_page_choices(st.session_state)
    if view == "hedge":
        hedge_view(q_method)
        return
    scale = ("Score: closeness to the true price (1 = exact, 0.5 = about 30–40% off, 0 = off by a factor of 2 or more), "
             "the same score as the tiles above." if q_method in LEVELS else "Score: share of answers scored correct.")
    if q_method == "headline_date":   # its generic score over time is biased by construction (plan §3)
        cut_rows = [{"model": m, "month": str(pd.Period(ui.cutoffs([m], meta)[m], "M"))}
                    for m in models if ui.cutoffs([m], meta)[m] is not None]
        rel_rows = [{"model": m, "month": str(pd.Period(ui.release_date(m, meta), "M"))}
                    for m in models if ui.release_date(m, meta) is not None]
        dating_view = c2.segmented_control("View", list(DATING_VIEWS), key="ev_q_dating_view",
                                           format_func=DATING_VIEWS.get, default="years") or "years"
        if dating_view == "years":
            own = {m: ui.scores(m, q_method, setting, fp) for m in models}
            frames = [ui.dating_years(df).assign(model=m) for m, df in own.items() if len(df)]
            blind = ui.dating_years(ui.scores(ui.BLIND_ID, q_method, setting, fp))
            if frames:
                ui.show_chart(ui.dating_year_chart(pd.concat(frames + [blind.assign(model=ui.BLIND_ID)]), models,
                                                   {m: ui.default_month(df) for m, df in own.items()},
                                                   cut_rows, rel_rows))
                st.caption("Each panel: the share of a model's headlines it dated within 3 months of the truth, per "
                           "year of publication (coloured line), next to a text-only baseline that guesses from the "
                           "headline's words (grey dashed). The shaded band is the year around the model's default "
                           "answer: the month it names most often, whatever the headline, when it does not recognise "
                           "it. Headlines from those months are dated correctly by habit, so correct answers there may "
                           "be luck, not memory. Real memory "
                           "of when things happened is the coloured line above the grey one outside the shaded band; "
                           "where the line falls to zero, the model's knowledge of Nordic news has ended. Dashed rule: "
                           "stated cutoff; solid rule: release. A refusal counts as a miss.")
            else:
                st.caption("No answered M18 questions for these models.")
            return
        frames = [ui.dating_frame(ui.scores(m, q_method, setting, fp)).assign(model=m) for m in models]
        frames = [f for f in frames if len(f)]
        if frames:
            ui.show_chart(ui.dating_chart(pd.concat(frames), models, cut_rows, rel_rows))
            st.caption("M18 is not shown as a score over time: a model cannot name a month after its knowledge ends, "
                       "so its score collapses after the cutoff whatever it remembers, and the months near its default "
                       "answer score high for free. Instead, each dot is one answered headline: where it was "
                       "published (across) and the month the model named (up). The dotted diagonal is a perfect "
                       "answer; a horizontal band of dots is the model's default month; strong dots lie within 3 "
                       "months of the truth. Real dating shows as strong dots on the diagonal away from the default "
                       "band. Dashed rule: cutoff; solid rule: release. The registered test (above) compares each "
                       "model with a text-only baseline on the months before its cutoff.")
        else:
            st.caption("No answered M18 questions for these models.")
        return
    if view == "rolling" and size == "compare":
        frames = [ui.rolling(m, q_method, setting, fp, size="compare").assign(model=m) for m in models]
        frames = [f for f in frames if len(f)]
        cut_rows = [{"model": m, "month": str(pd.Period(ui.cutoffs([m], meta)[m], "M"))}
                    for m in models if ui.cutoffs([m], meta)[m] is not None]
        rel_rows = [{"model": m, "month": str(pd.Period(ui.release_date(m, meta), "M"))}
                    for m in models if ui.release_date(m, meta) is not None]
        if frames:
            ui.show_chart(ui.rolling_size_chart(pd.concat(frames), models, cut_rows, rel_rows))
            st.caption("One line per company size in each model's panel: large (blue, solid), mid (grey, dashed), small "
                       "(orange, dotted). Each point is the mean over the 6-month window around that month; windows "
                       "with fewer than 15 questions of a size are left out. Dashed rule: cutoff; solid rule: release. "
                       "Hover a line for its 95% interval. " + scale)
            window_link(q_method, frames)
        else:
            st.caption("Not enough answered questions per size for a rolling window.")
        return
    if view == "rolling":
        frames = [ui.rolling(m, q_method, setting, fp, size=size).assign(model=m) for m in [*models, ui.BLIND_ID]]
        frames = [f for f in frames if len(f)]
        cut_rows = [{"model": m, "month": str(pd.Period(ui.cutoffs([m], meta)[m], "M"))}
                    for m in models if ui.cutoffs([m], meta)[m] is not None]
        rel_rows = [{"model": m, "month": str(pd.Period(ui.release_date(m, meta), "M"))}
                    for m in models if ui.release_date(m, meta) is not None]
        if frames:
            st.html(ui.rolling_chart_key())
            ui.show_chart(ui.rolling_chart(pd.concat(frames), models, cut_rows, rel_rows))
            st.caption("Each point: the mean score of the questions dated in a 6-month window around that month (from "
                       "2 months before to 3 after), with a 95% band from resampling companies. Windows with fewer than "
                       "15 questions are left out. Grey dashed: the blind guesser, which knows nothing; when it rises, "
                       "the questions of that period are simply easier, not better remembered. "
                       + (f"{SIZE_VIEWS[size]} companies only. " if size in ui.SIZES else "") + scale)
            window_link(q_method, frames, size)
        else:
            st.caption("Not enough answered questions for a rolling window.")
        return
    frames, cut_rows, rel_rows = [], [], []
    for m in models:
        df = ui.scores(m, q_method, setting, fp)
        if len(df) and size in ui.SIZES:
            df = df[ui.size_of(df, q_method) == size]
        if not len(df):
            continue
        q = analysis.by_quarter(ui.over_time_score(df, q_method))
        if len(q):
            frames.append(q.assign(model=m))
        c = ui.cutoffs([m], meta)[m]
        if c is not None:
            cut_rows.append({"model": m, "quarter": str(pd.Period(c, freq="Q"))})
        r = ui.release_date(m, meta)
        if r is not None:
            rel_rows.append({"model": m, "quarter": str(pd.Period(r, freq="Q"))})
    if frames:
        ui.show_chart(ui.quarter_chart(pd.concat(frames), cut_rows, models, all_models, rel_rows), height=300)
        st.caption(scale)


with st.container(border=True):
    over_time()

boundary = st.expander("Exploratory: where does knowledge end? An estimate the pilots found noisy", expanded=False)
boundary.caption("The single date where each model's score drops the most, found from the answers alone (all "
                 "selected methods pooled, chance-corrected) and compared with the stated cutoff. Pilots C and D found "
                 "these estimates noisy (years before the stated cutoffs, with wide intervals), and memory fades "
                 "gradually rather than stopping on one date; the headline-dating and cutoff-ladder summaries are the "
                 "better guide to where knowledge ends. Read as exploratory only.")
bounds = []
for m in models:
    eff = ui.effective(m, tuple(methods), setting, fp)
    stated = ui.cutoffs([m], meta)[m]
    if eff.get("detected"):
        rel = analysis.versus_stated(eff, stated)
        badges = [ui.badge(rel, "good" if rel == "consistent with stated" else "warn")] if rel else []
        bounds.append({"title": ui.display_name(m), "value": eff["breakpoint"].strftime("%b %Y"),
                       "dot": ui.model_color(m), "color": ui.model_color(m),
                       "sub": f"last known {eff['last_known']} · first unknown {eff['first_unknown']} · "
                              f"95% interval {eff['lo']:%b %Y} – {eff['hi']:%b %Y} · drop {eff['step']:+.2f} · "
                              f"stated cutoff {stated or 'unknown'}", "badges": badges})
    else:
        why = "too few items" if eff.get("n", 0) < 30 else "no clear drop in the scores"
        bounds.append({"title": ui.display_name(m), "value": "—", "dot": ui.model_color(m),
                       "color": ui.model_color(m), "tone": "muted",
                       "sub": f"{why} · stated cutoff {stated or 'unknown'}",
                       "badges": [ui.badge("not estimated", "muted")]})
with boundary:
    ui.cards(bounds)

# ---------------------------------------------------------------- breakdown
ui.section("Where does it come from?", "The same index within subgroups. Subgroups are small, so read this as "
                                       "exploratory (uncorrected p-values).")


@st.fragment
def where_from() -> None:
    """Changing model, method or split reruns only this section."""
    c1, c2, c3 = st.columns(3)
    ui.keep_choice(st.session_state, "ev_b_model", models, models[0])
    split_methods = ui.generic_methods(methods)
    if not split_methods:
        st.caption("The selected methods are read by their own registered tests (M14-M18), which are not split into "
                   "subgroups here.")
        return
    ui.keep_choice(st.session_state, "ev_b_method", split_methods, split_methods[0])
    b_model = c1.selectbox("Model", models, format_func=ui.display_name, key="ev_b_model")
    b_method = c2.selectbox("Method", split_methods, format_func=ui.method_option, key="ev_b_method",
                            help="M14-M18 are read by their own registered tests and are not split here.")
    ui.keep_choice(st.session_state, "ev_b_key", ui.STRATA[b_method], ui.STRATA[b_method][0])
    b_key = c3.selectbox("Split by", ui.STRATA[b_method], format_func=lambda k: k.replace("_", " "), key="ev_b_key",
                         help="company size = market-cap tercile of the company the item is about. Big companies' "
                              "news is copied far more widely online, so it is a proxy for training exposure.")
    ui.remember_page_choices(st.session_state)
    b = ui.breakdown(b_model, b_method, b_key, setting, fp, line)
    if len(b):
        st.dataframe(b, hide_index=True, width="stretch", column_config={
            "stratum": st.column_config.TextColumn(b_key.replace("_", " ")),
            "n_pre": st.column_config.NumberColumn("items before"),
            "n_post": st.column_config.NumberColumn("items after"),
            "mean_pre": st.column_config.NumberColumn("score before", format="%.3f"),
            "mean_post": st.column_config.NumberColumn("score after", format="%.3f"),
            "dkappa": st.column_config.NumberColumn("Δκ", format="%+.3f"),
            "ci_lo": st.column_config.NumberColumn("95% low", format="%+.3f"),
            "ci_hi": st.column_config.NumberColumn("95% high", format="%+.3f"),
            "p_value": st.column_config.NumberColumn("p", format="%.4f")})


with st.container(border=True):
    where_from()

# ---------------------------------------------------------------- robustness
ui.section("Robustness checks", "Two checks that a before/after difference is memory and not something else. Both "
                                "use the stored answers only.")
with st.container(border=True):
    ui.eyebrow(ui.placebo_note(models, ui.cutoffs(models, meta)))
    pl = ui.placebo_table(tuple(models), tuple(methods), setting, fp)
    if len(pl):
        flagged = pl[pl["p_holm"] < 0.05]
        (ui.callout if len(flagged) else st.caption)(
            f"<b>{len(flagged)} placebo cell(s) significant</b>: the matching results above may reflect drift, not "
            "memory." if len(flagged) else "No placebo cell is significant after Holm correction: no sign that the "
            "method mistakes drift over time for memory.", *(["warn"] if len(flagged) else []))
        st.dataframe(pl.assign(model=pl["model"].map(ui.display_name), method=pl["method"].map(LABELS))[
            ["model", "method", "n_pre", "n_post", "dkappa", "ci_lo", "ci_hi", "p_holm"]], hide_index=True,
            width="stretch", column_config={
                "n_pre": st.column_config.NumberColumn("items before 2022-07"),
                "n_post": st.column_config.NumberColumn("items 2022-07 to 2023-09"),
                "dkappa": st.column_config.NumberColumn("placebo Δκ", format="%+.3f"),
                "ci_lo": st.column_config.NumberColumn("95% low", format="%+.3f"),
                "ci_hi": st.column_config.NumberColumn("95% high", format="%+.3f"),
                "p_holm": st.column_config.NumberColumn("p (Holm)", format="%.3f")})
with st.container(border=True):
    ui.eyebrow("Staggered cutoffs · the same items, models with different cutoffs")
    sg = ui.staggered_table(tuple(models), tuple(methods), setting, fp, line)
    if len(sg):
        st.dataframe(sg.assign(pair=[f"{ui.display_name(a)} → {ui.display_name(b)}" for a, b in
                                     zip(sg["earlier"], sg["later"])])[
            ["pair", "n_window", "n_ref", "did", "ci_lo", "ci_hi", "p_holm", "answered_did", "answered_p_holm",
             "reading"]], hide_index=True, width="stretch",
            column_config={
                "n_window": st.column_config.NumberColumn("items between the cutoffs"),
                "n_ref": st.column_config.NumberColumn("items before both"),
                "did": st.column_config.NumberColumn("later model's extra advantage", format="%+.3f"),
                "ci_lo": st.column_config.NumberColumn("95% low", format="%+.3f"),
                "ci_hi": st.column_config.NumberColumn("95% high", format="%+.3f"),
                "p_holm": st.column_config.NumberColumn("p (Holm)", format="%.3f"),
                "answered_did": st.column_config.NumberColumn("same, both answered", format="%+.3f"),
                "answered_p_holm": st.column_config.NumberColumn("p (Holm), both answered", format="%.3f")})
        st.caption("For each pair: how much better the later-cutoff model does than the earlier one on items dated "
                   + ("after the earlier model's release" if line == "release" else "after the earlier model's cutoff")
                   + " and before the later model's cutoff, minus the same difference on items before both "
                   "(chance-corrected, pooled over the selected methods). Question difficulty and market regime "
                   "cancel, because both models answered the same items. Positive = memory only the later model can "
                   "have; it counts only if it also holds on the questions both answered (refusals scored as wrong "
                   "would otherwise pass for memory). Models whose cutoff we set by hand (Qwen3.8 Flash) are left "
                   "out. The registered version of this check is the cutoff-ladder report.")
    else:
        st.caption("Needs at least two selected real models with different cutoffs and items between them.")

# ---------------------------------------------------------------- how to read
ui.section("How to read these results")
with st.container(border=True):
    st.markdown(
        "- **Δκ near 0 with a narrow interval** — the model does no better on facts it could have seen in training. "
        "No memorization was detected *with this test and sample*; it is not proof the model never saw the data. "
        "The grey “≤ X” badges give the most memorization the data still allows.\n"
        "- **Before the cutoff means *could* have seen, not *did*.** Training data is a sample of the web, so the "
        "before-group mixes seen and unseen items; Δκ is therefore a lower bound on memorization of what was seen.\n"
        "- **Δκ clearly above 0** — the model knows the pre-cutoff facts better. Agreement across several methods is "
        "much stronger evidence than a single method.\n"
        "- **Wide intervals** — too few items on one side of the cutoff (common for recent cutoffs). Run more items.\n"
        "- **Controls** — the oracle must be flagged everywhere and random/blind must stay at zero; if not, the setup "
        "is broken, not the model.")

# ---------------------------------------------------------------- data
ui.section("Data")
cols_out = ["model", "method", "reading", "n_pre", "n_post", "mean_pre", "mean_post", "dkappa", "ci_lo", "ci_hi",
            "p_value", "p_holm", "p_perm", "coverage_pre", "coverage_post", "answered_mean_pre", "answered_mean_post",
            "abstain_rate", "refusal_rate", "malformed_rate", "hallucination_rate", "known_p", "n_samples",
            "n_best", "dkappa_best", "ci_lo_best", "ci_hi_best"]
out = table.reindex(columns=cols_out).assign(model=lambda d: d["model"].map(ui.display_name),
                                             method=lambda d: d["method"].map(LABELS))
with st.expander("Full table: coverage, hallucination and every p-value"):
    st.dataframe(out.round(4), hide_index=True, width="stretch")
    st.caption("coverage = share answered. abstain / refusal / malformed = share of items where the model said it "
               "did not know, declined, or gave no usable answer (kept apart from wrong answers). "
               "answered_mean = score among answered items only. "
               "hallucination_rate = share of fabricated companies given a price. known_p (DE-COP) = Fisher test of "
               "items recognised in ≥3 of 4 rotations before vs after the cutoff. p_perm = item-level permutation test.")
c1, c2, _ = st.columns([1, 1, 2])
c1.download_button("Summary CSV", out.to_csv(index=False), "evalbench_summary.csv", "text/csv",
                   icon=":material/download:", width="stretch")
items_csv = pd.concat([ui.analysis_frame(m, me, setting, fp, line, meta).assign(model=m, method=me, line=line)
                       for m in models for me in methods],
                      ignore_index=True)
c2.download_button("Item scores CSV", items_csv.to_csv(index=False), "evalbench_item_scores.csv", "text/csv",
                   icon=":material/download:", width="stretch")
