import streamlit as st

from evalbench import ui
from evalbench.llm.mock import is_local
from evalbench.llm.selfhosted import is_self_hosted
from evalbench.llm.openrouter import load_pins
from evalbench.methods import METHODS
from evalbench.pilot import FACT_TYPES

meta = ui.meta_by_id()
ui.page_header(
    "Knowledge profile",
    "What each model can recall, period by period, next to what it scores on facts it cannot know. A "
    "<b>diagnostic, not a contest</b>: there is no ranking and no medal for remembering the most.",
    meta="One cell per model and year, with baselines measured from each model's own answers "
         "instead of assumed.")
ui.callout("Diagnostic evidence, not a certificate of backtest safety. <b>None detected</b> means this sample and "
           "method found nothing. It never means a period is clean.", "muted")

FACT_ORDER = ("price", "return", "market", "event", "document")

with st.container(border=True):
    # the answer mode sits with the other display settings below; read it first, the data depends on it
    setting = ui.setting_picker_value("profile")
    calls = ui.load_calls(setting)
    ts = ui.analysis_testsets(setting)
    answered_sets = set(calls["testset"].unique()) if len(calls) else set()   # once: ~160,000 rows
    available = [m for m in ts if ts[m][0] in answered_sets]
    if not available:
        ui.setting_picker("profile")                 # keep the switch reachable, or there is no way back
        ui.empty_state(f"No answers stored for **{ui.setting_label(setting)}** yet.", "pages/run.py", "Start a run")
        st.stop()
    # two short choices instead of one long row: the fact type, then the method within it
    types = [t for t in FACT_ORDER if any(FACT_TYPES[m] == t for m in available)]
    counts = {t: sum(FACT_TYPES[m] == t for m in available) for t in types}
    ui.keep_choice(st.session_state, "prof_fact", types, types[0])
    fact = st.segmented_control("Fact type", types, key="prof_fact",
                                format_func=lambda t: t.capitalize() + (f" ({counts[t]})" if counts[t] > 1 else ""),
                                help="Price: exact price, price level, Split Clock, which price fits, known levels "
                                     "and choices (positive controls), US levels (private check) · Return: annual and "
                                     "monthly return, known returns (positive control) · Market: index levels and "
                                     "monthly changes (US: private check) · Event: hidden number, headline date · "
                                     "Document: spot the original, guided completion") or types[0]
    in_type = ui.methods_in_order([m for m in available if FACT_TYPES[m] == fact])
    ui.keep_choice(st.session_state, f"prof_method_{fact}", in_type, in_type[0])
    method = st.segmented_control(
        "Method", in_type, key=f"prof_method_{fact}", format_func=ui.method_md,
        help="Controls: facts every model should know. Private: built from data that may never be published.") \
        or in_type[0]
    answered = sorted(set(calls.loc[calls["testset"] == ts[method][0], "model"]))
    real = ui.by_cutoff([m for m in answered if not is_local(m)], meta)
    controls = ui.by_cutoff([m for m in answered if is_local(m)], meta)
    ui.restore_widgets(st.session_state, ["prof_hidden", "prof_controls", "prof_color", "prof_size"])
    # every model that answered this method is shown unless the reader switched it off; that choice is kept across
    # methods and pages (prof_hidden), so the picker is always derived from it
    if "prof_hidden" not in st.session_state:          # first visit: the auxiliary ladder models start switched off
        st.session_state["prof_hidden"] = sorted(ui.AUX_MODELS)
    hidden = set(st.session_state.get("prof_hidden") or [])
    pick_key = f"prof_models_{method}"

    def _picked():
        st.session_state["prof_hidden"] = sorted(ui.update_hidden(
            set(st.session_state.get("prof_hidden") or []), real, st.session_state.get(pick_key) or []))

    def _all(on: bool):
        st.session_state["prof_hidden"] = sorted(ui.update_hidden(
            set(st.session_state.get("prof_hidden") or []), real, real if on else []))

    st.session_state[pick_key] = ui.profile_selection(real, hidden)
    # full width, so the chips wrap onto a second line instead of being cut off
    chosen = st.pills("Models", real, selection_mode="multi", key=pick_key, format_func=ui.display_name,
                      on_change=_picked,
                      help="Every model that answered this method, earliest cutoff first. Click a model to switch it "
                           "off or on; models you switch off stay off on the other methods too.") or []
    p1, p2, _ = st.columns([1, 1, 10])
    p1.button("All", key="prof_all", on_click=_all, args=(True,), width="stretch",
              help="Show every model that answered this method.")
    p2.button("None", key="prof_none", on_click=_all, args=(False,), width="stretch",
              help="Clear the models, to pick a few.")
    ui.keep_choice(st.session_state, "prof_controls", controls, [], multi=True)
    picked_controls = st.pills("Controls", controls, selection_mode="multi", key="prof_controls",
                               format_func=ui.display_name,
                               help="Local check models (no memory, perfect memory, random answers). Off by "
                                    "default.") or [] if controls else []
    models = list(chosen) + list(picked_controls)
    d0, d1, d2, d3 = st.columns([1, 1.4, 1.4, 1.1], vertical_alignment="bottom")
    with d0:
        ui.setting_picker("profile")
    high_chance = METHODS[method].P0 > 0          # M3 (25%) and M6 (50%): raw recall looks high by guessing alone
    if st.session_state.get("prof_color_for") != method:
        st.session_state["prof_color"] = "excess" if high_chance else "recall"
        st.session_state["prof_color_for"] = method
    level = method in ui.LEVEL_METHODS   # scored by closeness, not as a rate
    color = d1.segmented_control("Colour", ["recall", "excess"], key="prof_color",
                                 format_func={"recall": ui.measure_name(method).removeprefix("Mean ").capitalize(),
                                              "excess": "Above baseline"}.get,
                                 help="The number in a cell is always the measured recall. The colour shows either "
                                      "that recall or how far it is above the model's own 'cannot know' level. Use "
                                      "'Above baseline' for methods with a high guessing level, such as DE-COP.") or "recall"
    ui.keep_choice(st.session_state, "prof_base", ["cutoff", "release"], ui.PROFILE_DEFAULT_BASELINE)
    if level:
        st.caption("Price levels are scored by **closeness** to the true price: 1 = exact, 0.5 = about 30–40% off, "
                   "0 = off by a factor of 2 or more. The number in a cell is the mean closeness (0.23), not a "
                   "percentage.")
    paired = method in ui.PAIRED_BLIND
    baseline = d2.segmented_control("Baseline", ["cutoff", "release"], key="prof_base", disabled=paired,
                                    format_func={"cutoff": "After cutoff", "release": "After release"}.get,
                                    help="This method compares each year with a blind baseline on the same "
                                         "questions, as its registered test does, so this switch does not apply."
                                         if paired else
                                         "What 'cannot know' means: items after the model's training cutoff (more "
                                         "items), or after the model became available (stricter).") or "cutoff"
    if method == "headline_date":
        st.caption("Headline dating is scored 1 − months off / 12 (1 = the right month, 0 = a year or more off). "
                   "Each year is compared with the **text-only guess on the same headlines**, as in the registered "
                   "test: a model's own score after its cutoff is near 0 by construction, so it cannot be the "
                   "baseline. **◇ habit** marks the year of the month a model names most often: when it does not "
                   "know, that answer lands near these headlines and scores by luck.")
    elif method in ui.INDEX_LEVELS:
        st.caption("Index levels are scored by closeness (1 = the exact close, 0 = 25% or more off). Each year is "
                   "compared with the **blind guesser on the same months** (last year's level), as in the "
                   "registered test: almost every model knows roughly where a main index trades, so beating that "
                   "guess is what shows memory. Years right after a large market move (such as 2021) make last "
                   "year's level a poor guess, so read single years with care.")
    elif method in ui.CORR_METHODS:
        st.caption("Each cell is the **correlation** between the model's answers and the real monthly moves that "
                   "year (−1 to +1; 0 = no relation), against the same correlation after the cutoff or release, as "
                   "in the registered test. A share of right up/down calls would mislead here: most months go up, "
                   "so 'always up' already scores about 0.6.")
    sized = ui.splits_by_size(method)
    by_size = d3.toggle("Split by company size", key="prof_size", disabled=not sized,
                        help="Rows per market-cap tercile. Cells get small quickly; small cells are hatched."
                        if sized else "This method's questions have no company sizes (indices, headlines, the "
                                      "largest companies only, or US shares).") and sized

ui.remember_page_choices(st.session_state)
if not models:
    st.info("Choose at least one model.")
    st.stop()

order = ui.by_cutoff(models, meta)
# one key for every method and selection, so the per-model cache (and the background warm-up) is shared
fp = ui.fingerprint(calls, ui.cutoffs(sorted(calls["model"].unique()), meta))
cells = ui.profile(tuple(order), method, setting, fp, by_size)
if not cells:
    st.info("None of the selected models has a known cutoff and answers for this method.")
    st.stop()
periods = sorted({c["period"] for c in cells})
strata = [s for s in ("all", "large", "mid", "small", "none") if any(c["stratum"] == s for c in cells)]
pins = load_pins()

changed, tested = ui.baseline_changes(cells)
if tested and not paired:
    other = "after-release" if baseline == "cutoff" else "after-cutoff"
    st.caption(f"Baseline: **{'after release' if baseline == 'release' else 'after cutoff'}**. {changed} of {tested} "
               f"readings on this page would differ with the {other} baseline"
               + (" (mostly cells that drop to 'insufficient' because fewer items come after release)." if changed
                  else ": the choice of baseline does not change any conclusion here."))

rc = ui.READING_COLORS[ui.theme_name()]
st.html('<div class="eb-key">'
        + "".join(f'<span>{ui.model_dot(m)}{ui._esc(ui.display_name(m))}</span>' for m in order)
        + '</div><div class="eb-key">'
        f'<span><b style="color:{rc["detected"]}">●</b> above baseline (outlined in red)</span>'
        f'<span><b style="color:{rc["none"]}">○</b> none detected</span>'
        '<span><b>–</b> too few items to judge (hatched; the grey number is the recall, with no verdict)</span>'
        f'<span><b style="color:{rc["warn"]}">?</b> unresolved</span>'
        '<span><span class="sw" style="outline:1px dashed #898781;outline-offset:-2px"></span>after the model\'s cutoff'
        '</span><span><i>grey italic year</i> = after release</span><span>▾ = cutoff year</span>'
        '<span>● = release year</span>'
        '<span><span class="sw" style="background:linear-gradient(90deg,var(--kh-lo,#eef4fc),#184f95)"></span>'
        'more recalled</span></div>')

for m in order:
    mc = [c for c in cells if c["model"] == m]
    if not mc:
        continue
    cut, rel = ui.cutoffs([m], meta)[m], ui.release_date(m, meta)
    if is_local(m):
        badges = [ui.badge("local control", "muted")]
    elif is_self_hosted(m):
        # own hardware, fixed weights: the configuration is known exactly, unlike a hosted model
        sh = (meta.get(m) or {}).get("self_hosted") or {}
        badges = [ui.badge("self-hosted" + "".join(f" · {sh[k]}" for k in ("quantization", "hardware") if sh.get(k)),
                           "info")]
        if sh.get("revision"):
            badges.append(ui.badge(f"weights {str(sh['revision'])[:12]}", "muted"))
    else:
        badges = [ui.badge("diagnostic · hosted via OpenRouter", "warn")]
    if not is_local(m) and not is_self_hosted(m):
        # how the answers shown were actually served, not what the pins say today
        served = calls[(calls["model"] == m) & (calls["testset"] == ts[method][0])]
        routes = set(served["route"].fillna("")) if "route" in served else {""}
        providers = sorted(set(served["provider"].dropna()))
        if len(routes) == 1 and "" not in routes:
            badges.append(ui.badge(f"pinned: {next(iter(routes))}", "info"))
        else:
            badges.append(ui.badge("served by " + ", ".join(providers[:3]) + (" …" if len(providers) > 3 else ""),
                                   "info" if len(providers) == 1 else "warn"))
        if m in pins and routes != {pins[m]}:
            badges.append(ui.badge(f"new runs pinned to {pins[m]}", "muted"))
    facts = sum(c["facts"] for c in mc if not c["after_cutoff"])
    with st.container(border=True):
        st.html(f'<div class="eb-model" style="border-left-color:{ui.model_color(m)}">'
                f'<div class="eb-title" style="gap:0.5rem"><b style="font-size:1.08rem">{ui.model_dot(m)}'
                f'{ui._esc(ui.display_name(m))}</b>{"".join(badges)}</div><div class="eb-meta" style="margin:0">'
                f'declared cutoff {cut or "unknown"} ({ui.cutoff_source(m, ui.load_overrides(), meta)}) · available since '
                f'{rel or "–"} · {facts} '
                f'facts before the cutoff · <b>{ui._esc(ui.baseline_summary(mc, baseline))}</b></div></div>')
        habit = ui.default_month(ui.scores(m, method, setting, fp)) if method == "headline_date" else None
        st.html(ui.profile_grid_html(mc, periods, strata, color=color, baseline=baseline, cutoff=cut, release=rel,
                                     stratum_labels={s: ui.row_label(method, s) for s in strata}, habit=habit))

st.caption("Recall = the method's item score (M3: share of the 4 orderings answered correctly, chance 25%; M4: "
           "overlap gain from the hint; M18: 1 − months off / 12; M15/M17: the correlation with the real moves). "
           "Refused questions are left out of every number and their share is shown on hover. Readings compare each period with the model's own level on facts it cannot "
           "know, using a cluster bootstrap by company (20,000 draws) and Holm correction across every cell on this "
           "page. M18 and M14/M16 cells compare with a blind baseline on the same questions instead, resampling "
           "months, as their registered tests do.")

ui.section("How to read a profile")
st.markdown(
    "Each cell is one model, one company-size group and one year, answered without tools in a fresh context. The "
    "comparison point is **measured**, not assumed: the same model's score on matched facts from after its cutoff "
    "(or after its release). That matters because some methods have a high guessing level. In pilot C, Gemini picked "
    "the original DE-COP passage 78% of the time even after its cutoff, and a fixed 25% baseline would have called "
    "every year exposed.\n\n"
    "Columns after a model became available are a check on the method: it cannot have seen those facts, so recall "
    "there should sit near its baseline.\n\n"
    "> *Under this protocol, the checkpoint showed limited detectable recall for the sampled Nordic facts in the "
    "stated period. The sample supports further exploratory testing; it does not certify an uncontaminated historical "
    "window.*\n\n"
    "That is the strongest statement a “none detected” cell supports. There is no “safe before” date.")

# after this view is drawn: compute the other methods' profiles in the background (this fact type first), so that
# switching method is instant once the warm-up has reached it
_order = sorted(available, key=lambda m: (FACT_TYPES[m] != fact, m == method))
_by_set = calls.groupby("testset")["model"].unique()
_jobs = [(m, tuple(ui.by_cutoff([x for x in _by_set.get(ts[m][0], []) if not is_local(x)], meta)))
         for m in _order if m != method]
ui.warm_profiles(_jobs, setting, fp)
