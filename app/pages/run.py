import hashlib
import math

import pandas as pd
import streamlit as st

from evalbench import runner, selections, store, ui, worker
from evalbench.cutoffs import save_edits
from evalbench.llm import selfhosted
from evalbench.llm.mock import is_local
from evalbench.llm.openrouter import load_pins
from evalbench.methods import PRIVATE_CHECKS, SECONDARY

ui.page_header("Run", "Choose models and methods, review exactly what will be asked and what it can cost, then "
                      "confirm. Every answer is stored, so nothing you already ran is paid for again. A run goes on in "
                      "the background: switch pages or close the tab, it keeps going.",
               meta="Three steps: choose · review · confirm")
meta, ts = ui.meta_by_id(), ui.testsets()
if not ts:
    ui.empty_state("No test sets found yet.", "pages/testsets.py", "Test sets")
    st.stop()
all_options = ui.LOCAL_MODELS + ui.picker_models(True)
RUN_KEYS = ["run_models", "run_all_variants", "run_methods", "run_n", "run_mode", "run_temp", "run_reasoning",
            "run_samples", "run_seed_on", "run_seed", "run_privacy"]
ui.restore_widgets(st.session_state, RUN_KEYS)       # back from another page: the choices made here last time
st.session_state.setdefault("run_models", ui.LOCAL_MODELS[:2])
st.session_state.setdefault("run_all_variants", False)
st.session_state.setdefault("run_methods", [m for m in ts if m not in SECONDARY | PRIVATE_CHECKS])
st.session_state.setdefault("run_n", 40)
st.session_state.setdefault("run_mode", "forced")
st.session_state.setdefault("run_temp", 0.0)
st.session_state.setdefault("run_reasoning", "auto")
st.session_state.setdefault("run_samples", 1)
st.session_state.setdefault("run_seed_on", False)
st.session_state.setdefault("run_seed", 42)
st.session_state.setdefault("run_privacy", "standard")
PRIVACY_LABELS = {"standard": "Standard (any provider)", "no_training": "No training on prompts",
                  "zdr": "Zero data retention"}
REASONING_LABELS = {"auto": "Auto (as little as the model allows)", "none": "None", "minimal": "Minimal",
                    "low": "Low", "medium": "Medium", "high": "High"}


def load_selection(name: str) -> None:
    s = selections.load()[name]
    skipped = [m for m in s["models"] if m not in all_options]
    st.session_state.update(run_models=[m for m in s["models"] if m in all_options],
                            run_methods=[m for m in s["methods"] if m in ts], run_n=s["n"], run_mode=s["mode"],
                            run_temp=float(s.get("temperature", 0.0)), run_reasoning=s.get("reasoning", "auto"),
                            run_samples=int(s.get("samples", 1)), run_seed_on=s.get("seed") is not None,
                            run_seed=int(s["seed"]) if s.get("seed") is not None else 42,
                            run_privacy=s.get("privacy", "standard"))
    note = f" ({len(skipped)} model(s) no longer available were skipped)" if skipped else ""
    st.session_state["sel_msg"] = ("good", f"Loaded <b>{ui._esc(name)}</b>{note}.")


def save_selection() -> None:
    name = st.session_state.get("sel_new_name", "")
    try:
        saved = selections.save(name, st.session_state["run_models"], st.session_state["run_methods"],
                                st.session_state["run_n"], st.session_state["run_mode"],
                                temperature=st.session_state["run_temp"],
                                reasoning=st.session_state["run_reasoning"], samples=st.session_state["run_samples"],
                                seed=st.session_state["run_seed"] if st.session_state["run_seed_on"] else None,
                                privacy=st.session_state["run_privacy"])
        st.session_state["sel_msg"] = ("good", f"Saved the current choices as <b>{ui._esc(saved)}</b>.")
        st.session_state["sel_pick"] = saved
    except ValueError as e:
        st.session_state["sel_msg"] = ("bad", ui._esc(str(e)))


def delete_selection(name: str) -> None:
    selections.delete(name)
    st.session_state["sel_pick"] = None
    st.session_state["sel_msg"] = ("muted", f"Deleted <b>{ui._esc(name)}</b>.")


def describe(name: str, s: dict) -> str:
    return (f"{name} · {len(s['models'])} models · {len(s['methods'])} methods · {s['n']} items · {s['mode']}"
            + f" · temperature {s.get('temperature', 0.0):.1f}"
            + (f" · reasoning {s['reasoning']}" if s.get("reasoning", "auto") != "auto" else "")
            + (f" · {s['samples']} samples" if s.get("samples", 1) > 1 else "")
            + (f" · seed {s['seed']}" if s.get("seed") is not None else "")
            + (f" · privacy {s['privacy']}" if s.get("privacy", "standard") != "standard" else "")
            + (" · built-in" if s["builtin"] else ""))


# ---------------------------------------------------------------- model browser
BROWSER_COLUMNS = ["pick", "name", "maker", "cutoff", "released", "input $/M", "output $/M", "reasoning", "context"]


def browser_key(view_ids: list[str]) -> str:
    # one editor per filtered view: its row numbers only mean something for the rows it was drawn with
    return "browser_" + hashlib.md5("|".join(view_ids).encode()).hexdigest()[:12]


def browser_ticked(view_ids: list[str]) -> None:
    """Folds the ticks in the table shown now into the pending selection, so changing a filter keeps them."""
    edits = st.session_state[browser_key(view_ids)]["edited_rows"]
    pending = st.session_state["browser_pending"]
    ticked = [m for i, m in enumerate(view_ids)
              if (edits.get(i) or edits.get(str(i)) or {}).get("pick", m in pending)]
    st.session_state["browser_pending"] = ui.apply_browser_selection(pending, view_ids, ticked)


def browser_apply() -> None:
    st.session_state["run_models"] = list(st.session_state["browser_pending"])
    st.session_state["browser_done"] = True


@st.dialog("Browse models", width="large")
def browse_models() -> None:
    table = ui.browser_table(st.session_state["browser_variants"], ui.cutoff_overrides())
    c1, c2 = st.columns(2)
    search = c1.text_input("Search", key="browser_search", placeholder="Name or id, e.g. llama, qwen3, gpt-4.1")
    makers = c2.multiselect("Makers", sorted(table["maker"].unique()), key="browser_makers", placeholder="All makers")
    years = sorted(int(y) for y in pd.to_numeric(table["cutoff"].str[:4], errors="coerce").dropna().unique())
    c1, c2, c3, c4 = st.columns([2.2, 1.3, 1.2, 1.1], vertical_alignment="bottom")
    span = (c1.select_slider("Knowledge cutoff year", years, value=(years[0], years[-1]), key="browser_years")
            if len(years) > 1 else None)
    max_out = c2.number_input("Max output $ per M", min_value=0.0, value=None, step=0.5, key="browser_max",
                              placeholder="any price")
    known = c3.toggle("Known cutoff only", key="browser_known",
                      help="Hide models without a knowledge cutoff (from OpenRouter or set by hand).")
    c4.toggle("Variants", key="browser_variants",
              help="Also list :free / :beta / :batch versions and dated snapshots.")
    narrowed = span if span and tuple(span) != (years[0], years[-1]) else None
    view = ui.filter_model_table(table, search, makers, narrowed, known, max_out)
    pending = st.session_state["browser_pending"]
    ids = list(view["id"])
    if ids:
        shown = view.assign(pick=view["id"].isin(pending)).reset_index(drop=True)
        st.data_editor(
            shown, key=browser_key(ids), on_change=browser_ticked, args=(ids,), hide_index=True, width="stretch",
            height=440, column_order=BROWSER_COLUMNS, disabled=[c for c in shown.columns if c != "pick"],
            column_config={"pick": st.column_config.CheckboxColumn("✓", width="small"),
                           "name": st.column_config.TextColumn("model", width="medium"),
                           "cutoff": st.column_config.TextColumn("cutoff", help="“(manual)” = set by hand"),
                           "input $/M": st.column_config.NumberColumn("in $/M", format="$%.2f"),
                           "output $/M": st.column_config.NumberColumn("out $/M", format="$%.2f"),
                           "context": st.column_config.NumberColumn("context", format="compact")})
    else:
        ui.callout("No model matches these filters.", "muted")
    current = st.session_state["run_models"]
    added = len([m for m in pending if m not in current])
    removed = len([m for m in current if m not in pending])
    st.caption(f"{len(view):,} of {len(table):,} models shown · {len(pending)} selected"
               + (f" · {added} to add" if added else "") + (f" · {removed} to remove" if removed else "")
               + " · click a column header to sort · ticks are kept when you change a filter")
    c1, c2, _ = st.columns([1, 1, 3])
    c1.button("Apply", type="primary", on_click=browser_apply, icon=":material/check:", width="stretch")
    if c2.button("Cancel", width="stretch"):
        st.rerun()
    if st.session_state.pop("browser_done", False):
        st.rerun()




# ---------------------------------------------------------------- self-hosted models
@st.fragment
def self_hosted_panel() -> None:
    entries = selfhosted.load()
    with st.expander(f"Self-hosted models ({len(entries)})", icon=":material/dns:"):
        st.caption("Open-weight models served on your own hardware by any OpenAI-compatible server (vLLM, "
                   "llama.cpp's llama-server, Ollama, LM Studio). They are free, need no OpenRouter key, and no "
                   "prompt leaves the machine or network the server runs on. Stored in config/self_hosted.yaml; "
                   "they appear in the model list as “Self-hosted: <name>”.")
        for model_id, e in entries.items():
            name = model_id[len(selfhosted.PREFIX):]
            c1, c2, c3 = st.columns([5, 1.2, 1], vertical_alignment="center")
            c1.markdown(f"**{name}** · `{e.get('model')}` at `{e.get('base_url')}`"
                        + (f" · {e['quantization']}" if e.get("quantization") else "")
                        + f" · cutoff {e.get('cutoff') or 'not set'}")
            if c2.button("Test connection", key=f"sh_test_{name}", icon=":material/lan:", width="stretch",
                         help="Asks the server which models it serves (GET /models). Free: no prompt is sent."):
                ok, msg = selfhosted.check_connection({**e, "base_url": str(e["base_url"])})
                ui.callout(ui._esc(msg), "good" if ok else "bad")
            if c3.button("Remove", key=f"sh_del_{name}", icon=":material/delete:", width="stretch"):
                selfhosted.delete_entry(name)
                st.rerun(scope="app")
        ui.eyebrow("Add or replace a model")
        with st.form("sh_add", border=False):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input("Short name", placeholder="qwen3-8b-q4",
                                 help="Lowercase letters, digits, '.', '_' or '-'. The id becomes self/<name>. "
                                      "Saving an existing name replaces that entry.")
            base_url = c2.text_input("Server address", placeholder="http://localhost:8000/v1",
                                     help="The OpenAI-compatible base URL, usually ending in /v1. Ollama: "
                                          "http://localhost:11434/v1 · LM Studio: http://localhost:1234/v1")
            served = c3.text_input("Model name on the server", placeholder="Qwen/Qwen3-8B",
                                   help="Exactly as the server's model list shows it (Test connection lists them).")
            c1, c2, c3 = st.columns(3)
            cutoff = c1.date_input("Knowledge cutoff", value=None, help="From the model card. Without it the "
                                   "model can run, but its results cannot be scored.")
            released = c2.date_input("Released", value=None, help="When the weights were published (for the "
                                     "after-release baseline).")
            quant = c3.text_input("Precision / quantization", placeholder="bf16, fp8, Q4_K_M …")
            c1, c2, c3 = st.columns(3)
            revision = c1.text_input("Weights revision", placeholder="Hugging Face commit or tag",
                                     help="Pins the exact weights, like a provider pin does for OpenRouter models.")
            sha = c2.text_input("Weights checksum (sha256)", placeholder="optional")
            hardware = c3.text_input("Hardware", placeholder="e.g. 1× RTX 4090 24 GB")
            c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1.4])
            params = c1.multiselect("Parameters the server accepts", list(selfhosted.PARAMETERS),
                                    default=list(selfhosted.PARAMETERS),
                                    help="Only these are sent. Most servers accept both.")
            extra_tokens = c2.number_input("Extra answer tokens", min_value=0, max_value=16000, value=0, step=256,
                                           help="Added to each method's answer budget, for models that think "
                                                "before answering. <think> blocks are removed from the answer.")
            concurrency = c3.number_input("Requests at once", min_value=1, max_value=64, value=2,
                                          help="A single GPU usually handles 1–4; vLLM handles more.")
            key_env = c4.text_input("API key variable (optional)", placeholder="VLLM_API_KEY",
                                    help="The NAME of an environment variable holding the server's key, never the "
                                         "key itself. Leave empty for a server without a key.")
            no_think = st.checkbox("Turn thinking off (Qwen3-style chat templates)",
                                   help="Sends chat_template_kwargs: {enable_thinking: false}, which vLLM and "
                                        "llama-server pass to the chat template.")
            c1, c2, _ = st.columns([1, 1, 3])
            test = c1.form_submit_button("Test connection", icon=":material/lan:", width="stretch")
            save = c2.form_submit_button("Save", type="primary", icon=":material/save:", width="stretch")
        entry = {"base_url": base_url.strip(), "model": served.strip(), "cutoff": cutoff, "released": released,
                 "quantization": quant.strip(), "revision": revision.strip(), "weights_sha256": sha.strip(),
                 "hardware": hardware.strip(), "parameters": params, "extra_tokens": int(extra_tokens),
                 "concurrency": int(concurrency), "api_key_env": key_env.strip(),
                 "extra_body": {"chat_template_kwargs": {"enable_thinking": False}} if no_think else None}
        if test or save:
            try:
                checked = selfhosted.validate(name.strip(), entry)
            except ValueError as e:
                ui.callout(ui._esc(str(e)), "bad")
                return
            if test:
                ok, msg = selfhosted.check_connection(checked)
                ui.callout(ui._esc(msg), "good" if ok else "bad")
            else:
                model_id = selfhosted.save_entry(name.strip(), entry)
                st.toast(f"Saved {model_id}. It is now in the model list.")
                st.rerun(scope="app")



# ---------------------------------------------------------------- the run going now, and the last one
@st.fragment(run_every=2)
def run_panel() -> None:
    """The run going now (started from any tab): progress and Stop, refreshed every 2 s. When it ends, the whole
    page reruns, so the Run button is free again and the result shows."""
    active = store.active_run(ui.conn())
    if active is None:
        if st.session_state.get("run_active_id"):
            st.session_state["run_active_id"] = None
            st.rerun()
        return
    st.session_state["run_active_id"] = active["run_id"]
    with st.container(border=True):
        c1, c2 = st.columns([5, 1], vertical_alignment="center")
        finished = active["done"] + active["failed"]
        c1.progress(min(1.0, finished / max(active["n_calls"] or 1, 1)),
                    text=f"**Run {active['run_id']}** · {ui.run_progress_text(active)}")
        c2.button("Stop", key=f"run_stop_{active['run_id']}", icon=":material/stop_circle:", width="stretch",
                  on_click=store.request_stop, args=(ui.conn(), active["run_id"]),
                  disabled=bool(active["stop_requested"]),
                  help="No new calls start; the ones already sent finish and are stored. Run the same selection "
                       "again later to continue.")
        st.caption("Running in the background: you can switch pages or close this tab, it keeps going. Keep the "
                   "computer awake.")


st.session_state.setdefault("run_active_id", None)
run_panel()
if st.session_state.get("run_active_id") is None:
    last = store.last_run(ui.conn())
    if last and last["status"] not in store.ACTIVE and last["heartbeat"]:   # runs from before background runs: no card
        message, kind = ui.run_outcome(last)
        ui.callout(message, kind)
        errors = store.run_errors(ui.conn(), last["run_id"])
        if errors:
            with st.expander(f"{len(errors)} failed calls — not stored, run again to retry"):
                st.code("\n".join(errors[:200]))

# ---------------------------------------------------------------- 1 choose
ui.section("1 · Choose", "Pick models and methods, or load a saved selection.")
with st.container(border=True):
    ui.eyebrow("Saved selections")
    saved = selections.load()
    c1, c2, c3, c4 = st.columns([4, 1, 1, 1.4], vertical_alignment="bottom")
    pick = c1.selectbox("Selection", list(saved), index=None, key="sel_pick", placeholder="Choose a saved selection…",
                        format_func=lambda n: describe(n, saved[n]) if n in saved else n, label_visibility="collapsed")
    c2.button("Load", on_click=load_selection, args=(pick,), disabled=pick is None, icon=":material/download:",
              width="stretch")
    c3.button("Delete", on_click=delete_selection, args=(pick,), icon=":material/delete:", width="stretch",
              disabled=pick is None or saved.get(pick, {}).get("builtin", True),
              help="Built-in selections cannot be deleted.")
    with c4.popover("Save current", icon=":material/bookmark_add:", width="stretch"):
        st.text_input("Name", key="sel_new_name", placeholder="e.g. Pilot, 4 models")
        st.button("Save", on_click=save_selection, type="primary", icon=":material/save:")
        st.caption("Saves the models, methods, items per method and answer mode below.")
    if "sel_msg" in st.session_state:
        kind, text = st.session_state.pop("sel_msg")
        ui.callout(text, kind)
    st.divider()
    shown = ui.LOCAL_MODELS + ui.picker_models(st.session_state["run_all_variants"])
    options = shown + [m for m in st.session_state["run_models"] if m not in shown]  # selected ones always stay
    c1, c2, c3 = st.columns([5, 1.3, 1.3], vertical_alignment="bottom")
    if c2.button("Browse models", icon=":material/table_view:", width="stretch",
                 help="A sortable table of every model with filters for maker, cutoff and price."):
        st.session_state["browser_pending"] = list(st.session_state["run_models"])
        st.session_state.setdefault("browser_variants", st.session_state["run_all_variants"])
        browse_models()
    models = c1.multiselect("Models", options, key="run_models", format_func=ui.model_option,
                            placeholder=f"Type to search {len(shown) - len(ui.LOCAL_MODELS)} OpenRouter models…")
    c3.toggle("Show all variants", key="run_all_variants",
              help="Also list :free / :beta / :batch versions and dated snapshots of models that exist in plain form.")
    self_hosted_panel()
    methods = st.pills("Methods", ui.methods_in_order(ts), selection_mode="multi", key="run_methods",
                       format_func=ui.method_md)
    ui.method_legend(ts)
    methods = ui.methods_in_order(methods or [])
    c1, c2, c3 = st.columns([1, 1.2, 1.4], vertical_alignment="bottom")
    n = int(c1.number_input("Items per method", min_value=5, max_value=2000, step=5, key="run_n",
                            help="The first N items of each frozen test set (they are stored in random order)."))
    with c2:
        mode = ui.mode_picker("run_mode")
    temperature = float(c3.slider(
        "Temperature", min_value=0.0, max_value=2.0, step=0.1, key="run_temp",
        help="Sampling randomness. 0 is the most repeatable and the usual choice for measuring memory. Answers at "
             "different temperatures are stored and scored separately. Models that do not accept a temperature "
             "(many reasoning models) ignore it."))
    ui.eyebrow("Model settings")
    c1, c2, c3, c4 = st.columns([1.5, 1, 1.2, 1.7], vertical_alignment="bottom")
    reasoning = c1.selectbox(
        "Reasoning effort", list(REASONING_LABELS), key="run_reasoning", format_func=REASONING_LABELS.get,
        help="How much a 'thinking' model reasons before answering. Reasoning tokens are billed and more reasoning "
             "raises the cost bound. Models without reasoning ignore it; models that must reason never get 'none'.")
    samples = int(c2.number_input(
        "Samples per item", min_value=1, max_value=10, step=1, key="run_samples",
        help="Ask each question this many times. The first attempt stays the headline result; best of k is "
             "reported next to it. Multiplies cost. Only meaningful above temperature 0."))
    with c3:
        seed_on = st.toggle("Fixed seed", key="run_seed_on",
                            help="Makes answers above temperature 0 repeatable where the provider supports it. "
                                 "Each sample gets seed + sample number.")
        seed = int(st.number_input("Seed", min_value=0, max_value=2**31 - 11, step=1, key="run_seed",
                                   disabled=not seed_on, label_visibility="collapsed"))
    privacy = c4.selectbox(
        "Data privacy", list(PRIVACY_LABELS), key="run_privacy", format_func=PRIVACY_LABELS.get,
        help="Which providers may serve the request. Standard: any provider (the widest choice and usually the "
             "cheapest). No training: only providers that don't train on prompts. Zero data retention: only providers "
             "that keep nothing. Stricter settings leave fewer providers per model.")

ui.remember_widgets(st.session_state, RUN_KEYS)
api_models = [m for m in models if not is_local(m)]          # real models: OpenRouter and self-hosted
paid_models = [m for m in api_models if runner.is_paid(m)]   # OpenRouter only: priced, pinned, need the key
current = ui.cutoffs(api_models, meta)
if api_models:
    with st.expander("Knowledge cutoffs", expanded=any(current[m] is None for m in api_models),
                     icon=":material/event:"):
        st.caption("Taken from OpenRouter unless you override it. The cutoff decides which items count as before "
                   "and after; items within ±3 months of it are left out.")
        edited = st.data_editor(pd.DataFrame({"model": api_models, "cutoff": [current[m] for m in api_models]}),
                                hide_index=True, disabled=["model"], key="cutoff_editor", width="stretch",
                                column_config={"cutoff": st.column_config.DateColumn("cutoff", format="YYYY-MM-DD")})
        if st.button("Save cutoffs", icon=":material/save:"):
            edits = {m: None if pd.isna(v) else pd.Timestamp(v).date() for m, v in zip(edited["model"], edited["cutoff"])}
            changed = save_edits(edits, current)
            st.toast(f"Saved {len(changed)} change(s)" if changed else "Nothing changed")
            st.rerun()
def unsupported(param: str) -> list[str]:
    return [m for m in api_models if param not in (meta.get(m, {}).get("supported_parameters") or [])]


if samples > 1 and temperature == 0:
    ui.callout(f"<b>{samples} samples at temperature 0</b> are usually identical answers paid for {samples} times. "
               "Raise the temperature to make best of k meaningful.", "warn")
if seed_on and temperature == 0:
    ui.callout("<b>A fixed seed has no effect at temperature 0.</b>", "info")
if seed_on and unsupported("seed"):
    ui.callout(f"<b>{len(unsupported('seed'))} selected model(s) ignore the seed:</b> "
               f"{ui._esc(', '.join(ui.display_name(m) for m in unsupported('seed')))}.", "info")
if reasoning != "auto" and unsupported("reasoning"):
    ui.callout(f"<b>{len(unsupported('reasoning'))} selected model(s) have no reasoning setting</b> and answer "
               f"normally: {ui._esc(', '.join(ui.display_name(m) for m in unsupported('reasoning')))}.", "info")
if paid_models and privacy == "zdr":
    st.caption("Zero data retention: a model with no provider that meets every requirement (zero retention and "
               "the parameters sent, e.g. a seed) fails with “No endpoints found” — listed after the run, nothing "
               "charged. Try “No training”, or turn the seed off, for those if permitted.")
ignores = unsupported("temperature")
if ignores and temperature != 0:
    ui.callout(f"<b>{len(ignores)} selected model(s) do not accept a temperature</b> and will run with their "
               f"provider default: {ui._esc(', '.join(ui.display_name(m) for m in ignores))}. Their answers are "
               f"still stored under temperature {temperature:.1f}.", "info")
missing = [m for m in api_models if current[m] is None]
if missing:
    ui.callout(f"<b>No cutoff for {', '.join(missing)}.</b> It can run, but its results cannot be scored until you "
               "set one under Knowledge cutoffs.", "warn")
if "decop" in (methods or []):
    paraphraser = ui.manifest("decop").get("paraphraser")
    if paraphraser == "mock":
        ui.callout("<b>DE-COP still uses mock distractors.</b> Fine for a pipeline check, not for real results — "
                   "generate paraphrases first (terminal command on the Test sets page).", "info")
    elif paraphraser in models:
        ui.callout(f"<b>{paraphraser} wrote the DE-COP paraphrases</b>, so its DE-COP score is not valid.", "bad")

if not models or not methods:
    st.stop()

# ---------------------------------------------------------------- 2 review
ui.section("2 · Review", "What this selection asks, and the most it can cost.")
pins = {m: r for m, r in load_pins().items() if m in paid_models}
pin_key = tuple(sorted(pins.items()))
priced, price_info, priced_at = ui.live_prices(tuple(sorted(paid_models)), pin_key) if paid_models else ({}, {}, "")
run_meta = {**meta, **priced}   # every paid model at its most expensive provider's current price
calls = runner.plan(models, methods, n, mode, ts, temperature=temperature, reasoning=reasoning,
                    seed=seed if seed_on else None, samples=samples, pins=pins)
plan = runner.plan_summary(calls, ui.conn(), run_meta, ui.cutoffs(models, meta))
todo = runner.pending(calls, ui.conn())
total = float(plan["est_cost"].sum())
paid = any(runner.is_paid(c.model) for c in todo)
ui.cards([
    {"label": "Calls to make", "value": f"{len(todo):,}", "sub": f"{len(calls) - len(todo):,} already stored"},
    {"label": "Cost, upper bound", "value": ui.money(total),
     "sub": f"most expensive provider · prices checked {priced_at}" if paid_models else "free models only"},
    {"label": "Models", "value": str(len(models)), "sub": f"{len(paid_models)} paid · {len(models) - len(paid_models)} free"},
    {"label": "Items per method", "value": str(n) + (f" × {samples}" if samples > 1 else ""),
     "sub": f"{len(methods)} methods · {mode} answers · temperature {temperature:.1f}"},
    {"label": "Model settings", "value": REASONING_LABELS[reasoning].split(" (")[0],
     "sub": f"reasoning · {samples} sample(s) · seed {seed if seed_on else '—'} · "
            f"{PRIVACY_LABELS[privacy].split(' (')[0].lower()}"},
])
view = plan.assign(model=plan["model"].map(ui.display_name))
st.dataframe(view, hide_index=True, width="stretch", column_config={
    "model": st.column_config.TextColumn("model", width="medium"),
    "est_cost": st.column_config.NumberColumn("cost ≤", format="$%.4f"),
    "cutoff": st.column_config.DateColumn("cutoff", format="YYYY-MM-DD"),
    "to_run": st.column_config.NumberColumn("to run"),
    "items_pre": st.column_config.NumberColumn("items before cutoff"),
    "items_post": st.column_config.NumberColumn("items after cutoff"),
    "items_excluded": st.column_config.NumberColumn("excluded"),
})
if paid_models:
    def spread(m: str) -> str:
        i = price_info.get(m) or {}
        if i.get("route"):
            return (f"{ui.display_name(m)}: pinned to {i['route']}" if i.get("ok") else
                    f"{ui.display_name(m)}: pinned to {i['route']}, which is not listed right now")
        if not i.get("ok"):
            return f"{ui.display_name(m)}: live check failed, using the listed price"
        lo, hi = float((i["listed"] or {}).get("completion") or 0), float(i["worst"].get("completion") or 0)
        return (f"{ui.display_name(m)}: {i['providers']} providers, output \\${lo * 1e6:.2f}–\\${hi * 1e6:.2f} per M"
                if hi > lo * 1.01 else f"{ui.display_name(m)}: {i['providers']} provider(s), one price")
    st.caption("The ceiling assumes every call goes to the model's most expensive provider (standard routing may use "
               "any of them; most calls go to cheaper ones); a pinned model only ever uses its pinned endpoint "
               "(config/provider_pins.yaml). " + " · ".join(spread(m) for m in paid_models))
    if any(not (price_info.get(m) or {}).get("ok") for m in paid_models):
        ui.callout("<b>The live price check failed for some models</b>, so their ceiling uses the listed price, "
                   "which can be lower than what some providers charge.", "warn")
thin = [ui.display_name(m) for m in plan[(plan["items_post"] < 20) & plan["cutoff"].notna()]["model"]]
if thin:
    st.caption(f"Few items after the cutoff for {', '.join(thin)}: expect a wide interval.")

# ---------------------------------------------------------------- 3 confirm
ui.section("3 · Confirm")
if not todo:
    ui.callout("<b>Everything in this selection is already stored.</b> Nothing to run — open Results.", "good")
    st.page_link("pages/results.py", label="Results", icon=":material/insights:")
    st.stop()
if paid and not ui.key_available():
    ui.callout("<b>OPENROUTER_API_KEY is not set</b>, so only the controls and self-hosted models can run.", "bad")
    st.stop()
with st.container(border=True):
    c1, c2 = st.columns([1, 2], vertical_alignment="center")
    default_cap = max(0.5, math.ceil(total * 1.2 * 100) / 100)
    cap = float(c1.number_input("Spend cap (USD)", min_value=0.0, max_value=50.0, value=float(default_cap),
                                step=0.25, help="Hard ceiling: a call only starts if its worst case still fits."))
    approved = True
    if paid:
        # the key carries the amount: a new ceiling (new prices, new selection) needs a new approval
        approved = c2.checkbox(f"I approve spending up to **{ui.money(min(cap, total))}** on OpenRouter for "
                               f"{len(todo):,} calls.", key=f"run_approve_{min(cap, total):.6f}_{len(todo)}")
    else:
        c2.html(ui.badge("free · controls and self-hosted models only", "good"))
    label = f"Run {len(todo):,} calls" + ("" if paid else " · free")
    busy = store.active_run(ui.conn())
    go = st.button(label, type="primary", icon=":material/play_arrow:", disabled=not approved or busy is not None)
    if busy:
        st.caption(f"Run {busy['run_id']} is going (top of the page). Start this one when it has finished, or stop "
                   "it first.")
if st.session_state.get("price_msg"):
    ui.callout(st.session_state.pop("price_msg"), "warn")
if go and paid:
    # prices are checked once more at the moment of paying; a higher ceiling than the one approved stops here
    fresh, _, fresh_at = ui.live_prices_now(tuple(sorted(paid_models)), pin_key)
    still_ok, fresh_total = runner.ceiling_still_approved(todo, {**meta, **fresh}, total)
    if not still_ok:
        ui.live_prices.clear()
        st.session_state["price_msg"] = (f"<b>Prices went up since this page was loaded</b> (checked {fresh_at}): the "
                                         f"ceiling is now {ui.money(fresh_total)} instead of {ui.money(total)}. "
                                         "Nothing was run — review the new ceiling and approve again.")
        st.rerun()
    run_meta = {**meta, **fresh}
if go:
    # the approved calls, prices and cap go to a background worker; the panel at the top follows it
    try:
        run_id, other = worker.submit(ui.conn(), todo, run_meta, spend_cap=cap, privacy=privacy)
    except Exception as e:
        st.error(f"The run could not be started: {e}")
        st.stop()
    if other:
        st.session_state["price_msg"] = f"<b>Run {other} is already going</b>, so nothing new was started."
    st.rerun()

# ---------------------------------------------------------------- history
ui.section("Run history")
runs = store.load_runs(ui.conn())
if len(runs):
    st.dataframe(runs.head(20), hide_index=True, width="stretch", column_config={
        "run_id": "run", "n_calls": st.column_config.NumberColumn("calls"),
        "est_cost": st.column_config.NumberColumn("cost ≤", format="$%.4f"),
        "cost": st.column_config.NumberColumn("cost", format="$%.4f")})
else:
    st.caption("No runs yet.")
