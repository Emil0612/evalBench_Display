import json

import pandas as pd
import streamlit as st

from evalbench import ui
from evalbench.items import load_testset
from evalbench.llm.mock import is_local
from evalbench.methods import METHODS
from evalbench.textutil import highlight_overlap

ui.page_header("Items", "Every test item with its ground truth, the exact prompt each model saw, and each model's raw "
                        "answer and score.", meta="Search, or filter by one model's side of the cutoff and outcome.")
meta = ui.meta_by_id()
if not ui.testsets():
    ui.empty_state("No test sets yet.", "pages/testsets.py", "Test sets")
    st.stop()
ALL = "all"
# arriving from a link (Evidence → Over time → "Open … in Items"): preset the filters once, then drop the link's params
link = dict(st.query_params)
if link.get("method"):
    st.session_state["items_method"] = link["method"]
    st.session_state["items_version"] = link.get("version", ALL)
    if link.get("model"):
        st.session_state["items_models"] = [link["model"]]
        st.session_state["items_focus"] = link["model"]
    st.session_state["items_months_req"] = (link.get("from"), link.get("to"))
    st.session_state["items_size"] = link.get("size", "all")
    st.query_params.clear()


def version_number(name: str) -> int:
    v = name.split("/")[1]
    return int(v[1:]) if v[1:].isdigit() else -1


with st.container(border=True):
    c0, c1, c2, c3 = st.columns([1, 1.2, 1, 2.3], vertical_alignment="bottom")
    with c0:
        setting = ui.setting_picker("items")
        mode = setting[0]
    calls = ui.load_calls(setting)
    ts = ui.analysis_testsets(setting)
    options = ui.methods_in_order(ts)
    if st.session_state.get("items_method") not in options:
        st.session_state["items_method"] = options[0]
    method = c1.selectbox("Method", options, key="items_method", format_func=ui.method_option)
    answered = sorted(set(calls[calls["method"] == method]["testset"]), key=version_number, reverse=True) \
        if len(calls) else []
    versions = (answered + [ALL]) if len(answered) > 1 else (answered or [ts[method][0]])
    if st.session_state.get("items_version") not in versions:
        st.session_state["items_version"] = ts[method][0] if ts[method][0] in versions else versions[0]
    version = c2.selectbox("Version", versions, key="items_version",
                           format_func=lambda v: "all versions" if v == ALL else v.split("/")[1],
                           help="The analysis pages use the newest version that has answers; the rolling view on "
                                "Evidence uses all answered versions.")
    names = answered if version == ALL else [version]
    items = [(n, it) for n in names for it in load_testset(*n.split("/"))]
    sub = calls[(calls["method"] == method) & (calls["testset"].isin(names))] if len(calls) else calls
    models = sorted(sub["model"].unique()) if len(sub) else []
    real = [m for m in models if not is_local(m)]
    ordered = ui.by_cutoff(models, meta)
    kept = [m for m in st.session_state.get("items_models", []) if m in ordered]
    st.session_state["items_models"] = kept if "items_models" in st.session_state and kept else \
        ui.by_cutoff(real or models, meta)[:8]
    chosen = c3.multiselect("Models", ordered, key="items_models", format_func=ui.display_name)
    c1, c2, c3, c4 = st.columns([2, 1.2, 1, 1])
    query = c1.text_input("Search", placeholder="item id, company, headline or text…").strip().lower()
    if st.session_state.get("items_focus") not in ["—"] + chosen:
        st.session_state["items_focus"] = "—"
    focus = c2.selectbox("Filter by model", ["—"] + chosen, key="items_focus",
                         format_func=lambda m: m if m == "—" else ui.display_name(m))
    side = c3.selectbox("Side of cutoff", ["all", "before cutoff", "after cutoff"], disabled=focus == "—")
    if st.session_state.get("items_size") not in ("all", *ui.SIZES):
        st.session_state["items_size"] = "all"
    outcome = c4.selectbox("Outcome", ["all", "correct", "wrong", "no direction", "abstained", "refused", "malformed"],
                           disabled=focus == "—")
    sizes = ui.company_sizes()
    if not ui.splits_by_size(method):
        st.session_state["items_size"] = "all"
    size = st.segmented_control("Company size", ["all", *ui.SIZES], key="items_size",
                                format_func=lambda v: "All sizes" if v == "all" else f"{v.capitalize()} caps",
                                disabled=not ui.splits_by_size(method),
                                help="Market-cap terciles of the Nordic listings." if ui.splits_by_size(method)
                                else "This method's questions have no company sizes.") or "all"
    months = sorted({it.date[:7] for _, it in items})
    if months:
        lo, hi = st.session_state.pop("items_months_req", (None, None)) or (None, None)
        if lo or hi:   # a window from the rolling chart, clipped to the months these questions cover
            lo = max(lo or months[0], months[0])
            hi = min(hi or months[-1], months[-1])
            lo, hi = (lo, hi) if lo <= hi else (months[0], months[-1])
            st.session_state["items_months"] = (min(m for m in months if m >= lo), max(m for m in months if m <= hi))
        elif tuple(st.session_state.get("items_months", ())) and not set(st.session_state["items_months"]) <= set(months):
            st.session_state["items_months"] = (months[0], months[-1])
        st.session_state.setdefault("items_months", (months[0], months[-1]))
        span = st.select_slider("Months", months, key="items_months", disabled=len(months) < 2,
                                help="The month each question is about. A link from Evidence → Over time sets the "
                                     "6-month window behind one point of the rolling chart.")
    else:
        span = ("", "9999")
fp = ui.fingerprint(calls, ui.cutoffs(chosen, meta))


def score_table(model: str) -> pd.DataFrame:
    df = ui.scores_all_versions(model, method, setting, fp)
    if not len(df):
        return pd.DataFrame()
    return df[df["testset"].isin(names)].set_index(["testset", "item_id"])


tables = {m: score_table(m) for m in chosen}


NON_ANSWER = {"abstain": "abstained", "refusal": "refused", "malformed": "malformed"}


def outcome_of(row) -> str:
    kind = row.get("outcome", "refusal" if row["refused"] else "answer")
    if kind in NON_ANSWER:
        return NON_ANSWER[kind]
    flag = row.get("no_direction")
    if pd.notna(flag) and bool(flag):   # a return of exactly 0: "I don't know", left out of every score (spec §7)
        return "no direction"
    if "hit_10pct" in row and pd.notna(row["hit_10pct"]):   # price levels: scored by closeness, labelled by ±10%
        return "correct" if row["hit_10pct"] >= 1 else "wrong"
    return "correct" if row["score"] >= (0.75 if method in ("decop", "price_choice", "known_choice") else 0.5) \
        else "wrong"


def num(value) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:.2f}"


def pct(value) -> str:
    return "–" if value is None or pd.isna(value) else f"{value:+.1f}%"


def sample_note(row) -> str:
    return f" · sample {int(row.sample) + 1}" if getattr(row, "sample", 0) else ""


def size_of_item(it) -> str:
    if it.method == "headline_date":
        return ui.headline_sizes().get(it.item_id, "unknown")
    own = it.strata.get("size")
    return own if own in ui.SIZES else sizes.get(it.cluster, "unknown")


def haystack(it) -> str:
    return " ".join([it.item_id, ui.truth_summary(method, it), json.dumps(it.payload, ensure_ascii=False),
                     json.dumps(it.truth, ensure_ascii=False)]).lower()


shown = []
for name, it in items:
    if not span[0] <= it.date[:7] <= span[1]:
        continue
    if size != "all" and size_of_item(it) != size:
        continue
    if query and query not in haystack(it):
        continue
    if focus != "—":
        t = tables[focus]
        if (name, it.item_id) not in t.index:
            continue
        row = t.loc[(name, it.item_id)]
        if side != "all" and ui.SIDE.get(int(row["group"])) != side:
            continue
        if outcome != "all" and outcome_of(row) != outcome:
            continue
    shown.append((name, it))

per_page = 15
pages = max(1, (len(shown) + per_page - 1) // per_page)
c1, c2 = st.columns([1, 4])
page = int(c1.number_input(f"Page (of {pages})", min_value=1, max_value=pages, value=1))
c2.caption(f"{len(shown)} of {len(items)} items · " + (" + ".join(names) if version == ALL else version))
mod = METHODS[method]
BADGE = {"correct": ":green-badge[correct]", "wrong": ":red-badge[wrong]", "no direction": ":gray-badge[no direction]",
         "abstained": ":gray-badge[abstained]",
         "refused": ":orange-badge[refused]", "malformed": ":violet-badge[malformed]"}

for name, it in shown[(page - 1) * per_page: page * per_page]:
    tag = (f" · {name.split('/')[1]}" if version == ALL else "") + (
        f" · {size_of_item(it)} cap" if size_of_item(it) in ui.SIZES else "")
    with st.expander(f"{it.item_id} · {it.date} · {ui.truth_summary(method, it)}{tag}"):
        t = it.truth
        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Ground truth**")
            if method == "price_recall":
                if it.fabricated:
                    st.markdown(":orange-badge[fabricated company] the right answer is “I don't know”.")
                else:
                    st.markdown(f"Close **{t['close']}** {it.payload['currency']} · adjusted {t['adj_close']} · "
                                f"one year earlier {t['baseline_close']}")
            elif method == "number_cloze":
                st.markdown(f"Masked number: **{t['number']}**")
                st.text(t["original"])
            elif method in ("price_level", "annual_return", "monthly_return") and it.fabricated:
                st.markdown(":orange-badge[fabricated company] the right answer is “I don't know”.")
            elif method in ("price_level", "known_level", "us_level"):
                st.markdown(f"Average close in {it.payload['month_name']}: **{t['level']:.2f}** {it.payload['currency']} · "
                            f"adjusted {num(t['adj_level'])} · same month a year earlier {num(t['baseline_level'])}")
            elif method == "split_clock":
                st.markdown(f"Average close in {it.payload['month_name']}: **{t['level']:.2f}** as quoted then · "
                            f"**{t['adj_level']:.2f}** as rewritten after the split on {t['event_date']} "
                            f"(factor {t['factor']:.1f}). An answer near the rewritten price means the model saw "
                            "data from after the split.")
            elif method in ("annual_return", "known_return"):
                st.markdown(f"Change in {it.payload['year']}: **{t['return_pct']:+.1f}%** (split-adjusted) · median "
                            f"share that year {t['median_return_pct']:+.1f}% · previous year {pct(t['baseline_return_pct'])}. "
                            "Scored on the direction (up or down); "
                            + ("an answer of exactly 0 means 'I don't know' and is left out of the scores."
                               if method == "annual_return" else
                               "an answer of exactly 0 gives no direction and scores 0.5."))
            elif method == "monthly_return":
                st.markdown(f"Change in {it.payload['month_name']}: **{t['return_pct']:+.1f}%** (split-adjusted) · "
                            f"median share that month {t['market_return_pct']:+.1f}% · previous month "
                            f"{pct(t['baseline_return_pct'])}. Scored correct when the answer is on the right side of "
                            "the median share; an answer of exactly 0 means 'I don't know' and is left out of the "
                            "scores.")
            elif method in ("index_level", "index_return", "us_index_level", "us_index_return"):
                st.markdown(f"{it.payload['index_name']} ({it.payload['symbol']}), last close of "
                            f"{it.payload['month_name']}: **{t['level']:.2f}** · previous month {t['prev_level']:.2f} · "
                            f"change **{t['return_pct']:+.2f}%** · a year earlier {num(t['baseline_level'])}")
            elif method in ("price_choice", "known_choice", "price_choice_prob", "known_choice_prob"):
                st.markdown(f"Average close in {it.payload['month_name']}: **{it.payload['answer_text']}** "
                            f"{it.payload['currency']}. Wrong options, the same share in other months: "
                            + ", ".join(f"{p} ({m})" for p, m in zip(it.payload["decoy_texts"], t["decoy_months"]))
                            + ". Asked four times, the correct option in each position once; the score is the "
                            "share answered right (chance 0.25).")
            elif method == "decop":
                st.text("Original: " + t["passage"])
                for k, p in enumerate(t["paraphrases"], 1):
                    st.text(f"Paraphrase {k}: {p}")
            else:
                st.text(it.payload["headline"] + "\n\n" + it.payload["prefix"])
                st.markdown("Original continuation:")
                st.text(t["reference"])
        with right:
            st.markdown("**Prompt**")
            tabs = st.tabs(mod.VARIANTS)
            for tab, variant in zip(tabs, mod.VARIANTS):
                tab.text(mod.render(it, mode, variant)[-1]["content"])
        for model in chosen:
            rows = sub[(sub["model"] == model) & (sub["item_id"] == it.item_id) & (sub["testset"] == name)]
            if rows.empty:
                continue
            tbl = tables[model]
            sc = tbl.loc[(name, it.item_id)] if (name, it.item_id) in tbl.index else None
            badge = BADGE[outcome_of(sc)] if sc is not None else ""
            where = ui.SIDE.get(int(sc["group"]), "") if sc is not None else ""
            score = ("incomplete" if sc is None else f"closeness {sc['score']:.2f} · within ±10%: "
                     f"{'yes' if sc.get('hit_10pct') == 1 else 'no'}" if "hit_10pct" in sc else f"score {sc['score']:.2f}")
            with st.container(border=True):
                st.markdown(f"**{ui.display_name(model)}** {badge} · {score} · {where}")
                for row in rows.itertuples():
                    if method == "completion":
                        st.caption(f"{row.variant}{sample_note(row)} — highlighted words also appear in the "
                                   "original continuation")
                        st.markdown(f"<div style='white-space:pre-wrap'>{highlight_overlap(row.response, t['reference'])}"
                                    "</div>", unsafe_allow_html=True)
                    else:
                        st.caption(f"{row.variant}{sample_note(row)} → parsed {json.loads(row.parsed)}")
                        st.text(row.response[:1500])
