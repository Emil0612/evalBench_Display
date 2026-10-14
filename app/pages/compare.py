import json

import pandas as pd
import streamlit as st

from evalbench import analysis, ui
from evalbench.llm.mock import is_local

ui.page_header("Compare two models", "Both models answer the same frozen items; each is judged against its own "
                                     "line. The question is whether their before/after gaps differ: a diagnostic, "
                                     "not a ranking.")
meta = ui.meta_by_id()
with st.container(border=True):
    c0, c1, c2, c3 = st.columns([1.1, 1.3, 1.3, 1.3], vertical_alignment="bottom")
    with c0:
        setting = ui.setting_picker("compare")
        mode = setting[0]
    calls = ui.load_calls(setting)
    ts = ui.analysis_testsets(setting)
    all_models = sorted(calls["model"].unique(), key=lambda m: (is_local(m), ui.display_name(m))) if len(calls) else []
    methods = ui.methods_in_order([m for m in ts if len(calls) and m in set(calls["method"])])
    if len(all_models) < 2 or not methods:
        st.stop()
    real = [m for m in all_models if not is_local(m)]
    answered = {m: set(calls.loc[calls["testset"] == ts[m][0], "model"]) for m in methods}
    order = ui.by_cutoff(ui.default_models(real or all_models, ui.cutoffs(real or all_models, meta)), meta)
    da0, db0, dm0 = ui.compare_defaults(answered, order + [m for m in all_models if m not in order], methods)
    ui.keep_choice(st.session_state, "cmp_a", all_models, da0)
    a = c1.selectbox("Model A", all_models, format_func=ui.display_name, key="cmp_a")
    others = [m for m in all_models if m != a]
    ui.keep_choice(st.session_state, "cmp_b", others, db0 if db0 in others else others[0])
    b = c2.selectbox("Model B", others, format_func=ui.display_name, key="cmp_b")
    ui.keep_choice(st.session_state, "cmp_method", methods, dm0)
    method = c3.selectbox("Method", methods, format_func=ui.method_option, key="cmp_method")
    ui.keep_choice(st.session_state, "cmp_line", list(ui.LINES), "release")
    line = st.segmented_control("Compare with", list(ui.LINES), key="cmp_line", format_func=ui.LINES.get) or "release"
    ui.method_legend(methods)
    ui.remember_page_choices(st.session_state)
if len(all_models) < 2:
    ui.empty_state("Compare needs results from at least two models.", "pages/run.py", "Start a run")
    st.stop()
fp = ui.fingerprint(calls, ui.cutoffs([a, b], meta))
da, db = ui.analysis_frame(a, method, setting, fp, line, meta), ui.analysis_frame(b, method, setting, fp, line, meta)
if da.empty or db.empty:
    ui.empty_state("Both models need results for this method.", "pages/run.py", "Start a run")
    st.stop()

name_a, name_b = ui.display_name(a), ui.display_name(b)
mtitle = ui.method_title(method)
res, merged = analysis.compare(da, db, method)
if method in analysis.REGISTERED:   # its own registered test per model; no registered paired test exists
    blind = {}
    if method in analysis.BLIND_REGISTERED:
        bl = ui.scores(ui.BLIND_ID, method, setting, fp)
        blind = {method: dict(zip(bl["item_id"], bl["score"]))} if len(bl) else {}
    cells = analysis.summary_from_frames({(a, method): da, (b, method): db}, B=2000, blind=blind)
    ui.callout(f"{mtitle} is read by its own registered test ({ui.registered_label(cells)}). No paired test between "
               "two models is registered for it, so each model's own result is shown with its 95% interval, and no "
               "difference between the two models is claimed here.", "muted")
    ui.cards([{"label": name, "value": f"{r['dkappa']:+.3f}" if pd.notna(r["dkappa"]) else "n/a",
               "sub": (f"95% interval [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] · " if pd.notna(r["ci_lo"]) else "")
                      + ui.verdict(r)} for name, (_, r) in zip((name_a, name_b), cells.iterrows())])
else:
    ans, _ = analysis.compare(da[~da["refused"].astype(bool)], db[~db["refused"].astype(bool)], method)
    text, tone = ui.compare_sentence(name_a, name_b, mtitle, res, ans)
    ui.callout(text, tone)
    ui.cards([
        {"label": "Δκ(A) − Δκ(B)", "value": f"{res['diff']:+.3f}" if pd.notna(res["diff"]) else "n/a",
         "sub": "positive: A's before/after gap is larger"},
        {"label": "95% interval", "value": f"[{res['ci_lo']:+.2f}, {res['ci_hi']:+.2f}]" if pd.notna(res["ci_lo"])
         else "n/a", "sub": "paired bootstrap over the shared items"},
        {"label": "Both answered", "value": f"{ans['diff']:+.3f}" if pd.notna(ans["diff"]) else "n/a",
         "sub": (f"[{ans['ci_lo']:+.2f}, {ans['ci_hi']:+.2f}] · the refusal guard" if pd.notna(ans["ci_lo"])
                 else "the refusal guard")},
        {"label": "McNemar p", "value": f"{res['mcnemar_p']:.4f}" if pd.notna(res["mcnemar_p"]) else "n/a",
         "sub": f"items before both cutoffs (n={res['n_both_pre']})" if pd.notna(res["mcnemar_p"])
                else ("no item falls before both models' cutoffs" if method in analysis.BINARY_METHODS
                      else "only for right/wrong methods")},
        {"label": "Shared items", "value": str(res["n_items"]), "sub": "same frozen items for both"},
    ])

if method == "headline_date":   # its generic score over time is biased by construction (M18 plan §3)
    ui.section("Over time", "Share of each model's headlines dated within 3 months, by year published, next to the "
                            "text-only baseline (grey dashed); the shaded band is the model's default-answer zone.")
    with st.container(border=True):
        own = {m: ui.scores(m, method, setting, fp) for m in (a, b)}
        frames = [ui.dating_years(df).assign(model=m) for m, df in own.items() if len(df)]
        blind_years = ui.dating_years(ui.scores(ui.BLIND_ID, method, setting, fp)).assign(model=ui.BLIND_ID)
        cut_m = [{"model": m, "month": str(pd.Period(c, "M"))} for m, c in ui.cutoffs([a, b], meta).items() if c]
        rel_m = [{"model": m, "month": str(pd.Period(ui.release_date(m, meta), "M"))}
                 for m in (a, b) if ui.release_date(m, meta) is not None]
        if frames:
            ui.show_chart(ui.dating_year_chart(pd.concat(frames + [blind_years]), [a, b],
                                               {m: ui.default_month(df) for m, df in own.items()}, cut_m, rel_m))
else:
    ui.section("Over time", "Mean item score by quarter; dashed lines mark each model's cutoff, solid lines its "
                            "release.")
    with st.container(border=True):
        qd = pd.concat([analysis.by_quarter(df).assign(model=m) for m, df in ((a, da), (b, db))])
        cut_rows = [{"model": m, "quarter": str(pd.Period(c, freq="Q"))} for m, c in ui.cutoffs([a, b], meta).items()
                    if c]
        rel_rows = [{"model": m, "quarter": str(pd.Period(ui.release_date(m, meta), freq="Q"))}
                    for m in (a, b) if ui.release_date(m, meta) is not None]
        ui.show_chart(ui.quarter_chart(qd, cut_rows, [a, b], all_models, rel_rows), height=260)

ui.section("Where they disagree", "Items the two models scored differently, with both raw answers.")
# price levels are scored by closeness, which nearly always differs: there, a disagreement is one model within ±10%
# and the other not
key = "hit_10pct" if "hit_10pct_a" in merged else "score"
dis = merged[merged[f"{key}_a"] != merged[f"{key}_b"]].copy()
if dis.empty:
    st.caption("The two models scored every shared item the same.")
    st.stop()
dis["A side"] = dis["group_a"].map(lambda g: ui.side_label(g, line))
dis["B side"] = dis["group_b"].map(lambda g: ui.side_label(g, line))
cols = ["item_id", "date_a", "A side", "score_a", "B side", "score_b"]
names = {"date_a": "date", "score_a": "A score", "score_b": "B score"}
if key == "hit_10pct":   # price levels: show why the row counts as a disagreement
    dis["A within ±10%"] = dis["hit_10pct_a"].map({1.0: "yes", 0.0: "no"})
    dis["B within ±10%"] = dis["hit_10pct_b"].map({1.0: "yes", 0.0: "no"})
    cols = ["item_id", "date_a", "A side", "score_a", "A within ±10%", "B side", "score_b", "B within ±10%"]
    names = {"date_a": "date", "score_a": "A closeness", "score_b": "B closeness"}
st.dataframe(dis[cols].rename(columns=names), hide_index=True, width="stretch",
             column_config={n: st.column_config.NumberColumn(format="%.2f") for n in
                            ("A score", "B score", "A closeness", "B closeness")})
items = {it.item_id: it for it in ts[method][1]}
sub = calls[(calls["method"] == method) & (calls["testset"] == ts[method][0]) & calls["model"].isin([a, b])]
for item_id in dis["item_id"].head(15):
    item = items[item_id]
    with st.expander(f"{item_id} · {item.date} · {ui.truth_summary(method, item)}"):
        left, right = st.columns(2, gap="medium")
        for col, model, df in ((left, a, da), (right, b, db)):
            score = df.set_index("item_id").loc[item_id, "score"]
            with col.container(border=True):
                st.markdown(f"**{ui.display_name(model)}** · score {score:.2f}")
                for row in sub[(sub["model"] == model) & (sub["item_id"] == item_id)].itertuples():
                    st.caption(f"{row.variant} → {json.loads(row.parsed)}")
                    st.text(row.response[:1500])
if len(dis) > 15:
    st.caption(f"Showing 15 of {len(dis)}; the Items page lists all of them.")
