import altair as alt
import pandas as pd
import streamlit as st

from evalbench import ui
from evalbench.items import next_version
from evalbench.methods import LABELS, METHODS

ts = ui.testsets()
ui.page_header("Test sets", "Frozen, versioned items that every model answers. Generating a new version never "
                            "overwrites an old one, so stored results always refer to exactly the items they used.",
               meta="Only the manifests are committed; the item files stay on this computer because they contain "
                    "licensed text.")
if not ts:
    ui.empty_state("No test sets yet. Build the free ones in the VS Code terminal:\n\n"
                   "`uv run python -m evalbench.build_testsets --mock-paraphrases`")
    st.stop()

# ---------------------------------------------------------------- overview cards
tiles = []
for method in METHODS:
    if method not in ts:
        continue
    name, items = ts[method]
    man = ui.manifest(method)
    years = [it.date[:4] for it in items if not it.fabricated]
    badges = [ui.badge(name, "muted"), ui.badge(f"{min(years)}–{max(years)}", "muted")]
    if method == "decop":
        para = man.get("paraphraser", "?")
        badges.append(ui.badge("mock distractors", "warn") if para == "mock" else ui.badge(f"paraphrases: {para}", "good"))
    if man.get("n_fabricated"):
        badges.append(ui.badge(f"{man['n_fabricated']} fabricated", "info"))
    tiles.append({"label": LABELS[method], "value": f"{len(items)} items", "color": ui.role_hex(method),
                  "sub": f"built {man.get('built_at', '')[:10]} · seed {man.get('seed')}", "badges": badges})
ui.cards(tiles)

ui.section("Items per year", "Every year needs items on both sides of the cutoffs you test.")
with st.container(border=True):
    per_year = pd.DataFrame([{"method": ui.method_short(m), "year": it.date[:4]}
                             for m, (_, items) in ts.items() for it in items if not it.fabricated])
    counts = per_year.value_counts().rename("items").reset_index()
    order = [ui.method_short(m) for m in ui.methods_in_order(ts)]
    chart = alt.Chart(counts).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X("year:O", title=None, axis=alt.Axis(labelAngle=0, ticks=False)),
        xOffset=alt.XOffset("method:N", sort=order),
        y=alt.Y("items:Q", title="items"),
        color=alt.Color("method:N", title=None, sort=order,
                        scale=alt.Scale(domain=order, range=ui.palette()["categorical"][: len(order)])),
        tooltip=["method:N", "year:O", "items:Q"])
    ui.show_chart(chart, height=240)
    with st.expander("How the items were selected"):
        for method in ts:
            role = ui.method_role(method)
            name = f":{ui.ROLE_COLORS[role][0]}[**{LABELS[method]}**]" if role else f"**{LABELS[method]}**"
            st.markdown(f"{name} — {ui.manifest(method).get('filters', '')}")

# ---------------------------------------------------------------- DE-COP generation
ui.section("DE-COP paraphrases", "M3 needs three paraphrases of every original passage. They are written once by a "
                                 "paraphraser model and then frozen.")
current = ui.manifest("decop").get("paraphraser")
current = ", ".join(current) if isinstance(current, list) else current
if current == "mock":
    ui.callout("<b>DE-COP currently uses mock distractors</b> (shuffled words). They only test the pipeline; "
               "generate real paraphrases before reading any M3 result.", "warn")
else:
    ui.callout(f"<b>DE-COP uses paraphrases written by {current}.</b> Do not test those model families on DE-COP.",
               "good")

with st.container(border=True):
    ui.eyebrow("Blind-baseline audit · can the original be found without any model?")
    st.dataframe(ui.decop_audit_table(ui.testsets_fingerprint()), hide_index=True, width="stretch")
    st.caption("How often each no-model rule picks the original (chance 25%; number of sets it decides in brackets). "
               "A rule passes at 35% or less; a form rule (only option with quote marks or some punctuation mark) "
               "passes only if it never decides. A set that fails cannot tell memory from spotting the original by "
               "its form, so M3 stays secondary until a version passes (pilots D and E: v1 and v2 fail).")

with st.container(border=True):
    ui.eyebrow(f"Generate real paraphrases · next version decop/{next_version('decop')} · paid, run in the terminal")
    st.markdown("Generation is a one-off setup step, so it runs in the VS Code terminal. The command first prints "
                "the upper-bound cost and asks **Continue? [y/N]** before anything is spent:")
    st.code("uv run python -m evalbench.build_testsets --decop-v3 --n 240", language="powershell")
    st.caption("v3 and later: three rewriting models from families we do not test (GLM, Kimi, Claude Haiku), two rewrites each; "
               "every rewrite checked for form tells; the three kept are chosen so the original is not the most "
               "central or longest option more often than chance. The audit above runs on the new version "
               "automatically; review a few items below before running DE-COP.")

# ---------------------------------------------------------------- review
ui.section("Review the DE-COP items", "Good paraphrases keep every fact and number, stay in the same language and "
                                      "read naturally; the original must not stand out by formatting or length.")
name, items = ts["decop"]
with st.container(border=True):
    idx = st.slider("Item", 1, len(items), 1, key="decop_item") - 1
    it = items[idx]
    st.html(" ".join([ui.badge(name, "muted"), ui.badge(it.item_id, "muted"), ui.badge(it.date, "muted"),
                      ui.badge(it.payload["distributor"], "muted"), ui.badge(f"language: {it.payload['language']}", "info")]))
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        ui.eyebrow("Original · verbatim")
        st.markdown(ui.plain(it.truth["passage"]))
    with c2:
        writers = it.truth.get("paraphrasers") or []
        for k, p in enumerate(it.truth["paraphrases"], 1):
            ui.eyebrow(f"Paraphrase {k}" + (f" · {writers[k - 1]}" if len(writers) >= k else ""))
            st.markdown(ui.plain(p))
