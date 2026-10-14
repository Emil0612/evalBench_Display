from pathlib import Path

import streamlit as st

from evalbench import devreload

SRC = Path(__file__).resolve().parents[1] / "src" / "evalbench"


@st.cache_resource
def _reload_state() -> dict:
    return {}


# pick up code changes in src/ without restarting the server (must run before the other evalbench imports);
# results cached by the old code are dropped too, so no page shows numbers computed by code that no longer exists
if devreload.refresh(_reload_state(), SRC):
    st.cache_data.clear()

from evalbench import demo, ui  # noqa: E402

st.set_page_config(page_title="evalBench · LLM contamination tests", page_icon=":material/biotech:", layout="wide")
pages = {
    "": [st.Page("pages/overview.py", title="Overview", icon=":material/home:", default=True)],
    "Workflow": [
        st.Page("pages/testsets.py", title="Test sets", icon=":material/dataset:"),
        st.Page("pages/run.py", title="Run", icon=":material/play_arrow:"),
    ],
    "Analysis": [
        st.Page("pages/profile.py", title="Profile", icon=":material/calendar_view_month:"),
        st.Page("pages/results.py", title="Evidence", icon=":material/insights:"),
        st.Page("pages/compare.py", title="Compare", icon=":material/compare_arrows:"),
        st.Page("pages/market.py", title="Market", icon=":material/show_chart:"),
        st.Page("pages/items.py", title="Items", icon=":material/list_alt:"),
    ],
}
nav = st.navigation(pages)
ui.inject_css()
st.logo(ui.logo_svg(), size="large")
ui.sidebar_status()
if demo.is_demo():
    st.sidebar.warning("**Demo mode:** synthetic data and free control models (`python -m evalbench.demo`). "
                       "No real prices, news or model answers.", icon=":material/science:")
nav.run()
