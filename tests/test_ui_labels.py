import pytest
import time

from evalbench import ui

META = {
    "meta-llama/llama-3.3-70b-instruct": {"name": "Meta: Llama 3.3 70B Instruct", "knowledge_cutoff": "2023-12-31",
                                          "pricing": {"prompt": "0.0000001", "completion": "0.00000032"}},
    "x/unknown": {},
}


def test_build_labels_for_local_and_api_models():
    labels = ui.build_labels(META, ui.LOCAL_MODELS)
    assert labels["meta-llama/llama-3.3-70b-instruct"] == \
        "Llama 3.3 70B Instruct · Meta · $0.10 / $0.32 per M · cutoff 2023-12"
    assert labels["mock/oracle@2024-06-30"].startswith("Oracle (control) · free")
    assert labels["x/unknown"].startswith("unknown")


def test_build_labels_is_fast_for_hundreds_of_models():
    meta = {f"v{i}/model-{i}": META["meta-llama/llama-3.3-70b-instruct"] for i in range(500)}
    t0 = time.perf_counter()
    ui.build_labels(meta, ui.LOCAL_MODELS)
    assert time.perf_counter() - t0 < 0.2


def test_picker_label_shows_a_manual_cutoff_marked_as_manual():
    meta = {"q/new": {"pricing": {"prompt": "0.00000015", "completion": "0.00000047"}, "knowledge_cutoff": None,
                      "name": "Qwen: New"}}
    labels = ui.build_labels(meta, [], overrides={"q/new": "2026-03-31"})
    assert "cutoff 2026-03 (manual)" in labels["q/new"]
    assert "cutoff ?" in ui.build_labels(meta, [])["q/new"]


# ---------------------------------------------------------------- model browser
BROWSER_META = {
    "openai/gpt-4.1-mini": {"name": "OpenAI: GPT-4.1 Mini", "pricing": {"prompt": "0.0000004", "completion": "0.0000016"},
                            "knowledge_cutoff": "2024-06-30", "created": 1744588800, "context_length": 1047576,
                            "supported_parameters": ["max_tokens", "temperature"]},
    "qwen/qwen3.8-flash": {"name": "Qwen: Qwen3.8 Flash", "pricing": {"prompt": "0.00000015", "completion": "0.00000047"},
                           "knowledge_cutoff": None, "created": 1787702400, "context_length": 1000000,
                           "supported_parameters": ["max_tokens", "reasoning"],
                           "reasoning": {"mandatory": False}},
    "x/nocutoff": {"name": "X: No Cutoff", "pricing": {"prompt": "0", "completion": "0"}},
}


def test_model_table_has_one_row_per_model_with_readable_columns():
    t = ui.model_table(BROWSER_META, overrides={"qwen/qwen3.8-flash": "2026-03-31"}).set_index("id")
    assert list(t.columns[:3]) == ["name", "maker", "cutoff"]
    assert t.loc["qwen/qwen3.8-flash", "cutoff"] == "2026-03-31 (manual)"
    assert t.loc["openai/gpt-4.1-mini", "cutoff"] == "2024-06-30"
    assert t.loc["openai/gpt-4.1-mini", "output $/M"] == pytest.approx(1.6)
    assert t.loc["qwen/qwen3.8-flash", "reasoning"] == "optional" and t.loc["openai/gpt-4.1-mini", "reasoning"] == "–"
    assert t.loc["qwen/qwen3.8-flash", "released"] == "2026-08-26"


def test_applying_ticks_only_changes_the_rows_shown():
    current = ["a/1", "b/2", "c/3"]
    shown = ["b/2", "c/3", "d/4"]                  # a/1 is filtered out of the table: it must stay selected
    assert ui.apply_browser_selection(current, shown, checked=["c/3", "d/4"]) == ["a/1", "c/3", "d/4"]


def test_browser_filters_combine_search_maker_cutoff_and_price():
    t = ui.model_table(BROWSER_META, overrides={"qwen/qwen3.8-flash": "2026-03-31"})
    ids = lambda **kw: sorted(ui.filter_model_table(t, **kw)["id"])
    assert ids() == sorted(BROWSER_META)
    assert ids(search="MINI") == ["openai/gpt-4.1-mini"]
    assert ids(search="qwen3.8") == ["qwen/qwen3.8-flash"]            # the id is searched as well as the name
    assert ids(makers=["OpenAI"]) == ["openai/gpt-4.1-mini"]
    assert ids(cutoff_years=(2025, 2026)) == ["qwen/qwen3.8-flash"]  # a manual cutoff counts as known
    assert ids(max_output=1.0) == [m for m in sorted(BROWSER_META) if m != "openai/gpt-4.1-mini"]
    assert "x/nocutoff" in ids() and "x/nocutoff" not in ids(known_cutoff_only=True)


# ---------------------------------------------------------------- choices survive visiting another page
def test_choices_come_back_after_streamlit_forgets_them_on_another_page():
    state = {"run_models": ["openai/gpt-4.1-mini"], "run_n": 150}
    ui.remember_widgets(state, ["run_models", "run_n"])
    del state["run_models"], state["run_n"]            # what Streamlit does to widget values on another page
    ui.restore_widgets(state, ["run_models", "run_n"])
    assert state["run_models"] == ["openai/gpt-4.1-mini"] and state["run_n"] == 150


def test_restoring_never_overwrites_a_value_that_is_still_there():
    state = {"run_n": 40}
    ui.remember_widgets(state, ["run_n"])
    state["run_n"] = 150                                 # changed on this page, e.g. by loading a selection
    ui.restore_widgets(state, ["run_n"])
    assert state["run_n"] == 150
    ui.restore_widgets(state, ["never_seen"])            # nothing remembered: nothing happens
    assert "never_seen" not in state


def test_every_page_choice_is_kept_but_approvals_and_other_keys_are_not():
    state = {"prof_fact": "return", "ev_methods_price-return": ["price_level"], "cmp_a": "x/a",
             "profile_mode": "abstain", "run_approve_0.007000_88": True, "sel_msg": ("good", "hi")}
    ui.remember_page_choices(state)
    kept = {k[len("_kept_"):] for k in state if k.startswith("_kept_")}
    assert kept == {"prof_fact", "ev_methods_price-return", "cmp_a", "profile_mode"}   # never a spend approval


def test_a_kept_choice_comes_back_only_if_it_is_still_an_option():
    state = {"_kept_cmp_a": "x/a", "_kept_ev_models": ["x/a", "x/gone"], "_kept_cmp_method": "decop"}
    ui.keep_choice(state, "cmp_a", ["x/a", "x/b"], default="x/b")
    ui.keep_choice(state, "ev_models", ["x/a", "x/b"], default=["x/b"], multi=True)
    ui.keep_choice(state, "cmp_method", ["price_level"], default="price_level")   # not offered any more
    ui.keep_choice(state, "prof_fact", ["price", "return"], default="price")       # never chosen: the default
    assert state["cmp_a"] == "x/a" and state["ev_models"] == ["x/a"]
    assert state["cmp_method"] == "price_level" and state["prof_fact"] == "price"


def test_an_emptied_multi_choice_stays_empty():
    state = {"ev_types": []}
    ui.keep_choice(state, "ev_types", ["price", "return"], default=["price", "return"], multi=True)
    assert state["ev_types"] == []
