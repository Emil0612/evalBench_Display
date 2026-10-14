"""The Profile view (one cell per model and year, with measured baselines) and the Evidence view's
fixed, unranked order."""
import json
from datetime import date

import pandas as pd

from evalbench import ui


def _cell(period, label, recall, **kw):
    base = {"model": "x/m", "method": "price_recall", "fact_type": "price", "region": "SE", "stratum": "all",
            "elicitation": "first_attempt", "period": period, "facts": 24, "responses": 24, "recall": recall,
            "ci": [recall - 0.1, recall + 0.1], "baseline": 0.05, "n_baseline": 40, "baseline_release": 0.04,
            "n_baseline_release": 30, "after_cutoff": False, "after_availability": False, "label": label}
    return {**base, **kw}


CELLS = [
    _cell("2021", "recall_above_baseline", 0.62, excess=0.57, excess_ci=[0.4, 0.7], label_release="recall_above_baseline"),
    _cell("2022", "none_detected", 0.08, excess=0.03, excess_ci=[-0.05, 0.12], detection_upper_bound=0.12,
          label_release="none_detected", detection_upper_bound_release=0.14),
    _cell("2023", "insufficient_coverage", 0.5, facts=6, responses=6),
    _cell("2025", "after_cutoff", 0.03, after_cutoff=True, after_availability=True),
]


def test_grid_shows_recall_label_marks_and_counts():
    html = ui.profile_grid_html(CELLS, periods=["2021", "2022", "2023", "2025"], strata=["all"])
    assert "62%" in html and "●" in html and "○" in html
    assert "n=6" in html                                   # small cells show their size instead of a number
    assert html.count("<td") == 4


def test_hover_text_carries_the_leads_fields_and_our_baseline():
    html = ui.profile_grid_html(CELLS, periods=["2021", "2022", "2023", "2025"], strata=["all"])
    assert "24 facts / 24 responses" in html
    assert "Matched baseline (own after-cutoff level): 5%" in html
    assert "Excess detection ≤ 12% (95%)" in html
    assert "cannot have seen" in html                     # after-availability note on the negative-control column


def test_header_marks_the_cutoff_and_greys_periods_after_availability():
    html = ui.profile_grid_html(CELLS, periods=["2021", "2022", "2023", "2025"], strata=["all"],
                                cutoff=date(2023, 12, 31), release=date(2024, 12, 6))
    assert 'class="after"' in html and "cutoff" in html
    assert html.index("cutoff") > html.index(">2023")    # the marker sits in the cutoff's year


def test_colour_modes_and_the_release_baseline():
    by_recall = ui.profile_grid_html(CELLS, periods=["2021"], strata=["all"], color="recall")
    by_excess = ui.profile_grid_html(CELLS, periods=["2021"], strata=["all"], color="excess")
    assert by_recall != by_excess and "62%" in by_excess              # the number is always the recall
    release = ui.profile_grid_html(CELLS, periods=["2022"], strata=["all"], baseline="release")
    assert "Excess detection ≤ 14% (95%)" in release and "after-release level" in release


def test_evidence_order_is_fixed_not_ranked():
    df = pd.DataFrame({"model": ["a/1", "b/2", "c/3"], "dkappa": [0.1, 0.9, 0.5]})
    order = ["c/3", "a/1", "b/2"]                                     # e.g. by cutoff date
    assert ui.fixed_order(df, order) == ["c/3", "a/1", "b/2"]


def test_card_header_names_the_baseline_in_use():
    assert ui.baseline_summary(CELLS, "cutoff") == "baseline: 5% (40 answers after the cutoff)"
    assert ui.baseline_summary(CELLS, "release") == "baseline: 4% (30 answers after release)"
    thin = [{**c, "n_baseline_release": 12} for c in CELLS]
    assert ui.baseline_summary(thin, "release").endswith("too few for a reading")


def test_count_of_readings_that_change_with_the_baseline():
    changed = [{**CELLS[1], "label_release": "insufficient_coverage"}, CELLS[0], CELLS[2], CELLS[3]]
    assert ui.baseline_changes(CELLS) == (0, 2)                 # two tested cells, both read the same
    assert ui.baseline_changes(changed) == (1, 2)


# ---------------------------------------------------------------- one colour system
def test_each_model_keeps_one_colour_and_controls_are_grey():
    models = ["x/early", "y/late", "mock/oracle@2024-06-30", "baseline/blind@2024-06-30", "z/mid"]
    order = ["x/early", "z/mid", "y/late", "mock/oracle@2024-06-30", "baseline/blind@2024-06-30"]
    colors = ui.assign_model_colors(order, slots=["#1", "#2", "#3"], greys={"mock/oracle": "#g1",
                                                                             "baseline/blind": "#g2"}, overflow="#o")
    assert colors == {"x/early": "#1", "z/mid": "#2", "y/late": "#3", "mock/oracle@2024-06-30": "#g1",
                      "baseline/blind@2024-06-30": "#g2"}
    more = ui.assign_model_colors(["a/1", "b/2", "c/3", "d/4"], slots=["#1", "#2", "#3"], greys={}, overflow="#o")
    assert more["d/4"] == "#o"                                  # never a generated ninth hue
    assert set(models) == set(colors)


def test_model_colours_never_use_the_reading_colours():
    for theme in ("light", "dark"):
        slots = ui.MODEL_SLOTS[theme]
        assert ui.READING_COLORS[theme]["detected"] not in slots and ui.READING_COLORS[theme]["warn"] not in slots


def test_readings_are_colour_coded_in_the_grid():
    html = ui.profile_grid_html(CELLS, periods=["2021", "2022", "2023"], strata=["all"])
    assert 'class="hit"' in html                                 # above baseline: outlined in the detection colour
    assert 'class="mk above"' in html and 'class="mk none"' in html


def test_cards_can_carry_a_model_colour_and_a_tone():
    html = ui.cards_html([{"title": "Llama", "value": "M3", "color": "#2a78d6", "tone": "bad", "dot": "#2a78d6"}])
    assert "border-top:3px solid #2a78d6" in html and 'class="v bad"' in html and "eb-dot" in html


def test_heatmap_colours_follow_the_reading_not_the_sign():
    assert ui.heat_reading(0.20, 0.048) == "memorization detected"
    assert ui.heat_reading(0.07, 1.0) == "none detected"          # positive but not significant: no red
    assert ui.heat_reading(-0.30, 0.01) == "better after the cutoff"
    assert ui.heat_reading(float("nan"), float("nan")) == "not scorable"


def test_colour_order_follows_first_appearance_so_old_models_keep_their_colour():
    first = {"x/llama": "2026-09-29T11:00", "y/gpt": "2026-09-29T11:00", "z/gemini": "2026-09-29T11:00",
             "n/oss": "2026-09-30T09:00", "mock/oracle@2024-06-30": "2026-09-29T08:00"}
    cut = {"x/llama": "2023-12-31", "y/gpt": "2024-06-30", "z/gemini": "2025-01-31", "n/oss": "2024-06-30"}
    order = ui.first_run_order(first, cut)
    # same first run: cutoff order; a model first run later comes after, even with an earlier cutoff
    assert [m for m in order if "mock" not in m] == ["x/llama", "y/gpt", "z/gemini", "n/oss"]


def test_first_seen_includes_archived_answers(tmp_path):
    from evalbench import store
    conn = store.connect(tmp_path / "t.sqlite")
    base = {"method": "price_recall", "item_id": "i", "variant": "main", "mode": "forced", "testset": "t",
            "prompt_version": "p1", "parsed": "{}"}
    store.upsert_run(conn, {"run_id": "r1", "created": "2026-09-29T10:00:00+00:00"})
    store.upsert_run(conn, {"run_id": "r2", "created": "2026-09-30T10:00:00+00:00"})
    store.insert_call(conn, {**base, "model": "a/m", "run_id": "r1"})
    store.insert_call(conn, {**base, "model": "a/m", "run_id": "r2", "route": "p"})    # replaces, archives r1's row
    store.insert_call(conn, {**base, "model": "b/m", "run_id": "r2"})
    seen = store.first_seen(conn)
    assert seen["a/m"].startswith("2026-09-29") and seen["b/m"].startswith("2026-09-30")


# ---------------------------------------------------------------- scaling to 6 models and 8 methods
def test_main_methods_leave_out_the_secondary_method_and_the_positive_control():
    from evalbench.methods import MAIN_METHODS, METHODS
    assert "completion" not in MAIN_METHODS and "known_return" not in MAIN_METHODS
    assert set(MAIN_METHODS) == set(METHODS) - {"completion", "decop", "annual_return", "price_recall", "number_cloze",
                                                "split_clock", "index_forecast", "path_date", "known_return", "known_level", "us_level",
                                                "known_choice", "price_choice_prob", "known_choice_prob", "us_index_level",
                                                "us_index_return"}


def test_heatmap_rows_carry_the_model_colour_dot():
    import pandas as pd
    t = pd.DataFrame({"model": ["x/a", "x/b"], "method": ["price_recall", "price_recall"], "dkappa": [0.2, 0.0],
                      "ci_lo": [0.1, -0.1], "ci_hi": [0.3, 0.1], "p_holm": [0.01, 1.0], "verdict": ["", ""],
                      "n_pre": [30, 30], "n_post": [30, 30]})
    spec = ui.heatmap_chart(t, ["x/a", "x/b"], ["price_recall"]).to_dict()
    marks = [layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"] for layer in spec["layer"]]
    assert "circle" in marks                                   # a coloured dot beside each model's name
    assert spec["resolve"]["scale"]["color"] == "independent"  # reading colours and model colours never mix


def test_insufficient_cells_still_show_their_measured_recall_greyed_with_the_count():
    html = ui.profile_grid_html(CELLS, periods=["2023"], strata=["all"])
    assert '<span class="v dim">50%</span>' in html        # the number is visible, but greyed: there is no verdict
    assert "n=6" in html and 'class="hatched"' in html and "–" in html


def test_an_insufficient_cell_without_items_shows_no_number():
    empty = _cell("2024", "insufficient_coverage", float("nan"), facts=0, responses=0, ci=[float("nan")] * 2)
    html = ui.profile_grid_html([empty], periods=["2024"], strata=["all"])
    assert 'class="v dim"' not in html and "n=0" in html


def test_method_pickers_use_short_names_and_put_the_main_methods_first():
    price = ["known_level", "us_level", "split_clock", "price_recall", "price_level"]
    assert ui.methods_in_order(price) == ["price_level", "known_level", "price_recall", "split_clock", "us_level"]
    assert ui.method_short("known_level") == "M9 Known levels · control"
    assert ui.method_short("us_level") == "M10 US levels · private"
    assert ui.method_short("completion") == "M4 Guided completion · secondary"
    assert ui.method_short("price_level") == "M5 Price level"
    assert all(len(ui.method_short(m)) <= 32 for m in ui.METHODS)


def test_heatmap_column_headers_wrap_onto_short_lines():
    assert ui.method_lines("completion") == ["M4", "Guided", "completion", "· secondary"]
    assert ui.method_lines("known_level") == ["M9", "Known", "levels", "· control"]
    assert ui.method_lines("price_level") == ["M5", "Price level"]
    assert all(len(line) <= 11 for m in ui.METHODS for line in ui.method_lines(m))
    t = pd.DataFrame({"model": ["x/a"] * 2, "method": ["completion", "known_level"], "dkappa": [0.1, 0.2],
                      "ci_lo": [0.0, 0.1], "ci_hi": [0.2, 0.3], "p_holm": [0.5, 0.01], "verdict": ["", ""],
                      "n_pre": [30, 30], "n_post": [30, 30]})
    spec = json.dumps(ui.heatmap_chart(t, ["x/a"], ["completion", "known_level"]).to_dict(), ensure_ascii=False)
    assert "split(datum.label" in spec and "M4|Guided|completion|· secondary" in spec
    assert '"labelPadding": 45' in spec                 # 4-line headers are lifted by 3 lines: never into the cells


def test_each_method_role_has_its_own_colour_in_every_kind_of_picker():
    assert [ui.method_role(m) for m in ("price_level", "known_level", "us_level", "completion", "price_choice_prob")] \
        == ["", "control", "private", "secondary", "probe"]
    assert ui.method_md("known_level") == ":green[M9 Known levels · control]"
    assert ui.method_md("price_level") == "M5 Price level"                    # main methods stay in the normal ink
    assert ui.method_option("us_level") == "🟠 M10 US levels · private"       # selectboxes cannot colour text
    assert ui.method_option("price_level") == "M5 Price level"
    assert ui.method_lines("price_choice_prob")[-1] == "· probe"
    assert len({ui.ROLE_COLORS[r][1] for r in ui.ROLE_COLORS}) == 4


def test_heatmap_headers_take_their_role_colour():
    t = pd.DataFrame({"model": ["x/a"] * 2, "method": ["price_level", "known_level"], "dkappa": [0.1, 0.2],
                      "ci_lo": [0.0, 0.1], "ci_hi": [0.2, 0.3], "p_holm": [0.5, 0.01], "verdict": ["", ""],
                      "n_pre": [30, 30], "n_post": [30, 30]})
    spec = json.dumps(ui.heatmap_chart(t, ["x/a"], ["price_level", "known_level"]).to_dict(), ensure_ascii=False)
    assert "datum.value == 'M9|Known|levels|· control' ? '#4cc38a'" in spec


def test_a_selected_role_button_lights_up_in_its_role_colour(monkeypatch):
    shown = []
    monkeypatch.setattr(ui.st, "html", shown.append)
    ui.inject_css()
    css = "".join(shown)
    # the private orange #f0a35e is rgb(240, 163, 94), the way Streamlit writes the label's colour
    assert ('button:is([aria-pressed="true"], [aria-checked="true"]):has(.stMarkdownColoredText[style*="240, 163, 94"])'
            '{border-color:#f0a35e !important; background-color:rgba(240, 163, 94, 0.1) !important;') in css
    assert css.count(":has(.stMarkdownColoredText") == len(ui.ROLE_COLORS)


def test_nine_real_models_each_get_their_own_colour():
    models = [f"m/{k}" for k in range(9)]
    for theme in ("light", "dark"):
        colors = ui.assign_model_colors(models, ui.MODEL_SLOTS[theme], ui.CONTROL_GREYS[theme], "#muted")
        assert len(set(colors.values())) == 9 and "#muted" not in colors.values()
        assert ui.PALETTES[theme]["muted"] not in ui.MODEL_SLOTS[theme]


def test_twelve_real_models_each_get_their_own_colour_and_hidden_helpers_come_last():
    models = [f"m/{k}" for k in range(12)]
    for theme in ("light", "dark"):
        colors = ui.assign_model_colors(models, ui.MODEL_SLOTS[theme], ui.CONTROL_GREYS[theme], "#muted")
        assert len(set(colors.values())) == 12 and "#muted" not in colors.values()
    aux = next(iter(ui.AUX_MODELS))
    order = ui.first_run_order({aux: "2026-10-06", "google/gemma-4-31b-it": "2026-10-08"}, {})
    assert order == ["google/gemma-4-31b-it", aux]
