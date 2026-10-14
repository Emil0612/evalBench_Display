"""Release-date visuals: the release line on Evidence and Overview, the model timeline, release markers in the
over-time chart and the Profile grid."""
import json
from datetime import date

import pandas as pd

from evalbench import ui


def _scores():
    return pd.DataFrame({"date": ["2024-09-01", "2024-12-15", "2025-06-01", "2025-09-30"], "group": [1, 1, 0, 0],
                         "score": [1.0, 1.0, 0.0, 0.0]})


# ---------------------------------------------------------------- 2: the release line
def test_the_release_line_regroups_items_and_leaves_the_grey_zone_out():
    out = ui.on_line(_scores(), date(2025, 3, 31), date(2025, 8, 21), "release")
    assert list(out["group"]) == [1, 1, -1, 0]
    assert list(ui.on_line(_scores(), date(2025, 3, 31), date(2025, 8, 21), "cutoff")["group"]) == [1, 1, 0, 0]


def test_a_model_without_a_release_date_keeps_the_cutoff_line():
    assert list(ui.on_line(_scores(), date(2024, 6, 30), None, "release")["group"]) == [1, 1, 0, 0]
    assert ui.on_line(pd.DataFrame(), date(2024, 6, 30), date(2025, 1, 1), "release").empty


# ---------------------------------------------------------------- 1: the model timeline
def test_timeline_has_three_phases_per_model_and_clips_at_the_end_of_our_data():
    rows = ui.timeline_rows({"a": date(2023, 12, 31), "q": date(2026, 3, 31)},
                            {"a": date(2024, 12, 6), "q": date(2026, 8, 26)},
                            start=date(2016, 1, 1), end=date(2026, 8, 10))
    a = rows[rows["model"] == "a"].set_index("phase")
    assert list(a.index) == list(ui.TIMELINE_PHASES)
    assert a.loc[ui.TIMELINE_PHASES[1], "start"] == "2023-12-31" and a.loc[ui.TIMELINE_PHASES[1], "end"] == "2024-12-06"
    assert a.loc[ui.TIMELINE_PHASES[2], "end"] == "2026-08-10"
    q = rows[rows["model"] == "q"].set_index("phase")
    assert ui.TIMELINE_PHASES[2] not in q.index                     # released after our data ends: nothing to test
    assert q.loc[ui.TIMELINE_PHASES[1], "end"] == "2026-08-10"


def test_timeline_chart_marks_cutoff_and_release_and_skips_models_without_a_release():
    meta = {"x/a": {"name": "X: A", "created": 1733443200}}           # 2024-12-06
    spec = json.dumps(ui.timeline_chart(["x/a", "mock/oracle@2024-06-30"], meta,
                                        cutoff_map={"x/a": date(2023, 12, 31), "mock/oracle@2024-06-30": date(2024, 6, 30)},
                                        end=date(2026, 8, 10)).to_dict(), ensure_ascii=False)
    assert "triangle-down" in spec and "circle" in spec and "2024-12-06" in spec
    assert "Oracle" not in spec                                     # controls have no release


# ---------------------------------------------------------------- 3: release markers over time
def test_over_time_chart_draws_a_solid_release_rule_next_to_the_dashed_cutoff():
    frames = pd.DataFrame({"model": ["x/a"] * 2, "quarter": ["2023Q1", "2025Q1"], "mean": [0.5, 0.2], "n": [10, 10]})
    spec = ui.quarter_chart(frames, [{"model": "x/a", "quarter": "2023Q4"}], ["x/a"], ["x/a"],
                            rel_rows=[{"model": "x/a", "quarter": "2024Q4"}]).to_dict()
    rules = [layer for layer in spec["layer"] if (layer.get("mark") or {}).get("type") == "rule"]
    assert len(rules) == 2 and sum("strokeDash" in r["mark"] for r in rules) == 1


# ---------------------------------------------------------------- 4: release marker in the Profile grid
def test_profile_grid_marks_the_release_year():
    cell = {"model": "x/a", "method": "price_level", "stratum": "all", "period": "2024", "facts": 20, "responses": 20,
            "recall": 0.2, "ci": [0.1, 0.3], "baseline": 0.1, "n_baseline": 30, "baseline_release": 0.1,
            "n_baseline_release": 25, "after_cutoff": True, "after_availability": False, "label": "after_cutoff"}
    html = ui.profile_grid_html([cell], periods=["2024"], strata=["all"], cutoff=date(2023, 12, 31),
                                release=date(2024, 12, 6))
    assert "● release" in html


# ---------------------------------------------------------------- the app's verdicts follow the registered rules
def _row(**kw):
    base = {"dkappa": 0.2, "p_holm": 0.01, "n_pre": 100, "n_post": 60, "answered_dkappa": 0.2, "answered_p_holm": 0.01,
            "malformed_rate": 0.0}
    return {**base, **kw}


def test_a_cell_with_fewer_than_20_items_on_a_side_gets_no_verdict():
    assert ui.cell_status(_row(n_post=8)) == "too few items"
    assert ui.cell_status(_row(n_pre=12, p_holm=0.9)) == "too few items"
    assert ui.verdict(_row(n_post=8)).startswith("too few items")


def test_a_detection_that_vanishes_without_refusals_is_flagged():
    assert ui.cell_status(_row()) == "memorization detected"
    assert ui.cell_status(_row(answered_p_holm=0.3)) == "refusal-driven"
    assert ui.cell_status(_row(answered_dkappa=-0.05)) == "refusal-driven"
    assert ui.verdict(_row(answered_p_holm=0.3)) == "not confirmed: refusal-driven"
    no_guard = {k: v for k, v in _row().items() if not k.startswith("answered")}
    assert ui.cell_status(no_guard) == "memorization detected"             # tables without the extra columns


def test_the_other_readings_are_unchanged():
    assert ui.cell_status(_row(dkappa=-0.3, answered_dkappa=-0.3)) == "better after the cutoff"
    assert ui.cell_status(_row(p_holm=0.4, answered_p_holm=0.4)) == "none detected"
    assert ui.cell_status(_row(dkappa=float("nan"))) == "not scorable"
    assert ui.cell_status(_row(malformed_rate=0.3)) == "unresolved"


def test_timeline_legend_labels_are_short_enough_to_show_in_full():
    assert all(len(p) <= 32 for p in ui.TIMELINE_PHASES)
