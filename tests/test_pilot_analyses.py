"""The pre-registered pilot C analyses (docs/preregistration/2026-09-29-pilot-c.md, §2-§5), on synthetic data
whose right answer is known."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from evalbench import pilot, stats
from evalbench.items import Item
from evalbench.methods import price_recall

CUT = date(2023, 12, 31)


def _items(years, per_year=20, clusters=10, start_month=1):
    rows = []
    for y in years:
        for k in range(per_year):
            month = start_month + (k % (13 - start_month))
            rows.append({"item_id": f"i{y}-{k}", "date": f"{y}-{month:02d}-15", "cluster": f"c{k % clusters}",
                         "fabricated": False, "outcome": "answer", "refused": False})
    return pd.DataFrame(rows)


def _scored(df, cutoff=CUT, score=lambda d: 0.0):
    from evalbench.cutoffs import group_for
    df = df.copy()
    df["group"] = [group_for(d, cutoff) for d in df["date"]]
    df["score"] = [score(d) for d in df["date"]]
    return df


# ---------------------------------------------------------------- M1 extras used by S1, S5 and S8
def test_price_answers_record_their_distance_to_both_price_conventions():
    item = Item("pr-1", "price_recall", "2021-05-03", "c1", {}, {"close": 400.0, "adj_close": 40.0})
    sc = price_recall.item_score(item, {"main": {"value": 41.0, "raw": "41", "refused": False, "outcome": "answer"}})
    assert sc.score == 0.0                                    # 2.5% off the adjusted close: no hit at 1%
    assert sc.extra["rel_err_adj"] == pytest.approx(0.025) and sc.extra["rel_err_raw"] == pytest.approx(0.8975)
    assert sc.extra["log_err_adj"] < sc.extra["log_err_raw"]
    no_adj = Item("pr-2", "price_recall", "2021-05-03", "c1", {}, {"close": 100.0, "adj_close": None})
    sc = price_recall.item_score(no_adj, {"main": {"value": 100.5, "raw": "100.5", "refused": False,
                                                   "outcome": "answer"}})
    assert sc.score == 1.0 and sc.extra["rel_err_adj"] is None


def test_bootstrap_mean_interval():
    s = np.r_[np.ones(30), np.zeros(30)]
    c = np.arange(60) % 12
    lo, hi = stats.bootstrap_mean(s, c, B=300)
    assert lo < 0.5 < hi and lo > 0.2 and hi < 0.8
    assert stats.bootstrap_mean(s, c, B=300) == (lo, hi)                   # deterministic with the seed


# ---------------------------------------------------------------- S1 tolerance bands and S8 raw convention
def _m1_frame():
    df = _scored(_items([2021, 2022, 2025]), score=lambda d: 0.0)
    pre = df["group"] == 1
    df["rel_err_raw"] = np.where(pre, 0.04, 0.5)      # before the cutoff: 4% off; after: 50% off
    df["rel_err_adj"] = np.where(pre, 0.005, 0.5)     # ... but within 0.5% of the adjusted close
    df["log_err_raw"], df["log_err_adj"] = np.log1p(df["rel_err_raw"]), np.log1p(df["rel_err_adj"])
    df["score"] = (np.minimum(df["rel_err_raw"], df["rel_err_adj"]) <= 0.01).astype(float)
    return df


def test_tolerance_bands_widen_monotonically():
    bands = pilot.tolerance_bands(_m1_frame(), B=200)
    assert list(bands["band"]) == [0.01, 0.03, 0.05, 0.10]
    assert (bands["hit_pre"] == 1.0).all() and (bands["hit_post"] == 0.0).all()
    assert bands["dkappa"].iloc[0] == pytest.approx(1.0)


def test_raw_convention_counts_only_unadjusted_hits():
    df = _m1_frame()
    assert pilot.raw_convention(df, B=200)["dkappa"] == pytest.approx(0.0)   # 4% off raw: never a hit
    df.loc[df["group"] == 1, "rel_err_raw"] = 0.005
    assert pilot.raw_convention(df, B=200)["dkappa"] == pytest.approx(1.0)


# ---------------------------------------------------------------- S2 staggered-cutoff comparison
def test_staggered_comparison_finds_knowledge_only_the_later_model_can_have():
    items = _items([2021, 2022, 2024], per_year=24)
    late_cut = date(2025, 1, 31)
    a = _scored(items, CUT, score=lambda d: 0.2)                         # earlier model: flat everywhere
    b = _scored(items, late_cut, score=lambda d: 0.9 if d.startswith("2024") else 0.2)  # later: knows 2024
    res = pilot.staggered({"price_recall": (a, b)}, CUT, late_cut, B=300)
    assert res["pooled"]["did"] == pytest.approx(0.7) and res["pooled"]["ci_lo"] > 0
    assert res["price_recall"]["n_window"] >= 10 and res["price_recall"]["n_ref"] > 0
    flat = pilot.staggered({"price_recall": (a, _scored(items, late_cut, score=lambda d: 0.5))}, CUT, late_cut, B=300)
    assert flat["pooled"]["did"] == pytest.approx(0.0)


def test_staggered_uses_chance_corrected_scores_and_skips_thin_methods():
    items = _items([2021, 2024], per_year=12)
    late_cut = date(2025, 1, 31)
    a = _scored(items, CUT, score=lambda d: 0.25)
    b = _scored(items, late_cut, score=lambda d: 0.625 if d.startswith("2024") else 0.25)
    res = pilot.staggered({"decop": (a, b)}, CUT, late_cut, B=200, min_window=20)
    assert res["pooled"]["did"] == pytest.approx(0.5)        # (0.625 - 0.25) / (1 - 0.25)
    assert res["decop"] is None                              # fewer than min_window window items: pooled only


# ---------------------------------------------------------------- S3 placebo cutoff
def test_placebo_is_zero_without_drift_and_flags_drift():
    items = _items([2021, 2022, 2023], per_year=24)
    flat = pilot.placebo(_scored(items, score=lambda d: 0.5), "price_recall", B=300)
    assert flat["dkappa"] == pytest.approx(0.0) and flat["ci_lo"] <= 0 <= flat["ci_hi"]
    drift = pilot.placebo(_scored(items, score=lambda d: 1.0 if d < "2022-07-01" else 0.0), "price_recall", B=300)
    assert drift["dkappa"] == pytest.approx(1.0) and drift["ci_lo"] > 0
    # only items older than 2023-10-01 enter
    assert flat["n_pre"] + flat["n_post"] == int((items["date"] < "2023-10-01").sum())


# ---------------------------------------------------------------- S4 boilerplate
@pytest.mark.parametrize("headline,flag", [
    ("Kallelse till årsstämma i Bolaget AB", True), ("Återköpsprogram avslutat", True),
    ("Notice of Annual General Meeting", True), ("Bolaget AB: AGM resolutions", True),
    ("Managers' transactions", True), ("Q3 report: sales up 12%", False), ("Magma Mining signs deal", False),
])
def test_boilerplate_headlines(headline, flag):
    assert pilot.is_boilerplate(headline) is flag


# ---------------------------------------------------------------- S5 Split Clock first look
def test_adjustment_events_are_read_from_the_raw_to_adjusted_ratio():
    d = pd.bdate_range("2021-01-01", "2023-12-31")
    close = pd.Series(400.0, index=d)
    close[d >= "2023-06-01"] = 40.0                                        # 1:10 split on 2023-06-01
    prices = pd.DataFrame({"insref": 7, "date": d, "close": close.values, "adj_close": 40.0, "price_type": 0})
    ev = pilot.adjustment_events(prices, min_factor=2.0)
    assert list(ev[7]) == [(date(2023, 6, 1), pytest.approx(10.0))]


def test_split_clock_classifies_answers_by_convention_and_event_side():
    items = {"a": Item("a", "price_recall", "2021-05-03", "c1", {}, {"close": 400.0, "adj_close": 40.0, "insref": 7}),
             "b": Item("b", "price_recall", "2021-05-03", "c2", {}, {"close": 400.0, "adj_close": 40.0, "insref": 8}),
             "c": Item("c", "price_recall", "2021-05-03", "c3", {}, {"close": 100.0, "adj_close": 100.0, "insref": 9})}
    events = {7: [(date(2023, 6, 1), 10.0)], 8: [(date(2025, 6, 1), 10.0)], 9: []}
    df = pd.DataFrame({"item_id": ["a", "b", "c"], "fabricated": False, "outcome": "answer",
                       "log_err_raw": [np.log(10) - 0.05, 0.02, 0.0], "log_err_adj": [0.05, np.log(10) - 0.02, 0.0]})
    counts = pilot.split_clock(df, items, events, cutoff=date(2024, 6, 30))
    assert counts == {("event_before_cutoff", "adjusted"): 1, ("event_after_cutoff", "raw"): 1}


# ---------------------------------------------------------------- S7 per-period profile
def _profile_frame():
    df = _scored(_items([2021, 2022, 2023, 2024, 2025], per_year=30),
                 score=lambda d: 1.0 if d < "2022-01-01" else 0.1)          # knows 2021 only
    return df


def test_profile_cells_have_the_leads_fields_and_a_measured_baseline():
    cells = pilot.label_profile(pilot.profile_cells(_profile_frame(), "price_recall", "m", CUT,
                                                    release=date(2024, 12, 6), B=300))
    by = {c["period"]: c for c in cells}
    assert {"model", "fact_type", "period", "facts", "responses", "recall", "ci", "baseline", "label",
            "after_availability", "after_cutoff"} <= set(by["2021"])
    assert by["2021"]["label"] == "recall_above_baseline" and by["2021"]["recall"] == pytest.approx(1.0)
    assert by["2021"]["baseline"] == pytest.approx(0.1)                    # own POST level, measured
    assert by["2022"]["label"] == "none_detected" and "detection_upper_bound" in by["2022"]
    assert by["2024"]["after_cutoff"] and by["2024"]["label"] == "after_cutoff"   # descriptive only
    assert by["2025"]["after_availability"] and not by["2023"]["after_availability"]
    assert by["2021"]["baseline_release"] == pytest.approx(0.1)             # B1r: items after 2024-12-06


def test_profile_coverage_and_unresolved_rules():
    df = _profile_frame()
    df = df[~((df["date"] < "2022-01-01") & (df["item_id"].str[-1].isin(list("0123456"))))]  # 2021 thinned
    df.loc[df["date"].str.startswith("2022"), "outcome"] = np.where(
        np.arange((df["date"].str.startswith("2022")).sum()) % 3 == 0, "malformed", "answer")
    cells = pilot.label_profile(pilot.profile_cells(df, "price_recall", "m", CUT, release=date(2024, 12, 6), B=200))
    by = {c["period"]: c for c in cells}
    assert by["2021"]["label"] == "insufficient_coverage"                  # fewer than 15 items left
    assert by["2022"]["label"] == "unresolved"                            # a third malformed (> 20%)


def test_profile_needs_enough_items_behind_the_baseline():
    df = _profile_frame()
    df = df[~df["date"].str.startswith(("2024", "2025"))]                 # no POST items at all
    cells = pilot.label_profile(pilot.profile_cells(df, "price_recall", "m", CUT, release=None, B=200))
    assert {c["label"] for c in cells if not c["after_cutoff"]} == {"insufficient_coverage"}


# ---------------------------------------------------------------- primary reading (§2)
@pytest.mark.parametrize("row,reading", [
    ({"dkappa": 0.4, "ci_lo": 0.2, "ci_hi": 0.6, "p_holm": 0.01, "n_pre": 50, "n_post": 30, "malformed_rate": 0.0},
     "memorization detected"),
    ({"dkappa": -0.4, "ci_lo": -0.6, "ci_hi": -0.2, "p_holm": 0.01, "n_pre": 50, "n_post": 30, "malformed_rate": 0.0},
     "better after the cutoff"),
    ({"dkappa": 0.05, "ci_lo": -0.1, "ci_hi": 0.2, "p_holm": 0.9, "n_pre": 50, "n_post": 30, "malformed_rate": 0.0},
     "none detected"),
    ({"dkappa": 0.4, "ci_lo": 0.2, "ci_hi": 0.6, "p_holm": 0.01, "n_pre": 50, "n_post": 12, "malformed_rate": 0.0},
     "insufficient coverage"),
    ({"dkappa": 0.4, "ci_lo": 0.2, "ci_hi": 0.6, "p_holm": 0.01, "n_pre": 50, "n_post": 30, "malformed_rate": 0.3},
     "unresolved"),
])
def test_primary_reading_follows_the_plan(row, reading):
    assert pilot.primary_reading(row).startswith(reading)


def test_median_log_error_of_answered_items_before_and_after():
    df = _m1_frame()
    df["log_error"] = np.where(df["group"] == 1, 0.1, 0.7)
    df.loc[df.index[:3], "log_error"] = np.nan                         # unanswered items don't count
    assert pilot.median_log_error(df) == {"pre": pytest.approx(0.1), "post": pytest.approx(0.7)}
