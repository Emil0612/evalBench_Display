"""Pilot E analysis rules (docs/preregistration/2026-09-30-pilot-e.md): the release line and the refusal guard."""
from datetime import date

import pandas as pd

from evalbench import pilot_e


def test_release_line_puts_the_grey_zone_aside():
    df = pd.DataFrame({"date": ["2024-09-01", "2024-12-15", "2025-06-01", "2025-09-30"]})
    groups = pilot_e.release_groups(df["date"], cutoff=date(2025, 3, 31), release=date(2025, 8, 21))
    # before cutoff - 91 days (2024-12-30) / grey zone / grey zone / after release
    assert list(groups) == [1, 1, -1, 0]


def test_the_primary_level_score_is_closeness_and_a_refusal_scores_zero():
    df = pd.DataFrame({"score": [1.0, 0.0, 0.0], "closeness": [0.9, 0.4, None], "refused": [False, False, True],
                       "in_range": [True, False, None]})
    out = pilot_e.with_score(df, "price_level", "primary")
    assert list(out["score"]) == [0.9, 0.4, 0.0]
    assert list(pilot_e.with_score(df, "price_level", "headline")["score"]) == [1.0, 0.0, 0.0]
    assert list(pilot_e.with_score(df, "monthly_return", "primary")["score"]) == [1.0, 0.0, 0.0]


def test_the_refusal_guard_can_only_take_a_detection_away():
    detected = "memorization detected: Δκ +0.20 [+0.10, +0.30]"
    assert pilot_e.guarded(detected, answered_p=0.01, answered_dk=0.15).startswith("memorization detected")
    assert pilot_e.guarded(detected, answered_p=0.30, answered_dk=0.05).startswith("not confirmed: refusal-driven")
    assert pilot_e.guarded(detected, answered_p=0.01, answered_dk=-0.1).startswith("not confirmed: refusal-driven")
    none = "none detected: Δκ +0.02, at most +0.10"
    assert pilot_e.guarded(none, answered_p=0.001, answered_dk=0.4) == none      # never adds a detection


def test_refusal_affected_when_answer_rates_differ_by_more_than_ten_points():
    assert pilot_e.refusal_affected(0.95, 0.80) and not pilot_e.refusal_affected(0.95, 0.88)
    assert not pilot_e.refusal_affected(float("nan"), 0.5)
