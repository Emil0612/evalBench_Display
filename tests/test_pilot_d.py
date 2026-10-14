"""Pilot D's registered analyses that pilot C did not have (docs/preregistration/2026-09-29-pilot-d.md)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from evalbench import pilot, pilot_d


def _frame(scores, dates, clusters=None, **cols):
    n = len(scores)
    return pd.DataFrame({"item_id": [f"i{k}" for k in range(n)], "date": dates, "score": scores,
                         "cluster": clusters or [f"c{k % 10}" for k in range(n)], "fabricated": False,
                         "refused": False, "outcome": "answer", **cols})


# ---------------------------------------------------------------- M7 positive control (E2)
def test_positive_control_is_sensitive_only_above_both_chance_and_the_blind_baseline():
    good = _frame([1.0] * 80 + [0.0] * 20, ["2020-12-30"] * 100)
    res = pilot_d.positive_control(good, blind_mean=0.41, B=2000)
    assert res["accuracy"] == pytest.approx(0.8) and res["ci_lo"] > 0.5 and res["sensitive"] is True
    coin = _frame([1.0] * 52 + [0.0] * 48, ["2020-12-30"] * 100)
    assert pilot_d.positive_control(coin, blind_mean=0.41, B=2000)["sensitive"] is False


def test_positive_control_needs_to_beat_a_strong_blind_baseline_too():
    ok = _frame([1.0] * 70 + [0.0] * 30, ["2020-12-30"] * 100)
    assert pilot_d.positive_control(ok, blind_mean=0.75, B=2000)["sensitive"] is False


# ---------------------------------------------------------------- M8 beyond the cutoff (E3)
def test_split_clock_beyond_cutoff_compares_rewritten_answers_with_the_blind_baseline():
    dates = ["2021-06-01"] * 20 + ["2025-06-01"] * 40                      # cutoff 2023-12-31
    model = _frame([1.0] * 20 + [1.0] * 20 + [0.0] * 20, dates, kind=["adjusted"] * 40 + ["raw"] * 20)
    blind = _frame([0.0] * 60, dates, kind=["raw"] * 60)
    res = pilot_d.split_clock_beyond(model, blind, cutoff=date(2023, 12, 31), B=2000)
    assert res["n"] == 40 and res["share"] == pytest.approx(0.5) and res["blind_share"] == pytest.approx(0.0)
    assert res["ci_lo"] > 0 and res["counts"] == {"adjusted": 20, "raw": 20}


# ---------------------------------------------------------------- M5 bands and convention, exclusions
def test_price_level_bands_and_raw_convention_use_their_own_tolerances():
    dates = ["2021-06-01"] * 30 + ["2025-06-01"] * 30
    df = _frame([0.0] * 60, dates, rel_err_raw=[0.07] * 30 + [0.5] * 30, rel_err_adj=[0.5] * 60)
    df["group"] = [1] * 30 + [0] * 30
    bands = pilot.tolerance_bands(df, bands=pilot_d.LEVEL_BANDS, method="price_level", B=200)
    assert list(bands["band"]) == [0.05, 0.10, 0.25] and list(bands["hit_pre"]) == [0.0, 1.0, 1.0]
    assert pilot.raw_convention(df, tolerance=0.10, method="price_level", B=200)["dkappa"] == pytest.approx(1.0)


def test_excluded_items_leave_every_analysis():
    df = _frame([1.0, 0.0, 1.0], ["2021-06-01"] * 3)
    df["item_id"] = ["dc-0003", "dc-0010", "dc-0035"]
    assert list(pilot_d.drop_excluded(df, "decop")["item_id"]) == ["dc-0010"]
    assert len(pilot_d.drop_excluded(df, "price_recall")) == 3


# ---------------------------------------------------------------- hallucination (E6)
def test_hallucination_and_abstention_in_abstain_mode():
    df = _frame([0.0] * 6, ["2021-06-01"] * 3 + ["2025-06-01"] * 3,
                outcome=["answer", "abstain", "abstain", "abstain", "abstain", "answer"])
    df["group"] = [1, 1, 1, 0, 0, 0]
    df["fabricated"] = [False, False, False, False, True, True]
    df["hallucinated"] = [None, None, None, None, False, True]
    res = pilot_d.hallucination(df)
    assert res["hallucination_rate"] == pytest.approx(0.5) and res["n_fabricated"] == 2
    assert res["abstain_pre"] == pytest.approx(2 / 3) and res["abstain_post"] == pytest.approx(1.0)


def test_typical_price_guesser_lands_where_a_knowledge_free_guess_would():
    from evalbench.items import Item
    items = {"r": Item("r", "split_clock", "2025-06-01", "c1", {"month": "2024-01"},
                       {"level": 0.55, "adj_level": 36.0, "factor": 0.55 / 36.0}),
             "o": Item("o", "split_clock", "2021-06-01", "c2", {"month": "2020-01"},
                       {"level": 400.0, "adj_level": 40.0, "factor": 10.0})}
    g = pilot_d.typical_guesser(items, {"2024-01": 40.0, "2020-01": 150.0}).set_index("item_id")
    assert g.loc["r", "kind"] == "adjusted" and g.loc["r", "score"] == 1.0      # a reverse split looks "normal" after
    assert g.loc["o", "kind"] == "raw" and g.loc["o", "score"] == 0.0
    assert set(g.columns) >= {"date", "cluster", "fabricated", "type"} and g.loc["r", "type"] == "reverse"
