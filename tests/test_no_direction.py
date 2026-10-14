"""Return methods: an answer of exactly 0 claims no direction, so it scores 0.5 (chance), not "down"
(scoring change of 2026-10-01; docs/scoring.md §6)."""
import pytest

from evalbench.items import Item
from evalbench.methods import annual_return as ar
from evalbench.methods import known_return as kr
from evalbench.methods import monthly_return as mr


def ans(v):
    return {"main": {"value": v, "raw": str(v), "refused": False, "outcome": "answer"}}


def _year(t, med=-1.0):
    return Item("ar-1", "annual_return", "2022-12-30", "c1", {"year": 2022}, {"return_pct": t, "median_return_pct": med})


def _month(t, mkt):
    return Item("mr-1", "monthly_return", "2023-03-31", "c1", {"month": "2023-03"},
                {"return_pct": t, "market_return_pct": mkt})


@pytest.mark.parametrize("truth", [-30.0, 172.1])
def test_a_zero_answer_to_a_yearly_return_scores_chance_whichever_way_the_share_went(truth):
    sc = ar.item_score(_year(truth), ans(0.0))
    assert sc.score == 0.5 and sc.extra["no_direction"] is True and sc.refused is False
    assert sc.extra["abs_err_pp"] == pytest.approx(abs(truth))       # the error is still recorded


def test_known_returns_share_the_rule():
    assert kr.item_score(_year(-30.0), ans(0)).score == 0.5


def test_a_directional_answer_is_scored_as_before():
    assert ar.item_score(_year(172.1), ans(15.3)).score == 1.0
    assert ar.item_score(_year(172.1), ans(-2.0)).score == 0.0
    assert ar.item_score(_year(172.1), ans(15.3)).extra["no_direction"] is False


def test_a_share_that_did_not_move_still_rewards_a_flat_answer():
    assert ar.item_score(_year(0.0), ans(0.0)).score == 1.0


@pytest.mark.parametrize("truth,market", [(-2.0, -6.0), (4.0, 1.0), (-8.0, 3.0)])
def test_a_zero_answer_to_a_monthly_return_scores_chance(truth, market):
    sc = mr.item_score(_month(truth, market), ans(0.0))
    assert sc.score == 0.5 and sc.extra["no_direction"] is True
    assert mr.item_score(_month(truth, market), ans(1.5)).extra["no_direction"] is False


# ---------------------------------------------------------------- the over-time views
def test_over_time_views_plot_closeness_for_price_levels_and_the_score_otherwise():
    import pandas as pd

    from evalbench import ui
    rows = [{"item_id": f"i{k}", "date": f"2024-{1 + k % 12:02d}-15", "cluster": f"c{k % 7}", "fabricated": False,
             "refused": False, "score": 1.0, "closeness": 0.4} for k in range(60)]
    df = pd.DataFrame(rows)
    level = ui.rolling_from_scores(df, "price_level")
    assert len(level) and (level["mean"].round(6) == 0.4).all()          # closeness, not the ±10% hit
    other = ui.rolling_from_scores(df.drop(columns="closeness"), "annual_return")
    assert (other["mean"] == 1.0).all()
