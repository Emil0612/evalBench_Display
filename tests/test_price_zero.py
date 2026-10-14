"""A share price of 0 is an abstention, not a price (scoring spec §10, 2026-10-07): no share trades at 0, so a model
that answers 0 says "I don't know". The app leaves it out of M5 and M10, like a 0% return under §7, and shows how often
it happens; the registered reports and the M9 control keep their own rule."""
import pandas as pd

from evalbench import analysis, ui
from evalbench.items import Item
from evalbench.methods import price_level


def _item():
    return Item(item_id="pl-1", method="price_level", date="2023-03-15", cluster="c1", payload={"name": "X"},
                truth={"level": 100.0, "adj_level": 100.0, "lo": 90.0, "hi": 110.0})


def test_a_price_of_zero_is_flagged_as_no_answer_and_a_real_price_is_not():
    zero = price_level.item_score(_item(), {"main": {"value": 0.0, "raw": "0"}})
    assert zero.extra.get("no_answer") is True and not zero.refused and zero.score == 0.0
    real = price_level.item_score(_item(), {"main": {"value": 95.0, "raw": "95"}})
    assert not real.extra.get("no_answer") and real.extra["closeness"] > 0.9


def _df():
    return pd.DataFrame({"item_id": list("abcd"), "date": ["2023-03-15"] * 4, "fabricated": False,
                         "refused": [False, False, True, False], "group": 1,
                         "no_answer": [True, None, None, False], "score": [0.0, 1.0, 0.0, 0.0]})


def test_the_app_leaves_price_zero_out_of_m5_and_m10_but_not_the_m9_control():
    assert analysis.drop_hedges(_df(), "price_level")["item_id"].tolist() == ["b", "c", "d"]
    assert analysis.drop_hedges(_df(), "us_level")["item_id"].tolist() == ["b", "c", "d"]
    assert analysis.drop_hedges(_df(), "known_level").equals(_df())     # registered positive control: unchanged


def test_the_zero_answer_rate_counts_price_zero():
    h = analysis.hedge_frame(_df())
    assert h["score"].tolist() == [1.0, 0.0, 0.0]


def test_evidence_offers_the_zero_answer_view_for_prices_with_its_own_label():
    assert ui.over_time_views("price_level")["hedge"] == "Answers of 0"
    assert ui.over_time_views("monthly_return")["hedge"] == "Answers of 0%"
