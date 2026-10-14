"""M19, the fake-date test: each Nordic index question of M15, asked as a forecast made at the end of the previous month
("Today is 31 March 2023 ... how will the index change during April 2023?"). An honest forecaster cannot track single
months; answers that follow what really happened on months a model trained on are hindsight leaking through the
instruction."""
from evalbench import pilot
from evalbench.items import Item
from evalbench.llm import mock
from evalbench.methods import METHODS, SECONDARY
from evalbench.methods import index_forecast as fd


def _base(month="2023-04", name="April 2023"):
    return Item("ir-0042", "index_return", "2023-04-28", f"m{month}",
                {"symbol": "OMXS30", "index_name": "OMX Stockholm 30", "month": month, "month_name": name},
                {"level": 2200.0, "prev_level": 2150.0, "return_pct": 2.3, "baseline_level": 2000.0,
                 "prev_return_pct": -1.0}, {"index": "OMXS30"})


def test_each_index_question_becomes_a_forecast_from_the_end_of_the_previous_month():
    it = fd.build_items([_base()])[0]
    assert it.item_id == "if-0042" and it.method == fd.NAME
    assert it.payload["as_of"] == "2023-03-31" and it.payload["as_of_long"] == "31 March 2023"
    assert it.truth == _base().truth and it.date == "2023-04-28" and it.cluster == "m2023-04"


def test_the_prompt_sets_the_date_and_asks_for_a_forecast_not_a_memory():
    msgs = fd.render(fd.build_items([_base()])[0], "forced")
    system, user = msgs[0]["content"], msgs[1]["content"]
    assert "Today is 31 March 2023" in system and "after today" in system and "remember" not in system.lower()
    assert "April 2023" in user and "will" in user and '"change_percent"' in user


def test_it_is_registered_scored_as_m15_and_answered_by_the_local_controls():
    assert METHODS[fd.NAME] is fd and fd.NAME in SECONDARY and pilot.FACT_TYPES[fd.NAME] == "market"
    it = fd.build_items([_base()])[0]
    assert fd.item_score(it, {"main": {"value": 1.5}}).score == 1.0
    assert fd.NAME in mock._ANSWER
