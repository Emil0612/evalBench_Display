"""M12/M13: which price fits this month (multiple choice). Wrong options are the same share's real prices in other
months: 4-24 months away, no split in between, every pair 15% apart, printed alike; asked four times with the
correct option in each position once."""
import math

import numpy as np
import pandas as pd

from evalbench.data import Data
from evalbench.items import Item
from evalbench.llm.mock import local_answer
from evalbench.methods import LABELS, METHODS, POSITIVE_CONTROLS, MAIN_METHODS
from evalbench.methods import known_choice as kc
from evalbench.methods import price_choice as pc


def _data(split_at: str | None = None) -> Data:
    """One share whose price swings strongly (a sine with a two-year period), optionally with a 10:1 split."""
    days = pd.bdate_range("2019-01-01", "2026-08-31")
    t = np.arange(len(days))
    close = 100 * np.exp(0.6 * np.sin(2 * np.pi * t / 520))
    adj = close.copy()
    if split_at:
        before = days < pd.Timestamp(split_at)
        adj = np.where(before, close / 10, close)          # history rewritten: old prices divided by 10
        close = np.where(before, close, close / 10)        # traded price drops tenfold at the split
    universe = pd.DataFrame({"insref": [1], "company_key": [10], "primary_insref": [1], "is_primary": True,
                             "n_listings": 1, "name": ["Swing AB"], "symbol": ["SWNG"], "tradecurrency": "SEK",
                             "instrumenttype": 4, "median_mcap_2015_2024": [5e9], "price_days": 1500})
    prices = pd.DataFrame({"insref": 1, "date": days, "price_type": 0, "close": close, "adj_close": adj})
    return Data(universe=universe, prices=prices, news=pd.DataFrame())


def _base(month: str, k: int = 0) -> Item:
    return Item(f"pl3-{k:04d}", "price_level", f"{month}-28", "c10",
                {"name": "Swing AB", "symbol": "SWNG", "currency": "SEK", "month": month,
                 "month_name": "Some month"}, {"insref": 1, "baseline_level": 120.0}, {"size": "large"})


def test_wrong_options_are_other_months_of_the_same_share_far_enough_away_and_far_enough_apart():
    data = _data()
    levels = pc.monthly_levels(data.prices)
    month = pd.Period("2023-06", "M")
    decoys = pc.pick_decoys(levels, month, np.random.default_rng(1))
    assert decoys is not None and len(decoys) == 3
    gaps = [abs((pd.Period(m, "M") - month).n) for m, _ in decoys]
    assert all(pc.MIN_GAP <= g <= pc.WINDOW for g in gaps)
    target = float(levels.set_index("month").loc[month, "level"])
    vals = [target] + [v for _, v in decoys]
    assert all(abs(math.log(a / b)) >= pc.MIN_LOG_GAP for i, a in enumerate(vals) for b in vals[i + 1:])
    for m, v in decoys:                                    # each is that month's real average close
        assert abs(v - float(levels.set_index("month").loc[pd.Period(m, "M"), "level"])) < 1e-9


def test_no_wrong_option_comes_from_the_other_side_of_a_split():
    data = _data(split_at="2023-01-02")
    levels = pc.monthly_levels(data.prices)
    for seed in range(5):
        decoys = pc.pick_decoys(levels, pd.Period("2022-06", "M"), np.random.default_rng(seed))
        assert decoys is None or all(pd.Period(m, "M") < pd.Period("2023-01", "M") for m, _ in decoys)


def test_a_share_that_barely_moves_gets_no_question():
    days = pd.bdate_range("2021-01-01", "2024-12-31")
    flat = pd.DataFrame({"insref": 1, "date": days, "close": 100.0, "adj_close": 100.0})
    assert pc.pick_decoys(pc.monthly_levels(flat), pd.Period("2022-06", "M"), np.random.default_rng(0)) is None


def test_all_four_options_are_printed_alike():
    assert pc.fmt_prices([216.4, 151.2, 247.9, 182.0]) == ["216", "151", "248", "182"]
    assert pc.fmt_prices([21.64, 15.12, 9.871, 30.0]) == ["21.64", "15.12", "9.87", "30.00"]
    assert pc.fmt_prices([0.2164, 0.1512, 0.3]) == ["0.216", "0.151", "0.300"]


def test_items_keep_the_source_question_and_rotate_the_correct_option_through_every_position():
    items, dropped = pc.build_items(_data(), [_base("2023-06"), _base("2024-03", 1)], seed=3)
    assert len(items) == 2 and dropped == 0
    it = items[0]
    assert it.date == "2023-06-28" and it.cluster == "c10" and it.strata == {"size": "large"}
    assert it.truth["source_item"] == "pl3-0000" and len(it.truth["decoy_months"]) == 3
    for k, v in enumerate(pc.VARIANTS):
        opts = pc.options(it, v)
        assert opts[k] == it.payload["answer_text"] and sorted(opts) == sorted([it.payload["answer_text"],
                                                                               *it.payload["decoy_texts"]])
        assert pc.correct_letter(v) == pc.LETTERS[k]
    text = pc.render(it, "forced", "r2")[-1]["content"]
    assert "C: " + it.payload["answer_text"] + " SEK" in text and "same share's average closing price" in text


def _ans(letter):
    return {"letter": letter, "refused": letter is None, "outcome": "answer" if letter else "abstain"}


def test_the_score_is_the_share_of_the_four_orderings_answered_right():
    it = pc.build_items(_data(), [_base("2023-06")], seed=3)[0][0]
    right = {v: _ans(pc.correct_letter(v)) for v in pc.VARIANTS}
    sc = pc.item_score(it, right)
    assert sc.score == 1.0 and sc.extra["known"] and sc.extra["consistent"] and not sc.extra["one_letter"]
    habit = pc.item_score(it, {v: _ans("A") for v in pc.VARIANTS})       # always "A": right once, by position
    assert habit.score == 0.25 and habit.extra["one_letter"] and not habit.extra["consistent"]
    none = pc.item_score(it, {v: _ans(None) for v in pc.VARIANTS})
    assert none.refused and none.score == 0.0
    assert pc.parse('{"answer": "C"}')["letter"] == "C"


def test_blind_rules_name_the_option_each_simple_rule_picks():
    it = Item("x", "price_choice", "2023-06-28", "c1", {}, {"level": 100.0, "decoy_levels": [140.0, 75.0, 60.0],
                                                           "baseline_level": 125.0})
    assert pc.blind_choices(it) == {"highest": 1, "lowest": 3, "middle": 0, "persistence": 1}
    assert pc.audit([it])["middle"] == 1.0 and pc.audit([it])["highest"] == 0.0
    assert pc.audit([it])["inner_pair"] == 0.5          # 100 is the second-cheapest of 60, 75, 100, 140


def test_controls_answer_as_designed():
    it = pc.build_items(_data(), [_base("2023-06")], seed=3)[0][0]
    import json
    for v in pc.VARIANTS:
        oracle = json.loads(local_answer("mock/oracle@2024-06-30", "price_choice", it, v, "forced"))["answer"]
        assert oracle == pc.correct_letter(v)                               # 2023 is before the oracle's cutoff
        blind = json.loads(local_answer("baseline/blind@2024-06-30", "price_choice", it, v, "forced"))["answer"]
        vals = pc.option_values(it, v)
        assert blind == pc.LETTERS[int(np.argmin([abs(math.log(x / 120.0)) for x in vals]))]


def test_registered_as_a_main_price_method_and_a_positive_control():
    from evalbench.pilot import FACT_TYPES
    assert LABELS["price_choice"].startswith("M12") and LABELS["known_choice"].startswith("M13")
    assert "price_choice" in MAIN_METHODS and "known_choice" in POSITIVE_CONTROLS
    assert FACT_TYPES["price_choice"] == FACT_TYPES["known_choice"] == "price"
    assert METHODS["known_choice"].item_score is pc.item_score and kc.P0 == 0.25


def test_the_number_of_wrong_options_priced_below_the_answer_can_be_set():
    levels = pc.monthly_levels(_data().prices)
    for below in range(4):
        decoys = pc.pick_decoys(levels, pd.Period("2023-01", "M"), np.random.default_rng(below), below=below,
                                window=36)
        assert decoys is not None
        target = float(levels.set_index("month").loc[pd.Period("2023-01", "M"), "level"])
        assert sum(v < target for _, v in decoys) == below


def test_the_answer_is_not_systematically_the_middle_price():
    # a steadily rising share: without balancing, the asked month sits between earlier and later prices
    days = pd.bdate_range("2017-01-01", "2026-08-31")
    rising = pd.DataFrame({"insref": 1, "date": days, "close": 50 * np.exp(np.arange(len(days)) / 900),
                           "adj_close": 50 * np.exp(np.arange(len(days)) / 900)})
    data = Data(universe=_data().universe, prices=rising, news=pd.DataFrame())
    months = [str(m) for m in pd.period_range("2020-01", "2024-12", freq="M")]
    items, _ = pc.build_items(data, [_base(m, k) for k, m in enumerate(months)], seed=5)
    rates = pc.audit(items)
    assert len(items) >= 50
    assert all(abs(rates[r] - 0.25) < 0.12 for r in ("highest", "lowest", "middle"))
