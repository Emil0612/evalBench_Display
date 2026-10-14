"""Pilot E question sets: M5 v2 (fresh months in two blocks around DeepSeek V3.1's cutoff and release), M8 v2
(ordinary splits only) and M11 monthly returns (scored against the market)."""
import json

import numpy as np
import pandas as pd
import pytest

from evalbench import pilot_e
from evalbench.items import Item
from evalbench.llm.mock import local_answer
from evalbench.methods import LABELS, MAIN_METHODS, METHODS
from evalbench.methods import monthly_return as mr
from evalbench.methods import price_level as pl
from evalbench.methods import split_clock as sc
from evalbench.pilot import FACT_TYPES


# ---------------------------------------------------------------- M5 v2
def test_price_level_months_can_be_restricted_and_pairs_excluded(data):
    months = pd.period_range("2025-09", "2026-07", freq="M")
    first = pl.build_items(data, 20, seed=1, months=months, fabricated_share=0)
    taken = {(it.truth["insref"], it.payload["month"]) for it in first}
    again = pl.build_items(data, 20, seed=2, months=months, fabricated_share=0, exclude=taken)
    assert all(it.payload["month"] >= "2025-09" for it in first + again)
    assert not taken & {(it.truth["insref"], it.payload["month"]) for it in again}


def test_price_level_v2_has_a_block_before_the_cutoff_and_one_after_the_release(data):
    items = pilot_e.build_price_level_v2(data, seed=3, n_per_block=20, exclude=set())
    real = [it for it in items if not it.fabricated]
    before = [it for it in real if it.payload["month"] <= "2024-11"]
    after = [it for it in real if it.payload["month"] >= "2025-09"]
    assert len(before) == 20 and len(after) == 20 and len(real) == 40
    assert sum(it.fabricated for it in items) == 4                       # 10%, as in v1
    assert len({it.item_id for it in items}) == len(items) and all(it.item_id.startswith("pl2-") for it in items)
    assert all(it.truth.get("lo") is not None for it in real)             # the month's range, for the spec's rule


def test_price_level_v2_never_repeats_a_v1_question(data):
    v1 = pl.build_items(data, 30, seed=9)
    used = {(it.truth["insref"], it.payload["month"]) for it in v1 if not it.fabricated}
    v2 = pilot_e.build_price_level_v2(data, seed=3, n_per_block=15, exclude=used)
    assert not used & {(it.truth["insref"], it.payload["month"]) for it in v2 if not it.fabricated}


# ---------------------------------------------------------------- M8 v2
def _split_prices():
    days = pd.bdate_range("2019-01-01", "2026-08-31")
    rows = []
    for ins, kind in ((1, "ordinary"), (2, "reverse")):
        close = np.where(days >= "2023-06-01", 40.0 if kind == "ordinary" else 4000.0, 400.0)
        adj = np.full(len(days), 40.0 if kind == "ordinary" else 4000.0)
        rows.append(pd.DataFrame({"insref": ins, "date": days, "price_type": 0, "close": close, "adj_close": adj}))
    return pd.concat(rows, ignore_index=True)


def test_split_clock_can_be_limited_to_ordinary_splits():
    from evalbench.data import Data
    universe = pd.DataFrame({"insref": [1, 2], "company_key": [10, 20], "primary_insref": [1, 2], "is_primary": True,
                             "n_listings": 1, "name": ["Up AB", "Down AB"], "symbol": ["UP", "DN"],
                             "tradecurrency": "SEK", "instrumenttype": 4, "median_mcap_2015_2024": [1e9, 2e9],
                             "price_days": 1500})
    data = Data(universe=universe, prices=_split_prices(), news=pd.DataFrame())
    both = sc.build_items(data, 10, seed=1)
    ordinary = sc.build_items(data, 10, seed=1, kinds=("ordinary",))
    assert {it.truth["factor"] > 1 for it in both} == {True, False}
    assert ordinary and all(it.truth["factor"] > 1 for it in ordinary)


# ---------------------------------------------------------------- M11 monthly returns
def test_monthly_return_items_have_a_return_the_market_median_and_a_blind_guess(data):
    items = mr.build_items(data, 30, seed=4)
    real = [it for it in items if not it.fabricated]
    assert len(items) == 30 and sum(it.fabricated for it in items) == 3
    assert all("2021-01" <= it.payload["month"] <= "2026-07" for it in real)
    assert all(it.truth["return_pct"] is not None and it.truth["market_return_pct"] is not None for it in real)
    assert len({it.item_id for it in items}) == 30


def _item(ret, market, fabricated=False):
    return Item(item_id="mr-x", method="monthly_return", date="2023-03-31", cluster="c1",
                payload={"name": "X", "symbol": "X", "month": "2023-03", "month_name": "March 2023"},
                truth={"return_pct": ret, "market_return_pct": market, "baseline_return_pct": 1.0, "insref": 1},
                fabricated=fabricated)


def _ans(v):
    return {"main": mr.parse(json.dumps({"change_percent": v}))}


def test_monthly_returns_are_scored_against_the_market():
    down_market = _item(ret=-2.0, market=-6.0)                   # fell, but beat a falling market
    assert mr.item_score(down_market, _ans(-1.0)).score == 1.0   # right side of the market
    assert mr.item_score(down_market, _ans(-9.0)).score == 0.0   # says it did worse than the market
    extra = mr.item_score(down_market, _ans(-9.0)).extra
    assert extra["abs_direction_hit"] is True and extra["abs_err_pp"] == pytest.approx(7.0)
    assert extra["within_5pp"] is False
    assert mr.P0 == 0.5


def test_fabricated_and_unanswered_monthly_returns():
    assert mr.item_score(_item(None, None, fabricated=True), _ans(3.0)).extra["hallucinated"] is True
    assert mr.item_score(_item(1.0, 0.0), {"main": mr.parse('{"change_percent": null}')}).refused is True


def test_monthly_return_is_registered_as_a_main_return_method(data):
    assert LABELS["monthly_return"].startswith("M11") and FACT_TYPES["monthly_return"] == "return"
    assert "monthly_return" in MAIN_METHODS and "monthly_return" in METHODS
    it = next(i for i in mr.build_items(data, 10, seed=1) if not i.fabricated)
    oracle = local_answer("mock/oracle@2024-06-30", "monthly_return", it, "main", "forced")
    assert mr.item_score(it, {"main": mr.parse(oracle)}).score == 1.0 or it.date > "2024-06-30"
    for kind in ("mock/random@2024-06-30", "baseline/blind@2024-06-30"):
        json.loads(local_answer(kind, "monthly_return", it, "main", "forced"))


# ---------------------------------------------------------------- the monthly grid (price level v3)
def test_monthly_grid_has_the_same_number_of_questions_every_month_across_sizes(data):
    months = pd.period_range("2024-01", "2024-06", freq="M")
    items = pl.build_monthly_grid(data, per_month=6, seed=2, months=months, exclude=set())
    per_month = pd.Series([it.payload["month"] for it in items]).value_counts()
    assert set(per_month.index) == {str(m) for m in months} and set(per_month) == {6}
    sizes = pd.Series([(it.payload["month"], it.strata["size"]) for it in items]).value_counts()
    assert set(sizes) == {2}                                   # 2 large, 2 mid, 2 small every month
    assert not any(it.fabricated for it in items)
    assert len({it.item_id for it in items}) == len(items) and all(it.item_id.startswith("pl3-") for it in items)
    assert all(it.truth.get("lo") is not None for it in items)


def test_monthly_grid_never_repeats_an_excluded_pair(data):
    months = pd.period_range("2024-01", "2024-03", freq="M")
    first = pl.build_monthly_grid(data, per_month=3, seed=1, months=months, exclude=set())
    used = {(it.truth["insref"], it.payload["month"]) for it in first}
    again = pl.build_monthly_grid(data, per_month=3, seed=2, months=months, exclude=used)
    assert not used & {(it.truth["insref"], it.payload["month"]) for it in again}
