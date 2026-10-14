"""M9 known price levels (positive control, Millistream) and M10 US price levels (private check), plus the two
extra readings the price-scoring spec proposes: 'within the month's own range' and a 0-1 closeness score."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from evalbench.items import Item
from evalbench.llm.mock import local_answer
from evalbench.methods import LABELS, MAIN_METHODS, METHODS, POSITIVE_CONTROLS, PRIVATE_CHECKS
from evalbench.methods import known_level as kl
from evalbench.methods import price_level as pl
from evalbench.methods import us_level as us
from evalbench.pilot import FACT_TYPES


def _answer(price):
    return {"main": pl.parse(json.dumps({"price": price}))}


def _level_item(**truth):
    base = {"level": 100.0, "adj_level": 50.0, "baseline_level": 90.0, "lo": 95.0, "hi": 104.0,
            "adj_lo": 47.5, "adj_hi": 52.0, "insref": 1}
    return Item(item_id="x", method="price_level", date="2023-03-31", cluster="c1", payload={}, truth={**base, **truth})


# ---------------------------------------------------------------- the spec's extra readings (score unchanged)
def test_an_answer_inside_the_months_own_range_is_marked_in_range():
    it = _level_item()
    assert pl.item_score(it, _answer(103.0)).extra["in_range"] is True
    assert pl.item_score(it, _answer(51.0)).extra["in_range"] is True       # inside the adjusted range
    assert pl.item_score(it, _answer(106.0)).extra["in_range"] is False     # within 10% (a hit), but outside the range
    assert pl.item_score(it, _answer(106.0)).score == 1.0


def test_closeness_is_one_when_exact_and_zero_once_off_by_a_factor_of_two():
    it = _level_item(adj_level=None, adj_lo=None, adj_hi=None)
    assert pl.item_score(it, _answer(100.0)).extra["closeness"] == pytest.approx(1.0)
    assert pl.item_score(it, _answer(200.0)).extra["closeness"] == pytest.approx(0.0)
    assert pl.item_score(it, _answer(50.0)).extra["closeness"] == pytest.approx(0.0)
    assert pl.item_score(it, _answer(100 * math.sqrt(2))).extra["closeness"] == pytest.approx(0.5)


def test_old_items_without_a_range_still_score():
    it = _level_item(lo=None, hi=None, adj_lo=None, adj_hi=None)
    out = pl.item_score(it, _answer(101.0))
    assert out.score == 1.0 and out.extra["in_range"] is None


def test_new_price_level_items_carry_the_months_range(data):
    items = [i for i in pl.build_items(data, 12, seed=3) if not i.fabricated]
    assert items and all(i.truth["lo"] <= i.truth["level"] <= i.truth["hi"] for i in items)


# ---------------------------------------------------------------- M9 known price levels
def test_known_levels_ask_only_about_the_largest_companies_before_every_cutoff(data):
    items = kl.build_items(data, 40, seed=1)
    biggest = set(data.universe.nlargest(kl.N_COMPANIES, "median_mcap_2015_2024")["insref"])
    assert len(items) == 40 and all(it.truth["insref"] in biggest for it in items)
    assert all("2016-01" <= it.payload["month"] <= "2022-12" for it in items)
    assert not any(it.fabricated for it in items)
    assert all(it.truth["lo"] <= it.truth["level"] <= it.truth["hi"] for it in items)
    assert len({it.item_id for it in items}) == 40 and all(it.item_id.startswith("kl-") for it in items)
    assert kl.render is pl.render and kl.item_score is pl.item_score and kl.parse is pl.parse


# ---------------------------------------------------------------- M10 US price levels (private)
def _us_prices(tmp_path):
    days = pd.bdate_range("2020-01-01", "2026-08-31")
    rows = []
    for symbol, name, split in (("AAA", "Alpha Corp", None), ("BBB", "Beta Inc", "2023-05-01")):
        close = np.linspace(100, 200, len(days))
        rows.append(pd.DataFrame({"symbol": symbol, "name": name, "date": days, "close": close,
                                  "split": [4.0 if split and d == pd.Timestamp(split) else 0.0 for d in days]}))
    path = tmp_path / "us_prices.parquet"
    pd.concat(rows).to_parquet(path)
    return path


def test_us_levels_skip_shares_that_split_and_use_the_year_before_as_the_blind_guess(tmp_path):
    items = us.build_items(None, 30, seed=2, path=_us_prices(tmp_path))
    assert len(items) == 30 and {it.payload["symbol"] for it in items} == {"AAA"}   # BBB split in the window
    assert all("2021-01" <= it.payload["month"] <= "2026-07" for it in items)
    assert all(it.payload["currency"] == "USD" and it.cluster == "us-AAA" for it in items)
    assert all(it.truth["baseline_level"] < it.truth["level"] for it in items)     # rising prices: a year earlier is lower
    assert all(it.truth["lo"] <= it.truth["level"] <= it.truth["hi"] for it in items)


def test_a_split_before_the_question_window_does_not_exclude_a_share(tmp_path):
    days = pd.bdate_range("2020-01-01", "2026-08-31")      # like Apple: a 4:1 split on 2020-08-31
    prices = pd.DataFrame({"symbol": "APL", "name": "Apple", "date": days, "close": 150.0,
                           "split": [4.0 if d == pd.Timestamp("2020-08-31") else 0.0 for d in days]})
    prices.to_parquet(tmp_path / "p.parquet")
    items = us.build_items(None, 10, seed=1, path=tmp_path / "p.parquet")
    assert len(items) == 10 and {it.payload["symbol"] for it in items} == {"APL"}


def test_us_levels_without_the_private_file_say_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_us_prices"):
        us.build_items(None, 5, seed=1, path=tmp_path / "missing.parquet")


# ---------------------------------------------------------------- registry
def test_the_level_controls_are_registered_but_kept_out_of_the_main_table():
    assert LABELS["known_level"].startswith("M9") and LABELS["us_level"].startswith("M10")
    assert FACT_TYPES["known_level"] == "price" and FACT_TYPES["us_level"] == "price"
    assert "known_level" in POSITIVE_CONTROLS and "us_level" in PRIVATE_CHECKS
    assert "known_level" not in MAIN_METHODS and "us_level" not in MAIN_METHODS
    assert set(METHODS) >= {"known_level", "us_level"}


def test_controls_can_answer_the_new_methods(data):
    it = kl.build_items(data, 3, seed=1)[0]
    for kind in ("mock/oracle@2024-06-30", "mock/random@2024-06-30", "baseline/blind@2024-06-30"):
        reply = local_answer(kind, "known_level", it, "main", "forced")
        assert kl.item_score(it, {"main": kl.parse(reply)}).refused is False
    oracle = local_answer("mock/oracle@2024-06-30", "known_level", it, "main", "forced")
    assert kl.item_score(it, {"main": kl.parse(oracle)}).score == 1.0
