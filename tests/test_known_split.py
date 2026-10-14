"""Pilot D additions: M7 known returns (positive control), M8 Split Clock, and fresh DE-COP passages."""
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from evalbench import build_testsets as bt
from evalbench.data import Data
from evalbench.items import Item, load_manifest, load_testset, save_testset
from evalbench.llm.mock import local_answer
from evalbench.methods import LABELS, METHODS, decop, known_return as kr, split_clock as sc
from evalbench.pilot import FACT_TYPES


# ---------------------------------------------------------------- M7 known returns (positive control)
def test_known_returns_use_only_the_largest_companies_and_old_years(data):
    items = kr.build_items(data, 30, seed=1)
    biggest = set(data.universe.nlargest(kr.N_COMPANIES, "median_mcap_2015_2024")["insref"])
    assert items and all(it.truth["insref"] in biggest for it in items)
    assert all(kr.YEARS[0] <= it.payload["year"] <= kr.YEARS[-1] <= 2022 for it in items)
    assert not any(it.fabricated for it in items)                 # positive controls ask about real facts only
    assert kr.parse is METHODS["annual_return"].parse and kr.item_score is METHODS["annual_return"].item_score


# ---------------------------------------------------------------- M8 Split Clock
def _split_data() -> Data:
    """Two companies: one with a 1:10 split on 2023-06-01, one with no split."""
    days = pd.bdate_range("2019-01-01", "2026-08-31")
    rows = []
    for ins, split in ((1, "2023-06-01"), (2, None)):
        close = np.full(len(days), 400.0)
        if split:
            close[days >= split] = 40.0
        adj = close / 10 if split else close
        if split:
            adj = np.full(len(days), 40.0)
        rows.append(pd.DataFrame({"insref": ins, "date": days, "price_type": 0, "close": close, "adj_close": adj}))
    universe = pd.DataFrame({"insref": [1, 2], "company_key": [10, 20], "primary_insref": [1, 2], "is_primary": True,
                             "n_listings": 1, "name": ["Alfa AB", "Beta AB"], "symbol": ["ALFA", "BETA"],
                             "tradecurrency": "SEK", "instrumenttype": 4, "median_mcap_2015_2024": [5e9, 6e9],
                             "price_days": 1500})
    return Data(universe=universe, prices=pd.concat(rows, ignore_index=True), news=pd.DataFrame())


def test_split_clock_items_ask_about_a_month_before_the_split_and_are_dated_at_the_split():
    items = sc.build_items(_split_data(), 10, seed=1)
    assert len(items) == 1                                         # one item per qualifying split
    it = items[0]
    assert it.date == "2023-06-01"                                 # the date that decides before/after cutoff
    month = pd.Period(it.payload["month"])
    assert month < pd.Period("2023-06", "M") - 5                   # well before the split
    assert it.truth["level"] == pytest.approx(400.0) and it.truth["adj_level"] == pytest.approx(40.0)
    assert it.truth["factor"] == pytest.approx(10.0)


def test_split_clock_scores_an_answer_near_the_rewritten_price():
    it = Item("sc-1", "split_clock", "2023-06-01", "c1", {}, {"level": 400.0, "adj_level": 40.0, "factor": 10.0})
    ans = lambda v: {"main": {"value": v, "raw": str(v), "refused": False, "outcome": "answer"}}  # noqa: E731
    assert sc.item_score(it, ans(42.0)).score == 1.0              # the post-split (rewritten) price: seen later data
    raw = sc.item_score(it, ans(390.0))
    assert raw.score == 0.0 and raw.extra["kind"] == "raw"
    assert sc.item_score(it, ans(5000.0)).extra["kind"] == "neither"


@pytest.mark.parametrize("cutoff,expected", [("2024-06-30", 1.0), ("2022-06-30", 0.0)])
def test_split_clock_oracle_knows_the_rewritten_price_only_after_the_split(cutoff, expected):
    it = sc.build_items(_split_data(), 10, seed=1)[0]
    reply = local_answer(f"mock/oracle@{cutoff}", "split_clock", it, "main", "forced")
    assert sc.item_score(it, {"main": sc.parse(reply)}).score == expected
    for kind in ("mock/random@2024-06-30", "baseline/blind@2024-06-30"):
        json.loads(local_answer(kind, "split_clock", it, "main", "forced"))


def test_new_methods_are_registered():
    assert LABELS["known_return"].startswith("M7") and LABELS["split_clock"].startswith("M8")
    assert FACT_TYPES["known_return"] == "return" and FACT_TYPES["split_clock"] == "price"


# ---------------------------------------------------------------- DE-COP v2: fresh passages
def test_pick_articles_can_exclude_articles_already_used(data):
    first = decop.pick_articles(data, 5, 1)
    used = {r["newsid"] for r, _ in first}
    fresh = decop.pick_articles(data, 5, 1, exclude=used)
    assert fresh and not used & {r["newsid"] for r, _ in fresh}


def test_fresh_generation_skips_every_article_of_earlier_versions(data, tmp_path):
    _, v1 = bt.generate_decop(data, 5, 1, decop.mock_paraphrases, "mock", root=tmp_path)
    _, v2 = bt.generate_decop(data, 5, 1, decop.mock_paraphrases, "mock", root=tmp_path, fresh=True)
    assert not {it.truth["newsid"] for it in v1} & {it.truth["newsid"] for it in v2}
    assert load_manifest("decop", "v2", root=tmp_path)["fresh"] is True
