"""M16/M17: the same index questions for four US indices (S&P 500, Dow Jones, Nasdaq Composite, Nasdaq-100), as the
reference the Nordic result is compared with. PRIVATE: Yahoo data, never published."""
import json

import numpy as np
import pandas as pd

from evalbench.llm.mock import local_answer
from evalbench.methods import LABELS, MAIN_METHODS, METHODS, PRIVATE_CHECKS
from evalbench.methods import index_level as il
from evalbench.methods import us_index_level as ul
from evalbench.methods import us_index_return as ur


def _daily():
    days = pd.bdate_range("2014-01-01", "2026-09-16")
    rows = []
    for k, (sym, name) in enumerate(ul.NAMES.items()):
        level = 2000 * (k + 1) * np.exp(np.arange(len(days)) / 3000)
        rows.append(pd.DataFrame({"symbol": sym, "name": name, "currency": "USD", "date": days, "close": level}))
    return pd.concat(rows, ignore_index=True)


def test_us_questions_match_the_nordic_ones_month_for_month():
    us = ul.build_items(_daily(), seed=1)
    assert len(us) == 4 * len(il.MONTHS)
    assert {i.payload["symbol"] for i in us} == set(ul.NAMES)
    assert all(i.item_id.startswith("ul-") and i.method == "us_index_level" for i in us)
    ret = ur.build_items(_daily(), seed=1)
    assert [i.payload["month"] for i in ret] == [i.payload["month"] for i in us]      # same questions, same order
    assert all(i.method == "us_index_return" for i in ret)


def test_the_wording_and_scoring_are_exactly_the_nordic_ones():
    it = ul.build_items(_daily(), seed=1)[0]
    text = ul.render(it, "forced")[-1]["content"]
    assert text.startswith(f"What was the closing level of the {it.payload['index_name']} index ({it.payload['symbol']})")
    assert ul.item_score is il.item_score and ul.parse is il.parse and ur.item_score is not None
    assert ul.CLOSENESS_ZERO == il.CLOSENESS_ZERO


def test_registered_as_private_checks_with_controls():
    assert LABELS["us_index_level"].startswith("M16") and LABELS["us_index_return"].startswith("M17")
    assert {"us_index_level", "us_index_return"} <= PRIVATE_CHECKS
    assert not {"us_index_level", "us_index_return"} & set(MAIN_METHODS)
    assert METHODS["us_index_return"].P0 == 0.5
    it = ul.build_items(_daily(), seed=1)[0]
    json.loads(local_answer("mock/oracle@2024-06-30", "us_index_level", it, "main", "forced"))
    json.loads(local_answer("baseline/blind@2024-06-30", "us_index_return", it, "main", "forced"))
