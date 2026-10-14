"""M14 index level and M15 index monthly change: market-level memory (the S&P 500's monthly direction is recalled
98% right inside training windows; Lopez-Lira et al.). Month-end closes of four Nordic indices."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from evalbench.llm.mock import local_answer
from evalbench.methods import LABELS, MAIN_METHODS, METHODS
from evalbench.methods import index_level as il
from evalbench.methods import index_return as ir


def _daily() -> pd.DataFrame:
    days = pd.bdate_range("2014-01-01", "2026-09-16")
    rows = []
    for k, (ins, sym) in enumerate(il.INDICES.items()):
        level = 1000 * (k + 1) * np.exp(0.15 * np.sin(np.arange(len(days)) / 60 + k))
        rows.append(pd.DataFrame({"insref": ins, "symbol": sym, "name": il.NAMES[sym], "currency": "SEK",
                                  "date": days, "close": level}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def items():
    return il.build_items(_daily(), seed=1)


def test_one_question_per_index_and_month_with_month_end_truths(items):
    months = pd.period_range("2016-01", "2026-08", freq="M")
    assert len(items) == 4 * len(months)
    daily = _daily()
    it = next(i for i in items if i.payload["symbol"] == "OMXS30" and i.payload["month"] == "2023-03")
    m = daily[(daily.symbol == "OMXS30") & (daily.date.dt.to_period("M") == pd.Period("2023-03", "M"))]
    prev = daily[(daily.symbol == "OMXS30") & (daily.date.dt.to_period("M") == pd.Period("2023-02", "M"))]
    assert it.truth["level"] == pytest.approx(m.close.iloc[-1])
    assert it.truth["return_pct"] == pytest.approx((m.close.iloc[-1] / prev.close.iloc[-1] - 1) * 100)
    assert it.date == m.date.iloc[-1].date().isoformat()
    assert it.cluster == "m2023-03"                     # the four indices move together: resample months
    assert it.truth["baseline_level"] is not None and it.truth["prev_return_pct"] is not None


def test_the_first_24_questions_are_the_micro_pilot_block(items):
    block = items[:24]
    assert {i.payload["symbol"] for i in block} == {"OMXS30"}
    old = [i for i in block if i.payload["month"] <= "2023-12"]
    new = [i for i in block if i.payload["month"] >= "2025-09"]
    assert len(old) == 12 and len(new) == 12
    assert len({i.item_id for i in items}) == len(items)


def _parsed(v, key="level"):
    return {"main": (il if key == "level" else ir).parse(json.dumps({key: v}))}


def test_index_level_closeness_reaches_zero_at_25_percent_off(items):
    it = items[0]
    t = it.truth["level"]
    assert il.item_score(it, _parsed(t)).score == pytest.approx(1.0)
    assert il.item_score(it, _parsed(t * 1.1)).score == pytest.approx(1 - math.log(1.1) / math.log(1.25))
    assert il.item_score(it, _parsed(t * 1.3)).score == 0.0
    sc = il.item_score(it, _parsed(t * 1.03))
    assert sc.extra["within_5pct"] and not sc.extra["within_2pct"]
    assert il.parse('{"level": null}')["refused"] is True


def test_index_return_is_scored_on_direction_and_a_zero_is_no_direction(items):
    it = next(i for i in items if abs(i.truth["return_pct"]) > 1)
    t = it.truth["return_pct"]
    assert ir.item_score(it, _parsed(t / 2, "change_percent")).score == 1.0
    assert ir.item_score(it, _parsed(-t, "change_percent")).score == 0.0
    zero = ir.item_score(it, _parsed(0, "change_percent"))
    assert zero.score == 0.5 and zero.extra["no_direction"]
    assert ir.item_score(it, _parsed(t + 1.5, "change_percent")).extra["abs_err_pp"] == pytest.approx(1.5)


def test_controls(items):
    old = next(i for i in items if i.payload["month"] == "2019-06" and i.payload["symbol"] == "OMXS30")
    oracle = json.loads(local_answer("mock/oracle@2024-06-30", "index_level", old, "main", "forced"))
    assert oracle["level"] == pytest.approx(old.truth["level"], abs=1e-3)
    blind = json.loads(local_answer("baseline/blind@2024-06-30", "index_level", old, "main", "forced"))
    assert blind["level"] == pytest.approx(old.truth["baseline_level"], abs=1e-3)   # the same month a year earlier
    blind_r = json.loads(local_answer("baseline/blind@2024-06-30", "index_return", old, "main", "forced"))
    assert blind_r["change_percent"] == pytest.approx(old.truth["prev_return_pct"], abs=1e-3)   # last month again


def test_registered_as_main_methods():
    from evalbench.pilot import FACT_TYPES
    assert LABELS["index_level"].startswith("M14") and LABELS["index_return"].startswith("M15")
    assert {"index_level", "index_return"} <= set(MAIN_METHODS)
    assert FACT_TYPES["index_level"] == "market" and FACT_TYPES["index_return"] == "market"   # spec §9, 2026-10-06
    assert METHODS["index_return"].P0 == 0.5 and METHODS["index_level"].P0 == 0.0
