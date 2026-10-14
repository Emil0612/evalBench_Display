"""M5 price level (approximate monthly price) and M6 annual return: rougher price knowledge than an exact daily
close, which pilot C showed is too hard (docs/preregistration/2026-09-29-pilot-c.md)."""
import json

import pandas as pd
import pytest

from evalbench.items import Item
from evalbench.llm.mock import local_answer
from evalbench.methods import LABELS, METHODS, annual_return as ar, price_level as pl


# ---------------------------------------------------------------- M5 price level
def test_level_items_carry_monthly_averages_under_both_conventions(data):
    items = pl.build_items(data, 30, seed=1)
    assert len(items) == 30 and sum(it.fabricated for it in items) == 3
    px = data.prices.assign(month=data.prices["date"].dt.strftime("%Y-%m"))
    for it in (it for it in items if not it.fabricated):
        g = px[(px["insref"] == it.truth["insref"]) & (px["month"] == it.payload["month"])]
        assert it.truth["level"] == pytest.approx(g["close"].mean())
        assert it.truth["adj_level"] == pytest.approx(g["adj_close"].mean())
        assert it.date == g["date"].max().date().isoformat()     # known once the month is over
        assert "2021-01" <= it.payload["month"] <= "2026-07"


def test_level_scoring_uses_a_ten_percent_band_on_either_convention():
    it = Item("pl-1", "price_level", "2023-03-31", "c1", {}, {"level": 200.0, "adj_level": 100.0})
    ans = lambda v: {"main": {"value": v, "raw": str(v), "refused": False, "outcome": "answer"}}  # noqa: E731
    assert pl.item_score(it, ans(215.0)).score == 1.0                 # 7.5% off the raw level
    assert pl.item_score(it, ans(108.0)).score == 1.0                 # 8% off the adjusted level
    sc = pl.item_score(it, ans(150.0))
    assert sc.score == 0.0 and sc.extra["rel_err_adj"] == pytest.approx(0.5)
    fake = Item("pl-f", "price_level", "2023-03-31", "fake1", {}, {"level": None, "adj_level": None}, fabricated=True)
    assert pl.item_score(fake, ans(50.0)).extra["hallucinated"] is True


def test_level_prompt_asks_for_an_approximate_monthly_average():
    it = Item("pl-1", "price_level", "2023-03-31", "c1", {"name": "Volvo B", "symbol": "VOLV B", "currency": "SEK",
                                                         "month": "2023-03", "month_name": "March 2023"}, {})
    text = pl.render(it, "forced")[-1]["content"]
    assert "average" in text and "March 2023" in text and "Volvo B" in text


# ---------------------------------------------------------------- M6 annual return
def test_return_items_use_split_adjusted_year_end_closes(data):
    items = ar.build_items(data, 30, seed=2)
    assert len(items) == 30 and sum(it.fabricated for it in items) == 3
    px = data.prices
    for it in (it for it in items if not it.fabricated):
        y = it.payload["year"]
        g = px[px["insref"] == it.truth["insref"]]
        start = g[g["date"].dt.year == y - 1]["adj_close"].iloc[-1]
        end = g[g["date"].dt.year == y]["adj_close"].iloc[-1]
        assert it.truth["return_pct"] == pytest.approx((end / start - 1) * 100)
        assert 2021 <= y <= 2025 and it.date.startswith(str(y))
        assert it.truth["median_return_pct"] is not None


@pytest.mark.parametrize("text,value", [
    ('{"change_percent": -12.5}', -12.5), ('{"change_percent": "−12,5 %"}', -12.5), ('{"change_percent": "+8"}', 8.0),
    ("The share price fell by 23% during the year.", -23.0), ("Aktien sjönk med 12,5 procent.", -12.5),
    ("It rose about 40 percent.", 40.0), ("I estimate -3.2%", -3.2),
])
def test_return_parsing_keeps_the_sign(text, value):
    p = ar.parse(text)
    assert p["outcome"] == "answer" and p["value"] == pytest.approx(value)


def test_return_parsing_non_answers():
    assert ar.parse('{"change_percent": null}')["outcome"] == "abstain"
    assert ar.parse("I'm sorry, I can't help with that.")["outcome"] == "refusal"


def test_return_scoring_direction_magnitude_and_versus_the_median():
    it = Item("ar-1", "annual_return", "2022-12-30", "c1", {"year": 2022},
              {"return_pct": -30.0, "median_return_pct": -20.0})
    ans = lambda v: {"main": {"value": v, "raw": str(v), "refused": False, "outcome": "answer"}}  # noqa: E731
    sc = ar.item_score(it, ans(-25.0))
    assert sc.score == 1.0 and sc.extra["abs_err_pp"] == pytest.approx(5.0)
    assert sc.extra["within_10pp"] is True and sc.extra["vs_median_hit"] is True
    sc = ar.item_score(it, ans(-10.0))                     # right direction, but says it beat the median: it didn't
    assert sc.score == 1.0 and sc.extra["vs_median_hit"] is False
    assert ar.item_score(it, ans(15.0)).score == 0.0


# ---------------------------------------------------------------- registry and controls
def test_new_methods_are_registered_with_fact_types():
    from evalbench import pilot
    assert {"price_level", "annual_return"} <= set(METHODS)
    assert LABELS["price_level"].startswith("M5") and LABELS["annual_return"].startswith("M6")
    assert pilot.FACT_TYPES["price_level"] == "price" and pilot.FACT_TYPES["annual_return"] == "return"


@pytest.mark.parametrize("method", ["price_level", "annual_return"])
def test_controls_answer_the_new_methods(data, method):
    mod = METHODS[method]
    items = [it for it in mod.build_items(data, 20, seed=3) if not it.fabricated]
    early = next(it for it in items if it.date < "2024-01-01")
    oracle = mod.item_score(early, {"main": mod.parse(local_answer("mock/oracle@2024-06-30", method, early, "main",
                                                                   "forced"))})
    assert oracle.score == 1.0
    for kind in ("mock/random@2024-06-30", "baseline/blind@2024-06-30"):
        reply = local_answer(kind, method, early, "main", "forced")
        assert mod.parse(reply)["outcome"] in ("answer", "abstain")
        json.loads(reply)
