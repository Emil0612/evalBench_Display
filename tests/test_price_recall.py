import math

import pandas as pd
import pytest

from evalbench.items import Item
from evalbench.methods import price_recall as pr


def test_build_items(data):
    items = pr.build_items(data, 30, seed=1)
    assert len(items) == 30
    fakes = [it for it in items if it.fabricated]
    assert len(fakes) == 3
    real = [it for it in items if not it.fabricated]
    assert {it.strata["size"] for it in real} == {"small", "mid", "large"}
    px = data.prices.set_index(["insref", "date"])["close"]
    for it in real:
        assert "2021-01-01" <= it.date <= "2026-08-31"
        assert it.truth["close"] == pytest.approx(px[(it.truth["insref"], pd.Timestamp(it.date))])
        assert it.truth["baseline_close"] is not None


def test_build_items_is_deterministic(data):
    assert pr.build_items(data, 12, seed=5) == pr.build_items(data, 12, seed=5)


def test_render_modes():
    it = Item("pr-0", "price_recall", "2023-03-01", "c1", {"name": "Volvo B", "symbol": "VOLV B", "currency": "SEK"}, {})
    assert "best numeric estimate" in pr.render(it, "forced", "main")[-1]["content"]
    assert "null" in pr.render(it, "abstain", "main")[-1]["content"]
    assert "Volvo B" in pr.render(it, "forced", "main")[-1]["content"]


@pytest.mark.parametrize("text,value,refused", [
    ('{"price": 210.65}', 210.65, False),
    ('```json\n{"price": "210,65"}\n```', 210.65, False),
    ('{"price": null}', None, True),
    ("I think it was about 210,65 SEK", 210.65, False),
    ("I don't know.", None, True),
    ("", None, True),
])
def test_parse_variants(text, value, refused):
    out = pr.parse(text)
    assert out["refused"] is refused
    assert (out["value"] == pytest.approx(value)) if value is not None else out["value"] is None


def test_item_score():
    it = Item("pr-0", "price_recall", "2023-03-01", "c1", {}, {"close": 100.0, "adj_close": 50.0})
    assert pr.item_score(it, {"main": {"value": 100.9, "refused": False}}).score == 1.0
    assert pr.item_score(it, {"main": {"value": 50.4, "refused": False}}).score == 1.0
    miss = pr.item_score(it, {"main": {"value": 102.0, "refused": False}})
    assert miss.score == 0.0 and miss.extra["log_error"] == pytest.approx(math.log(1.02))
    assert pr.item_score(it, {"main": {"value": None, "refused": True}}).refused


def test_item_score_fabricated():
    it = Item("pr-f", "price_recall", "2023-03-01", "f0", {}, {"close": None, "adj_close": None}, fabricated=True)
    s = pr.item_score(it, {"main": {"value": 55.0, "refused": False}})
    assert s.score == 0.0 and s.extra["hallucinated"] is True
    assert pr.item_score(it, {"main": {"value": None, "refused": True}}).extra["hallucinated"] is False


def test_swedish_three_decimal_price_is_scored_correctly():
    it = Item("pr-0", "price_recall", "2023-03-01", "c1", {}, {"close": 0.125, "adj_close": 0.125})
    assert pr.item_score(it, {"main": pr.parse('{"price": "0,125"}')}).score == 1.0
    assert pr.item_score(it, {"main": pr.parse("About 0,125 SEK")}).score == 1.0


def test_prose_fallback_takes_the_answer_not_the_year():
    assert pr.parse("In 2021, Volvo closed at 210,65")["value"] == pytest.approx(210.65)
    assert pr.parse("It closed at 210,65 on that day in 2021.")["value"] == pytest.approx(210.65)
