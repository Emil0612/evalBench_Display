"""M20, the masking test: an unnamed Nordic index's 12 monthly changes, no dates and no levels: in which month does the
period end? A model that places it in time can recover the dates a backtest tried to hide."""
import pandas as pd

from evalbench import pilot
from evalbench.items import Item
from evalbench.llm import mock
from evalbench.methods import METHODS, SECONDARY
from evalbench.methods import path_date as pd_


def _base(symbol, months):
    return [Item(f"ir-{symbol}{i:03d}", "index_return", f"{m}-28", f"m{m}",
                 {"symbol": symbol, "index_name": f"Index {symbol}", "month": m, "month_name": m},
                 {"return_pct": 0.1 * (i + 1) * (-1) ** i}, {"index": symbol}) for i, m in enumerate(months)]


MONTHS = [str(p) for p in pd.period_range("2020-01", "2021-02", freq="M")]          # 14 months


def test_every_12_month_window_of_every_index_becomes_one_question():
    its = pd_.build_items(_base("A", MONTHS) + _base("B", MONTHS))
    assert len(its) == 6                                                          # 3 end months x 2 indices
    first = [i for i in its if i.truth["symbol"] == "A" and i.truth["month"] == "2020-12"][0]
    assert first.payload["changes"] == [round(0.1 * (k + 1) * (-1) ** k, 1) for k in range(12)]
    assert first.truth["first_month"] == "2020-01" and first.cluster == "m2020-12"
    assert first.item_id.startswith("pd-") and "blind_month" in first.truth


def test_a_gap_in_the_months_breaks_the_window():
    gap = [m for m in MONTHS if m != "2020-06"]
    assert pd_.build_items(_base("A", gap)) == []


def test_the_prompt_shows_changes_only_no_name_no_levels_no_dates():
    it = pd_.build_items(_base("A", MONTHS))[0]
    text = " ".join(m["content"] for m in pd_.render(it, "forced"))
    assert "Index A" not in text and "A)" not in text and "2020" not in text and "2021" not in text
    assert "+0.1" in text and '"year"' in text and '"month"' in text


def test_it_is_registered_scored_by_months_off_and_answered_by_the_controls():
    it = pd_.build_items(_base("A", MONTHS))[0]
    assert METHODS[pd_.NAME] is pd_ and pd_.NAME in SECONDARY and pilot.FACT_TYPES[pd_.NAME] == "market"
    s = pd_.item_score(it, {"main": pd_.parse('{"year": 2020, "month": 11}')})
    assert s.extra["err_months"] == -1 and abs(s.score - (1 - 1 / 12)) < 1e-9
    assert pd_.NAME in mock._ANSWER
