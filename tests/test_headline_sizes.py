"""M18 by company size: a headline takes the market-cap tercile of the listed company it is about (the largest, when it
names several); headlines about no listed company form their own group."""
import numpy as np
import pandas as pd
import pytest

from evalbench import headline_report as hr
from evalbench import ui
from evalbench.items import Item
from evalbench.methods import headline_date as hd


def _item(i, newsid):
    return Item(f"hd-{i:04d}", "headline_date", "2020-05-14", "m2020-05", {"headline": "x", "language": "sv"},
                {"month": "2020-05", "blind_month": "2020-01", "newsid": newsid})


def test_a_headline_takes_the_size_of_its_largest_listed_company():
    items = [_item(1, "a"), _item(2, "b"), _item(3, "c"), _item(4, "d")]
    refs = {"a": np.array([10]), "b": np.array([10, 20]), "c": None, "d": np.array([99])}
    company = {10: 1, 20: 2}
    sizes = {"c1": "small", "c2": "large"}
    out = hd.company_sizes(items, refs, company, sizes)
    assert out == {"hd-0001": "small", "hd-0002": "large", "hd-0003": "none", "hd-0004": "unknown"}


def test_headlines_split_by_size_on_the_profile_with_their_own_row_for_no_company():
    assert ui.splits_by_size("headline_date")
    assert ui.row_label("headline_date", "all") == "Nordic · news headlines"
    assert ui.row_label("headline_date", "none") == "Nordic · no listed company"
    df = pd.DataFrame({"s_size": ["large", "none", "unknown"], "s_company_size": ["unknown"] * 3})
    assert list(ui.size_of(df, "headline_date")) == ["large", "none", "unknown"]


def test_size_rows_compare_model_and_text_only_on_the_months_before_the_cutoff():
    rng = np.random.default_rng(0)
    rows = []
    for k in range(120):
        size = ("large", "small")[k % 2]
        rows.append({"item_id": f"hd-{1000 + k}", "side": 1, "month": pd.Period(f"20{16 + k % 5}-0{1 + k % 9}", "M"),
                     "score": (0.6 if size == "large" else 0.3) + rng.normal(0, 0.05), "blind_score": 0.3,
                     "refused": False, "size": size})
    out = {r["size"]: r for r in hr.size_rows(pd.DataFrame(rows), B=500)}
    assert out["large"]["d"] == pytest.approx(0.3, abs=0.02) and out["large"]["n"] == 60
    assert abs(out["small"]["d"]) < 0.03
