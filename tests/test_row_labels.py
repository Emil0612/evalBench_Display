"""Profile rows name what a method actually asks about: the market (Nordic or US) and the population (company sizes,
the 15 largest companies, indices, headlines). Only methods with real company sizes can be split by size."""
import pytest

from evalbench import ui


@pytest.mark.parametrize("method,stratum,label", [
    ("price_level", "all", "Nordic · all sizes"),
    ("price_level", "large", "Nordic · large caps"),
    ("monthly_return", "small", "Nordic · small caps"),
    ("index_return", "all", "Nordic · indices"),
    ("us_index_return", "all", "US · indices"),
    ("us_index_level", "all", "US · indices"),
    ("us_level", "all", "US · large companies"),
    ("known_level", "all", "Nordic · 15 largest companies"),
    ("headline_date", "all", "Nordic · news headlines"),
])
def test_row_labels_name_the_market_and_the_population(method, stratum, label):
    assert ui.row_label(method, stratum) == label


def test_only_methods_with_company_sizes_can_be_split():
    for m in ("price_level", "price_recall", "monthly_return", "annual_return", "price_choice", "split_clock"):
        assert ui.splits_by_size(m)
    for m in ("index_level", "index_return", "us_index_level", "us_index_return", "us_level", "known_level",
              "known_return", "known_choice"):
        assert not ui.splits_by_size(m)
