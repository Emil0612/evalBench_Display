"""Loading the extract in data/ and the shared news eligibility filter."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .items import ROOT
from .textutil import is_templated

DATA_DIR = ROOT / "data"
DATE_MIN, DATE_MAX = "2021-01-01", "2026-08-31"
NEWS_LANGS = ("sv", "en")
NEWS_TYPES = (1, 2, 4)
NEWS_SOURCES = ("press_release", "editorial")


@dataclass
class Data:
    universe: pd.DataFrame
    prices: pd.DataFrame
    news: pd.DataFrame


def str_or_none(v) -> str | None:
    return v if isinstance(v, str) and v else None


def load_data(data_dir: Path = DATA_DIR) -> Data:
    universe = pd.read_parquet(data_dir / "universe.parquet")
    prices = pd.read_parquet(data_dir / "prices_daily.parquet",
                             columns=["insref", "date", "price_type", "close", "adj_close"])
    prices = prices[prices["close"].notna() & (prices["price_type"].fillna(0) == 0)].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    news = pd.read_parquet(data_dir / "news_sample.parquet")
    news["date"] = pd.to_datetime(news["date"])
    return Data(universe=universe, prices=prices, news=news)


def eligible_news(news: pd.DataFrame, universe: pd.DataFrame, min_chars: int = 500, max_chars: int = 6000) -> pd.DataFrame:
    """sv/en press releases + editorial in the date window, non-templated; adds company name and cluster."""
    n = news[
        news["lang_group"].isin(NEWS_LANGS)
        & news["newstype"].isin(NEWS_TYPES)
        & news["source_group"].isin(NEWS_SOURCES)
        & news["date"].between(DATE_MIN, DATE_MAX)
        & news["text"].str.len().between(min_chars, max_chars)
    ]
    n = n[~n["headline"].map(is_templated)].copy()
    names = dict(zip(universe["insref"].astype(int), universe["name"]))
    keys = dict(zip(universe["insref"].astype(int), universe["company_key"].astype(int)))

    def first_known(refs):
        if refs is None:
            return None
        return next((int(r) for r in refs if int(r) in names), None)

    first = [first_known(r) for r in n["insrefs"]]
    n["company"] = [str_or_none(names.get(i)) if i is not None else None for i in first]
    n["cluster"] = [f"c{keys[i]}" if i is not None else f"n{nid}" for i, nid in zip(first, n["newsid"])]
    return n.reset_index(drop=True)


def company_size_map(universe: pd.DataFrame) -> dict[str, str]:
    """Bootstrap cluster id ("c<company_key>") -> market-cap tercile, as used for the price items.
    Big companies' news is reproduced far more widely online, so size is a proxy for training exposure."""
    u = universe[universe["is_primary"] & (universe["instrumenttype"] == 4)
                 & universe["median_mcap_2015_2024"].notna() & (universe["price_days"] >= 500)]
    sizes = pd.qcut(u["median_mcap_2015_2024"], 3, labels=["small", "mid", "large"]).astype(str)
    return {f"c{int(k)}": s for k, s in zip(u["company_key"], sizes)}
