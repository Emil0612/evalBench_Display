import numpy as np
import pandas as pd
import pytest

from evalbench.data import Data

FILLER = ("The board believes the outlook for the coming year remains stable and the company will keep investing "
          "in growth across all of its markets while holding costs under firm control.")


def _article(k: int) -> str:
    sales, prev, ebit = 1000 + 37 * k, 900 + 11 * k, 150 + 7 * k
    p1 = (f"Company {k} reported net sales of SEK {sales} million for the period, compared with SEK {prev} "
          f"million a year earlier. {FILLER}")
    p2 = f"Operating profit amounted to SEK {ebit} million and the order intake grew during the quarter. {FILLER}"
    p3 = f"The chief executive said demand was strong in the Nordic region. {FILLER} {FILLER}"
    return "\n\n".join([p1, p2, p3])


@pytest.fixture(scope="session")
def data() -> Data:
    ins = list(range(1, 10))
    universe = pd.DataFrame({
        "insref": ins, "company_key": [i * 10 for i in ins], "primary_insref": ins, "is_primary": True,
        "n_listings": 1, "name": [f"Company {i} B" for i in ins], "symbol": [f"CO{i} B" for i in ins],
        "tradecurrency": "SEK", "instrumenttype": 4, "median_mcap_2015_2024": [i * 1e9 for i in ins],
        "price_days": 1500,
    })
    days = pd.bdate_range("2019-01-01", "2026-08-31")
    prices = pd.concat(
        [pd.DataFrame({"insref": i, "date": days, "price_type": 0, "close": 100 + i + np.arange(len(days)) * 0.01})
         for i in ins],
        ignore_index=True,
    )
    prices["adj_close"] = prices["close"] / 2
    n = 60
    lang = ["sv" if k % 2 else "en" for k in range(n)]
    news = pd.DataFrame({
        "newsid": [f"n{k}" for k in range(n)],
        "date": pd.date_range("2021-01-01", periods=n, freq="MS") + pd.Timedelta(days=14),
        "distributor": "Cision", "source_group": "press_release", "language": lang, "lang_group": lang,
        "newstype": 2, "headline": [f"Company {k} publishes interim report" for k in range(n)],
        "text": [_article(k) for k in range(n)], "insrefs": [[k % 9 + 1] for k in range(n)],
    })
    template = news.iloc[[0]].assign(newsid="tpl", headline="Company 0 share buyback programme")
    news = pd.concat([news, template], ignore_index=True)
    return Data(universe=universe, prices=prices, news=news)
