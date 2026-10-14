"""Fetch daily closes of famous US shares for M10 (us_level), the private positive-control check.

PRIVATE: Yahoo Finance data is for personal use only. The file goes to
data/private/ (gitignored); nothing built from it may be published or pushed.

    uv run --with yfinance python -m evalbench.fetch_us_prices

yfinance is not a project dependency on purpose; --with installs it for this one command. Free, no key."""
from __future__ import annotations

import pandas as pd

from .methods.us_level import US_PRICES

# widely followed large caps; ones that split since 2020 are listed too but dropped by the M10 builder
TICKERS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "META": "Meta Platforms", "JPM": "JPMorgan Chase", "KO": "Coca-Cola",
    "V": "Visa", "MA": "Mastercard", "JNJ": "Johnson & Johnson", "PG": "Procter & Gamble", "XOM": "Exxon Mobil",
    "MCD": "McDonald's", "BRK-B": "Berkshire Hathaway B", "HD": "Home Depot", "UNH": "UnitedHealth Group",
    "DIS": "Walt Disney", "NKE": "Nike", "PFE": "Pfizer", "PEP": "PepsiCo",
    "NVDA": "Nvidia", "AMZN": "Amazon", "GOOGL": "Alphabet A", "TSLA": "Tesla", "WMT": "Walmart",
}


def main() -> None:
    import yfinance as yf

    raw = yf.download(list(TICKERS), start="2019-01-01", end="2026-09-01", auto_adjust=False, actions=True,
                      progress=False, group_by="ticker")
    rows = []
    for symbol, name in TICKERS.items():
        if symbol not in raw.columns.get_level_values(0):
            print(f"{symbol}: no data")
            continue
        d = raw[symbol].dropna(subset=["Close"])
        # 'Close' is split-adjusted (not dividend-adjusted); for shares without a split in the window it is the
        # price as traded, which is why the builder drops every share that split
        rows.append(pd.DataFrame({"symbol": symbol, "name": name, "date": d.index.tz_localize(None),
                                  "close": d["Close"].to_numpy(), "split": d["Stock Splits"].fillna(0).to_numpy()}))
        print(f"{symbol}: {len(d)} days, {int((d['Stock Splits'].fillna(0) > 0).sum())} split(s)")
    out = pd.concat(rows, ignore_index=True)
    US_PRICES.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(US_PRICES)
    print(f"wrote {len(out):,} rows -> {US_PRICES} (private, gitignored)")


if __name__ == "__main__":
    main()
