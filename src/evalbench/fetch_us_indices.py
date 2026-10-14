"""Fetch daily closes of four US stock indices for M16/M17 (us_index_level, us_index_return), the private reference
for the Nordic market-level test.

PRIVATE: Yahoo Finance data is for personal use only. The file goes to data/private/ (gitignored); nothing built from
it may be published or pushed.

    uv run --with yfinance python -m evalbench.fetch_us_indices

yfinance is not a project dependency on purpose; --with installs it for this one command. Free, no key."""
from __future__ import annotations

import pandas as pd

from .methods.us_index_level import NAMES, US_INDEX_FILE, YAHOO


def main() -> None:
    import yfinance as yf

    rows = []
    for symbol, ticker in YAHOO.items():
        d = yf.download(ticker, start="2014-01-01", end="2026-09-17", auto_adjust=False, progress=False)
        if d.empty:
            print(f"{symbol} ({ticker}): no data")
            continue
        close = d["Close"].squeeze("columns") if isinstance(d["Close"], pd.DataFrame) else d["Close"]
        rows.append(pd.DataFrame({"symbol": symbol, "name": NAMES[symbol], "currency": "USD",
                                  "date": close.index.tz_localize(None), "close": close.to_numpy(float)}))
        print(f"{symbol} ({ticker}): {len(close)} days, {close.index.min().date()} .. {close.index.max().date()}")
    US_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(US_INDEX_FILE, index=False)
    print(f"wrote {US_INDEX_FILE}")


if __name__ == "__main__":
    main()
