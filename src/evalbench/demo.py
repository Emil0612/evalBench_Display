"""Demo mode: run the whole pipeline without the licensed data and without any model costs.

Writes synthetic data to data/ (120 fictional companies with simulated prices, invented news, simulated index paths),
builds the test sets from it, and answers them with the free control models: an oracle that knows everything before
its cutoff, a random guesser and a blind baseline. The app then shows what a memorization signal looks like (the
oracle) and what no signal looks like (random, blind). Nothing here is real market data.

uv run python -m evalbench.demo          # then: uv run evalbench-ui
"""
from __future__ import annotations

import hashlib
import sys
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

from .data import DATA_DIR
from .items import TESTSETS

MARKER = "DEMO"                      # data/DEMO marks the folder as synthetic, so a rerun may replace it
START, END = "2014-01-01", "2026-08-31"
N_COMPANIES = 120
SEED = 20261014
CONTROLS = ["mock/oracle@2024-06-30", "mock/random@2024-06-30", "baseline/blind@2024-06-30"]
INDICES = {6485: ("OMXS30", "OMX Stockholm 30", "SEK", 2000.0), 6486: ("OMXSPI", "OMX Stockholm PI", "SEK", 900.0),
           746: ("OMXN40", "OMX Nordic 40", "EUR", 1500.0), 768: ("OMXHPI", "OMX Helsinki PI", "EUR", 9000.0)}

_FIRST = ["Norr", "Fjäll", "Ström", "Sjö", "Berg", "Lund", "Ek", "Gran", "Is", "Sol", "Vind", "Skog", "Älv", "Dal", "Ljus"]
_LAST = ["kraft", "tech", "data", "bygg", "medica", "logistik", "energi", "gruppen", "systems", "invest", "foods", "metall"]
_SUFFIX = {"SEK": "AB", "NOK": "ASA", "EUR": "Oyj", "DKK": "A/S"}
_EVENTS_EN = ["reports higher net sales for the {q} quarter", "signs a framework agreement with a Nordic customer",
              "opens a new production site in {city}", "appoints a new chief executive officer",
              "lowers its outlook for the full year", "completes the acquisition of a regional competitor",
              "launches a new product line for industrial customers", "reports a stronger operating margin",
              "wins an order worth SEK {n} million", "plans to hire {k} engineers in {city}",
              "raises its dividend to SEK {d} per share", "cuts {k} jobs at its plant in {city}",
              "receives regulatory approval for its new service", "starts a strategic review of its {city} unit"]
_EVENTS_SV = ["redovisar ökad omsättning för {q_sv} kvartalet", "tecknar ramavtal med nordisk kund",
              "öppnar ny produktionsanläggning i {city}", "utser ny verkställande direktör",
              "sänker prognosen för helåret", "slutför förvärvet av en regional konkurrent",
              "får order värd {n} miljoner kronor", "anställer {k} ingenjörer i {city}",
              "höjer utdelningen till {d} kronor per aktie", "varslar {k} anställda i {city}"]
_MACRO_EN = ["Central bank leaves its policy rate unchanged", "Inflation slows more than expected in {month_name}",
             "Nordic stocks fall as oil prices climb", "Krona strengthens after strong export figures"]
_CITIES = ["Umeå", "Tampere", "Aarhus", "Bergen", "Örebro", "Turku", "Luleå", "Odense"]
_Q = ["first", "second", "third", "fourth"]
_Q_SV = ["första", "andra", "tredje", "fjärde"]
_FILLER_EN = ("The company said demand from its main markets remained stable during the period. Management expects "
              "continued investment in production capacity and digital services. Order intake rose compared with the "
              "same period last year, while costs for raw materials and freight were broadly unchanged. ")
_FILLER_SV = ("Bolaget uppger att efterfrågan på de viktigaste marknaderna var stabil under perioden. Ledningen räknar "
              "med fortsatta investeringar i produktionskapacitet och digitala tjänster. Orderingången ökade jämfört "
              "med samma period föregående år, medan kostnaderna för råvaror och frakt var i stort sett oförändrade. ")


def _days() -> pd.DatetimeIndex:
    return pd.bdate_range(START, END)


def companies(rng: np.random.Generator) -> pd.DataFrame:
    names = sorted({a + b for a in _FIRST for b in _LAST})
    picked = rng.choice(names, N_COMPANIES, replace=False)
    currency = rng.choice(list(_SUFFIX), N_COMPANIES, p=[0.6, 0.15, 0.15, 0.1])
    shares = np.exp(rng.uniform(np.log(5e6), np.log(2e9), N_COMPANIES))
    insref = np.arange(100001, 100001 + N_COMPANIES)
    return pd.DataFrame({"insref": insref, "name": [f"{n} {_SUFFIX[c]}" for n, c in zip(picked, currency)],
                         "symbol": [n[:4].upper() for n in picked], "tradecurrency": currency, "shares": shares})


def prices(comp: pd.DataFrame, market: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Daily closes from a one-factor model (market move times beta plus own noise), one row per company and day."""
    days = _days()
    frames = []
    for row in comp.itertuples():
        beta, vol = rng.uniform(0.6, 1.4), rng.uniform(0.01, 0.03)
        r = beta * market + rng.normal(0, vol, len(days))
        adj = np.round(rng.uniform(20, 400) * np.exp(np.cumsum(r)), 2)
        close = adj.copy()
        if row.Index % 15 == 0:                        # a few splits: raw prices before them were higher
            at, factor = int(rng.integers(500, len(days) - 300)), float(rng.choice([2.0, 4.0, 10.0]))
            close[:at] = np.round(close[:at] * factor, 2)
        volume = rng.integers(1_000, 500_000, len(days))
        frames.append(pd.DataFrame({
            "insref": row.insref, "date": days.date, "price_type": 0, "open": close, "high": np.round(close * 1.01, 2),
            "low": np.round(close * 0.99, 2), "close": close, "vwap": close, "volume": volume,
            "turnover": close * volume, "numtrades": volume // 50, "mcap": close * row.shares,
            "adj_open": close, "adj_high": np.round(close * 1.01, 2), "adj_low": np.round(close * 0.99, 2),
            "adj_close": adj}))
    return pd.concat(frames, ignore_index=True)


def universe(comp: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    span = px[(px["date"] >= pd.Timestamp("2015-01-01").date()) & (px["date"] <= pd.Timestamp("2024-12-31").date())]
    med = span.groupby("insref")[["mcap", "turnover"]].median()
    days = px.groupby("insref")["date"].agg(["min", "max", "size"])
    u = comp.drop(columns="shares").assign(
        company_key=comp["insref"], primary_insref=comp["insref"], is_primary=True, n_listings=1,
        isin=[f"XX{i:010d}" for i in comp["insref"]], instrumenttype=4, instrumentsubtype=0, country="SE",
        marketplace=0, primarymarketplace=0, shareclass="", issuername=comp["name"],
        first_price=comp["insref"].map(days["min"]), last_price=comp["insref"].map(days["max"]),
        price_days=comp["insref"].map(days["size"]).astype("int64"),
        median_mcap_2015_2024=comp["insref"].map(med["mcap"]),
        median_turnover_2015_2024=comp["insref"].map(med["turnover"]), company_articles=0,
        first_article=pd.NaT, last_article=pd.NaT)
    return u


def index_daily(market: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    days = _days()
    frames = []
    for insref, (symbol, name, currency, level) in INDICES.items():
        close = np.round(level * np.exp(np.cumsum(market + rng.normal(0, 0.002, len(days)))), 2)
        frames.append(pd.DataFrame({"insref": insref, "symbol": symbol, "name": name, "currency": currency,
                                    "date": days.date, "close": close, "high": np.round(close * 1.005, 2),
                                    "low": np.round(close * 0.995, 2)}))
    return pd.concat(frames, ignore_index=True)


def news(comp: pd.DataFrame, rng: np.random.Generator, per_month: int = 60) -> pd.DataFrame:
    """Invented press releases and editorial items, Swedish and English, about the fictional companies (a few macro
    items name no company)."""
    rows = []
    for month in pd.period_range("2016-01", END[:7], freq="M"):
        for k in range(per_month):
            day = month.start_time + pd.Timedelta(days=int(rng.integers(0, month.days_in_month)))
            sv = rng.random() < 0.45
            q = (month.month - 1) // 3
            fill = {"q": _Q[q], "q_sv": _Q_SV[q], "city": rng.choice(_CITIES), "month_name": month.strftime("%B"),
                    "n": int(rng.integers(20, 900)), "k": int(rng.integers(12, 400)), "d": int(rng.integers(2, 25))}
            if not sv and rng.random() < 0.15:
                company, headline = None, rng.choice(_MACRO_EN).format(**fill)
            else:
                company = comp.iloc[int(rng.integers(0, len(comp)))]
                event = rng.choice(_EVENTS_SV if sv else _EVENTS_EN).format(**fill)
                headline = f"{company['name']} {event}"
            sales, ebit = int(rng.integers(120, 9800)), round(float(rng.uniform(1.5, 19.5)), 1)
            margin = f"{ebit:.1f}".replace(".", ",") if sv else f"{ebit:.1f}"     # Swedish decimal comma
            figures = (f"Nettoomsättningen uppgick till {sales} MSEK och rörelsemarginalen var {margin} procent. " if sv
                       else f"Net sales amounted to SEK {sales} million and the operating margin was {margin} percent. ")
            text = headline + ". " + figures + (_FILLER_SV if sv else _FILLER_EN) * int(rng.integers(3, 8))
            newsid = f"demo-{month}-{k:03d}"
            rows.append({"newsid": newsid, "date": day.date(), "time": dtime(8, 30),
                         "distributor": "Demo Wire", "source_group": "press_release" if company is not None else "editorial",
                         "language": "sv" if sv else "en", "newstype": int(rng.choice([0, 1, 2, 4])),
                         "headline": headline, "text_chars": len(text),
                         "text_md5": hashlib.md5(text.encode("utf-8")).hexdigest(), "same_text_count": 1,
                         "insrefs": [int(company["insref"])] if company is not None else [],
                         "raw_insrefs": [int(company["insref"])] if company is not None else [],
                         "year": month.year, "lang_group": "sv" if sv else "en", "text": text})
    return pd.DataFrame(rows)


def write_data(data_dir: Path = DATA_DIR, seed: int = SEED) -> None:
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0003, 0.011, len(_days()))
    comp = companies(rng)
    px = prices(comp, market, rng)
    data_dir.mkdir(parents=True, exist_ok=True)
    universe(comp, px).to_parquet(data_dir / "universe.parquet", index=False)
    px.to_parquet(data_dir / "prices_daily.parquet", index=False)
    articles = news(comp, rng)
    articles.to_parquet(data_dir / "news_sample.parquet", index=False)
    articles.drop(columns="text").to_parquet(data_dir / "news_index.parquet", index=False)   # metadata + headline
    index_daily(market, rng).to_parquet(data_dir / "index_daily.parquet", index=False)
    (data_dir / MARKER).write_text("Synthetic demo data written by `python -m evalbench.demo`. Not market data.\n",
                                   encoding="utf-8")


def is_demo(data_dir: Path = DATA_DIR) -> bool:
    return (data_dir / MARKER).exists()


def check_safe(data_dir: Path = DATA_DIR, testsets: Path = TESTSETS) -> str | None:
    """Why the demo must not run here (real data or real test sets present), or None."""
    if any(data_dir.glob("*.parquet")) and not is_demo(data_dir):
        return f"{data_dir} holds data that is not demo data; the demo will not overwrite it"
    if any(testsets.glob("*/*.jsonl")) and not is_demo(data_dir):
        return f"{testsets} holds built test sets; the demo will not mix with them"
    return None


def main(argv=None) -> None:
    reason = check_safe()
    if reason:
        sys.exit(reason)
    from . import build_testsets, runner, store
    from .items import load_current_testsets

    if is_demo():                                      # a rerun rebuilds the demo sets and answers from scratch
        for f in [*TESTSETS.glob("*/*"), *store.DB_PATH.parent.glob(store.DB_PATH.name + "*")]:
            f.unlink()
    print("writing synthetic data to data/ ...")
    write_data()
    print("building test sets ...")
    build_testsets.main(["--mock-paraphrases"])
    sets = load_current_testsets()
    calls = runner.plan(CONTROLS, list(sets), 200, "forced", sets)
    assert calls and not any(runner.is_paid(c.model) for c in calls), "the demo only uses free controls"
    print(f"answering {len(calls):,} questions with the free control models ...")
    runner.run(calls, store.connect(), {}, spend_cap=0.0)
    print("done: start the app with `uv run evalbench-ui`")


if __name__ == "__main__":
    main()
