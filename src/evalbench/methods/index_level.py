"""M14: market-level memory, the month-end close of a Nordic stock index. Index-level look-ahead is far larger than
stock-level (GPT-4o recalls S&P 500 levels within 1% and monthly direction 98% right inside its training window;
Lopez-Lira, Tang & Zhu 2025), and knowing the market's path contaminates any backtest with market exposure.
Four indices with long daily histories in Millistream: OMXS30, OMXSPI, OMXN40, OMXHPI (data/index_daily.parquet).

Score: closeness on a tighter scale than M5 (an index moves a few percent a month): 1 = exact, 0 = 25% or more off.
Fact type: price."""
from __future__ import annotations

import calendar
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import DATA_DIR
from ..items import Item, ItemScore
from ..textutil import classify_non_answer, extract_json
from . import price_recall
from .common import SYSTEM_MEMORY

NAME = "index_level"
PROMPT_VERSION = "p1"
P0 = 0.0
VARIANTS = ["main"]
MAX_TOKENS = 60
CLOSENESS_ZERO = 1.25
INDEX_FILE = DATA_DIR / "index_daily.parquet"
INDICES = {6485: "OMXS30", 6486: "OMXSPI", 746: "OMXN40", 768: "OMXHPI"}
NAMES = {"OMXS30": "OMX Stockholm 30", "OMXSPI": "OMX Stockholm All-Share", "OMXN40": "OMX Nordic 40",
         "OMXHPI": "OMX Helsinki All-Share"}
MONTHS = pd.period_range("2016-01", "2026-08", freq="M")
PILOT = 12                   # the micro-pilot block: OMXS30, 12 old months and 12 months after every release
MIN_DAYS = 5
FILTERS = ("one question per index and calendar month 2016-01..2026-08 with at least 5 trading days and a previous "
           "month; truth = the last close of the month; the first 24 questions are a micro-pilot block (OMXS30, 12 "
           "months 2016-2023 and 12 months 2025-09..2026-08), the rest in random order")


def is_pilot(item_id: str) -> bool:
    """One of the Nordic sets' first 24 questions (the micro-pilot block, il-/ir-0000..0023), left out of every
    confirmatory test (index plan §6.2). The US sets have no pilot block."""
    prefix, _, number = item_id.partition("-")
    return prefix in ("il", "ir") and number.isdigit() and int(number) < 2 * PILOT


def load_index(path: Path = INDEX_FILE) -> pd.DataFrame:
    daily = pd.read_parquet(path)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def build_items(daily: pd.DataFrame, seed: int, months=MONTHS, method: str = NAME, id_prefix: str = "il",
                names: dict | None = None, pilot_symbol: str | None = "OMXS30") -> list[Item]:
    """M14 and M15 share these questions (same index-months, same order); method names the set they belong to.
    names: symbol -> index name (default: the four Nordic indices); pilot_symbol: the index of the micro-pilot block
    put first (None: no block)."""
    names = names or NAMES
    rng = np.random.default_rng(seed)
    months = list(months)
    items = []
    for symbol in names:
        d = daily[daily["symbol"] == symbol].sort_values("date")
        if d.empty:
            continue
        d = d.assign(month=d["date"].dt.to_period("M"))
        ends = d.groupby("month").agg(level=("close", "last"), date=("date", "last"), days=("close", "size"))
        for month in months:
            if month not in ends.index or (month - 1) not in ends.index or ends.loc[month, "days"] < MIN_DAYS:
                continue
            level, prev = float(ends.loc[month, "level"]), float(ends.loc[month - 1, "level"])
            before_prev = ends.loc[month - 2, "level"] if (month - 2) in ends.index else None
            year_ago = ends.loc[month - 12, "level"] if (month - 12) in ends.index else None
            items.append(Item(
                item_id="", method=method, date=ends.loc[month, "date"].date().isoformat(), cluster=f"m{month}",
                payload={"symbol": symbol, "index_name": names[symbol], "month": str(month),
                         "month_name": f"{calendar.month_name[month.month]} {month.year}"},
                truth={"level": level, "prev_level": prev, "return_pct": (level / prev - 1) * 100,
                       "baseline_level": None if year_ago is None else float(year_ago),
                       "prev_return_pct": None if before_prev is None else (prev / float(before_prev) - 1) * 100},
                strata={"index": symbol}))
    pilot = []
    if pilot_symbol:
        is_old = [i for i, it in enumerate(items) if it.payload["symbol"] == pilot_symbol and it.payload["month"] <= "2023-12"]
        is_new = [i for i, it in enumerate(items) if it.payload["symbol"] == pilot_symbol and it.payload["month"] >= "2025-09"]
        pilot = [int(i) for i in rng.choice(is_old, PILOT, replace=False)] + \
                [int(i) for i in rng.choice(is_new, min(PILOT, len(is_new)), replace=False)]
        pilot = [pilot[int(i)] for i in rng.permutation(len(pilot))]
    rest = [i for i in rng.permutation(len(items)) if int(i) not in set(pilot)]
    ordered = [items[i] for i in pilot] + [items[int(i)] for i in rest]
    for k, it in enumerate(ordered):
        it.item_id = f"{id_prefix}-{k:04d}"
    return ordered


def render(item: Item, mode: str, variant: str = "main") -> list[dict]:
    p = item.payload
    question = (f"What was the closing level of the {p['index_name']} index ({p['symbol']}) on the last trading day "
                f"of {p['month_name']}? An approximate figure is fine.")
    if mode == "forced":
        instr = 'You must give your best numeric estimate, even if you are unsure. Respond only with JSON: {"level": <number>}'
    else:
        instr = 'If you do not know, answer null. Respond only with JSON: {"level": <number or null>}'
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{question}\n{instr}"}]


_LEVEL_KEY = re.compile(r'"level"\s*:\s*"?\s*([-+]?\d[\d.,]*(?: \d{3}[\d.,]*)*)', re.IGNORECASE)
# index names with digits, removed from an answer string before it is read ("OMXS30 was about 2200" -> 2200)
_INDEX_NAMES = re.compile(r"\b(?:OMX[A-Z]*\d+|S&P\s*500|Nasdaq[- ]?100|DJIA)\b", re.IGNORECASE)


def number_after_key(text: str, key: re.Pattern) -> str | None:
    """The number written right after the answer's JSON key in a reply whose JSON is broken ('{"level": 13,200}',
    Novita's '{"level":!!!!!3635}'); None when the key has no number after it (a reply cut off mid-JSON)."""
    m = key.search((text or "").replace("!", ""))
    return m.group(1) if m else None


def no_answer(text: str) -> dict:
    return {"value": None, "raw": None, "refused": True, "outcome": classify_non_answer(text)}


def json_value(value):
    """The answer field as written, with index names taken out of a string answer."""
    return _INDEX_NAMES.sub(" ", value) if isinstance(value, str) else value


def finite(parsed: dict) -> dict:
    """NaN or Infinity in the JSON is no answer (malformed)."""
    v = parsed.get("value")
    if v is not None and not math.isfinite(v):
        return {"value": None, "raw": None, "refused": True, "outcome": "malformed"}
    return parsed


def parse(text: str) -> dict:
    """The JSON answer only. A reply without one is no answer: a number elsewhere in the text is never read, since the
    index names (OMXS30, S&P 500, Nasdaq-100) and the model's own dates ("my cutoff is June 2024") contain numbers."""
    obj = extract_json(text)
    if obj is not None and "level" in obj:
        return finite(price_recall.parse(json.dumps({"price": json_value(obj["level"])})))
    raw = number_after_key(text, _LEVEL_KEY)
    return no_answer(text) if raw is None else finite(price_recall.parse(json.dumps({"price": raw})))


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    """Closeness to the month-end close: 1 - |ln(answer/truth)| / ln 1.25, floored at 0. A '2 214,53' style answer is
    read both ways (sv/en) and the closer reading counts, as in M1/M5."""
    if parsed["main"].get("value") is None:
        return ItemScore(0.0, refused=True)
    readings = price_recall._readings(parsed["main"])
    if not readings:
        return ItemScore(0.0, refused=False)
    rel, log_err = price_recall._distance(readings, item.truth["level"])
    return ItemScore(max(0.0, 1.0 - log_err / math.log(CLOSENESS_ZERO)), refused=False, extra={
        "log_error": log_err, "within_2pct": bool(rel <= 0.02), "within_5pct": bool(rel <= 0.05)})
