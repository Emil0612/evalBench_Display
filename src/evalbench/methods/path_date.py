"""M20, the masking test (exploratory): the monthly percentage changes of an unnamed Nordic stock index over 12
consecutive months, oldest first, with no dates, no index name and no levels: in which year and month does the period
end? A model that can place such a path in time can recover the dates a backtest tried to hide, and with them its
hindsight (Lopez-Lira, Tang & Zhu 2025: "masking fails as LLMs reconstruct entities and dates from minimal context").
Scored by months off as M18; the analysis (mask_date) uses hits within a month beyond the model's own answer habits.
Fact type: market."""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from ..items import Item, ItemScore
from . import headline_date
from .common import SYSTEM_MEMORY

NAME = "path_date"
PROMPT_VERSION = "p1"
P0 = 0.0
VARIANTS = ["main"]
MAX_TOKENS = headline_date.MAX_TOKENS
LENGTH = 12
BLIND_MONTH = "2021-10"          # the blind control names one fixed month: a guesser with a habit and no memory
FILTERS = ("every 12-month window of consecutive months of each index in index_return/v1 (the micro-pilot block "
           "included), changes rounded to 0.1 point; truth = the window's last month")


def build_items(base_items: list[Item], length: int = LENGTH) -> list[Item]:
    """One question per index and window of `length` consecutive months (a gap breaks the window)."""
    by_symbol = defaultdict(dict)
    for it in base_items:
        by_symbol[it.payload["symbol"]][pd.Period(it.payload["month"], "M")] = it
    out = []
    for symbol in sorted(by_symbol):
        months = by_symbol[symbol]
        for end in sorted(months):
            window = [end - k for k in range(length - 1, -1, -1)]
            if not all(m in months for m in window):
                continue
            last = months[end]
            out.append(Item(
                item_id=f"pd-{len(out):04d}", method=NAME, date=last.date, cluster=f"m{end}",
                payload={"changes": [round(float(months[m].truth["return_pct"]), 1) for m in window],
                         "length": length},
                truth={"month": str(end), "first_month": str(window[0]), "symbol": symbol,
                       "blind_month": BLIND_MONTH},
                strata={"index": symbol}))
    return out


def render(item: Item, mode: str, variant: str = "main") -> list[dict]:
    changes = ", ".join(f"{c:+.1f}%" for c in item.payload["changes"])
    question = (f"Below are the monthly percentage changes of a Nordic stock index over {item.payload['length']} "
                "consecutive calendar months, oldest first. The index is not named and the dates are removed. In which "
                f"year and month does this period end (the month of the last change)?\n\nChanges: {changes}")
    if mode == "forced":
        instr = ('You must give your best estimate, even if you are unsure. Respond only with JSON: '
                 '{"year": <yyyy>, "month": <1-12>}')
    else:
        instr = ('If you do not know, answer null. Respond only with JSON: '
                 '{"year": <yyyy or null>, "month": <1-12 or null>}')
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{question}\n{instr}"}]


parse = headline_date.parse


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    value = parsed["main"].get("value")
    if value is None:
        return ItemScore(0.0, refused=True)
    err = headline_date.month_error(value, item.truth["month"])
    return ItemScore(max(0.0, 1.0 - abs(err) / 12), refused=False, extra={
        "err_months": err, "exact": err == 0, "within_1": abs(err) <= 1,
        "same_year": value[:4] == item.truth["month"][:4], "answer": value})
