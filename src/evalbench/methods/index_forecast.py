"""M19, the fake-date test (exploratory): each Nordic index question of M15, asked as a forecast made at the end of the
previous month: "Today is 31 March 2023 [...] how will the OMX Stockholm 30 change during April 2023?". An honest
forecaster cannot track single months, so its answers should not follow what really happened; answers that do, on
months a model trained on, are hindsight leaking through the instruction (Lopez-Lira, Tang & Zhu 2025; Eliseev &
Seleznev 2026, "fake date tests"). Same truth, parser and scoring as M15. Fact type: market."""
from __future__ import annotations

import calendar

import pandas as pd

from ..items import Item, ItemScore
from . import index_return

NAME = "index_forecast"
PROMPT_VERSION = "p1"
P0 = 0.5
VARIANTS = ["main"]
MAX_TOKENS = index_return.MAX_TOKENS
FILTERS = ("the index-months of M15 (index_return/v1), same order and truth; asked as a forecast dated the last "
           "calendar day of the previous month")


def build_items(base_items: list[Item]) -> list[Item]:
    """One forecast question per M15 question: as of the last day of the month before."""
    out = []
    for it in base_items:
        prev = pd.Period(it.payload["month"], "M") - 1
        as_of = prev.end_time.date()
        out.append(Item(
            item_id="if-" + it.item_id.split("-", 1)[1], method=NAME, date=it.date, cluster=it.cluster,
            payload={**it.payload, "as_of": as_of.isoformat(),
                     "as_of_long": f"{as_of.day} {calendar.month_name[as_of.month]} {as_of.year}"},
            truth=dict(it.truth), strata=dict(it.strata), fabricated=it.fabricated))
    return out


def render(item: Item, mode: str, variant: str = "main") -> list[dict]:
    p = item.payload
    system = (f"Today is {p['as_of_long']}. You are a financial analyst. You only know information that was public on "
              "or before today, and nothing about anything that happened after today.")
    question = (f"Forecast: by how many percent will the {p['index_name']} index ({p['symbol']}) change during "
                f"{p['month_name']}, from today's close to the last close of {p['month_name']}? Give a signed "
                "percentage: negative if you expect it to fall.")
    if mode == "forced":
        instr = ('You must give your best estimate, even if you are unsure. Respond only with JSON: '
                 '{"change_percent": <number>}')
    else:
        instr = 'If you cannot estimate it, answer null. Respond only with JSON: {"change_percent": <number or null>}'
    return [{"role": "system", "content": system}, {"role": "user", "content": f"{question}\n{instr}"}]


parse = index_return.parse


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    return index_return.item_score(item, parsed)
