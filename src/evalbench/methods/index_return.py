"""M15: market-level memory, the monthly change of a Nordic stock index (the questions of M14, asked as a return).
Scored on the direction (chance 0.5; the measured baseline decides); an answer of exactly 0 gives no direction and
scores 0.5, as on M6/M11. Fact type: return."""
from __future__ import annotations

import json
import re

from ..items import Item, ItemScore
from ..textutil import extract_json
from . import annual_return
from . import index_level
from .common import SYSTEM_MEMORY

NAME = "index_return"
PROMPT_VERSION = "p1"
P0 = 0.5
VARIANTS = ["main"]
MAX_TOKENS = 60
FILTERS = "the index-months of M14 (index_level), same order; truth = last close vs the previous month's last close"

_CHANGE_KEY = re.compile(r'"change_?percent"\s*:\s*"?\s*([-+−–]?\s?\d[\d.,]*)', re.IGNORECASE)


def parse(text: str) -> dict:
    """The JSON answer only, as M14: a number elsewhere in a refusal ("The S&P 500 index for November 2024 ...") or in
    a reply cut off before its JSON ("closed at 15,257.76 on July 31") is never read as the change."""
    obj = extract_json(text)
    if obj is not None and "change_percent" in obj:
        value = index_level.json_value(obj["change_percent"])
        return index_level.finite(annual_return.parse(json.dumps({"change_percent": value})))
    raw = index_level.number_after_key(text, _CHANGE_KEY)
    if raw is None:
        return index_level.no_answer(text)
    return index_level.finite(annual_return.parse(json.dumps({"change_percent": raw})))


def build_items(daily, seed: int) -> list[Item]:
    return index_level.build_items(daily, seed, method=NAME, id_prefix="ir")


def render(item: Item, mode: str, variant: str = "main") -> list[dict]:
    p = item.payload
    question = (f"By how many percent did the {p['index_name']} index ({p['symbol']}) change during {p['month_name']}, "
                f"from the last close of the previous month to the last close of {p['month_name']}? Give a signed "
                "percentage: negative if it fell.")
    if mode == "forced":
        instr = ('You must give your best estimate, even if you are unsure. Respond only with JSON: '
                 '{"change_percent": <number>}')
    else:
        instr = 'If you do not know, answer null. Respond only with JSON: {"change_percent": <number or null>}'
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{question}\n{instr}"}]


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    value = parsed["main"].get("value")
    if value is None:
        return ItemScore(0.0, refused=True)
    t = item.truth["return_pct"]
    no_direction = value == 0 and t != 0
    hit = (value > 0) == (t > 0)
    err = abs(value - t)
    return ItemScore(0.5 if no_direction else float(hit), refused=False, extra={
        "abs_err_pp": err, "within_2pp": bool(err <= 2), "no_direction": no_direction,
        "answer": float(value), "truth": float(t)})   # for the registered correlation test (analysis.registered_index_cell)
