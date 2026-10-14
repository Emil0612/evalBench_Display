"""Local stand-in models used as controls.

mock/oracle@<cutoff>  answers correctly for items dated before <cutoff>, plausibly-wrong after (positive control)
mock/random@<cutoff>  plausible random answers everywhere (negative control)
baseline/blind@<cutoff> model-free heuristics from the item itself (blind baseline)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from datetime import date

import numpy as np
import pandas as pd

from ..items import Item
from ..methods import (annual_return, completion, decop, headline_date, index_forecast, index_level, index_return, known_choice,
                       known_choice_prob, path_date,
                       known_level, known_return, monthly_return, number_cloze, price_choice, price_choice_prob,
                       price_level, price_recall, split_clock, us_index_level, us_index_return, us_level)
from ..textutil import extract_numbers, is_year, significant_digits

KINDS = ("mock/oracle", "mock/random", "baseline/blind")
GENERIC = "The company said it expects continued growth and will provide further information in due course."


def is_local(model_id: str) -> bool:
    return model_id.split("@")[0] in KINDS


def _rng(*parts: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.md5("|".join(parts).encode()).hexdigest()[:8], 16))


def local_answer(model_id: str, method: str, item: Item, variant: str, mode: str, sample: int = 0) -> str:
    kind, cutoff = model_id.split("@")
    rng = _rng(model_id, item.item_id, variant, *([str(sample)] if sample else []))
    knows = kind == "mock/oracle" and not item.fabricated and date.fromisoformat(item.date) < date.fromisoformat(cutoff)
    return _ANSWER[method](item, variant, kind, knows, rng)


def _price(item, variant, kind, knows, rng) -> str:
    t = item.truth
    if kind == "baseline/blind":
        v = t["baseline_close"]
    elif item.fabricated:
        v = None if kind == "mock/oracle" else float(rng.uniform(10, 500))
    elif knows:
        v = t["close"]
    else:
        v = t["close"] * float(rng.uniform(0.6, 1.4))
    return json.dumps({"price": None if v is None else round(v, 4)})


def _cloze(item, variant, kind, knows, rng) -> str:
    truth = item.truth["number"]
    if kind == "baseline/blind":
        nums = [m.group() for m in extract_numbers(item.payload["context"])
                if significant_digits(m.group()) >= 3 and not is_year(m.group())]
        ans = Counter(nums).most_common(1)[0][0] if nums else "0"
    elif knows:
        ans = truth
    else:
        digits = max(1, len(re.sub(r"\D", "", truth)))
        ans = str(int(rng.integers(10 ** (digits - 1), 10 ** digits)))
    return json.dumps({"answer": ans}, ensure_ascii=False)


def _decop(item, variant, kind, knows, rng) -> str:
    if kind == "baseline/blind":
        opts = decop.options(item, variant)
        letter = decop.LETTERS[max(range(4), key=lambda i: len(opts[i]))]
    elif knows:
        letter = decop.correct_letter(variant)
    else:
        letter = decop.LETTERS[int(rng.integers(4))]
    return json.dumps({"answer": letter})


def _choice(item, variant, kind, knows, rng) -> str:
    if kind == "baseline/blind" and item.truth.get("baseline_level"):
        # persistence: the option closest to the same month a year earlier
        base = item.truth["baseline_level"]
        vals = price_choice.option_values(item, variant)
        letter = price_choice.LETTERS[min(range(4), key=lambda i: abs(math.log(vals[i] / base)))]
    elif knows and kind != "baseline/blind":
        letter = price_choice.correct_letter(variant)
    else:
        letter = price_choice.LETTERS[int(rng.integers(4))]
    return json.dumps({"answer": letter})


def _index_level(item, variant, kind, knows, rng) -> str:
    t = item.truth
    if kind == "baseline/blind":
        v = t["baseline_level"]               # the same month a year earlier
    elif knows:
        v = t["level"]
    else:
        v = t["level"] * float(rng.uniform(0.7, 1.3))
    return json.dumps({"level": None if v is None else round(v, 4)})


def _index_return(item, variant, kind, knows, rng) -> str:
    t = item.truth
    if kind == "baseline/blind":
        v = t["prev_return_pct"]              # momentum: last month's change again
    elif knows:
        v = t["return_pct"]
    else:
        v = float(rng.normal(0, 4))
    return json.dumps({"change_percent": None if v is None else round(v, 4)})


def _headline(item, variant, kind, knows, rng) -> str:
    if kind == "baseline/blind":
        m = pd.Period(item.truth["blind_month"], "M")         # the model-free dating baseline
    elif knows:
        m = pd.Period(item.truth["month"], "M")
    else:
        m = headline_date.MONTHS[int(rng.integers(len(headline_date.MONTHS)))]
    return json.dumps({"year": m.year, "month": m.month})


def _completion(item, variant, kind, knows, rng) -> str:
    return item.truth["reference"] if knows and variant == "guided" else GENERIC


def _level(item, variant, kind, knows, rng) -> str:
    t = item.truth
    if kind == "baseline/blind":
        v = t["baseline_level"]               # the same month a year earlier
    elif item.fabricated:
        v = None if kind == "mock/oracle" else float(rng.uniform(10, 500))
    elif knows:
        v = t["level"]
    else:
        v = t["level"] * float(rng.uniform(0.6, 1.4))
    return json.dumps({"price": None if v is None else round(v, 4)})


def _return(item, variant, kind, knows, rng) -> str:
    t = item.truth
    if kind == "baseline/blind":
        v = t["baseline_return_pct"]          # momentum: last year's return
    elif item.fabricated:
        v = None if kind == "mock/oracle" else float(rng.uniform(-40, 40))
    elif knows:
        v = t["return_pct"]
    else:
        v = float(rng.uniform(-50, 50))
    return json.dumps({"change_percent": None if v is None else round(v, 4)})


def _month_return(item, variant, kind, knows, rng) -> str:
    t = item.truth
    if kind == "baseline/blind":
        v = t["baseline_return_pct"]          # momentum: last month's return
    elif item.fabricated:
        v = None if kind == "mock/oracle" else float(rng.uniform(-15, 15))
    elif knows:
        v = t["return_pct"]
    else:
        v = float(rng.uniform(-15, 15))
    return json.dumps({"change_percent": None if v is None else round(v, 4)})


def _split(item, variant, kind, knows, rng) -> str:
    t = item.truth
    if kind == "baseline/blind":
        v = t["baseline_level"]               # a year earlier, as quoted then: the unadjusted price
    elif knows:
        v = t["adj_level"]                    # has seen data from after the split: the rewritten price
    elif kind == "mock/oracle":
        v = t["level"]                        # knows the period, not the later split
    else:
        v = (t["level"] if rng.random() < 0.5 else t["adj_level"]) * float(rng.uniform(0.6, 1.4))
    return json.dumps({"price": None if v is None else round(v, 4)})


_ANSWER = {
    price_recall.NAME: _price,
    number_cloze.NAME: _cloze,
    decop.NAME: _decop,
    completion.NAME: _completion,
    price_level.NAME: _level,
    annual_return.NAME: _return,
    known_return.NAME: _return,
    known_level.NAME: _level,
    us_level.NAME: _level,
    monthly_return.NAME: _month_return,
    split_clock.NAME: _split,
    price_choice.NAME: _choice,
    known_choice.NAME: _choice,
    index_level.NAME: _index_level,
    index_return.NAME: _index_return,
    us_index_level.NAME: _index_level,
    us_index_return.NAME: _index_return,
    price_choice_prob.NAME: _choice,
    known_choice_prob.NAME: _choice,
    headline_date.NAME: _headline,
    index_forecast.NAME: _index_return,
    path_date.NAME: _headline,
}
