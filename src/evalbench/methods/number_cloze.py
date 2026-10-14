"""M2: masked-number cloze on news text (TS-Guessing, Deng et al. 2024)."""
from __future__ import annotations

import re

import numpy as np

from ..data import Data, eligible_news, str_or_none
from ..items import Item, ItemScore
from ..textutil import (answer_number, classify_non_answer, extract_json, extract_numbers, has_unit, is_year,
                        numbers_equal, parse_number, significant_digits)
from .common import SYSTEM_MEMORY, about

NAME = "number_cloze"
PROMPT_VERSION = "p1"
P0 = 0.0
VARIANTS = ["main"]
MAX_TOKENS = 40
CONTEXT_BEFORE, CONTEXT_AFTER = 1000, 500
FILTERS = ("sv/en press releases + editorial, newstype 1/2/4, 500-6000 chars, templated headlines excluded; "
           "masked number has >=3 significant digits, is not a year, not in the headline, unique in the text, "
           "and sits next to a currency/percent/magnitude word; since v2 also not derivable from two other numbers in "
           "the shown context (sum, difference, percent change, share, percent of)")


def _line_of(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start: end if end != -1 else len(text)]


def _decimals(s: str, lang: str) -> int:
    """Digits after the decimal separator, as the number is written in the article's language."""
    sep = "," if lang == "sv" else "."
    tail = "".join(ch for ch in s if not ch.isspace())
    head, found, frac = tail.rpartition(sep)
    return len(frac) if found and frac.isdigit() and head else 0


def inferable(value: float, decimals: int, others: list[float]) -> bool:
    """Whether a number follows from two other numbers in the text, to the precision it is written with: a sum or
    difference, a percent change, a share in percent, or a percent of. Pilot C: models got 13-23% of masked
    numbers right even after their cutoff, largely numbers like these."""
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    vals = [o for o in dict.fromkeys(others) if o]
    for i, a in enumerate(vals):
        for b in vals[i + 1:]:
            derived = (a + b, abs(a - b), (a / b - 1) * 100, (b / a - 1) * 100, a / b * 100, b / a * 100, a * b / 100)
            if any(abs(d - value) <= tol for d in derived):
                return True
    return False


def find_mask_candidates(text: str, headline: str, lang: str = "en") -> list[re.Match]:
    out = []
    numbers = extract_numbers(text)
    for m in numbers:
        s = m.group()
        if significant_digits(s) < 3 or is_year(s) or s in (headline or "") or text.count(s) != 1:
            continue
        # the tail of a longer number ("-1 730 288" -> "730 288") or a table cell (often derivable from its row)
        if re.search(r"\d[ \u00a0\u202f]?$", text[max(0, m.start() - 2):m.start()]) or "|" in _line_of(text, m.start()):
            continue
        if not has_unit(text, m.start(), m.end()):
            continue
        lo, hi = m.start() - CONTEXT_BEFORE, m.end() + CONTEXT_AFTER
        others = [parse_number(o.group(), lang) for o in numbers
                  if o is not m and lo <= o.start() and o.end() <= hi and not is_year(o.group())]
        value = parse_number(s, lang)
        if value is not None and inferable(value, _decimals(s, lang), [v for v in others if v is not None]):
            continue
        out.append(m)
    return out


def build_items(data: Data, n: int, seed: int) -> list[Item]:
    rng = np.random.default_rng(seed)
    news = eligible_news(data.news, data.universe)
    items: list[Item] = []
    for i in rng.permutation(len(news)):
        r = news.iloc[int(i)]
        cands = find_mask_candidates(r["text"], r["headline"], lang=r["lang_group"])
        if not cands:
            continue
        m = cands[int(rng.integers(len(cands)))]
        text = r["text"]
        start, end = max(0, m.start() - CONTEXT_BEFORE), min(len(text), m.end() + CONTEXT_AFTER)
        items.append(Item(
            item_id=f"nc-{len(items):04d}", method=NAME, date=r["date"].date().isoformat(), cluster=r["cluster"],
            payload={"headline": r["headline"], "distributor": r["distributor"], "company": str_or_none(r["company"]),
                     "language": r["lang_group"], "context": text[start:m.start()] + "[MASK]" + text[m.end():end],
                     "cut_start": start > 0, "cut_end": end < len(text)},
            truth={"number": m.group(), "original": text[start:end], "newsid": r["newsid"]},
            strata={"language": r["lang_group"], "source_group": r["source_group"]},
        ))
        if len(items) >= n:
            break
    return items


def render(item: Item, mode: str, variant: str = "main") -> list[dict]:
    p = item.payload
    context = ("…" if p["cut_start"] else "") + p["context"] + ("…" if p["cut_end"] else "")
    head = (f"The excerpt below is from a news text published on {item.date} by {p['distributor']}{about(p['company'])}. "
            "One number in the text has been replaced by [MASK]. What exact number was originally there?")
    if mode == "forced":
        instr = 'You must give your best guess. Respond only with JSON: {"answer": "<the number, written as in the text>"}'
    else:
        instr = ('If you do not know, answer null. Respond only with JSON: '
                 '{"answer": "<the number, written as in the text>" or null}')
    user = f"{head}\n\nHeadline: {p['headline']}\n\nText:\n{context}\n\n{instr}"
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": user}]


def parse(text: str) -> dict:
    obj = extract_json(text)
    if obj is not None and "answer" in obj:
        a = obj["answer"]
        if a is None or str(a).strip() == "":
            return {"answer": None, "refused": True, "outcome": "abstain"}
        if not any(ch.isdigit() for ch in str(a)):
            return {"answer": None, "refused": True, "outcome": classify_non_answer(str(a))}
        return {"answer": str(a).strip(), "refused": False, "outcome": "answer"}
    m = answer_number(text or "")
    kind = classify_non_answer(text) if m and is_year(m.group()) else "answer"
    if m and kind in ("answer", "malformed"):
        return {"answer": m.group(), "refused": False, "outcome": "answer"}
    return {"answer": None, "refused": True, "outcome": kind if m else classify_non_answer(text)}


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    a = parsed["main"]
    if a.get("refused", True) or not a.get("answer"):
        return ItemScore(0.0, refused=True)
    return ItemScore(float(numbers_equal(a["answer"], item.truth["number"], item.payload["language"])), refused=False)
