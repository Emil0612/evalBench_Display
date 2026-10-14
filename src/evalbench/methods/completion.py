"""M4: guided vs general instruction completion (Golchin & Surdeanu, ICLR 2024)."""
from __future__ import annotations

import re

import numpy as np

from ..data import Data, eligible_news, str_or_none
from ..items import Item, ItemScore
from ..textutil import classify_non_answer, number_recall, rouge_l_f1
from .common import SYSTEM_MEMORY, about, kind

NAME = "completion"
PROMPT_VERSION = "p1"
P0 = 0.0
VARIANTS = ["guided", "general"]
MAX_TOKENS = 250
PREFIX_CHARS, MIN_PREFIX = 250, 40
REF_CHARS, MIN_REST = 500, 400
FILTERS = ("same article filter as number_cloze; prefix = headline + first sentence(s) up to 250 chars; "
           "reference = the next ~500 chars; articles need >=400 chars after the prefix")
REFUSAL_RE = re.compile(
    r"^\s*(?:unknown\b|i (?:can(?:'|no)t|am unable|don't have|do not have)|i'm sorry|sorry|jag kan inte|tyvärr)",
    re.IGNORECASE,
)


def split_prefix(text: str) -> tuple[str, str]:
    body = text.strip()
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s)", body[: PREFIX_CHARS + 1]) if m.end() >= MIN_PREFIX]
    cut = ends[-1] if ends else body[:PREFIX_CHARS].rfind(" ")
    if cut <= 0:
        cut = PREFIX_CHARS
    return body[:cut].rstrip(), body[cut:].lstrip()


def reference_from(rest: str) -> str:
    if len(rest) <= REF_CHARS:
        return rest
    cut = rest[:REF_CHARS].rfind(" ")
    return rest[: cut if cut > 0 else REF_CHARS]


def build_items(data: Data, n: int, seed: int) -> list[Item]:
    rng = np.random.default_rng(seed)
    news = eligible_news(data.news, data.universe)
    items: list[Item] = []
    for i in rng.permutation(len(news)):
        r = news.iloc[int(i)]
        prefix, rest = split_prefix(r["text"])
        if len(rest) < MIN_REST:
            continue
        items.append(Item(
            item_id=f"cp-{len(items):04d}", method=NAME, date=r["date"].date().isoformat(), cluster=r["cluster"],
            payload={"headline": r["headline"], "prefix": prefix, "distributor": r["distributor"],
                     "company": str_or_none(r["company"]), "source_group": r["source_group"],
                     "language": r["lang_group"]},
            truth={"reference": reference_from(rest), "newsid": r["newsid"]},
            strata={"language": r["lang_group"], "source_group": r["source_group"]},
        ))
        if len(items) >= n:
            break
    return items


def render(item: Item, mode: str, variant: str) -> list[dict]:
    p = item.payload
    text = f"{p['headline']}\n\n{p['prefix']}"
    instr = "Reply with the continuation only (about 80 words)."
    if variant == "guided":
        head = (f"Below is the beginning of a {kind(p['source_group'])} published by {p['distributor']} on "
                f"{item.date}{about(p['company'])}. Continue the text exactly as it appeared in the original publication.")
        if mode == "abstain":
            instr += " If you do not know the original text, reply exactly: UNKNOWN"
        return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{head}\n\n{text}\n\n{instr}"}]
    return [{"role": "user", "content": f"Continue the following text.\n\n{text}\n\n{instr}"}]


def parse(text: str) -> dict:
    t = re.sub(r"^```\w*\n?|\n?```$", "", (text or "").strip()).strip()
    if not t:
        outcome = "malformed"
    elif t.upper().startswith("UNKNOWN"):
        outcome = "abstain"
    elif REFUSAL_RE.match(t):
        c = classify_non_answer(t)
        outcome = c if c != "malformed" else "refusal"
    else:
        outcome = "answer"
    return {"text": t, "refused": outcome != "answer", "outcome": outcome}


def strip_echo(output: str, prefix: str) -> str:
    """Drop a repeated prefix if the model echoed the given text before continuing."""
    tail = prefix[-40:]
    i = output.find(tail)
    if i != -1 and i < len(prefix) + 200:
        return output[i + len(tail):].lstrip()
    return output


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    ref, prefix = item.truth["reference"], item.payload["prefix"]

    def rouge(variant: str) -> float:
        p = parsed[variant]
        return 0.0 if p.get("refused", True) else rouge_l_f1(strip_echo(p.get("text", ""), prefix), ref)

    rg, rn = rouge("guided"), rouge("general")
    g = parsed["guided"]
    g_refused = g.get("refused", True)
    nr = None if g_refused else number_recall(strip_echo(g.get("text", ""), prefix), ref, item.payload["language"])
    return ItemScore(0.0 if g_refused else rg - rn, refused=g_refused,
                     extra={"rouge_guided": rg, "rouge_general": rn, "number_recall": nr})
