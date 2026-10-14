"""M3: DE-COP, pick the verbatim passage among three paraphrases (Duarte et al. 2024).

Frozen at v7 (decided 2026-10-06): seven versions failed the after-release gate, because models pick the original by
the style of machine rewrites, not from memory (Gemini 2.5 Flash 44-74% after its release, chance 25%). The method stays
secondary; build_testsets refuses new paid versions unless --unfreeze is passed."""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

import numpy as np

from ..data import Data, eligible_news, str_or_none
from ..items import Item, ItemScore
from ..textutil import classify_non_answer, extract_json, is_boilerplate
from .common import SYSTEM_MEMORY, about, kind

NAME = "decop"
PROMPT_VERSION = "p1"
P0 = 0.25
VARIANTS = ["r0", "r1", "r2", "r3"]
LETTERS = ("A", "B", "C", "D")
MAX_TOKENS = 60  # room for a short preamble before the letter (Llama sometimes explains first)
MIN_WORDS, MAX_WORDS = 40, 80
LANG_NAMES = {"sv": "Swedish", "en": "English"}
FILTERS = ("same article filter as number_cloze; one plain-text paragraph of 40-80 words per article; "
           "3 paraphrases from a fixed paraphraser model (see manifest)")

# "B", "B.", "B)", "(B)", "b) The second", "B: Nokia", "Option B" -> B
JSON_LETTER_RE = re.compile(r"^\W*(?:(?:option|answer|alternativ|svar)\s*)?\(?([ABCD])(?![A-Za-z0-9])", re.IGNORECASE)

ParaphraseFn = Callable[[list[tuple[str, str]]], list[list[str] | None]]
_NUM = re.compile(r"\d+(?:[ \u00a0.,]\d+)*")
DASHES = "—–"                                   # em and en dash
QUOTES = "“”„«»\""               # typographic and plain double quotes
# every character whose presence alone could mark one option out: all four options must use the same set
SPECIALS = DASHES + QUOTES + "()[]:;%&/!?…'’*§+=@#"
MAX_LENGTH_RATIO = 1.15


def _company_words(company: str | None) -> list[str]:
    """The distinctive words of a company name ('Freemelt' from 'Freemelt Holding AB')."""
    return [w for w in re.findall(r"[^\W\d_][\w&-]{3,}", company or "")
            if w.lower() not in ("group", "holding", "publ", "class", "bank")]


def style_tells(passage: str, paraphrases: list[str], company: str | None = None) -> list[str]:
    """Ways a rewrite differs from the original in form rather than wording, which would let a model spot the
    original without having seen it (pilot C: Gemini picked it 78% of the time even after its cutoff). Empty when
    the set is usable: the same numbers written the same way, the same punctuation marks in both directions (pilot
    E: rewrites dropped the original's quotation marks), the company name kept, similar length."""
    nums = Counter(_NUM.findall(passage))
    names = [w for w in _company_words(company) if w in passage]
    reasons = []
    for k, p in enumerate(paraphrases, 1):
        if Counter(_NUM.findall(p)) != nums:
            reasons.append(f"rewrite {k}: numbers differ from the original (missing, changed or reformatted)")
        elif set(p) & set(DASHES) - set(passage):
            reasons.append(f"rewrite {k}: uses a dash the original does not")
        elif set(p) & set(QUOTES) - set(passage):
            reasons.append(f"rewrite {k}: uses quote marks the original does not")
        elif set(passage) & set(QUOTES) - set(p):
            reasons.append(f"rewrite {k}: drops the quote marks the original uses")
        elif set(p) & set(SPECIALS) != set(passage) & set(SPECIALS):
            added = "".join(sorted(set(p) & set(SPECIALS) - set(passage)))
            dropped = "".join(sorted(set(passage) & set(SPECIALS) - set(p)))
            reasons.append(f"rewrite {k}: punctuation differs from the original (adds {added or '-'}, "
                           f"drops {dropped or '-'})")
        elif any(w not in p for w in names):
            reasons.append(f"rewrite {k}: changes or drops the company name")
        elif not 1 / MAX_LENGTH_RATIO <= len(p) / len(passage) <= MAX_LENGTH_RATIO:
            reasons.append(f"rewrite {k}: length {len(p) / len(passage):.2f}x the original")
    return reasons


def pick_passage(text: str, rng: np.random.Generator) -> str | None:
    cands = []
    for para in re.split(r"\n\s*\n", text or ""):
        para = " ".join(para.split())
        if not para or para[0] in "|#*[-\\>!":
            continue
        # markdown syntax would let a model spot the original by formatting instead of wording
        if re.search(r"\*\*|__|\\|\||\]\(", para) or re.match(r"\d+\.\s", para) or is_boilerplate(para):
            continue
        words = para.split()
        if len(words) > MAX_WORDS:
            acc: list[str] = []
            for sent in re.split(r"(?<=[.!?])\s+", para):
                if len(acc) + len(sent.split()) > MAX_WORDS:
                    break
                acc.extend(sent.split())
            words = acc
        if len(words) >= MIN_WORDS:
            cands.append(" ".join(words))
    return cands[int(rng.integers(len(cands)))] if cands else None


def mock_paraphrases(batch: list[tuple[str, str]]) -> list[list[str] | None]:
    """Deterministic fake distractors, for pipeline tests only; they are NOT real paraphrases."""
    out = []
    for passage, _lang in batch:
        w = passage.split()
        out.append([" ".join(w[1:] + w[:1]), " ".join(reversed(w)), " ".join(w[::2] + w[1::2])])
    return out


def pick_articles(data: Data, n: int, seed: int, exclude: set | None = None) -> list[tuple]:
    """(article row, verbatim passage) for up to n eligible articles, deterministic for a seed; articles whose newsid
    is in exclude are skipped (fresh passages for a confirmatory set)."""
    rng = np.random.default_rng(seed)
    news = eligible_news(data.news, data.universe)
    if exclude:
        news = news[~news["newsid"].isin(exclude)]
    picked = []
    for i in rng.permutation(len(news)):
        r = news.iloc[int(i)]
        passage = pick_passage(r["text"], rng)
        if passage:
            picked.append((r, passage))
        if len(picked) >= n:
            break
    return picked


def build_items(data: Data, n: int, seed: int, paraphrase_fn: ParaphraseFn, exclude: set | None = None) -> list[Item]:
    picked = pick_articles(data, n, seed, exclude)
    paraphrases = paraphrase_fn([(p, r["lang_group"]) for r, p in picked])
    items: list[Item] = []
    for (r, passage), ps in zip(picked, paraphrases):
        if not ps or len(ps) != 3 or len({passage, *ps}) != 4 or style_tells(passage, ps, str_or_none(r["company"])):
            continue
        items.append(Item(
            item_id=f"dc-{len(items):04d}", method=NAME, date=r["date"].date().isoformat(), cluster=r["cluster"],
            payload={"distributor": r["distributor"], "company": str_or_none(r["company"]),
                     "language": r["lang_group"], "source_group": r["source_group"]},
            truth={"passage": passage, "paraphrases": list(ps), "newsid": r["newsid"]},
            strata={"language": r["lang_group"], "source_group": r["source_group"]},
        ))
    return items


def options(item: Item, variant: str) -> list[str]:
    opts = list(item.truth["paraphrases"])
    opts.insert(int(variant[1:]), item.truth["passage"])
    return opts


def correct_letter(variant: str) -> str:
    return LETTERS[int(variant[1:])]


def render(item: Item, mode: str, variant: str) -> list[dict]:
    p = item.payload
    head = (f"Below are four passages. Exactly one of them is the verbatim original text from a "
            f"{kind(p['source_group'])} published on {item.date} by {p['distributor']}{about(p['company'])}. "
            "The other three are paraphrases. Which option is the original?")
    body = "\n\n".join(f"{letter}: {text}" for letter, text in zip(LETTERS, options(item, variant)))
    if mode == "forced":
        instr = 'Respond only with JSON: {"answer": "A" | "B" | "C" | "D"}'
    else:
        instr = 'If you cannot tell, answer null. Respond only with JSON: {"answer": "A" | "B" | "C" | "D" | null}'
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{head}\n\n{body}\n\n{instr}"}]


def parse(text: str) -> dict:
    obj = extract_json(text)
    if obj is not None and "answer" in obj:
        a = obj["answer"]
        if a is None:
            return {"letter": None, "refused": True, "outcome": "abstain"}
        m = JSON_LETTER_RE.match(str(a))
        if m:
            return {"letter": m.group(1).upper(), "refused": False, "outcome": "answer"}
        return {"letter": None, "refused": True, "outcome": classify_non_answer(str(a))}
    stated = re.search(r"(?:answer|svar|option|alternativ)\s*(?:is|är|:)?\s*\(?([ABCD])\b", text or "", re.IGNORECASE)
    letters = re.findall(r"\b([ABCD])\b", text or "")
    letter = stated.group(1).upper() if stated else (letters[-1] if letters else None)
    if letter is None:
        return {"letter": None, "refused": True, "outcome": classify_non_answer(text)}
    return {"letter": letter, "refused": False, "outcome": "answer"}


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    k = sum(parsed[v].get("letter") == correct_letter(v) for v in VARIANTS)
    refused = all(parsed[v].get("refused", True) for v in VARIANTS)
    return ItemScore(k / len(VARIANTS), refused=refused, extra={"n_correct": k, "known": k >= 3})
