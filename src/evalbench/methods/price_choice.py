"""M12: which price fits this month? The M5 question as recognition instead of recall: four options, one the share's
real average price that month, three the same share's real average price in other months. Models that will not
produce a number (gpt-oss-20b fails the recall controls) may still recognise one, and since every option is a price
the share really had, knowing its usual level does not help: only knowing when each price happened does.

Asked four times with the correct option in each position once (permutation balancing: LLMs' letter preferences
swing multiple-choice accuracy by 13-85%); the score is the share of the four answered right. Fact type: price."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data import Data
from ..items import Item, ItemScore
from . import decop
from .common import SYSTEM_MEMORY

NAME = "price_choice"
PROMPT_VERSION = "p1"
P0 = 0.25
VARIANTS = ["r0", "r1", "r2", "r3"]
LETTERS = decop.LETTERS
MAX_TOKENS = 60
WINDOW = 24                      # wrong options come from months at most this far from the asked month...
WIDE_WINDOW = 36                 # ...or this far, for calm shares that have too few distinct prices within WINDOW
MIN_GAP = 4                      # ...and at least this far, so they belong to other periods of the same share
MIN_LOG_GAP = math.log(1.15)     # every pair of the four prices at least 15% apart: distinct after rounding
SPLIT_LOG_TOL = math.log(1.2)    # adjusted/raw ratio within 20% of the asked month's: no split in between
MIN_DAYS = 10
FILTERS = (f"the company-months of a source set; three wrong options = the same share's average close (as traded) in "
           f"other months {MIN_GAP}-{WINDOW} months away ({WIDE_WINDOW} when needed) with no split in between, every "
           f"pair of the four prices at least 15% apart; the number of wrong options priced below the answer drawn "
           f"uniformly from 0-3 (so highest, lowest and middle are no tell); all four printed with the same number of "
           f"decimals (three significant digits on the smallest); options in a fixed random order, the correct one "
           f"inserted at each position once")

parse = decop.parse


def monthly_levels(prices: pd.DataFrame) -> pd.DataFrame:
    """Per (insref, month): the mean close as traded, the mean adjusted close and the number of trading days."""
    px = prices.assign(month=prices["date"].dt.to_period("M"))
    return (px.groupby(["insref", "month"])
            .agg(level=("close", "mean"), adj_level=("adj_close", "mean"), days=("close", "size")).reset_index())


def _months_apart(a: pd.Period, b: pd.Period) -> int:
    return abs((a - b).n)


def is_central(values: list[float]) -> bool:
    """Is the first price the one closest to the four prices' geometric mean (what the 'middle' rule picks)?"""
    logs = [math.log(v) for v in values]
    centre = sum(logs) / len(logs)
    return int(np.argmin([abs(x - centre) for x in logs])) == 0


def pick_decoys(levels: pd.DataFrame, month: pd.Period, rng: np.random.Generator, window: int = WINDOW,
                min_gap: int = MIN_GAP, below: int | None = None, central: bool | None = None,
                tries: int = 400) -> list[tuple[str, float]] | None:
    """Three (month, price) wrong options for one share's month, or None when it has too few suitable months.
    below: how many of the three must be priced below the answer; central: whether the answer must (not) be the
    price closest to the four's geometric mean (None: either). Random triples are drawn until one fits."""
    lv = levels.set_index("month")
    if month not in lv.index:
        return None
    target = lv.loc[month]
    ratio = target["adj_level"] / target["level"] if pd.notna(target["adj_level"]) else None
    rows = []
    for m, r in lv.iterrows():
        gap = _months_apart(m, month)
        if not min_gap <= gap <= window or r["days"] < MIN_DAYS or not r["level"] > 0:
            continue
        if ratio is not None and pd.notna(r["adj_level"]) and \
                abs(math.log((r["adj_level"] / r["level"]) / ratio)) > SPLIT_LOG_TOL:
            continue                       # a split lies between: the price would be on another scale
        rows.append((str(m), float(r["level"])))
    level = float(target["level"])
    far = [r for r in rows if abs(math.log(r[1] / level)) >= MIN_LOG_GAP]
    lower, higher = [r for r in far if r[1] < level], [r for r in far if r[1] > level]
    if below is not None and (len(lower) < below or len(higher) < 3 - below) or len(far) < 3:
        return None
    for _ in range(tries):
        if below is None:
            triple = [far[int(i)] for i in rng.choice(len(far), 3, replace=False)]
        else:                              # draw each side's share directly: rare ranks are found too
            triple = [lower[int(i)] for i in rng.choice(len(lower), below, replace=False)] + \
                     [higher[int(i)] for i in rng.choice(len(higher), 3 - below, replace=False)]
            triple = [triple[int(i)] for i in rng.permutation(3)]
        values = [level] + [v for _, v in triple]
        if any(abs(math.log(a / b)) < MIN_LOG_GAP for i, a in enumerate(values) for b in values[i + 1:]):
            continue
        if below is not None and sum(v < level for _, v in triple) != below:
            continue
        if central is not None and is_central(values) != central:
            continue
        return triple
    return None


def balanced_decoys(levels: pd.DataFrame, month: pd.Period, rng: np.random.Generator):
    """Wrong options placed so that no simple rule finds the answer: its rank among the four drawn uniformly (0-3
    cheaper options; the 'highest' and 'lowest' rules), and whether it is the price closest to the four's geometric
    mean drawn with probability 1/4 (the 'middle' rule: wrong options come from months around the answer, so in a
    trending share they centre on its price). Relaxed step by step for shares with too few distinct prices: the
    other ranks, then the wide window, then the centre condition. Returns (decoys, below, window)."""
    want_central = bool(rng.random() < 0.25)
    for central in (want_central, None):
        for window in (WINDOW, WIDE_WINDOW):
            for below in rng.permutation(4):
                decoys = pick_decoys(levels, month, rng, window=window, below=int(below), central=central)
                if decoys is not None:
                    return decoys, int(below), window
    return None, None, None


def fmt_prices(values: list[float]) -> list[str]:
    """The same number of decimals for every option (three significant digits on the smallest): no option stands out
    by how it is written."""
    decimals = min(4, max(0, 2 - math.floor(math.log10(min(values)))))
    return [f"{v:.{decimals}f}" for v in values]


def build_items(data: Data, base_items: list[Item], seed: int, id_prefix: str = "pc",
                method: str = NAME) -> tuple[list[Item], int]:
    """One multiple-choice item per (real) source item, keeping its company, month, date, cluster and strata.
    Returns (items, number of source items dropped for lack of suitable wrong options)."""
    rng = np.random.default_rng(seed)
    real = [it for it in base_items if not it.fabricated and it.truth.get("insref") is not None]
    insrefs = {int(it.truth["insref"]) for it in real}
    levels = monthly_levels(data.prices[data.prices["insref"].isin(insrefs)])
    by_ins = {int(k): g for k, g in levels.groupby("insref")}
    items, dropped = [], 0
    for it in real:
        insref, month = int(it.truth["insref"]), pd.Period(it.payload["month"], "M")
        decoys, below, window = balanced_decoys(by_ins.get(insref, levels.iloc[:0]), month, rng)
        if decoys is None:
            dropped += 1
            continue
        target = float(by_ins[insref].set_index("month").loc[month, "level"])
        order = rng.permutation(3)
        decoys = [decoys[int(i)] for i in order]
        texts = fmt_prices([target] + [v for _, v in decoys])
        items.append(Item(
            item_id=f"{id_prefix}-{len(items):04d}", method=method, date=it.date, cluster=it.cluster,
            payload={**{k: it.payload[k] for k in ("name", "symbol", "currency", "month", "month_name")},
                     "answer_text": texts[0], "decoy_texts": texts[1:]},
            truth={"level": target, "decoy_levels": [v for _, v in decoys], "decoy_months": [m for m, _ in decoys],
                   "baseline_level": it.truth.get("baseline_level"), "insref": insref, "source_item": it.item_id,
                   "n_below": below, "window": window},
            strata=dict(it.strata)))
    return items, dropped


def options(item: Item, variant: str) -> list[str]:
    opts = list(item.payload["decoy_texts"])
    opts.insert(int(variant[1:]), item.payload["answer_text"])
    return opts


def option_values(item: Item, variant: str) -> list[float]:
    vals = list(item.truth["decoy_levels"])
    vals.insert(int(variant[1:]), item.truth["level"])
    return vals


def correct_letter(variant: str) -> str:
    return LETTERS[int(variant[1:])]


def render(item: Item, mode: str, variant: str) -> list[dict]:
    p = item.payload
    head = (f"Which of these was the average closing price of {p['name']} (ticker: {p['symbol']}) during "
            f"{p['month_name']}, in {p['currency']}? The other three options are the same share's average closing "
            "price in other months.")
    body = "\n".join(f"{letter}: {text} {p['currency']}" for letter, text in zip(LETTERS, options(item, variant)))
    if mode == "forced":
        instr = 'You must choose one, even if you are unsure. Respond only with JSON: {"answer": "A" | "B" | "C" | "D"}'
    else:
        instr = 'If you do not know, answer null. Respond only with JSON: {"answer": "A" | "B" | "C" | "D" | null}'
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{head}\n\n{body}\n\n{instr}"}]


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    """The share of the four orderings answered right (chance 0.25). Extras: n_correct, known (3+ of 4), one_letter
    (the same letter all four times: a position habit, not a choice) and consistent (the same price all four times)."""
    letters = [parsed[v].get("letter") for v in VARIANTS]
    k = sum(letter == correct_letter(v) for v, letter in zip(VARIANTS, letters))
    refused = all(parsed[v].get("refused", True) for v in VARIANTS)
    picked = {options(item, v)[LETTERS.index(letter)] for v, letter in zip(VARIANTS, letters) if letter in LETTERS}
    answered = all(letter in LETTERS for letter in letters)
    return ItemScore(k / len(VARIANTS), refused=refused, extra={
        "n_correct": k, "known": k >= 3, "one_letter": answered and len(set(letters)) == 1,
        "consistent": answered and len(picked) == 1})


# ---------------------------------------------------------------- probability probes (M12p/M13p)
def _letter(token: str) -> str | None:
    s = token.strip().strip('"').strip("'").strip()
    return s if s in LETTERS else None


def letter_probs(content: list | None) -> dict | None:
    """The letters' probabilities at the token where the answer letter is written: the top alternatives there that
    are letters, renormalised over A-D (mass = their share before renormalising). None when no such token."""
    if not content:
        return None
    text, pos = "", None
    for i, t in enumerate(content):
        if _letter(t.get("token", "")) and "answer" in text.lower():
            pos = i
            break
        text += t.get("token", "")
    if pos is None:                            # no JSON key seen: the last letter token
        pos = next((i for i in range(len(content) - 1, -1, -1) if _letter(content[i].get("token", ""))), None)
    if pos is None:
        return None
    probs = {letter: 0.0 for letter in LETTERS}
    for alt in content[pos].get("top_logprobs") or [{"token": content[pos]["token"],
                                                     "logprob": content[pos]["logprob"]}]:
        letter = _letter(alt.get("token", ""))
        if letter:
            probs[letter] += math.exp(alt["logprob"])
    mass = sum(probs.values())
    if mass <= 0:
        return None
    return {"probs": {k: v / mass for k, v in probs.items()}, "mass": mass}


def with_logprobs(parsed: dict, content: list | None) -> dict:
    lp = letter_probs(content)
    return {**parsed, "letter_probs": lp} if lp else parsed


def item_score_prob(item: Item, parsed: dict[str, dict]) -> ItemScore:
    """The mean probability of the correct letter over the four orderings (chance 0.25). An ordering without
    probabilities counts its chosen letter fully (1 or 0); a refusal counts 0. Extras: n_correct (the chosen letters,
    as M12), from_logprobs (orderings scored from probabilities)."""
    ps, from_lp = [], 0
    for v in VARIANTS:
        lp = parsed[v].get("letter_probs")
        if lp:
            ps.append(float(lp["probs"].get(correct_letter(v), 0.0)))
            from_lp += 1
        else:
            ps.append(1.0 if parsed[v].get("letter") == correct_letter(v) else 0.0)
    base = item_score(item, parsed)
    return ItemScore(float(np.mean(ps)), refused=base.refused, extra={**base.extra, "from_logprobs": from_lp})


# ---------------------------------------------------------------- blind-baseline audit (no model involved)
def blind_choices(item: Item) -> dict[str, int]:
    """Which option (0 = the correct one, 1-3 = the wrong ones in stored order) each simple rule picks. Option-only
    rules must stay near chance (0.25), otherwise the options give the answer away; 'persistence' (closest to the
    same month a year earlier) is the blind guesser, a measure of how much a price's history alone tells."""
    vals = [item.truth["level"], *item.truth["decoy_levels"]]
    logs = [math.log(v) for v in vals]
    centre = sum(logs) / len(logs)
    out = {"highest": int(np.argmax(vals)), "lowest": int(np.argmin(vals)),
           "middle": int(np.argmin([abs(x - centre) for x in logs]))}
    base = item.truth.get("baseline_level")
    if base:
        out["persistence"] = int(np.argmin([abs(math.log(v / base)) for v in vals]))
    return out


def audit(items: list[Item]) -> dict[str, float]:
    """Each rule's hit rate (the share of items where it picks the correct option), plus 'inner_pair': guessing
    between the two middle-ranked prices (right half the time when the answer is one of them)."""
    hits: dict[str, list[float]] = {}
    for it in items:
        for rule, k in blind_choices(it).items():
            hits.setdefault(rule, []).append(int(k == 0))
        rank = sum(v < it.truth["level"] for v in it.truth["decoy_levels"])
        hits.setdefault("inner_pair", []).append(0.5 if rank in (1, 2) else 0.0)
    return {rule: float(np.mean(h)) for rule, h in hits.items()}
