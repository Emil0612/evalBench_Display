"""M18: headline dating. A Millistream headline with every date cue removed; the model names the year and month it
was published. Knowing when something happened is memory a backtest can leak (the date of an event); text cues alone
date a headline only roughly, which a model-free baseline trained on other headlines measures (Lopez-Lira, Tang & Zhu
2025: GPT-4o dates headlines to month-year 90% of the time before its cutoff).

Score: 1 - |months off| / 12, zero at a year or more. Fact type: event."""
from __future__ import annotations

import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

from ..data import DATA_DIR
from ..items import Item, ItemScore, file_md5
from ..textutil import classify_non_answer, extract_json
from .common import SYSTEM_MEMORY

NAME = "headline_date"
PROMPT_VERSION = "p1"
P0 = 0.0
VARIANTS = ["main"]
MAX_TOKENS = 40
NEWS_FILE = DATA_DIR / "news_index.parquet"
MONTHS = pd.period_range("2016-01", "2026-07", freq="M")
PER_MONTH = 12
PILOT_N = len(MONTHS)                  # the first 127 items: one per month, the micro-pilot block
TRAIN_PER_MONTH = 300                  # the blind baseline's training headlines per month
TEMPLATE_MIN = 10                      # a headline shape seen this often is a template
DIM = 2 ** 18
YEARS = (2000, 2030)
NEWSTYPES = {0: "macro", 1: "editorial", 2: "press_release"}
FILTERS = ("Swedish or English headlines, newstype 0-2, 30-200 characters, published once (same_text_count 1, a unique "
           "headline), no date cue (year, month name, quarter or report period, numeric date, week number), not a "
           "template (the same shape 10+ times); 13 per month 2016-01..2026-07: the first 127 items one per month (the "
           "micro-pilot block), then 12 per month; truth = the publication month; blind_month = a hashed bag-of-words "
           "ridge regression on 300 other headlines per month, refit leaving out the test month and its neighbours")

_MONTH_WORDS = ("january february march april may june july august september october november december "
                "januari februari mars maj juni juli augusti oktober "
                "jan feb mar apr jun jul aug sep sept oct okt nov dec").split()
_CUES = [re.compile(p, re.IGNORECASE) for p in (
    r"\b(19|20)\d{2}\b",                                   # a year
    r"\b(" + "|".join(_MONTH_WORDS) + r")\b",              # a month name (English, Swedish)
    r"\bq[1-4]\b|\b[1-4]q\b|\bh[12]\b",                    # Q3, 3Q, H1
    r"quarter|kvartal|half[- ]?year|halvår|interim|delårs|bokslut|year[- ]end|helår|full[- ]year|årsredovisning|"
    r"annual report|bokslutskommuniké",
    r"\b\d{1,2}[./]\d{1,2}([./]\d{2,4})?\b",               # a numeric date
    r"\b(week|vecka|v\.)\s*\d{1,2}\b",                     # a week number
)]


def has_date_cue(headline: str) -> bool:
    return any(p.search(headline or "") for p in _CUES)


def template_key(headline: str) -> str:
    """The headline's shape: the company prefix before ':' dropped, digits as '#', lower case, spaces collapsed."""
    h = (headline or "").strip()
    head, sep, rest = h.partition(":")
    if sep and len(head) <= 60:
        h = rest
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", h.lower())).strip()


def load_headlines(path: Path = NEWS_FILE) -> pd.DataFrame:
    news = pd.read_parquet(path, columns=["newsid", "date", "language", "newstype", "distributor", "headline",
                                          "same_text_count"])
    news["date"] = pd.to_datetime(news["date"])
    return news


def eligible(news: pd.DataFrame) -> pd.DataFrame:
    """The filters of FILTERS, in that order; adds 'month'."""
    n = news[news["language"].isin(["sv", "en"]) & news["newstype"].isin(list(NEWSTYPES))].copy()
    n["headline"] = n["headline"].astype(str).str.strip()
    n = n[n["headline"].str.len().between(30, 200) & (n["same_text_count"] <= 1)]
    n = n[n["headline"].map(n["headline"].value_counts()) == 1]
    n = n[~n["headline"].map(has_date_cue)]
    keys = n["headline"].map(template_key)
    n = n[keys.map(keys.value_counts()) < TEMPLATE_MIN]
    n["month"] = pd.PeriodIndex(n["date"], freq="M")
    return n


def _tokens(headline: str) -> list[str]:
    words = re.findall(r"\w+", headline.lower())
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def features(headlines: list[str]) -> csr_matrix:
    """Hashed unigrams and bigrams, binary, each row scaled to unit length."""
    rows, cols = [], []
    for i, h in enumerate(headlines):
        idx = sorted({zlib.crc32(t.encode("utf-8")) % DIM for t in _tokens(h)})
        rows += [i] * len(idx)
        cols += idx
    X = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(headlines), DIM))
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return csr_matrix(X.multiply(1.0 / norms[:, None]))


def blind_predict(train: pd.DataFrame, test_month: pd.Period, test_headlines: list[str], damp: float = 0.3) -> list:
    """The blind baseline's month for each test headline: a ridge regression of the month on the headline's words,
    trained on `train` (with 'month' and 'headline') without the test month and its two neighbours."""
    keep = train[(train["month"] < test_month - 1) | (train["month"] > test_month + 1)]
    y = np.array([m.ordinal for m in keep["month"]], dtype=float)
    mean = y.mean()
    w = lsqr(features(keep["headline"].tolist()), y - mean, damp=damp)[0]
    pred = mean + features(test_headlines) @ w
    lo, hi = MONTHS[0].ordinal, MONTHS[-1].ordinal
    return [pd.Period(ordinal=int(round(min(max(p, lo), hi))), freq="M") for p in pred]


def build_items(news: pd.DataFrame, seed: int) -> list[Item]:
    """The pilot block (one headline per month), then 12 per month; each with the blind baseline's month. A frame
    that already has 'month' is taken as already filtered."""
    rng = np.random.default_rng(seed)
    pool = news if "month" in news.columns else eligible(news)
    picked = {}
    for m in MONTHS:
        cand = pool[pool["month"] == m]
        if len(cand) < PER_MONTH + 1:
            continue
        picked[m] = cand.iloc[rng.permutation(len(cand))[:PER_MONTH + 1]]
    used = set(pd.concat(picked.values())["newsid"]) if picked else set()
    rest = pool[~pool["newsid"].isin(used)]
    train = pd.concat([g.iloc[rng.permutation(len(g))[:TRAIN_PER_MONTH]] for _, g in rest.groupby("month")])
    rows = [(m, df.iloc[0]) for m, df in picked.items()] + \
        [(m, r) for m, df in picked.items() for _, r in df.iloc[1:].iterrows()]
    blind = {}
    for m, df in picked.items():
        for (_, r), g in zip(df.iterrows(), blind_predict(train, m, df["headline"].tolist())):
            blind[r["newsid"]] = g
    items = []
    for k, (m, r) in enumerate(rows):
        items.append(Item(
            item_id=f"hd-{k:04d}", method=NAME, date=r["date"].date().isoformat(), cluster=f"m{m}",
            payload={"headline": r["headline"], "language": r["language"]},
            truth={"month": str(m), "blind_month": str(blind[r["newsid"]]), "newsid": r["newsid"],
                   "newstype": int(r["newstype"]), "distributor": r["distributor"]},
            strata={"language": r["language"], "newstype": NEWSTYPES[int(r["newstype"])]}))
    return items


def manifest_extra(path: Path = NEWS_FILE) -> dict:
    """The set's real provenance for its manifest: the headline index it is built from and the months it covers."""
    return {"source_md5": {path.name: file_md5(path)},
            "date_window": [MONTHS[0].start_time.date().isoformat(), MONTHS[-1].end_time.date().isoformat()]}


def is_pilot(item_id: str) -> bool:
    prefix, _, number = item_id.partition("-")
    return prefix == "hd" and number.isdigit() and int(number) < PILOT_N


def render(item: Item, mode: str, variant: str = "main") -> list[dict]:
    lang = "Swedish" if item.payload["language"] == "sv" else "English"
    question = (f"Below is the headline of a Nordic financial news item ({lang}). Any dates have been removed from "
                f"the headline set. In which year and month was it published?\n\nHeadline: {item.payload['headline']}")
    if mode == "forced":
        instr = ('You must give your best estimate, even if you are unsure. Respond only with JSON: '
                 '{"year": <yyyy>, "month": <1-12>}')
    else:
        instr = ('If you do not know, answer null. Respond only with JSON: '
                 '{"year": <yyyy or null>, "month": <1-12 or null>}')
    return [{"role": "system", "content": SYSTEM_MEMORY}, {"role": "user", "content": f"{question}\n{instr}"}]


def _int(v):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def parse(text: str) -> dict:
    """The JSON answer only; a year or month outside the range is malformed, never scored as a far-off date."""
    obj = extract_json(text)
    if obj is None or "year" not in obj:
        return {"value": None, "refused": True, "outcome": classify_non_answer(text) if obj is None else "malformed"}
    if obj.get("year") is None and obj.get("month") is None:
        return {"value": None, "refused": True, "outcome": "abstain"}
    year, month = _int(obj.get("year")), _int(obj.get("month"))
    if year is None or month is None or not (YEARS[0] <= year <= YEARS[1]) or not 1 <= month <= 12:
        return {"value": None, "refused": True, "outcome": "malformed"}
    return {"value": f"{year:04d}-{month:02d}", "refused": False, "outcome": "answer"}


def month_error(answer: str, truth: str) -> int:
    return pd.Period(answer, "M").ordinal - pd.Period(truth, "M").ordinal


SIZE_RANK = {"small": 0, "mid": 1, "large": 2}


def company_sizes(items, refs_by_newsid: dict, company_of: dict, sizes: dict) -> dict[str, str]:
    """item_id -> the market-cap tercile of the listed company the headline is about (the largest when it names
    several), 'none' when it names no listed company, 'unknown' when its companies have no size (exploratory,
    2026-10-06; the frozen set is unchanged)."""
    out = {}
    for it in items:
        refs = refs_by_newsid.get(it.truth.get("newsid"))
        refs = [] if refs is None else list(refs)
        if not refs:
            out[it.item_id] = "none"
            continue
        found = [sizes.get(f"c{int(company_of[r])}") for r in refs if r in company_of and pd.notna(company_of[r])]
        found = [s for s in found if s in SIZE_RANK]
        out[it.item_id] = max(found, key=SIZE_RANK.get) if found else "unknown"
    return out


def blind_score(item: Item) -> float:
    """The text-only baseline's score on this headline: the M18 score of its blind month."""
    return max(0.0, 1.0 - abs(month_error(item.truth["blind_month"], item.truth["month"])) / 12)


def item_score(item: Item, parsed: dict[str, dict]) -> ItemScore:
    value = parsed["main"].get("value")
    if value is None:
        return ItemScore(0.0, refused=True)
    err = month_error(value, item.truth["month"])
    return ItemScore(max(0.0, 1.0 - abs(err) / 12), refused=False, extra={
        "err_months": err, "exact": err == 0, "within_3": abs(err) <= 3,
        "same_year": value[:4] == item.truth["month"][:4], "answer": value})
