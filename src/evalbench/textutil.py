"""Text helpers shared by the methods: number handling, template filter, JSON extraction, ROUGE-L."""
from __future__ import annotations

import html
import json
import math
import re

# A number: grouped thousands with one consistent separator (space, nbsp, comma or dot) or a plain
# number with an optional decimal part. Not glued to letters, dates (15.06.2021, 2021-06-15) or slashes.
NUMBER_RE = re.compile(
    r"(?<![\w.,/-])"
    r"(?:\d{1,3}(?:(?P<sep>[ \u00a0\u202f,.])\d{3})(?:(?P=sep)\d{3})*(?:[.,]\d+)?"
    r"|\d+(?:[.,]\d+)?)"
    r"(?![\w/-]|[.,]\d)"
)
UNIT_AFTER_RE = re.compile(
    r"\s?(?:%|procent\b|percent\b|per cent\b|[mt]?sek\b|[mt]?nok\b|[mt]?dkk\b|[mt]?eur\b|[mt]?usd\b"
    r"|kr\b|kronor\b|mkr\b|mdkr\b|mnkr\b|miljoner\b|miljarder\b|million|billion|bn\b|mn\b|m\b)",
    re.IGNORECASE,
)
UNIT_BEFORE_RE = re.compile(r"(?:sek|nok|dkk|eur|usd|€|\$|£)\s?$", re.IGNORECASE)
TEMPLATE_RE = re.compile(
    r"buy.?back|återköp|insider|kallelse|notice of|flaggning|transactions? in own|own shares|egna aktier"
    r"|major shareholder|managers.? transactions|share repurchase|årsstämma|annual general meeting"
    r"|bolagsstämma|extraordinary general|market making|likviditetsgarant|commercial paper",
    re.IGNORECASE,
)
BOILERPLATE_RE = re.compile(
    r"market abuse regulation|marknadsmissbruksförordning|obliged to make public|skyldig\w* att offentliggöra"
    r"|for further information|för ytterligare information|för mer information|contact person|kontaktperson"
    r"|was submitted for publication|lämnades för offentliggörande|certified adviser"
    r"|^\s*(?:om|about) [^.:\n]{1,60}:|disclaimer|trademark|forward-looking statement|framåtriktade uttalanden",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"\w+")
ABSTAIN_RE = re.compile(
    r"\b(?:i do(?:n't| not) know|not sure|unsure|unknown|(?:do(?:n't| not)|does not) have (?:that|this|the|any|enough) "
    r"(?:information|data)|no (?:reliable )?(?:data|information)|can(?:'|no)t recall|cannot recall|unable to recall"
    r"|jag vet inte|vet ej|vet inte)\b|(?<![\w/])n/a(?![\w/])",
    re.IGNORECASE,
)
REFUSAL_RE = re.compile(
    r"\b(?:i can(?:'|no)t (?:help|assist|provide|share|comply|reproduce|answer(?! with (?:certainty|confidence|precision)))"
    r"|i (?:won't|will not)|i'm sorry|i am sorry"
    r"|as an ai|not able to (?:provide|help|share)|unable to (?:provide|help|share|reproduce)"
    r"|cannot (?:provide|share|reproduce|answer(?! with (?:certainty|confidence|precision)))|decline|tyvärr)\b",
    re.IGNORECASE,
)


def extract_numbers(text: str) -> list[re.Match]:
    return list(NUMBER_RE.finditer(text or ""))


def parse_number(s: str, lang: str) -> float | None:
    """Parse a number written in Swedish (space thousands, comma decimal) or English conventions."""
    s = re.sub(r"[ \u00a0\u202f]", "", str(s).strip())
    s = re.sub(r"[^\d.,-]", "", s)
    if not re.search(r"\d", s):
        return None
    if lang == "sv":
        if "," in s:
            s = s.replace(".", "").replace(",", ".")
        elif re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", s):
            s = s.replace(".", "")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def number_values(s: str) -> set[float]:
    """Every value a string could mean under Swedish or English conventions."""
    return {v for lang in ("sv", "en") if (v := parse_number(s, lang)) is not None}


def numbers_equal(answer: str, truth: str, lang: str) -> bool:
    """Exact match: the truth is read in the article's language, the answer in either convention."""
    t = parse_number(truth, lang)
    if t is None:
        return False
    return any(math.isclose(a, t, rel_tol=1e-9, abs_tol=1e-9) for a in number_values(answer))


def parse_answer_number(value) -> float | None:
    """A price as a model wrote it: JSON number, or a string like '210,65' / 'SEK 1 234.50'."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    digits = re.sub(r"[^\d.,]", "", s)
    lang = "sv" if re.search(r",\d{1,2}$", digits) else "en"
    return parse_number(s, lang)


def answer_number(text: str) -> re.Match | None:
    """The number a prose answer most likely gives: the last one that is not a year (else the last one)."""
    nums = extract_numbers(text)
    non_years = [m for m in nums if not is_year(m.group())]
    return (non_years or nums or [None])[-1]


def classify_non_answer(text: str) -> str:
    """For a reply that gave no usable answer: 'abstain' (says it doesn't know), 'refusal' (declines), or
    'malformed' (empty, garbled or off-format). Abstention wins when both appear ("Sorry, I don't know")."""
    t = (text or "").strip()
    if not t:
        return "malformed"
    if ABSTAIN_RE.search(t):
        return "abstain"
    if REFUSAL_RE.search(t):
        return "refusal"
    return "malformed"


def significant_digits(s: str) -> int:
    return len(re.sub(r"\D", "", s).lstrip("0"))


def is_year(s: str) -> bool:
    return bool(re.fullmatch(r"(?:19|20)\d\d", s.strip()))


def has_unit(text: str, start: int, end: int) -> bool:
    """True if a currency/percent/magnitude word sits right after or a currency right before the number."""
    return bool(UNIT_AFTER_RE.match(text, end)) or bool(UNIT_BEFORE_RE.search(text[max(0, start - 6):start]))


def is_templated(headline: str) -> bool:
    """Formulaic disclosures (buybacks, insider trades, AGM notices...) that score high without memory."""
    return bool(TEMPLATE_RE.search(headline or ""))


def is_boilerplate(text: str) -> bool:
    """Legal/contact boilerplate that appears in thousands of releases (known without seeing the article)."""
    return bool(BOILERPLATE_RE.search(text or ""))


def extract_json(text: str) -> dict | None:
    """First flat JSON object in a model reply (tolerates prose and code fences around it)."""
    for m in re.finditer(r"\{[^{}]*\}", text or ""):
        try:
            obj = json.loads(m.group(), strict=False)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def lcs_mask(a: list[str], b: list[str]) -> list[bool]:
    """Which tokens of a belong to one longest common subsequence with b."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i][j] = dp[i + 1][j + 1] + 1 if a[i] == b[j] else max(dp[i + 1][j], dp[i][j + 1])
    mask, i, j = [False] * n, 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            mask[i] = True
            i, j = i + 1, j + 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return mask


def rouge_l_f1(candidate: str, reference: str) -> float:
    """ROUGE-L F1 with the candidate cut to the reference's token length (Golchin & Surdeanu)."""
    ref = tokens(reference)
    cand = tokens(candidate)[: len(ref)]
    if not cand or not ref:
        return 0.0
    lcs = lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def highlight_overlap(candidate: str, reference: str) -> str:
    """HTML of the candidate with its LCS tokens (shared with the reference) wrapped in <mark>."""
    matches = list(_TOKEN_RE.finditer(candidate or ""))
    mask = lcs_mask([m.group().lower() for m in matches], tokens(reference))
    out, pos = [], 0
    for m, hit in zip(matches, mask):
        out.append(html.escape(candidate[pos:m.start()]))
        word = html.escape(m.group())
        out.append(f"<mark>{word}</mark>" if hit else word)
        pos = m.end()
    out.append(html.escape((candidate or "")[pos:]))
    return "".join(out)


def number_recall(output: str, reference: str, lang: str) -> float | None:
    """Share of the reference's numbers that the output reproduces (None if the reference has none)."""
    ref_vals = [v for m in extract_numbers(reference) if (v := parse_number(m.group(), lang)) is not None]
    if not ref_vals:
        return None
    out_vals: set[float] = set().union(*(number_values(m.group()) for m in extract_numbers(output)))
    hits = sum(any(math.isclose(r, o, rel_tol=1e-9, abs_tol=1e-9) for o in out_vals) for r in ref_vals)
    return hits / len(ref_vals)
