"""
Scoring helpers for the evaluation: compare answers and SQL results to ground truth.

Ground truth comes from a question's `reference_sql`, run directly on DuckDB. The values
it returns (the first row, or the first column with `"check": "first_column"`) are the
facts a correct answer has to state.
"""

import re
from typing import Any

# Legal-name endings people (and models) drop: "CAPITAL ONE FINANCIAL CORPORATION" is
# matched by "Capital One", "BANK OF AMERICA, NATIONAL ASSOCIATION" by "Bank of America".
_NAME_SUFFIXES = re.compile(
    r"(,?\s*\b(national association|n\.?a\.?|financial corporation|financial|corporation|"
    r"corp\.?|company|& co\.?|co\.?|inc\.?|llc|l\.l\.c\.|group|holdings|bank usa|"
    r"intermediate)\b\.?)+\s*$",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")
PARAPHRASE_THRESHOLD = 0.75
_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "on",
    "in",
    "with",
    "your",
    "you",
    "to",
    "for",
    "or",
    "and",
    "by",
    "from",
    "about",
    "at",
    "is",
    "are",
    "was",
    "other",
}


def _content_words(text: str) -> set[str]:
    """Lower-cased words minus stopwords, with a plural 's' stripped."""
    words = set()
    for word in re.findall(r"[a-z']+", text.lower()):
        word = word.strip("'")
        if word in _STOPWORDS or len(word) < 3:
            continue
        words.add(word[:-1] if word.endswith("s") and len(word) > 3 else word)
    return words


def expected_values(rows: list[dict[str, Any]], check: str = "first_row") -> list[Any]:
    """The facts to look for: all values of the first row, or every value of column one."""
    if not rows:
        return []
    if check == "first_column":
        first = next(iter(rows[0]))
        return [row[first] for row in rows]
    return list(rows[0].values())


def _normalize_text(value: str) -> str:
    text = value.strip()
    while True:
        shorter = _NAME_SUFFIXES.sub("", text).strip(" ,.&")
        if shorter == text:
            break
        text = shorter
    return re.sub(r"\s+", " ", text).lower()


def _numbers_in(text: str) -> list[float]:
    numbers = []
    for match in _NUMBER_RE.findall(text):
        try:
            numbers.append(float(match.replace(",", "").rstrip(".")))
        except ValueError:
            continue
    return numbers


def number_mentioned(value: float, text: str, exact: bool = False) -> bool:
    """
    True if `text` states `value`. Counts (`exact`) must match exactly; other numbers
    allow rounding (97.4 vs 97.38) and fraction vs percent (0.974 vs 97.4%).
    """
    numbers = _numbers_in(text)
    if exact:
        return float(value) in numbers
    candidates = [float(value)]
    if 0 < abs(value) <= 1:
        candidates.append(value * 100)  # a fraction written as a percentage
    return any(
        abs(number - target) <= max(0.051, abs(target) * 0.005)
        for number in numbers
        for target in candidates
    )


def value_mentioned(value: Any, text: str) -> bool:
    """True if one expected value appears in `text` (numbers and names are normalized)."""
    if value is None:
        return False
    if isinstance(value, bool):
        return str(value).lower() in text.lower()
    if isinstance(value, int):
        return number_mentioned(value, text, exact=True)
    if isinstance(value, float):
        return number_mentioned(value, text)
    if hasattr(value, "isoformat"):  # dates
        return value.isoformat()[:10] in text or str(value) in text
    needle = _normalize_text(str(value))
    if not needle:
        return False
    haystack = _normalize_text(text)
    # Whole words only, so "CA" doesn't match inside "Oscar".
    pattern = r"(?<!\w)" + re.escape(needle) + r"(?!\w)"
    if re.search(pattern, haystack):
        return True
    # Long category labels get paraphrased ("Problem with a purchase shown on your
    # statement" -> "purchases shown on statements"): accept most of the content words.
    words = _content_words(needle)
    if len(words) >= 3:
        found = words & _content_words(haystack)
        return len(found) / len(words) >= PARAPHRASE_THRESHOLD
    return False


def fraction_mentioned(values: list[Any], text: str) -> float | None:
    """Share of `values` that appear in `text`; None when there is nothing to check."""
    if not values:
        return None
    return sum(value_mentioned(v, text) for v in values) / len(values)


def result_matches(expected: list[Any], model_rows: list[dict[str, Any]]) -> bool | None:
    """
    True if every expected value appears somewhere in the model's SQL result, whatever
    the column names or order. None when there is no ground truth.
    """
    if not expected:
        return None
    flat = " | ".join(str(v) for row in model_rows for v in row.values())
    return all(value_mentioned(v, flat) for v in expected)


def cites_sources(answer: str, documents: list[dict[str, Any]]) -> bool | None:
    """True if a text-search answer mentions at least one retrieved record id."""
    ids = [str(doc.get("id")) for doc in documents or [] if doc.get("id")]
    if not ids:
        return None
    return any(doc_id in answer for doc_id in ids)


def keyword_rate(answer: str, keywords: list[str]) -> float | None:
    """Share of loose keywords found in the answer (the original, rough check)."""
    if not keywords:
        return None
    lower = answer.lower()
    return sum(k.lower() in lower for k in keywords) / len(keywords)
