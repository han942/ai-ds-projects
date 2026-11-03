"""Korean text helpers built on Kiwi: sentences, morphemes, nouns, numbers, and sentence endings."""

from __future__ import annotations

import re
from functools import lru_cache

from kiwipiepy import Kiwi

_KIWI: Kiwi | None = None


def kiwi() -> Kiwi:
    global _KIWI
    if _KIWI is None:
        _KIWI = Kiwi()
    return _KIWI


@lru_cache(maxsize=50_000)
def analyze(text: str) -> tuple[tuple[str, str], ...]:
    return tuple((t.form, t.tag) for t in kiwi().tokenize(text))


@lru_cache(maxsize=50_000)
def sentences(text: str) -> tuple[str, ...]:
    sents = tuple(s.text.strip() for s in kiwi().split_into_sents(text) if s.text.strip())
    return sents or ((text.strip(),) if text.strip() else ())


def morphemes(text: str) -> list[str]:
    return [form for form, _ in analyze(text)]


def morph_string(text: str) -> str:
    return " ".join(morphemes(text))


def nouns(text: str, tag: str = "NNG", min_len: int = 2) -> list[str]:
    return [form for form, t in analyze(text) if t == tag and len(form) >= min_len]


# Numbers -------------------------------------------------------------------

_SMALL_UNITS = {"십": 10, "백": 100, "천": 1_000}
_BIG_UNITS = {"만": 10**4, "억": 10**8, "조": 10**12, "경": 10**16}
_DIGIT_NUMBER = re.compile(
    r"(?<![\d.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)([십백천]?[만억조경]|[십백천])?"
)
_NATIVE_NUMBERS = {
    "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "석": 3, "네": 4, "넷": 4,
    "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10, "스무": 20, "스물": 20,
}
_COUNTERS = "명|곳|개|가지|번|살|시간|달|해|권|대|채|마리|군데|차례|배|층|척|건"
_NATIVE_NUMBER = re.compile(
    r"(?<![가-힣])(" + "|".join(sorted(_NATIVE_NUMBERS, key=len, reverse=True)) + r")\s?(?=(" + _COUNTERS + r"))"
)


def _magnitude(unit: str | None) -> int:
    if not unit:
        return 1
    value = 1
    for ch in unit:
        value *= _SMALL_UNITS.get(ch, 1) * _BIG_UNITS.get(ch, 1)
    return value


def numbers(text: str) -> set[float]:
    """Numeric values with Korean magnitudes resolved, so '100만원' == '100만 원' and '2000억' == '2천억'."""
    values: set[float] = set()
    for match in _DIGIT_NUMBER.finditer(text):
        base = float(match.group(1).replace(",", ""))
        values.add(round(base * _magnitude(match.group(2)), 6))
    for match in _NATIVE_NUMBER.finditer(text):
        values.add(float(_NATIVE_NUMBERS[match.group(1)]))
    return values


# Sentence endings ------------------------------------------------------------

REGISTERS = ("haeyo", "hapsho", "haera", "banmal", "none")


def sentence_register(sentence: str) -> str:
    """Speech level of a sentence from its final ending (EF)."""
    endings = [form for form, tag in analyze(sentence) if tag == "EF"]
    if not endings:
        return "none"
    ending = endings[-1]
    if ending.endswith(("요", "죠")):
        return "haeyo"
    if ending.endswith(("니다", "니까")):
        return "hapsho"
    if ending.endswith("다") or ending in {"라", "냐", "자"}:
        return "haera"
    return "banmal"


_FINAL_PUNCT = tuple(".!?…。")
_TRAILING = "\"'”’」』)] \n\t"


def looks_truncated(text: str) -> bool:
    """True when the text stops mid-sentence: no final punctuation and no final ending."""
    stripped = text.rstrip(_TRAILING)
    if not stripped:
        return True
    if stripped.endswith(_FINAL_PUNCT):
        return False
    tokens = analyze(stripped)
    return not tokens or tokens[-1][1] != "EF"
