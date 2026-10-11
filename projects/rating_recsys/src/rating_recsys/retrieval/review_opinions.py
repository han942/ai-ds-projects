"""Conservative Korean opinion rules with raw-text evidence coordinates.

This is a lexical experiment, not an ABSA model or LLM extraction. It never
accepts ratings. A review may retain several targets and opposite opinions;
unmentioned categories remain absent. Explicit negation is kept with its
predicate, and absence of a bad quality is not fabricated as praise.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


CATEGORIES = ("taste", "value", "service", "atmosphere", "portion")
VERSION = "raw-multi-opinion-rules-v2"
POLARITIES = ("positive", "negative", "neutral", "unknown")

# These boundaries retain their connective in the preceding evidence. In
# particular, negation in "친절하지 않지만" remains in that same clause.
_BOUNDARY = re.compile(
    r"[.!?。！？;；,，\r\n]+|지만|는데|은데|으나|"
    r"(?:었고|았고|였고|했고|됐고|졌고|겼고|있고|없고|럽고|하고|좋고|많고|적고|"
    r"작고|크고|길고|짧고|비싸고|질기고)|"
    r"\s+(?:하지만|반면(?:에)?|다만|그런데)\s+"
)

_TARGETS = {
    "taste": re.compile(
        r"(?:돈까스|돈가스|쌀국수|짜장면|탕수육|떡볶이|스테이크|햄버거|"
        r"파스타|냉면|라면|짬뽕|국물|육수|고기|식감|신선도|소스|양념|"
        r"피자|초밥|생선|김밥|김치|커피|디저트|음식|요리|국수|카레|매운맛|매운|맵|"
        r"면(?=[은는이가을도\s]|$)|밥(?=[은는이가을도\s]|$)|"
        r"맛(?=[은이가도을\s]|$))"
    ),
    "value": re.compile(r"가격|가성비|금액|비용|요금|값(?=[은는이가을도\s]|$)"),
    "service": re.compile(r"대기\s*시간|직원(?:분)?|종업원|점원|사장(?:님)?|응대|웨이팅|대기|서비스"),
    "atmosphere": re.compile(r"분위기|인테리어|화장실|테이블|좌석|소음|공간|청결|위생|조명|매장|내부"),
    "portion": re.compile(r"분량|포션|양(?=[은는이가을도\s]|많|적|부족|$)"),
}

# Predicate-only category assignment is permitted only when the word itself
# identifies that category. Its target is explicitly marked implicit rather
# than represented as an invented source quote.
_IMPLICIT_LABELS = {"taste": "맛", "value": "가격", "service": "응대",
                    "atmosphere": "청결·소음", "portion": "양"}


@dataclass(frozen=True)
class _Rule:
    name: str
    pattern: re.Pattern
    polarity: str
    category: str | None = None
    # Positive lexical predicates can become negative under explicit negation.
    # A negated negative/neutral predicate is unknown/neutral, not automatic praise.
    negation: bool = True


_RULES = (
    _Rule("taste-delicious", re.compile(r"맛\s*(?:이\s*)?있"), "positive", "taste"),
    _Rule("taste-not-delicious", re.compile(r"맛\s*(?:이\s*)?없"), "negative", "taste"),
    _Rule("taste-tough", re.compile(r"질기|질겼|질겨"), "negative", "taste"),
    _Rule("taste-rancid", re.compile(r"비리|비렸|비려|누린내|잡내"), "negative", "taste"),
    _Rule("taste-fresh", re.compile(r"신선하|신선했"), "positive", "taste"),
    _Rule("taste-spicy-fact", re.compile(r"맵|매운|매워|매웠"), "neutral", "taste"),
    _Rule("taste-salty-fact", re.compile(r"짜다|짜요|짠|짰"), "neutral", "taste"),
    _Rule("value-expensive", re.compile(r"비싸|비쌌"), "negative", "value"),
    _Rule("value-affordable", re.compile(r"저렴하|저렴했|합리적"), "positive", "value"),
    _Rule("service-unfriendly", re.compile(r"불친절하|불친절했|불친절"), "negative", "service"),
    _Rule("service-friendly", re.compile(r"(?<!불)친절하|(?<!불)친절했|(?<!불)친절"), "positive", "service"),
    _Rule("service-rude", re.compile(r"무례하|무례했"), "negative", "service"),
    _Rule("atmosphere-noisy", re.compile(r"시끄럽|시끄러|지저분하|지저분했|더럽|더러웠"), "negative", "atmosphere"),
    _Rule("atmosphere-clean", re.compile(r"깨끗하|깨끗했"), "positive", "atmosphere"),
    _Rule("atmosphere-quiet-fact", re.compile(r"조용하|조용했|조용"), "neutral", "atmosphere"),
    _Rule("portion-plentiful-fact", re.compile(r"푸짐하|푸짐했|푸짐"), "neutral", "portion"),
    _Rule("money-amount-fact", re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:만\s*원|천\s*원|원)"), "neutral", "value", False),
    _Rule("positive-judgement", re.compile(r"좋(?=[다았아은지고네요음습])|만족하|만족했|만족스|훌륭하|훌륭했"), "positive"),
    _Rule("negative-judgement", re.compile(r"나쁘|나빴|아쉽|아쉬웠|실망하|실망했|불쾌하|불쾌했|불만족|별로"), "negative"),
    _Rule("quantity-fact", re.compile(r"많[다아요았은]|적[다어요었은]|부족하|부족했"), "neutral"),
    _Rule("length-fact", re.compile(r"길[다어요었은]|오래|짧[다아요았은]"), "neutral"),
    _Rule("uncertain-judgement", re.compile(r"애매하|애매했|모르겠|고민되"), "unknown"),
)

_NEG_BEFORE = re.compile(r"(?:안|못)\s*$")
_NEG_ENDING = r"(?:하|했|았|었)?(?:습니다|는다|아요|어요|에요|네요|지만|는데|다|음|아|어|고|라)?"
# These patterns must start immediately after this predicate. A later negative
# about another food, service or quantity cannot invert its sentiment. Qualified
# denial ("좋은 편은 아니다") and cognitive denial are kept unknown: rejecting
# praise does not by itself establish dissatisfaction.
_NEGATION_PATTERNS = (
    ("preference", re.compile(r"^아하(?:지는?|진|지)\s*(?:않|못)" + _NEG_ENDING)),
    ("cognitive", re.compile(
        r"^(?:다고|다는)?\s*(?:느껴지|느끼|생각하|생각되|생각(?:은|을)?\s*하)"
        r"(?:지는?|진|지)\s*(?:않|못)" + _NEG_ENDING)),
    ("qualified", re.compile(
        r"^(?:아하는|다는|는|은|다)?\s*(?:느낌|편|것|건|정도|수준)(?:은|는|이)?\s*아니"
        + _NEG_ENDING)),
    ("qualified", re.compile(r"^(?:다|다는)?\s*까지(?:는)?\s*아니" + _NEG_ENDING)),
    ("direct", re.compile(r"^(?:하)?(?:지는?|진|지)\s*(?:않|못)" + _NEG_ENDING)),
    ("direct", re.compile(r"^(?:다는|는|은)?\s*(?:건|것은|것이)?\s*아니" + _NEG_ENDING)),
)
_PREDICATE_END = re.compile(
    r"^(?:해서는|해서도|해서|아서|어서|었|았|다|아요|어요|요|네요|은|는|고|지만|는데|은데)*"
)
_UNCERTAINTY = re.compile(r"모르겠|애매|을\s*수도|을\s*것\s*같|다면|소문|다는\s*말")


def _clauses(text):
    start = 0
    for end in [match.end() for match in _BOUNDARY.finditer(text)] + [len(text)]:
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right-1].isspace():
            right -= 1
        if left < right:
            yield left, right
        start = end


def _scope(text, match, rule):
    """Return predicate coordinates and whether an explicit local negator occurs."""
    left, right = match.span()
    before = _NEG_BEFORE.search(text[:left]) if rule.negation else None
    after, after_kind = None, None
    if rule.negation:
        for kind, pattern in _NEGATION_PATTERNS:
            after = pattern.match(text[right:])
            if after is not None:
                after_kind = kind
                break
    if before:
        left = before.start()
    if after:
        right += after.end()
    else:
        right += _PREDICATE_END.match(text[right:]).end()
    negations = int(before is not None) + int(after is not None)
    polarity = rule.polarity
    if negations:
        if negations > 1:
            polarity = "unknown"
        elif after_kind in ("qualified", "cognitive"):
            polarity = "unknown"
        elif polarity == "positive":
            polarity = "negative"
        elif polarity == "negative":
            polarity = "unknown"
        # Objective absence of a neutral attribute remains an objective statement.
    return left, right, polarity, negations, after_kind


def extract_opinions(text):
    """Return deterministic target/opinion records, grounded in the raw input.

    Coordinates are Python character offsets [start, end). ``target_kind``
    distinguishes quoted nouns from category labels inferred from an intrinsic
    predicate. Neutral and unknown are retained in output; callers can omit
    them from positive/negative model channels without calling them unmentioned.
    A missing category has no fabricated record or zero-valued sentiment.
    """
    if text is None:
        return []
    if not isinstance(text, str):
        raise TypeError("Review text must be a string or None")
    result = []
    for start, end in _clauses(text):
        clause = text[start:end]
        targets = [(category, match) for category, pattern in _TARGETS.items()
                   for match in pattern.finditer(clause)]
        candidates = []
        for rule in _RULES:
            for match in rule.pattern.finditer(clause):
                # A short predicate already inside a more specific predicate is
                # not another opinion ("불친절" vs "친절", "불만족" vs "만족").
                if any(a < match.start() < b for a, b in
                       ((m.start(), m.end()) for m in re.finditer(r"불친절|불만족", clause))):
                    continue
                left, right, polarity, negations, negation_kind = _scope(clause, match, rule)
                options = [(category, target) for category, target in targets
                           if (rule.category is None or category == rule.category)]
                preceding = [(category, target) for category, target in options if target.start() <= match.start()]
                # Prefer an explicit noun before the predicate. Stop generic
                # judgement from jumping backward across a different subject.
                if preceding:
                    category, target = max(preceding, key=lambda value: value[1].end())
                    intervening = [value for value in targets if target.end() <= value[1].start() < match.start()
                                   and value[0] != category]
                    if intervening:
                        continue
                elif rule.category is not None:
                    category, target = rule.category, None
                else:
                    # "좋아요" by itself identifies neither food nor service.
                    continue
                if rule.name == "quantity-fact":
                    if category != "portion":
                        continue
                    if clause[match.start():match.end()].startswith("부족"):
                        polarity = "unknown" if negations else "negative"
                if rule.name == "length-fact":
                    if category != "service" or target is None or not re.search(r"대기|웨이팅", target.group()):
                        continue
                    stem = clause[match.start():match.end()]
                    if stem.startswith(("길", "오래")):
                        polarity = "unknown" if negations else "negative"
                if polarity in ("positive", "negative") and _UNCERTAINTY.search(clause):
                    polarity = "unknown"
                # Price numbers are facts even when a different category's noun
                # occurs in the clause; the category itself remains price.
                target_text = target.group() if target is not None else _IMPLICIT_LABELS[category]
                trace = [rule.name, "target:explicit" if target is not None else "target:implicit-category"]
                if negations:
                    trace.append(f"local-negation:{negations}")
                if negation_kind is not None:
                    trace.append("negation-kind:"+negation_kind)
                if _UNCERTAINTY.search(clause):
                    trace.append("uncertain-or-attributed-clause")
                candidates.append({
                    "version": VERSION, "category": category, "target": target_text,
                    "target_kind": "explicit" if target is not None else "implicit_category",
                    "target_start": start+target.start() if target is not None else None,
                    "target_end": start+target.end() if target is not None else None,
                    "polarity": polarity, "opinion": text[start+left:start+right],
                    "opinion_start": start+left, "opinion_end": start+right,
                    "evidence": text[start:end], "start": start, "end": end,
                    "rule_trace": trace,
                })
        # A judgement about the same target supersedes an objective attribute
        # in that same clause ("양이 많아서 좋다", "맵지 않아 좋다"). It does
        # not remove distinct concrete targets or contradictory judgements.
        for record in candidates:
            if record["polarity"] == "neutral":
                same = [other for other in candidates if other["category"] == record["category"]
                        and other["target"] == record["target"]
                        and other["polarity"] in ("positive", "negative")]
                if same:
                    continue
            result.append(record)
    # Stable provenance-based deduplication: repeated source clauses at different
    # positions remain distinct rather than being silently given one weight.
    deduplicated = {}
    for record in sorted(result, key=lambda r: (r["start"], r["opinion_start"], CATEGORIES.index(r["category"]),
                                                r["opinion_end"], r["target"])):
        key = (record["start"], record["end"], record["category"], record["target"],
               record["opinion_start"], record["opinion_end"], record["polarity"])
        if key in deduplicated:
            deduplicated[key]["rule_trace"].extend(record["rule_trace"][:1])
        else:
            deduplicated[key] = record
    return list(deduplicated.values())
