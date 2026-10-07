"""Deterministic source sentences for review-aspect evidence selection.

Offsets refer to the rating-redacted extraction text, not the raw review.
This verifies provenance, not whether a sentence supports an interpretation.
"""
from __future__ import annotations

import re


SENTENCE_VERSION = "source-sentences-v1"
# Keep decimals, punctuation without following whitespace and contrast clauses
# intact. Newlines also delimit fragments. No linguistic segmentation claim.
BOUNDARY = re.compile(r"[.!?。！？]+(?=\s|$)|\r\n|[\r\n]")
REDACTED_RATING = re.compile(re.escape("[별점표현 제거]"))


def source_sentences(text):
    # Isolate rating markers, so rejecting them does not discard the adjacent
    # opinion in a review without punctuation ("5점 맛은 좋다").
    boundaries = sorted({match.end() for match in BOUNDARY.finditer(text)} |
                        {offset for match in REDACTED_RATING.finditer(text)
                         for offset in (match.start(), match.end())})
    sentences = []
    start = 0
    for end in [*boundaries, len(text)]:
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            sentences.append({"sentence_id": len(sentences) + 1,
                              "start": left, "end": right, "text": text[left:right]})
        start = end
    return sentences


def extraction_inputs(rows):
    return [{"review_id": row["review_id"], "text": row["extraction_text"],
             "sentences": source_sentences(row["extraction_text"])} for row in rows]


def model_inputs(inputs):
    """Only review IDs and numbered source text cross the model boundary."""
    return [{"review_id": row["review_id"],
             "sentences": [{"sentence_id": s["sentence_id"], "text": s["text"]}
                           for s in row["sentences"]]} for row in inputs]


def resolve_evidence(sentence_ids, source):
    if (not isinstance(sentence_ids, list) or not 1 <= len(sentence_ids) <= 2
            or any(type(i) is not int for i in sentence_ids)
            or len(set(sentence_ids)) != len(sentence_ids)):
        raise ValueError("Evidence requires one or two distinct integer sentence IDs")
    sentences = {s["sentence_id"]: s for s in source["sentences"]}
    quotes = []
    for sentence_id in sentence_ids:
        if sentence_id not in sentences:
            raise ValueError("Evidence sentence ID is outside this review")
        sentence = sentences[sentence_id]
        quote = source["text"][sentence["start"]:sentence["end"]]
        if not quote or quote != sentence["text"] or "[별점표현 제거]" in quote:
            raise ValueError("Evidence sentence is invalid or contains a redacted rating")
        quotes.append(quote)
    return quotes
