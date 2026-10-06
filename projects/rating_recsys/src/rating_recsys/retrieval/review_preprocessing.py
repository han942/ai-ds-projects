"""Versioned optional BM25 cleanup; preserve food, service and price terms."""

from __future__ import annotations

import html
import re
import unicodedata


PREPROCESSING_VERSIONS = {
    "baseline": "positive-recent-kiwi-content-v1",
    "clean": "positive-recent-kiwi-clean-v1",
    "clean_stopwords": "positive-recent-kiwi-clean-stopwords-v1",
}
# Fixed before evaluation. Apply to Kiwi lemmas, not substrings of food names.
# These are common praise/visit expressions; retain taste, menu and service words.
REVIEW_STOPWORDS = frozenset((
    "여기", "거기", "저기", "오늘", "이번", "다음", "방문", "재방문",
    "곳", "집", "번", "것", "수", "맛있", "맛나", "좋", "최고", "짱", "굿", "추천",
))
URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_TAG = re.compile(r"<[^>]*>")
LAUGHTER = re.compile(r"[ㅋㅎㅠㅜ]{2,}")
EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\ufe0f\u200d]")


def clean_review_profile(text: str) -> str:
    """Clean the already selected, truncated profile without fetching more text."""
    text = html.unescape(text)
    text = URL.sub(" ", text)
    text = HTML_TAG.sub(" ", text)
    text = LAUGHTER.sub(" ", text)
    text = EMOJI.sub(" ", text)
    return unicodedata.normalize("NFKC", text)
