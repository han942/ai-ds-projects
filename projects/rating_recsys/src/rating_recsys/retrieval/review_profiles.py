"""Shared positive-review selection for cutoff-specific text retrieval."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Protocol, Sequence

from rating_recsys.datasets.models import Interaction


class ProfileConfig(Protocol):
    min_rating: float
    max_review_chars: int
    max_user_reviews: int
    max_item_reviews: int


def _selected_reviews(rows: Sequence[Interaction], texts: Mapping[int, str | None],
                      limit: int, config: ProfileConfig) -> list[tuple[Interaction, str]]:
    parts = []
    for row in sorted(rows, key=lambda row: (row.event_date, row.review_id), reverse=True):
        if row.rating < config.min_rating:
            continue
        text = " ".join((texts[row.review_id] or "").split())[:config.max_review_chars]
        if text:
            parts.append((row, text))
        if len(parts) == limit:
            break
    return parts


def _document(rows, texts, limit, config) -> str:
    return "\n".join(text for _, text in _selected_reviews(rows, texts, limit, config))


def build_profiles(history: Sequence[Interaction], texts: Mapping[int, str | None],
                   query_users: Sequence[int], config: ProfileConfig
                   ) -> tuple[dict[int, str], dict[int, str]]:
    """Use only supplied history, never inspect other rows in the text mapping."""
    by_user: dict[int, list[Interaction]] = defaultdict(list)
    by_item: dict[int, list[Interaction]] = defaultdict(list)
    users = set(query_users)
    for row in history:
        if row.user_id in users:
            by_user[row.user_id].append(row)
        by_item[row.restaurant_id].append(row)
    user_docs = {
        uid: doc for uid, rows in by_user.items()
        if (doc := _document(rows, texts, config.max_user_reviews, config))
    }
    item_docs = {
        iid: doc for iid, rows in by_item.items()
        if (doc := _document(rows, texts, config.max_item_reviews, config))
    }
    return user_docs, item_docs
