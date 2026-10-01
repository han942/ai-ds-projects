"""Leakage-aware query builders for training and evaluation."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from typing import Iterable

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.models import RecommendationQuery, WindowQuery


def interaction_key(item: Interaction) -> tuple[object, int, int]:
    return (item.event_date, item.review_id, item.restaurant_id)


def global_interaction_key(item: Interaction) -> tuple[object, int, int, int]:
    return (item.event_date, item.review_id, item.user_id, item.restaurant_id)


def build_prefix_queries(
    interactions: Iterable[Interaction],
    *,
    config: ExperimentConfig,
    phase: str = "train",
) -> tuple[RecommendationQuery, ...]:
    """Ranker training queries: each non-first visit with its preceding history."""

    grouped: dict[int, list[Interaction]] = defaultdict(list)
    for item in interactions:
        grouped[item.user_id].append(item)

    queries: list[RecommendationQuery] = []
    for user_id in sorted(grouped):
        history = sorted(grouped[user_id], key=interaction_key)
        for index in range(1, len(history)):
            target = history[index]
            queries.append(
                RecommendationQuery(
                    query_id=f"{phase}:u{user_id}:r{target.review_id}",
                    phase=phase,
                    user_id=user_id,
                    cutoff=target.event_date,
                    history=tuple(history[:index]),
                    target=target,
                    relevance=config.relevance(target.rating),
                )
            )
    return tuple(sorted(queries, key=lambda query: query.query_id))


def build_window_queries(
    history_interactions: Iterable[Interaction],
    window_interactions: Iterable[Interaction],
    *,
    config: ExperimentConfig,
    phase: str,
    cutoff: date,
) -> tuple[tuple[WindowQuery, ...], int]:
    """Evaluation queries for one window: one per user with history at ``cutoff``.

    Returns the queries and the number of window users without any history
    (new users). New users cannot be personalised and are not evaluated.
    """

    history: dict[int, list[Interaction]] = defaultdict(list)
    for item in history_interactions:
        if item.event_date > cutoff:
            raise ValueError("History interaction is after the window cutoff")
        history[item.user_id].append(item)
    window: dict[int, list[Interaction]] = defaultdict(list)
    for item in window_interactions:
        if item.event_date <= cutoff:
            raise ValueError("Window interaction is not after the cutoff")
        window[item.user_id].append(item)

    queries: list[WindowQuery] = []
    new_users = 0
    for user_id in sorted(window):
        user_history = tuple(sorted(history.get(user_id, ()), key=interaction_key))
        if not user_history:
            new_users += 1
            continue
        visits = tuple(sorted(window[user_id], key=interaction_key))
        queries.append(
            WindowQuery(
                query_id=f"{phase}:u{user_id}",
                phase=phase,
                user_id=user_id,
                cutoff=cutoff,
                history=user_history,
                window=visits,
                relevance_by_item={
                    item.restaurant_id: config.relevance(item.rating) for item in visits
                },
            )
        )
    return tuple(queries), new_users


def build_training_windows(
    interactions: Iterable[Interaction],
    *,
    config: ExperimentConfig,
    through: date,
    phase: str = "train",
) -> tuple[tuple[date, tuple[WindowQuery, ...]], ...]:
    """Non-overlapping calendar windows with frozen history before each start.

    The final partial window stops at ``through``. A final refit must rebuild
    these windows through T2 instead of appending a second copy of the partial
    T1 window. Targets never enter their own retrieval context or features.
    """
    months = config.lightgcn_checkpoint_months
    rows = sorted((row for row in interactions if row.event_date <= through), key=global_interaction_key)
    if not rows:
        return ()
    first = rows[0].event_date
    start = date(first.year, (first.month - 1) // months * months + 1, 1)
    history: list[Interaction] = []
    result = []
    index = 0
    while start <= through:
        next_month = start.year * 12 + start.month - 1 + months
        next_start = date(next_month // 12, next_month % 12 + 1, 1)
        visits = []
        while index < len(rows) and rows[index].event_date < next_start:
            visits.append(rows[index])
            index += 1
        queries, _ = build_window_queries(
            history, visits, config=config,
            phase=f"{phase}:{start.isoformat()}", cutoff=start - timedelta(days=1),
        )
        if queries:
            result.append((start, queries))
        history.extend(visits)
        start = next_start
    return tuple(result)
