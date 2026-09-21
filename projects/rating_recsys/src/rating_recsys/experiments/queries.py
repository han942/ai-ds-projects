"""Build leakage-aware offline recommendation queries."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.models import RecommendationQuery


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
    """Use each non-first interaction as a target with its preceding history."""

    grouped: dict[int, list[Interaction]] = defaultdict(list)
    for item in interactions:
        grouped[item.user_id].append(item)

    queries: list[RecommendationQuery] = []
    for user_id in sorted(grouped):
        history = sorted(grouped[user_id], key=interaction_key)
        for index in range(1, len(history)):
            target = history[index]
            prefix = tuple(history[:index])
            queries.append(
                RecommendationQuery(
                    query_id=f"{phase}:u{user_id}:r{target.review_id}",
                    phase=phase,
                    user_id=user_id,
                    cutoff=target.event_date,
                    history=prefix,
                    target=target,
                    relevance=config.relevance(target.rating),
                )
            )
    return tuple(sorted(queries, key=lambda query: query.query_id))


def build_holdout_queries(
    history_interactions: Iterable[Interaction],
    targets: Iterable[Interaction],
    *,
    config: ExperimentConfig,
    phase: str,
) -> tuple[RecommendationQuery, ...]:
    """Build one validation or test query for every held-out interaction."""

    grouped: dict[int, list[Interaction]] = defaultdict(list)
    for item in history_interactions:
        grouped[item.user_id].append(item)
    for user_history in grouped.values():
        user_history.sort(key=interaction_key)

    queries = [
        RecommendationQuery(
            query_id=f"{phase}:u{target.user_id}:r{target.review_id}",
            phase=phase,
            user_id=target.user_id,
            cutoff=target.event_date,
            history=tuple(
                item
                for item in grouped.get(target.user_id, ())
                if interaction_key(item) < interaction_key(target)
            ),
            target=target,
            relevance=config.relevance(target.rating),
        )
        for target in targets
    ]
    return tuple(sorted(queries, key=lambda query: query.query_id))


def interactions_before(
    interactions: Iterable[Interaction],
    query: RecommendationQuery,
) -> tuple[Interaction, ...]:
    """Return only interactions ordered before a query target."""

    cutoff_key = global_interaction_key(query.target)
    return tuple(
        sorted(
            (
                item
                for item in interactions
                if global_interaction_key(item) < cutoff_key
                and item.review_id != query.target.review_id
            ),
            key=global_interaction_key,
        )
    )
