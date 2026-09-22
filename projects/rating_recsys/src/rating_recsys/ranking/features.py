"""Cutoff-safe feature construction for the baseline ranker."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from rating_recsys.experiments.models import Candidate, FeatureRow, RecommendationQuery
from rating_recsys.retrieval.baselines import (
    ITEM_ITEM,
    POPULARITY,
    RetrievalContext,
)


FEATURE_NAMES = (
    "popularity_score",
    "item_average_rating",
    "item_item_sum_similarity",
    "item_item_max_similarity",
    "popularity_source_present",
    "item_item_source_present",
    "inverse_popularity_rank",
    "inverse_item_item_rank",
    "rrf_score",
    "candidate_rank_inverse",
    "user_history_length",
    "user_average_rating",
    "region_affinity",
)


def feature_values(
    query: RecommendationQuery,
    candidate: Candidate,
    context: RetrievalContext,
) -> dict[str, float]:
    region_counts = Counter(item.region for item in query.history)
    history_length = len(query.history)
    average_rating = (
        sum(item.rating for item in query.history) / history_length
        if history_length
        else 0.0
    )
    return {
        "popularity_score": candidate.source_scores.get(POPULARITY, 0.0),
        "item_average_rating": context.average_rating(candidate.restaurant_id),
        "item_item_sum_similarity": candidate.source_scores.get(ITEM_ITEM, 0.0),
        "item_item_max_similarity": candidate.source_scores.get(
            "item_item_max", 0.0
        ),
        "popularity_source_present": float(POPULARITY in candidate.candidate_sources),
        "item_item_source_present": float(ITEM_ITEM in candidate.candidate_sources),
        "inverse_popularity_rank": _inverse_rank(
            candidate.source_ranks.get(POPULARITY)
        ),
        "inverse_item_item_rank": _inverse_rank(
            candidate.source_ranks.get(ITEM_ITEM)
        ),
        "rrf_score": candidate.rrf_score,
        "candidate_rank_inverse": 1.0 / candidate.candidate_rank,
        "user_history_length": float(history_length),
        "user_average_rating": average_rating,
        "region_affinity": (
            region_counts.get(candidate.region, 0) / history_length
            if history_length
            else 0.0
        ),
    }


def build_feature_rows(
    query: RecommendationQuery,
    candidates: Iterable[Candidate],
    context: RetrievalContext,
) -> tuple[FeatureRow, ...]:
    return tuple(
        FeatureRow(
            query_id=query.query_id,
            phase=query.phase,
            user_id=query.user_id,
            restaurant_id=candidate.restaurant_id,
            target_restaurant_id=query.target.restaurant_id,
            relevance=(
                query.relevance
                if candidate.restaurant_id == query.target.restaurant_id
                else 0
            ),
            history_depth=query.history_depth,
            candidate=candidate,
            features=feature_values(query, candidate, context),
        )
        for candidate in candidates
    )


def _inverse_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0
