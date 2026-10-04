"""Cutoff-safe feature construction for the Stage 2 ranker."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

from rating_recsys.experiments.models import Candidate, FeatureRow, RecommendationQuery
from rating_recsys.retrieval.baselines import (
    ITEM_ITEM,
    POPULARITY,
    RetrievalContext,
)


# Same value as ``retrieval.lightgcn.LIGHTGCN`` (not imported to keep this
# module free of NumPy/SciPy for config validation).
LIGHTGCN = "lightgcn"
# LightGCN score standardised over one query's Stage 1 candidates. Raw dot
# products are not comparable across models refit for different periods.
LIGHTGCN_Z = "lightgcn_z"

# ``region_affinity`` must stay last: NO_REGION_FEATURE_NAMES drops it.
FEATURE_NAMES = (
    "popularity_score",
    "item_average_rating",
    "item_item_sum_similarity",
    "item_item_max_similarity",
    "popularity_source_present",
    "item_item_source_present",
    "inverse_popularity_rank",
    "inverse_item_item_rank",
    "lightgcn_score_z",
    "lightgcn_source_present",
    "inverse_lightgcn_rank",
    "rrf_score",
    "candidate_rank_inverse",
    "user_history_length",
    "user_average_rating",
    "region_affinity",
)
NO_REGION_FEATURE_NAMES = FEATURE_NAMES[:-1]


def shrink_mean(total: float, count: int, prior: float, strength: float) -> float:
    """Regularize a past-only average toward a past-only global prior."""
    if strength < 0 or not math.isfinite(strength):
        raise ValueError("rating shrinkage strength must be finite and nonnegative")
    denominator = count + strength
    return (total + strength * prior) / denominator if denominator else 0.0


def global_rating_prior(context: RetrievalContext) -> float:
    count = sum(context.item_counts.values())
    return sum(context.item_rating_sum.values()) / count if count else 0.0


def feature_values(
    query: RecommendationQuery,
    candidate: Candidate,
    context: RetrievalContext,
    *,
    include_region: bool = True,
    rating_shrinkage_strength: float = 0.0,
    rating_prior: float | None = None,
) -> dict[str, float]:
    """Features of one candidate, all from data before the query.

    ``*_source_present`` and ``inverse_*_rank`` describe the candidate's
    position in each source's Top-k list (C0 popularity, C1 item-item, C4
    LightGCN), whether or not that source is fused into the Stage 1 output.
    """

    history_length = len(query.history)
    average_rating = (
        sum(item.rating for item in query.history) / history_length
        if history_length
        else 0.0
    )
    item_average = context.average_rating(candidate.restaurant_id)
    if rating_shrinkage_strength:
        prior = global_rating_prior(context) if rating_prior is None else rating_prior
        average_rating = shrink_mean(
            sum(item.rating for item in query.history), history_length,
            prior, rating_shrinkage_strength,
        )
        item_average = shrink_mean(
            context.item_rating_sum.get(candidate.restaurant_id, 0.0),
            context.item_counts.get(candidate.restaurant_id, 0),
            prior, rating_shrinkage_strength,
        )
    ranks = candidate.source_ranks
    values = {
        "popularity_score": candidate.source_scores.get(POPULARITY, 0.0),
        "item_average_rating": item_average,
        "item_item_sum_similarity": candidate.source_scores.get(ITEM_ITEM, 0.0),
        "item_item_max_similarity": candidate.source_scores.get(
            "item_item_max", 0.0
        ),
        "popularity_source_present": float(POPULARITY in ranks),
        "item_item_source_present": float(ITEM_ITEM in ranks),
        "inverse_popularity_rank": _inverse_rank(ranks.get(POPULARITY)),
        "inverse_item_item_rank": _inverse_rank(ranks.get(ITEM_ITEM)),
        "lightgcn_score_z": candidate.source_scores.get(LIGHTGCN_Z, 0.0),
        "lightgcn_source_present": float(LIGHTGCN in ranks),
        "inverse_lightgcn_rank": _inverse_rank(ranks.get(LIGHTGCN)),
        "rrf_score": candidate.rrf_score,
        "candidate_rank_inverse": 1.0 / candidate.candidate_rank,
        "user_history_length": float(history_length),
        "user_average_rating": average_rating,
    }
    if include_region:
        region_counts = Counter(item.region for item in query.history)
        values["region_affinity"] = (
            region_counts.get(candidate.region, 0) / history_length
            if history_length
            else 0.0
        )
    return values


def build_feature_rows(
    query: RecommendationQuery,
    candidates: Iterable[Candidate],
    context: RetrievalContext,
    *,
    include_region: bool = True,
    rating_shrinkage_strength: float = 0.0,
) -> tuple[FeatureRow, ...]:
    prior = global_rating_prior(context) if rating_shrinkage_strength else None
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
            candidate=candidate,
            features=feature_values(
                query, candidate, context, include_region=include_region,
                rating_shrinkage_strength=rating_shrinkage_strength, rating_prior=prior,
            ),
        )
        for candidate in candidates
    )


def _inverse_rank(rank: int | None) -> float:
    return 1.0 / rank if rank is not None else 0.0
