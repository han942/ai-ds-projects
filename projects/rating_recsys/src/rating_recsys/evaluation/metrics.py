"""Stage-specific metrics for single-held-out-item recommendation queries."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence

from rating_recsys.experiments.models import Candidate, RankedCandidate, RecommendationQuery


@dataclass(frozen=True, slots=True)
class RankingObservation:
    query_id: str
    user_id: int
    target_restaurant_id: int
    relevance: int
    history_depth: str
    ordered_restaurant_ids: tuple[int, ...]


def candidate_observations(
    queries: Iterable[RecommendationQuery],
    candidates_by_query: dict[str, Sequence[Candidate]],
) -> tuple[RankingObservation, ...]:
    return tuple(
        RankingObservation(
            query_id=query.query_id,
            user_id=query.user_id,
            target_restaurant_id=query.target.restaurant_id,
            relevance=query.relevance,
            history_depth=query.history_depth,
            ordered_restaurant_ids=tuple(
                item.restaurant_id
                for item in candidates_by_query.get(query.query_id, ())
            ),
        )
        for query in queries
    )


def ranked_observations(
    queries: Iterable[RecommendationQuery],
    ranked: Iterable[RankedCandidate],
) -> tuple[RankingObservation, ...]:
    grouped: defaultdict[str, list[RankedCandidate]] = defaultdict(list)
    for item in ranked:
        grouped[item.row.query_id].append(item)
    return tuple(
        RankingObservation(
            query_id=query.query_id,
            user_id=query.user_id,
            target_restaurant_id=query.target.restaurant_id,
            relevance=query.relevance,
            history_depth=query.history_depth,
            ordered_restaurant_ids=tuple(
                item.row.restaurant_id
                for item in sorted(
                    grouped.get(query.query_id, ()), key=lambda row: row.final_rank
                )
            ),
        )
        for query in queries
    )


def evaluate_rankings(
    observations: Iterable[RankingObservation],
    *,
    cutoffs: Sequence[int],
    catalog_ids: Iterable[int],
    item_popularity: dict[int, int] | None = None,
    item_regions: dict[int, str] | None = None,
) -> dict[str, object]:
    all_observations = tuple(observations)
    relevant = tuple(item for item in all_observations if item.relevance > 0)
    zero_relevance = len(all_observations) - len(relevant)
    metrics: dict[str, object] = {
        "queries": len(all_observations),
        "evaluated_queries": len(relevant),
        "zero_relevance_queries": zero_relevance,
        "zero_relevance_query_rate": (
            zero_relevance / len(all_observations) if all_observations else 0.0
        ),
    }
    for cutoff in cutoffs:
        metrics[f"recall_at_{cutoff}"] = _mean_or_zero(
            _hit(item, cutoff) for item in relevant
        )
        metrics[f"hit_rate_at_{cutoff}"] = metrics[f"recall_at_{cutoff}"]
        metrics[f"ndcg_at_{cutoff}"] = _mean_or_zero(
            _ndcg(item, cutoff) for item in relevant
        )
        metrics[f"mrr_at_{cutoff}"] = _mean_or_zero(
            _reciprocal_rank(item, cutoff) for item in relevant
        )
        metrics[f"catalog_coverage_at_{cutoff}"] = _catalog_coverage(
            all_observations,
            cutoff,
            catalog_ids,
        )
        if item_popularity is not None:
            metrics[f"novelty_at_{cutoff}"] = _novelty(
                all_observations,
                cutoff,
                item_popularity,
            )
        if item_regions is not None:
            metrics[f"intra_list_region_diversity_at_{cutoff}"] = _region_diversity(
                all_observations,
                cutoff,
                item_regions,
            )
    metrics["history_breakdown"] = {
        bucket: _breakdown_metrics(
            tuple(item for item in relevant if item.history_depth == bucket),
            cutoffs,
        )
        for bucket in ("history_1_2", "history_3_plus")
    }
    return metrics


def source_contribution(
    queries: Iterable[RecommendationQuery],
    candidates_by_query: dict[str, Sequence[Candidate]],
) -> dict[str, dict[str, int]]:
    contribution: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {"retrieved_targets": 0, "unique_targets": 0, "candidates": 0}
    )
    for query in queries:
        if query.relevance <= 0:
            continue
        candidates = candidates_by_query.get(query.query_id, ())
        for candidate in candidates:
            for source in candidate.candidate_sources:
                contribution[source]["candidates"] += 1
            if candidate.restaurant_id != query.target.restaurant_id:
                continue
            for source in candidate.candidate_sources:
                contribution[source]["retrieved_targets"] += 1
            if len(candidate.candidate_sources) == 1:
                contribution[candidate.candidate_sources[0]]["unique_targets"] += 1
    return {key: value for key, value in sorted(contribution.items())}


def latency_summary(values_ms: Iterable[float]) -> dict[str, float]:
    values = sorted(float(value) for value in values_ms)
    if not values:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    return {
        "p50_ms": _percentile(values, 0.50),
        "p95_ms": _percentile(values, 0.95),
        "mean_ms": mean(values),
    }


def _breakdown_metrics(
    observations: Sequence[RankingObservation],
    cutoffs: Sequence[int],
) -> dict[str, float | int]:
    result: dict[str, float | int] = {"queries": len(observations)}
    for cutoff in cutoffs:
        result[f"recall_at_{cutoff}"] = _mean_or_zero(
            _hit(item, cutoff) for item in observations
        )
        result[f"ndcg_at_{cutoff}"] = _mean_or_zero(
            _ndcg(item, cutoff) for item in observations
        )
    return result


def _target_rank(item: RankingObservation, cutoff: int) -> int | None:
    try:
        rank = item.ordered_restaurant_ids[:cutoff].index(
            item.target_restaurant_id
        ) + 1
    except ValueError:
        return None
    return rank


def _hit(item: RankingObservation, cutoff: int) -> float:
    return float(_target_rank(item, cutoff) is not None)


def _reciprocal_rank(item: RankingObservation, cutoff: int) -> float:
    rank = _target_rank(item, cutoff)
    return 1.0 / rank if rank else 0.0


def _ndcg(item: RankingObservation, cutoff: int) -> float:
    rank = _target_rank(item, cutoff)
    if rank is None:
        return 0.0
    gain = (2**item.relevance) - 1
    dcg = gain / math.log2(rank + 1)
    ideal_dcg = gain
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _catalog_coverage(
    observations: Sequence[RankingObservation],
    cutoff: int,
    catalog_ids: Iterable[int],
) -> float:
    catalog = set(catalog_ids)
    if not catalog:
        return 0.0
    recommended = {
        restaurant_id
        for item in observations
        for restaurant_id in item.ordered_restaurant_ids[:cutoff]
    }
    return len(recommended & catalog) / len(catalog)


def _novelty(
    observations: Sequence[RankingObservation],
    cutoff: int,
    item_popularity: dict[int, int],
) -> float:
    total = sum(item_popularity.values())
    if total <= 0:
        return 0.0
    values = [
        -math.log2(max(item_popularity.get(restaurant_id, 0), 1) / total)
        for item in observations
        for restaurant_id in item.ordered_restaurant_ids[:cutoff]
    ]
    return _mean_or_zero(values)


def _region_diversity(
    observations: Sequence[RankingObservation],
    cutoff: int,
    item_regions: dict[int, str],
) -> float:
    query_scores: list[float] = []
    for observation in observations:
        items = observation.ordered_restaurant_ids[:cutoff]
        pairs = [
            (left, right)
            for index, left in enumerate(items)
            for right in items[index + 1 :]
        ]
        if not pairs:
            continue
        query_scores.append(
            sum(
                item_regions.get(left) != item_regions.get(right)
                for left, right in pairs
            )
            / len(pairs)
        )
    return _mean_or_zero(query_scores)


def _mean_or_zero(values: Iterable[float]) -> float:
    collected = tuple(values)
    return mean(collected) if collected else 0.0


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return values[lower]
    weight = index - lower
    return values[lower] * (1 - weight) + values[upper] * weight
