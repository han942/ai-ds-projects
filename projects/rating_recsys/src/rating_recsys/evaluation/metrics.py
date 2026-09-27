"""Ranking metrics for fixed-cutoff queries with one or more positives.

Accuracy metrics (Recall, Precision, NDCG, MAP, MRR @K) average only queries
with at least one relevant positive. Coverage, novelty and region diversity
describe the recommendation lists themselves and use every query.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence


ACCURACY_METRICS = ("recall", "precision", "ndcg", "map", "mrr")


@dataclass(frozen=True, slots=True)
class RankingObservation:
    """One query's ordered list and the graded relevance of its window visits.

    ``relevance_by_item`` holds 0/1/2 for every restaurant visited in the window.
    Relevance 0 (a low rating) counts as non-relevant, like an unvisited item.
    """

    query_id: str
    user_id: int
    relevance_by_item: dict[int, int]
    ordered_restaurant_ids: tuple[int, ...]

    @property
    def relevant_items(self) -> dict[int, int]:
        return {
            restaurant_id: relevance
            for restaurant_id, relevance in self.relevance_by_item.items()
            if relevance > 0
        }


def query_scores(observation: RankingObservation, cutoff: int) -> dict[str, float]:
    """Per-query accuracy values; used for averages and the paired bootstrap."""

    relevant = observation.relevant_items
    if not relevant:
        raise ValueError("A query needs at least one relevant item to be scored")
    hits = 0
    precision_sum = 0.0
    dcg = 0.0
    first_rank: int | None = None
    for rank, restaurant_id in enumerate(
        observation.ordered_restaurant_ids[:cutoff], start=1
    ):
        relevance = relevant.get(restaurant_id, 0)
        if relevance <= 0:
            continue
        hits += 1
        precision_sum += hits / rank
        dcg += ((2**relevance) - 1) / math.log2(rank + 1)
        if first_rank is None:
            first_rank = rank
    ideal = sorted(relevant.values(), reverse=True)[:cutoff]
    ideal_dcg = sum(
        ((2**relevance) - 1) / math.log2(rank + 1)
        for rank, relevance in enumerate(ideal, start=1)
    )
    return {
        "recall": hits / len(relevant),
        "precision": hits / cutoff,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
        "map": precision_sum / min(cutoff, len(relevant)),
        "mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def evaluate_rankings(
    observations: Iterable[RankingObservation],
    *,
    cutoffs: Sequence[int],
    catalog_ids: Iterable[int],
    item_popularity: dict[int, int] | None = None,
    item_regions: dict[int, str] | None = None,
) -> dict[str, object]:
    all_observations = tuple(observations)
    relevant = tuple(item for item in all_observations if item.relevant_items)
    catalog = set(catalog_ids)
    metrics: dict[str, object] = {
        "queries": len(all_observations),
        "evaluated_queries": len(relevant),
        "mean_relevant_items": _mean_or_zero(
            len(item.relevant_items) for item in relevant
        ),
    }
    for cutoff in cutoffs:
        scores = [query_scores(item, cutoff) for item in relevant]
        for name in ACCURACY_METRICS:
            metrics[f"{name}_at_{cutoff}"] = _mean_or_zero(s[name] for s in scores)
        metrics[f"catalog_coverage_at_{cutoff}"] = _catalog_coverage(
            all_observations, cutoff, catalog
        )
        if item_popularity is not None:
            metrics[f"novelty_at_{cutoff}"] = _novelty(
                all_observations, cutoff, item_popularity
            )
        if item_regions is not None:
            metrics[f"intra_list_region_diversity_at_{cutoff}"] = _region_diversity(
                all_observations, cutoff, item_regions
            )
    return metrics


def _catalog_coverage(
    observations: Sequence[RankingObservation],
    cutoff: int,
    catalog: set[int],
) -> float:
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
    return _mean_or_zero(
        -math.log2(max(item_popularity.get(restaurant_id, 0), 1) / total)
        for item in observations
        for restaurant_id in item.ordered_restaurant_ids[:cutoff]
    )


def _region_diversity(
    observations: Sequence[RankingObservation],
    cutoff: int,
    item_regions: dict[int, str],
) -> float:
    query_scores_: list[float] = []
    for observation in observations:
        items = observation.ordered_restaurant_ids[:cutoff]
        pairs = [
            (left, right)
            for index, left in enumerate(items)
            for right in items[index + 1 :]
        ]
        if not pairs:
            continue
        query_scores_.append(
            sum(item_regions.get(left) != item_regions.get(right) for left, right in pairs)
            / len(pairs)
        )
    return _mean_or_zero(query_scores_)


def _mean_or_zero(values: Iterable[float]) -> float:
    collected = tuple(values)
    return mean(collected) if collected else 0.0
