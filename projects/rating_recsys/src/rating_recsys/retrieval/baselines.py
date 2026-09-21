"""Deterministic popularity and item-item candidate baselines."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.models import Candidate, RecommendationQuery


POPULARITY = "popularity"
ITEM_ITEM = "item_item"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    popularity: tuple[Candidate, ...]
    item_item: tuple[Candidate, ...]
    union: tuple[Candidate, ...]
    eligible_catalog: frozenset[int]
    target_available: bool
    latency_ms: float


@dataclass(frozen=True, slots=True)
class RetrievalContext:
    item_counts: dict[int, int]
    item_rating_sum: dict[int, float]
    item_regions: dict[int, str]
    item_names: dict[int, str]
    item_frequencies: dict[int, int]
    cooccurrence: dict[tuple[int, int], int]

    def average_rating(self, restaurant_id: int) -> float:
        count = self.item_counts.get(restaurant_id, 0)
        return self.item_rating_sum.get(restaurant_id, 0.0) / count if count else 0.0

    def similarity(self, left: int, right: int) -> float:
        if left == right:
            return 0.0
        pair = (left, right) if left < right else (right, left)
        denominator = math.sqrt(
            self.item_frequencies.get(left, 0)
            * self.item_frequencies.get(right, 0)
        )
        return self.cooccurrence.get(pair, 0) / denominator if denominator else 0.0


def build_context(interactions: Iterable[Interaction]) -> RetrievalContext:
    item_counts: Counter[int] = Counter()
    item_rating_sum: defaultdict[int, float] = defaultdict(float)
    item_regions: dict[int, str] = {}
    item_names: dict[int, str] = {}
    items_by_user: defaultdict[int, set[int]] = defaultdict(set)

    for item in interactions:
        item_counts[item.restaurant_id] += 1
        item_rating_sum[item.restaurant_id] += item.rating
        item_regions[item.restaurant_id] = item.region
        item_names[item.restaurant_id] = item.restaurant_name
        items_by_user[item.user_id].add(item.restaurant_id)

    item_frequencies: Counter[int] = Counter()
    cooccurrence: Counter[tuple[int, int]] = Counter()
    for items in items_by_user.values():
        ordered = sorted(items)
        item_frequencies.update(ordered)
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1 :]:
                cooccurrence[(left, right)] += 1

    return RetrievalContext(
        item_counts=dict(item_counts),
        item_rating_sum=dict(item_rating_sum),
        item_regions=item_regions,
        item_names=item_names,
        item_frequencies=dict(item_frequencies),
        cooccurrence=dict(cooccurrence),
    )


class BaselineCandidateGenerator:
    """Generate source candidates and fuse them with reciprocal rank fusion."""

    def __init__(self, *, candidate_k: int = 100, rrf_constant: int = 60) -> None:
        if candidate_k < 1 or rrf_constant < 1:
            raise ValueError("candidate_k and rrf_constant must be positive")
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant

    def retrieve(
        self,
        query: RecommendationQuery,
        available_interactions: Iterable[Interaction],
    ) -> tuple[RetrievalResult, RetrievalContext]:
        started = time.perf_counter()
        context = build_context(available_interactions)
        seen = {item.restaurant_id for item in query.history}
        eligible = sorted(set(context.item_counts) - seen)

        popularity_scores = {
            restaurant_id: float(context.item_counts[restaurant_id])
            for restaurant_id in eligible
        }
        item_sum_scores: dict[int, float] = {}
        item_max_scores: dict[int, float] = {}
        for restaurant_id in eligible:
            similarities = [
                context.similarity(history_item.restaurant_id, restaurant_id)
                for history_item in query.history
            ]
            item_sum_scores[restaurant_id] = sum(similarities)
            item_max_scores[restaurant_id] = max(similarities, default=0.0)

        popularity_ids = _top_ids(popularity_scores, self.candidate_k)
        item_item_ids = _top_ids(
            {key: value for key, value in item_sum_scores.items() if value > 0},
            self.candidate_k,
        )
        popularity_ranks = {
            restaurant_id: index + 1
            for index, restaurant_id in enumerate(popularity_ids)
        }
        item_item_ranks = {
            restaurant_id: index + 1
            for index, restaurant_id in enumerate(item_item_ids)
        }

        popularity = tuple(
            self._candidate(
                query,
                restaurant_id,
                context,
                popularity_scores,
                item_sum_scores,
                item_max_scores,
                popularity_ranks,
                {},
                rank,
            )
            for rank, restaurant_id in enumerate(popularity_ids, start=1)
        )
        item_item = tuple(
            self._candidate(
                query,
                restaurant_id,
                context,
                popularity_scores,
                item_sum_scores,
                item_max_scores,
                {},
                item_item_ranks,
                rank,
            )
            for rank, restaurant_id in enumerate(item_item_ids, start=1)
        )

        union_ids = set(popularity_ids) | set(item_item_ids)
        rrf_scores = {
            restaurant_id: sum(
                1.0 / (self.rrf_constant + rank)
                for rank in (
                    popularity_ranks.get(restaurant_id),
                    item_item_ranks.get(restaurant_id),
                )
                if rank is not None
            )
            for restaurant_id in union_ids
        }
        fused_ids = _top_ids(rrf_scores, self.candidate_k)
        union = tuple(
            self._candidate(
                query,
                restaurant_id,
                context,
                popularity_scores,
                item_sum_scores,
                item_max_scores,
                popularity_ranks,
                item_item_ranks,
                rank,
                rrf_score=rrf_scores[restaurant_id],
            )
            for rank, restaurant_id in enumerate(fused_ids, start=1)
        )
        return (
            RetrievalResult(
                popularity=popularity,
                item_item=item_item,
                union=union,
                eligible_catalog=frozenset(eligible),
                target_available=query.target.restaurant_id in context.item_counts,
                latency_ms=(time.perf_counter() - started) * 1000,
            ),
            context,
        )

    def inject_target(
        self,
        query: RecommendationQuery,
        candidates: tuple[Candidate, ...],
        context: RetrievalContext,
    ) -> tuple[Candidate, ...]:
        """Inject an available positive into training rows, never evaluation."""

        target_id = query.target.restaurant_id
        if any(item.restaurant_id == target_id for item in candidates):
            return candidates
        if target_id not in context.item_counts:
            return candidates
        injected = Candidate(
            query_id=query.query_id,
            user_id=query.user_id,
            restaurant_id=target_id,
            restaurant_name=context.item_names[target_id],
            region=context.item_regions[target_id],
            candidate_sources=(),
            source_scores={
                POPULARITY: float(context.item_counts[target_id]),
                ITEM_ITEM: sum(
                    context.similarity(item.restaurant_id, target_id)
                    for item in query.history
                ),
                "item_item_max": max(
                    (
                        context.similarity(item.restaurant_id, target_id)
                        for item in query.history
                    ),
                    default=0.0,
                ),
            },
            source_ranks={},
            rrf_score=0.0,
            candidate_rank=len(candidates) + 1,
            injected_for_training=True,
        )
        return candidates + (injected,)

    def _candidate(
        self,
        query: RecommendationQuery,
        restaurant_id: int,
        context: RetrievalContext,
        popularity_scores: dict[int, float],
        item_sum_scores: dict[int, float],
        item_max_scores: dict[int, float],
        popularity_ranks: dict[int, int],
        item_item_ranks: dict[int, int],
        rank: int,
        *,
        rrf_score: float | None = None,
    ) -> Candidate:
        ranks = {
            source: source_rank
            for source, source_rank in (
                (POPULARITY, popularity_ranks.get(restaurant_id)),
                (ITEM_ITEM, item_item_ranks.get(restaurant_id)),
            )
            if source_rank is not None
        }
        sources = tuple(source for source in (POPULARITY, ITEM_ITEM) if source in ranks)
        return Candidate(
            query_id=query.query_id,
            user_id=query.user_id,
            restaurant_id=restaurant_id,
            restaurant_name=context.item_names[restaurant_id],
            region=context.item_regions[restaurant_id],
            candidate_sources=sources,
            source_scores={
                POPULARITY: popularity_scores.get(restaurant_id, 0.0),
                ITEM_ITEM: item_sum_scores.get(restaurant_id, 0.0),
                "item_item_max": item_max_scores.get(restaurant_id, 0.0),
            },
            source_ranks=ranks,
            rrf_score=rrf_score if rrf_score is not None else 0.0,
            candidate_rank=rank,
        )


def _top_ids(scores: dict[int, float], limit: int) -> tuple[int, ...]:
    return tuple(
        restaurant_id
        for restaurant_id, _ in sorted(
            scores.items(), key=lambda pair: (-pair[1], pair[0])
        )[:limit]
    )
