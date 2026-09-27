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
REGION_POPULARITY = "region_popularity"
BASE_RRF = "base_rrf"


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    popularity: tuple[Candidate, ...]
    item_item: tuple[Candidate, ...]
    region_popularity: tuple[Candidate, ...]
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
    neighbors: dict[int, dict[int, int]]

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


class IncrementalRetrievalContext:
    """Maintain the exact batch context while interactions become visible.

    Interactions must be added in global chronological order. Each interaction is
    applied once, replacing the previous per-query full scan and context rebuild.
    """

    def __init__(self) -> None:
        self._context = RetrievalContext(
            item_counts={},
            item_rating_sum={},
            item_regions={},
            item_names={},
            item_frequencies={},
            cooccurrence={},
            neighbors={},
        )
        self._items_by_user: defaultdict[int, set[int]] = defaultdict(set)

    @property
    def context(self) -> RetrievalContext:
        return self._context

    def add(self, interaction: Interaction) -> None:
        context = self._context
        restaurant_id = interaction.restaurant_id
        context.item_counts[restaurant_id] = (
            context.item_counts.get(restaurant_id, 0) + 1
        )
        context.item_rating_sum[restaurant_id] = (
            context.item_rating_sum.get(restaurant_id, 0.0) + interaction.rating
        )
        context.item_regions[restaurant_id] = interaction.region
        context.item_names[restaurant_id] = interaction.restaurant_name

        user_items = self._items_by_user[interaction.user_id]
        if restaurant_id in user_items:
            return
        for other_id in user_items:
            pair = (
                (restaurant_id, other_id)
                if restaurant_id < other_id
                else (other_id, restaurant_id)
            )
            count = context.cooccurrence.get(pair, 0) + 1
            context.cooccurrence[pair] = count
            context.neighbors.setdefault(restaurant_id, {})[other_id] = count
            context.neighbors.setdefault(other_id, {})[restaurant_id] = count
        user_items.add(restaurant_id)
        context.item_frequencies[restaurant_id] = (
            context.item_frequencies.get(restaurant_id, 0) + 1
        )


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

    neighbors: dict[int, dict[int, int]] = {}
    for (left, right), count in cooccurrence.items():
        neighbors.setdefault(left, {})[right] = count
        neighbors.setdefault(right, {})[left] = count

    return RetrievalContext(
        item_counts=dict(item_counts),
        item_rating_sum=dict(item_rating_sum),
        item_regions=item_regions,
        item_names=item_names,
        item_frequencies=dict(item_frequencies),
        cooccurrence=dict(cooccurrence),
        neighbors=neighbors,
    )


class BaselineCandidateGenerator:
    """Generate source candidates and fuse them with reciprocal rank fusion."""

    def __init__(
        self, *, candidate_k: int = 100, rrf_constant: int = 60, include_region: bool = True
    ) -> None:
        if candidate_k < 1 or rrf_constant < 1:
            raise ValueError("candidate_k and rrf_constant must be positive")
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant
        self.include_region = include_region

    def retrieve(
        self,
        query: RecommendationQuery,
        available_interactions: Iterable[Interaction],
    ) -> tuple[RetrievalResult, RetrievalContext]:
        context = build_context(available_interactions)
        return self.retrieve_from_context(query, context)

    def retrieve_from_context(
        self,
        query: RecommendationQuery,
        context: RetrievalContext,
    ) -> tuple[RetrievalResult, RetrievalContext]:
        """Retrieve from a cutoff-safe precomputed context."""

        started = time.perf_counter()
        seen = {item.restaurant_id for item in query.history}
        eligible = sorted(set(context.item_counts) - seen)

        popularity_scores = {
            restaurant_id: float(context.item_counts[restaurant_id])
            for restaurant_id in eligible
        }
        item_sum_scores: dict[int, float] = {}
        item_max_scores: dict[int, float] = {}
        for history_item in query.history:
            history_id = history_item.restaurant_id
            history_frequency = context.item_frequencies.get(history_id, 0)
            if not history_frequency:
                continue
            for restaurant_id, co_count in context.neighbors.get(history_id, {}).items():
                if restaurant_id in seen or restaurant_id not in context.item_counts:
                    continue
                item_frequency = context.item_frequencies.get(restaurant_id, 0)
                if not item_frequency:
                    continue
                similarity = co_count / math.sqrt(history_frequency * item_frequency)
                item_sum_scores[restaurant_id] = (
                    item_sum_scores.get(restaurant_id, 0.0) + similarity
                )
                item_max_scores[restaurant_id] = max(
                    item_max_scores.get(restaurant_id, 0.0), similarity
                )

        region_popularity_scores: dict[int, float] = {}
        if self.include_region and query.history:
            history_region_counts: Counter[str] = Counter(
                item.region for item in query.history
            )
            history_length = len(query.history)
            region_popularity_scores = {
                restaurant_id: (
                    context.item_counts[restaurant_id]
                    * history_region_counts.get(context.item_regions[restaurant_id], 0)
                    / history_length
                )
                for restaurant_id in eligible
                if history_region_counts.get(context.item_regions[restaurant_id], 0) > 0
            }

        popularity_ids = _top_ids(popularity_scores, self.candidate_k)
        item_item_ids = _top_ids(
            {key: value for key, value in item_sum_scores.items() if value > 0},
            self.candidate_k,
        )
        region_popularity_ids = _top_ids(
            region_popularity_scores,
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
        region_popularity_ranks = {
            restaurant_id: index + 1
            for index, restaurant_id in enumerate(region_popularity_ids)
        }

        popularity = tuple(
            self._candidate(
                query,
                restaurant_id,
                context,
                popularity_scores,
                item_sum_scores,
                item_max_scores,
                region_popularity_scores,
                popularity_ranks,
                {},
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
                region_popularity_scores,
                {},
                item_item_ranks,
                {},
                rank,
            )
            for rank, restaurant_id in enumerate(item_item_ids, start=1)
        )

        region_popularity = tuple(
            self._candidate(
                query,
                restaurant_id,
                context,
                popularity_scores,
                item_sum_scores,
                item_max_scores,
                region_popularity_scores,
                {},
                {},
                region_popularity_ranks,
                rank,
            )
            for rank, restaurant_id in enumerate(region_popularity_ids, start=1)
        )

        union_ids = (
            set(popularity_ids) | set(item_item_ids) | set(region_popularity_ids)
        )
        base_union_ids = set(popularity_ids) | set(item_item_ids)
        base_rrf_scores = {
            restaurant_id: sum(
                1.0 / (self.rrf_constant + rank)
                for rank in (
                    popularity_ranks.get(restaurant_id),
                    item_item_ranks.get(restaurant_id),
                )
                if rank is not None
            )
            for restaurant_id in base_union_ids
        }
        rrf_scores = {
            restaurant_id: sum(
                1.0 / (self.rrf_constant + rank)
                for rank in (
                    popularity_ranks.get(restaurant_id),
                    item_item_ranks.get(restaurant_id),
                    region_popularity_ranks.get(restaurant_id),
                )
                if rank is not None
            )
            for restaurant_id in union_ids
        }
        base_quota = min(self.candidate_k, max(1, self.candidate_k // 2))
        base_ids = _top_ids(base_rrf_scores, base_quota)
        base_id_set = set(base_ids)
        expanded_ids = _top_ids(rrf_scores, len(rrf_scores))
        fused_ids = tuple(
            list(base_ids)
            + [
                restaurant_id
                for restaurant_id in expanded_ids
                if restaurant_id not in base_id_set
            ][: self.candidate_k - len(base_ids)]
        )
        union = tuple(
            self._candidate(
                query,
                restaurant_id,
                context,
                popularity_scores,
                item_sum_scores,
                item_max_scores,
                region_popularity_scores,
                popularity_ranks,
                item_item_ranks,
                region_popularity_ranks,
                rank,
                rrf_score=rrf_scores[restaurant_id],
                base_rrf_score=base_rrf_scores.get(restaurant_id, 0.0),
            )
            for rank, restaurant_id in enumerate(fused_ids, start=1)
        )
        return (
            RetrievalResult(
                popularity=popularity,
                item_item=item_item,
                region_popularity=region_popularity,
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
                **(
                    {REGION_POPULARITY: _region_popularity_score(query, target_id, context)}
                    if self.include_region
                    else {}
                ),
                BASE_RRF: 0.0,
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
        region_popularity_scores: dict[int, float],
        popularity_ranks: dict[int, int],
        item_item_ranks: dict[int, int],
        region_popularity_ranks: dict[int, int],
        rank: int,
        *,
        rrf_score: float | None = None,
        base_rrf_score: float | None = None,
    ) -> Candidate:
        ranks = {
            source: source_rank
            for source, source_rank in (
                (POPULARITY, popularity_ranks.get(restaurant_id)),
                (ITEM_ITEM, item_item_ranks.get(restaurant_id)),
                (
                    REGION_POPULARITY,
                    region_popularity_ranks.get(restaurant_id),
                ),
            )
            if source_rank is not None
        }
        sources = tuple(
            source
            for source in (POPULARITY, ITEM_ITEM, REGION_POPULARITY)
            if source in ranks
        )
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
                **(
                    {REGION_POPULARITY: region_popularity_scores.get(restaurant_id, 0.0)}
                    if self.include_region
                    else {}
                ),
                BASE_RRF: base_rrf_score if base_rrf_score is not None else 0.0,
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


def _region_popularity_score(
    query: RecommendationQuery,
    restaurant_id: int,
    context: RetrievalContext,
) -> float:
    if not query.history:
        return 0.0
    preferred = sum(
        item.region == context.item_regions[restaurant_id] for item in query.history
    ) / len(query.history)
    return context.item_counts[restaurant_id] * preferred
