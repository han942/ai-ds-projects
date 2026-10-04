"""Stage 1 baseline C5: reciprocal rank fusion of C1 item-item and C4 LightGCN.

Chosen from the validation window of the 2026-09-28 LightGCN comparison
(history in ``PLAN.md`` and ``legacy/v2_experiments.zip``): among
LightGCN alone and its RRF fusions with C0/C1/C2, C1 + LightGCN had the highest
validation Recall@100. C0 (popularity), C2 (region popularity) and the previous
C3 quota union are still built from the same context. They are reported as
reference lists and feed ranker features, but they are not fused into C5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

from rating_recsys.experiments.models import Candidate, RecommendationQuery
from rating_recsys.retrieval.baselines import (
    ITEM_ITEM,
    POPULARITY,
    REGION_POPULARITY,
    BaselineCandidateGenerator,
    RetrievalContext,
    RetrievalResult,
)
from rating_recsys.retrieval.lightgcn import LIGHTGCN
from rating_recsys.ranking.features import LIGHTGCN_Z


FUSED_SOURCES = (ITEM_ITEM, LIGHTGCN)
RANKED_SOURCES = (POPULARITY, ITEM_ITEM, REGION_POPULARITY, LIGHTGCN)


class GraphScorer(Protocol):
    def score(self, user_id: int, restaurant_ids: Iterable[int]) -> dict[int, float]: ...


@dataclass(frozen=True, slots=True)
class HybridResult:
    reference: RetrievalResult
    """C0, C1, C2 lists and the C3 quota union from the same context."""
    lightgcn: tuple[int, ...]
    """C4: LightGCN Top-k restricted to unseen restaurants in the context."""
    union: tuple[Candidate, ...]
    """C5: the Stage 1 output passed to the ranker."""


def rrf(
    rankings: Sequence[Sequence[int]], *, k: int, constant: int
) -> tuple[int, ...]:
    """Reciprocal rank fusion of ordered lists; ties by restaurant id."""

    scores = rrf_scores(rankings, constant=constant)
    return tuple(
        restaurant_id
        for restaurant_id, _ in sorted(scores.items(), key=lambda p: (-p[1], p[0]))[:k]
    )


def rrf_scores(rankings: Sequence[Sequence[int]], *, constant: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, restaurant_id in enumerate(ranking, start=1):
            scores[restaurant_id] = scores.get(restaurant_id, 0.0) + 1.0 / (
                constant + rank
            )
    return scores


class HybridCandidateGenerator:
    """C5 = RRF(C1 Top-k, C4 LightGCN Top-k) cut to Top-k, without quotas.

    ``legacy_c3_quota`` only shapes the reference C3 list. A query whose user is
    not in the LightGCN graph (``graph_ranked`` empty) gets C1 order alone.
    """

    def __init__(
        self,
        *,
        candidate_k: int = 100,
        rrf_constant: int = 60,
        include_region: bool = True,
        legacy_c3_quota: float = 0.75,
    ) -> None:
        self.reference = BaselineCandidateGenerator(
            candidate_k=candidate_k,
            rrf_constant=rrf_constant,
            include_region=include_region,
            base_quota_fraction=legacy_c3_quota,
        )

    @property
    def candidate_k(self) -> int:
        return self.reference.candidate_k

    @property
    def rrf_constant(self) -> int:
        return self.reference.rrf_constant

    @property
    def include_region(self) -> bool:
        return self.reference.include_region

    def retrieve(
        self,
        query: RecommendationQuery,
        context: RetrievalContext,
        *,
        graph_ranked: Sequence[int] = (),
        scorer: GraphScorer | None = None,
    ) -> HybridResult:
        reference, _ = self.reference.retrieve_from_context(query, context)
        eligible = reference.eligible_catalog
        lightgcn = tuple(item for item in graph_ranked if item in eligible)[
            : self.candidate_k
        ]
        ranks = {
            POPULARITY: _ranks(c.restaurant_id for c in reference.popularity),
            ITEM_ITEM: _ranks(c.restaurant_id for c in reference.item_item),
            REGION_POPULARITY: _ranks(
                c.restaurant_id for c in reference.region_popularity
            ),
            LIGHTGCN: _ranks(lightgcn),
        }
        fused = rrf_scores(
            [
                tuple(c.restaurant_id for c in reference.item_item),
                lightgcn,
            ],
            constant=self.rrf_constant,
        )
        ordered = [
            item
            for item, _ in sorted(fused.items(), key=lambda p: (-p[1], p[0]))[
                : self.candidate_k
            ]
        ]
        graph_scores = (
            scorer.score(query.user_id, ordered) if scorer is not None and lightgcn else {}
        )
        stats = _mean_std(graph_scores.values())
        union = tuple(
            self._candidate(
                query, item, rank, context, reference, ranks, fused, graph_scores, stats
            )
            for rank, item in enumerate(ordered, start=1)
        )
        return HybridResult(
            reference=reference,
            lightgcn=lightgcn,
            union=union,
        )

    def _candidate(
        self,
        query: RecommendationQuery,
        restaurant_id: int,
        rank: int,
        context: RetrievalContext,
        reference: RetrievalResult,
        ranks: dict[str, dict[int, int]],
        fused: dict[int, float],
        graph_scores: dict[int, float],
        stats: tuple[float, float] | None,
    ) -> Candidate:
        maps = reference.score_maps
        raw = graph_scores.get(restaurant_id)
        z = 0.0
        if raw is not None and stats is not None and stats[1] > 0:
            z = (raw - stats[0]) / stats[1]
        scores = {
            POPULARITY: maps[POPULARITY].get(restaurant_id, 0.0),
            ITEM_ITEM: maps[ITEM_ITEM].get(restaurant_id, 0.0),
            "item_item_max": maps["item_item_max"].get(restaurant_id, 0.0),
            LIGHTGCN: raw if raw is not None else 0.0,
            LIGHTGCN_Z: z,
        }
        if self.include_region:
            scores[REGION_POPULARITY] = maps[REGION_POPULARITY].get(restaurant_id, 0.0)
        return Candidate(
            query_id=query.query_id,
            user_id=query.user_id,
            restaurant_id=restaurant_id,
            restaurant_name=context.item_names[restaurant_id],
            region=context.item_regions[restaurant_id],
            candidate_sources=tuple(
                source for source in FUSED_SOURCES if restaurant_id in ranks[source]
            ),
            source_scores=scores,
            source_ranks={
                source: ranks[source][restaurant_id]
                for source in RANKED_SOURCES
                if restaurant_id in ranks[source]
            },
            rrf_score=fused.get(restaurant_id, 0.0),
            candidate_rank=rank,
        )


def _ranks(ids: Iterable[int]) -> dict[int, int]:
    return {item: rank for rank, item in enumerate(ids, start=1)}


def _mean_std(values: Iterable[float]) -> tuple[float, float] | None:
    collected = list(values)
    if not collected:
        return None
    mean = math.fsum(collected) / len(collected)
    variance = math.fsum((value - mean) ** 2 for value in collected) / len(collected)
    return mean, math.sqrt(variance)
