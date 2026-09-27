"""Shared records passed between retrieval, ranking, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from rating_recsys.datasets.models import Interaction


@dataclass(frozen=True, slots=True)
class RecommendationQuery:
    """One single-positive query: history before ``cutoff`` and the next visit.

    Ranker training rows are built from these prefix queries. Evaluation uses
    :class:`WindowQuery`, which can hold several positives.
    """

    query_id: str
    phase: str
    user_id: int
    cutoff: date
    history: tuple[Interaction, ...]
    target: Interaction
    relevance: int


@dataclass(frozen=True, slots=True)
class WindowQuery:
    """One user at a fixed cutoff with every visit in the following window."""

    query_id: str
    phase: str
    user_id: int
    cutoff: date
    history: tuple[Interaction, ...]
    window: tuple[Interaction, ...]
    relevance_by_item: dict[int, int]

    @property
    def relevant(self) -> bool:
        return any(value > 0 for value in self.relevance_by_item.values())

    def retrieval_query(self) -> RecommendationQuery:
        # The candidate generator API needs a target. The first window visit is
        # only an id anchor: candidates depend on history and context alone.
        return RecommendationQuery(
            query_id=self.query_id,
            phase=self.phase,
            user_id=self.user_id,
            cutoff=self.cutoff,
            history=self.history,
            target=self.window[0],
            relevance=max(self.relevance_by_item.values()),
        )


@dataclass(frozen=True, slots=True)
class Candidate:
    query_id: str
    user_id: int
    restaurant_id: int
    restaurant_name: str
    region: str
    candidate_sources: tuple[str, ...]
    source_scores: dict[str, float]
    source_ranks: dict[str, int]
    rrf_score: float
    candidate_rank: int
    injected_for_training: bool = False


@dataclass(frozen=True, slots=True)
class FeatureRow:
    query_id: str
    phase: str
    user_id: int
    restaurant_id: int
    target_restaurant_id: int
    relevance: int
    candidate: Candidate
    features: dict[str, float]
