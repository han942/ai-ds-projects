"""Shared records passed between retrieval, ranking, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from rating_recsys.datasets.models import Interaction


@dataclass(frozen=True, slots=True)
class RecommendationQuery:
    query_id: str
    phase: str
    user_id: int
    cutoff: date
    history: tuple[Interaction, ...]
    target: Interaction
    relevance: int

    @property
    def history_depth(self) -> str:
        return "history_1_2" if len(self.history) <= 2 else "history_3_plus"


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
    history_depth: str
    candidate: Candidate
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    row: FeatureRow
    ranking_score: float
    final_rank: int
