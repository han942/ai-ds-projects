"""Versioned configuration for the first two-stage baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """All choices that can change the offline baseline result."""

    schema_version: str = "baseline-v5-region-ablation"
    protocol: str = "seen-user-leave-last-two-out"
    minimum_user_items: int = 3
    candidate_k: int = 100
    ranking_k: int = 10
    rrf_constant: int = 60
    random_seed: int = 42
    region_mode: str = "with_region"
    relevance_high_threshold: float = 4.0
    relevance_low_threshold: float = 3.0

    def __post_init__(self) -> None:
        if self.region_mode not in {"with_region", "without_region"}:
            raise ValueError("region_mode must be with_region or without_region")
        if self.minimum_user_items < 3:
            raise ValueError("minimum_user_items must be at least 3")
        if self.candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        if not 1 <= self.ranking_k <= self.candidate_k:
            raise ValueError("ranking_k must be between 1 and candidate_k")
        if self.rrf_constant < 1:
            raise ValueError("rrf_constant must be positive")
        if self.relevance_low_threshold >= self.relevance_high_threshold:
            raise ValueError("low relevance threshold must be below high threshold")

    @property
    def include_region(self) -> bool:
        return self.region_mode == "with_region"

    def relevance(self, rating: float) -> int:
        if rating >= self.relevance_high_threshold:
            return 2
        if rating >= self.relevance_low_threshold:
            return 1
        return 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
