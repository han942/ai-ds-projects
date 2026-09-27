"""Versioned configuration for the two-stage recommender experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rating_recsys.ranking.features import FEATURE_NAMES, NO_REGION_FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """All choices that can change the offline result.

    The split uses one global date cutoff pair from interaction quantiles.
    Grid fields are the options tried on the validation window; the test
    window is evaluated once with the selected options.
    """

    schema_version: str = "global-cutoff-v1"
    protocol: str = "global-temporal-multi-positive"
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    candidate_k: int = 100
    ranking_k: int = 10
    rrf_constant: int = 60
    random_seed: int = 42
    region_mode: str = "with_region"
    relevance_high_threshold: float = 4.0
    relevance_low_threshold: float = 3.0
    quota_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    num_leaves_grid: tuple[int, ...] = (15, 31, 63)
    min_child_samples_grid: tuple[int, ...] = (10, 100)
    learning_rate: float = 0.05
    max_estimators: int = 1000
    early_stopping_rounds: int = 50
    reg_lambda: float = 1.0
    # The earlier fixed ranker (150 trees, 15 leaves, 10 min samples) is also a
    # selection candidate, so tuning is compared against it on validation.
    reference_estimators: int = 150
    reference_num_leaves: int = 15
    reference_min_child_samples: int = 10
    bootstrap_samples: int = 2000
    n_jobs: int = 8

    def __post_init__(self) -> None:
        if self.region_mode not in {"with_region", "without_region"}:
            raise ValueError("region_mode must be with_region or without_region")
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("fractions must be between 0 and 1")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("train and validation fractions must sum below 1")
        if self.candidate_k < 1 or not 1 <= self.ranking_k <= self.candidate_k:
            raise ValueError("ranking_k must be between 1 and candidate_k")
        if self.rrf_constant < 1:
            raise ValueError("rrf_constant must be positive")
        if not self.quota_grid or any(not 0 <= q <= 1 for q in self.quota_grid):
            raise ValueError("quota_grid values must be between 0 and 1")
        if not self.num_leaves_grid or not self.min_child_samples_grid:
            raise ValueError("LightGBM grids must not be empty")
        if self.relevance_low_threshold >= self.relevance_high_threshold:
            raise ValueError("low relevance threshold must be below high threshold")

    @property
    def include_region(self) -> bool:
        return self.region_mode == "with_region"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES if self.include_region else NO_REGION_FEATURE_NAMES

    @property
    def candidate_cutoffs(self) -> tuple[int, ...]:
        return tuple(k for k in (20, 50, 100) if k <= self.candidate_k) or (
            self.candidate_k,
        )

    @property
    def ranking_cutoffs(self) -> tuple[int, ...]:
        return tuple(k for k in (5, 10) if k <= self.ranking_k) or (self.ranking_k,)

    def relevance(self, rating: float) -> int:
        if rating >= self.relevance_high_threshold:
            return 2
        if rating >= self.relevance_low_threshold:
            return 1
        return 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
