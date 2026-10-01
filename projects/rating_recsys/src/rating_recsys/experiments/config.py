"""Versioned configuration for the two-stage recommender experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from rating_recsys.ranking.features import FEATURE_NAMES, NO_REGION_FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """All choices that can change the offline result.

    The split uses one global date cutoff pair from interaction quantiles.
    Stage 1 is C5 = RRF(C1 item-item, C4 LightGCN). The LightGCN settings are
    fixed here; they were chosen on the validation window of the LightGCN
    comparison run (3 layers, L2 1e-4, 20 epochs; see BASELINE_MODEL.md).
    Grid fields are the ranker options tried on the validation window; the
    test window is evaluated once with the selected options.
    """

    schema_version: str = "global-cutoff-v2"
    protocol: str = "global-temporal-multi-positive"
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    candidate_k: int = 100
    ranking_k: int = 10
    rrf_constant: int = 60
    random_seed: int = 42
    region_mode: str = "with_region"
    # Historical baseline defaults remain available for controlled comparisons.
    ranker_training_mode: str = "prefix"
    ranker_label_mode: str = "relevance"
    relevance_high_threshold: float = 4.0
    relevance_low_threshold: float = 3.0
    # C4 LightGCN. Training queries use models refit every
    # ``lightgcn_checkpoint_months`` (calendar aligned) on earlier interactions.
    lightgcn_dimension: int = 64
    lightgcn_layers: int = 3
    lightgcn_epochs: int = 20
    lightgcn_learning_rate: float = 0.005
    lightgcn_regularization: float = 1e-4
    lightgcn_batch_size: int = 2048
    lightgcn_checkpoint_months: int = 3
    # Reference only: the previous C3 quota union, with the quota the
    # 2026-09-27 run chose on validation. Not fused into Stage 1.
    legacy_c3_quota: float = 0.75
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
        if self.ranker_training_mode not in {"prefix", "window"}:
            raise ValueError("ranker_training_mode must be prefix or window")
        if self.ranker_label_mode not in {"relevance", "rating"}:
            raise ValueError("ranker_label_mode must be relevance or rating")
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
        if not 0 <= self.legacy_c3_quota <= 1:
            raise ValueError("legacy_c3_quota must be between 0 and 1")
        if self.lightgcn_checkpoint_months not in {1, 2, 3, 4, 6, 12}:
            raise ValueError("lightgcn_checkpoint_months must divide 12")
        if not self.num_leaves_grid or not self.min_child_samples_grid:
            raise ValueError("LightGBM grids must not be empty")
        if self.relevance_low_threshold >= self.relevance_high_threshold:
            raise ValueError("low relevance threshold must be below high threshold")
        self.lightgcn_config  # validates the LightGCN fields

    @property
    def include_region(self) -> bool:
        return self.region_mode == "with_region"

    @property
    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_NAMES if self.include_region else NO_REGION_FEATURE_NAMES

    @property
    def lightgcn_config(self):
        from rating_recsys.retrieval.lightgcn import LightGCNConfig

        return LightGCNConfig(
            dimension=self.lightgcn_dimension,
            layers=self.lightgcn_layers,
            epochs=self.lightgcn_epochs,
            batch_size=self.lightgcn_batch_size,
            learning_rate=self.lightgcn_learning_rate,
            regularization=self.lightgcn_regularization,
            seed=self.random_seed,
        )

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

    def training_label(self, rating: float) -> int:
        """Half-star rating indexes preserve ratings without three-bin grading.

        Zero is reserved for an unobserved candidate, not a measured zero-star
        rating. Evaluation relevance remains unchanged in both training modes.
        """
        if self.ranker_label_mode == "relevance":
            return self.relevance(rating)
        label = round(rating * 2)
        if not 1 <= rating <= 5 or abs(label / 2 - rating) > 1e-8:
            raise ValueError("rating labels require ratings from 1 to 5 in half-star steps")
        return label

    @property
    def label_gain(self) -> tuple[float, ...]:
        if self.ranker_label_mode == "rating":
            return tuple(index / 2 for index in range(11))
        return (0.0, 1.0, 3.0)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
