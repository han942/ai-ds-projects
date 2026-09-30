"""Candidate models the comparison runner can put next to Stage 1 (C5).

Adding a model takes two steps. Implement it under ``retrieval/``; it needs
``fit(..., callback=...)`` that calls ``callback(epoch_stats, model)`` after
every epoch (returning True stops training), a ``history`` list of those
stats, and ``recommend(user_ids, exclude, k)``. Then add one
:class:`CandidateModel` subclass below and list it in ``CANDIDATE_MODELS``.
The runner (``experiments.comparison``), the CLI (``rating-recsys-compare``)
and the report (``evaluation.comparison_report``) are shared, so no new
experiment, CLI or report file is needed.

Scalar config fields automatically become CLI flags. Declare only grid axes
in ``grid_parameters``; coupled choices can override ``variants``. Shared
tests use one small config entry per model in ``tests/support.py``.

Model code is imported inside the methods, so comparing LightGCN never
imports PyTorch.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import fields
from itertools import product
from typing import Any, Callable, Iterable, Mapping, Sequence

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.models import WindowQuery


def _names(text: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in text.split(",") if item.strip())


class CandidateModel:
    """How the comparison runner trains and describes one candidate model.

    Model configs must be frozen dataclasses with ``name``, ``epochs`` and
    ``to_dict()``. The runner replaces ``epochs`` with the selected epoch
    count for the test refit.
    """

    name: str
    """Stage key, output folder ``comparisons/<name>/`` and CLI subcommand."""
    title: str
    letter: str
    """Short name in fusion labels, e.g. ``RRF C1+L``."""
    needs_review_texts = False
    default_max_epochs = 20
    default_eval_every = 1
    default_patience = 3
    packages: tuple[str, ...] = ()
    """Extra distributions recorded in the manifest environment."""
    loss_description = "epoch 평균 학습 loss"
    description: tuple[str, ...] = ()
    """Report bullets on the method (section 2)."""
    leakage_checks: dict[str, object] = {}
    # field -> (CLI flag, default values). Other scalar config fields become
    # --field-name automatically. epochs/seed use the shared CLI flags.
    grid_parameters: Mapping[str, tuple[str, tuple[Any, ...]]] = {}
    variant_fields: tuple[str, ...] = ()

    # ---- grid ------------------------------------------------------------
    def config_type(self) -> type:
        """Import the model's frozen config lazily (optional dependencies)."""

        raise NotImplementedError

    def add_arguments(self, group: argparse._ArgumentGroup) -> None:
        defaults = self.config_type()()
        for field in fields(defaults):
            name = field.name
            if not field.init or name in ("epochs", "seed", *self.variant_fields):
                continue
            value = getattr(defaults, name)
            if type(value) not in (bool, int, float, str):
                raise TypeError(f"{self.name}.{name}: override add_arguments for non-scalar fields")
            if name in self.grid_parameters:
                flag, values = self.grid_parameters[name]

                def parse_values(text: str, value_type=type(value)) -> tuple:
                    return tuple(value_type(item.strip()) for item in text.split(",") if item.strip())

                if isinstance(value, bool):
                    raise TypeError(f"{self.name}.{name}: boolean grid requires a custom parser")
                group.add_argument(flag, dest=name, type=parse_values, default=values)
            elif isinstance(value, bool):
                group.add_argument(
                    f"--{name.replace('_', '-')}", dest=name,
                    action=argparse.BooleanOptionalAction, default=value,
                )
            else:
                group.add_argument(f"--{name.replace('_', '-')}", dest=name, type=type(value), default=value)

    def variants(self, args: argparse.Namespace | None = None) -> tuple[dict[str, Any], ...]:
        """Coupled config choices, e.g. objective + activation in DeepCoNN."""

        return ({},)

    def _grid(self, values: dict[str, Any], axes: Mapping[str, tuple], variants) -> tuple[Any, ...]:
        keys = tuple(axes)
        config_type = self.config_type()
        return tuple(
            config_type(**{**values, **variant, **dict(zip(keys, combination))})
            for variant in variants
            for combination in product(*(axes[key] for key in keys))
        )

    def grid_from_args(self, args: argparse.Namespace) -> tuple[Any, ...]:
        defaults = self.config_type()()
        values = {
            field.name: getattr(args, field.name)
            for field in fields(defaults)
            if field.init and field.name not in ("epochs", "seed", *self.variant_fields)
            and field.name not in self.grid_parameters
        }
        values.update(epochs=args.max_epochs, seed=args.seed)
        return self._grid(
            values, {name: getattr(args, name) for name in self.grid_parameters}, self.variants(args)
        )

    def default_grid(self, seed: int = 42) -> tuple[Any, ...]:
        defaults = self.config_type()()
        values = {field.name: getattr(defaults, field.name) for field in fields(defaults) if field.init}
        values.update(epochs=self.default_max_epochs, seed=seed)
        return self._grid(
            values, {name: spec[1] for name, spec in self.grid_parameters.items()}, self.variants()
        )

    # ---- training --------------------------------------------------------
    def fit(
        self,
        model_config: Any,
        interactions: Sequence[Interaction],
        texts: Mapping[int, str | None] | None,
        callback: Callable[[Any, Any], bool | None] | None = None,
    ) -> Any:
        raise NotImplementedError

    def loss_fields(self, stats: Any) -> dict[str, float]:
        """Curve fields of one epoch; ``loss`` is plotted and tabulated."""

        return {"loss": float(stats.loss)}

    def diagnostics(
        self, model: Any, model_config: Any, queries: Sequence[WindowQuery]
    ) -> dict[str, float]:
        """Extra values recorded at every validation evaluation (and on test)."""

        return {}

    def model_summary(self, model: Any) -> dict[str, object]:
        """Size of what the model was fitted on; ``training_interactions`` is required."""

        return {"training_interactions": int(model.edge_count)}

    def references(
        self,
        train: Sequence[Interaction],
        validation_queries: Sequence[WindowQuery],
        test_history: Sequence[Interaction],
        test_queries: Sequence[WindowQuery],
    ) -> dict[str, float]:
        """Model-free reference values for the report (e.g. mean-rating RMSE)."""

        return {}


# ---------------------------------------------------------------------------
# LightGCN
# ---------------------------------------------------------------------------


class LightGCNCandidate(CandidateModel):
    name = "lightgcn"
    title = "LightGCN"
    letter = "L"
    default_max_epochs = 200
    default_eval_every = 5
    default_patience = 6
    packages = ("numpy", "scipy")
    loss_description = "epoch 평균 BPR loss(정규화 항 제외)"
    description = (
        "LightGCN: 사용자–식당 이분 그래프, 정규화 인접행렬로 L층 전파 후 0~L층 평균. "
        "BPR loss(방문 식당 vs 균등 샘플 미방문 식당), batch ego embedding L2, Adam. "
        "NumPy/SciPy 구현, seed 고정.",
        "그래프 edge는 cutoff 이전 방문만 쓴다. 이미 방문한 식당은 후보에서 뺀다. "
        "C4는 config에 고정된 LightGCN이고, 여기서 학습하는 L은 grid로 새로 고른 설정이다.",
    )

    grid_parameters = {
        "layers": ("--layers-grid", (1, 2, 3)),
        "regularization": ("--regularization-grid", (1e-4, 1e-2)),
    }

    def config_type(self) -> type:
        from rating_recsys.retrieval.lightgcn import LightGCNConfig

        return LightGCNConfig

    def fit(self, model_config, interactions, texts, callback=None):
        from rating_recsys.retrieval.lightgcn import LightGCN

        return LightGCN(model_config).fit(interactions, callback=callback)

    def loss_fields(self, stats) -> dict[str, float]:
        return {"loss": float(stats.bpr_loss), "reg_loss": float(stats.reg_loss)}

    def model_summary(self, model) -> dict[str, object]:
        return {
            "training_interactions": int(model.edge_count),
            "users": len(model.user_ids),
            "restaurants": len(model.item_ids),
        }


# ---------------------------------------------------------------------------
# DeepCoNN
# ---------------------------------------------------------------------------


class DeepCoNNCandidate(CandidateModel):
    name = "deepconn"
    title = "DeepCoNN"
    letter = "D"
    needs_review_texts = True
    default_max_epochs = 12
    default_eval_every = 1
    default_patience = 3
    packages = ("numpy", "torch")
    description = (
        "DeepCoNN: 사용자 문서(본인의 과거 리뷰)와 식당 문서(그 식당의 과거 리뷰)를 각각 "
        "embedding → 1D CNN → max-over-time → FC로 읽고, 두 벡터를 FM으로 결합한다.",
        "토큰은 글자(주로 한글 음절)이고 embedding은 처음부터 학습한다. 문서는 최신 리뷰부터 "
        "리뷰당 `max_review_length`자, 구분 토큰을 넣어 `doc_length`자까지 이어 붙인다.",
        "누수 차단: 문서에는 cutoff 이전 리뷰만 들어간다. 학습 visit (u, i)의 리뷰는 u와 i의 "
        "문서에서 모두 뺀다(평가 때 정답 리뷰는 항상 미래라 문서에 없다).",
        "`mse`는 논문의 평점 회귀, `bpr`은 방문 식당을 균등 샘플 미방문 식당보다 높게 두는 "
        "순위 학습이다. 후보는 두 경우 모두 FM 점수 순서다.",
    )
    leakage_checks = {"training_visit_review_removed_from_both_documents": True}

    grid_parameters = {"kernel_size": ("--kernel-sizes", (3,))}
    variant_fields = ("objective", "latent_activation")
    default_variants = ("bpr:relu", "bpr:linear", "mse:relu")

    def config_type(self) -> type:
        from rating_recsys.retrieval.deepconn import DeepCoNNConfig

        return DeepCoNNConfig

    def add_arguments(self, group: argparse._ArgumentGroup) -> None:
        super().add_arguments(group)
        group.add_argument(
            "--variants", type=_names, default=self.default_variants,
            help="objective:activation 목록 (objective bpr, mse; activation relu, linear)",
        )

    def variants(self, args: argparse.Namespace | None = None) -> tuple[dict[str, Any], ...]:
        result = []
        for variant in self.default_variants if args is None else args.variants:
            objective, _, activation = variant.partition(":")
            result.append({"objective": objective, "latent_activation": activation or "relu"})
        return tuple(result)

    def fit(self, model_config, interactions, texts, callback=None):
        from rating_recsys.retrieval.deepconn import DeepCoNN

        if texts is None:
            raise ValueError("DeepCoNN needs review texts")
        return DeepCoNN(model_config).fit(interactions, texts, callback=callback)

    def diagnostics(self, model, model_config, queries) -> dict[str, float]:
        values: dict[str, float] = {}
        if model_config.objective == "mse":
            values.update(window_rmse(model, queries))
        values.update(model.dead_latent_rates())
        return values

    def model_summary(self, model) -> dict[str, object]:
        return {
            "training_interactions": int(model.edge_count),
            "users": len(model.user_ids),
            "restaurants": len(model.item_ids),
            "vocabulary_size": len(model.vocabulary) if model.vocabulary else 0,
        }

    def references(self, train, validation_queries, test_history, test_queries):
        return {
            "validation_mean_rating_rmse": mean_rating_rmse(train, validation_queries),
            "test_mean_rating_rmse": mean_rating_rmse(test_history, test_queries),
        }


def window_rmse(model, queries: Sequence[WindowQuery]) -> dict[str, float]:
    """RMSE of predicted ratings on every window visit the model can score."""

    users, items, ratings = [], [], []
    for query in queries:
        for visit in query.window:
            users.append(visit.user_id)
            items.append(visit.restaurant_id)
            ratings.append(visit.rating)
    predicted = model.score_pairs(users, items)
    pairs = [(p, r) for p, r in zip(predicted, ratings) if not math.isnan(p)]
    if not pairs:
        return {"rmse": float("nan"), "rated_visits": 0}
    return {
        "rmse": math.sqrt(sum((p - r) ** 2 for p, r in pairs) / len(pairs)),
        "rated_visits": len(pairs),
    }


def mean_rating_rmse(train: Iterable[Interaction], queries: Sequence[WindowQuery]) -> float:
    """RMSE of predicting the training mean rating for every window visit."""

    ratings = [row.rating for row in train]
    mean = sum(ratings) / len(ratings)
    visits = [visit.rating for query in queries for visit in query.window]
    return math.sqrt(sum((mean - r) ** 2 for r in visits) / len(visits)) if visits else float("nan")


CANDIDATE_MODELS: dict[str, CandidateModel] = {
    model.name: model for model in (LightGCNCandidate(), DeepCoNNCandidate())
}
