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
from pathlib import Path
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
    def prepare_fit(self, queries: Sequence[WindowQuery]) -> None:
        """Optionally restrict expensive preprocessing to evaluation users."""

    def fit(
        self,
        model_config: Any,
        interactions: Sequence[Interaction],
        texts: Mapping[int, str | None] | None,
        callback: Callable[[Any, Any], bool | None] | None = None,
    ) -> Any:
        raise NotImplementedError

    def prepare_experiment(self, config, snapshot, texts_meta) -> None:
        """Validate external frozen inputs before model fitting."""

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

    def save_artifacts(self, model: Any, run_dir: Path) -> dict[str, str]:
        """Optionally persist a fitted index in the comparison run."""
        return {}

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


class BM25Candidate(CandidateModel):
    name = "bm25"
    title = "Kiwi + BM25"
    letter = "B"
    needs_review_texts = True
    default_max_epochs = 1
    default_eval_every = 1
    default_patience = 1
    packages = ("numpy", "scipy", "bm25s", "kiwipiepy", "kiwipiepy_model")
    loss_description = "BM25는 비학습 검색이며 1은 색인 생성 단계, loss 0은 자리표시자"
    description = (
        "사용자 query는 cutoff 이전의 최근 평점 4점 이상 리뷰 최대 5개, 식당 문서는 같은 시점까지의 최근 평점 4점 이상 리뷰 최대 10개다. 리뷰당 공백을 정리한 본문 240자를 사용하며 메뉴·상호·지역 메타데이터는 추가하지 않는다.",
        "Kiwi cong 모델을 로컬 CPU에서 실행한다. 명사 NNG·NNP, 동사 VV, 형용사 VA, 어근 XR, 외국어 SL의 형태를 소문자로 추출한다. 사용자 query의 중복 단어는 제거하고 식당 문서의 단어 빈도는 유지한다.",
        "preprocessing=baseline은 기존 처리만 적용한다. clean은 같은 리뷰·240자 선택 이후 URL·HTML·이모티콘·반복 웃음/울음 문자를 정리하고 Unicode NFKC를 적용한다. clean_stopwords는 추가로 고정된 공통 칭찬·방문 표현을 Kiwi 형태 단위로 제거한다. 정확한 버전·불용어 목록·원본/처리 프로필 hash는 retriever_preprocessing에 기록한다.",
        "bm25s의 Lucene 방식, k1=1.2·b=0.75를 첫 고정 설정으로 사용한다(실제 값은 config에 기록). cutoff별 식당 문서만으로 IDF·평균 문서 길이를 새로 계산하며, 점수가 같은 식당은 ID 오름차순이다.",
        "기존 Liquid 실험과 리뷰 선택 기준은 같지만, BM25에는 500 BPE 토큰 제한을 적용하지 않는다. 따라서 Liquid와의 차이는 검색 방식과 입력 잘림 차이를 함께 포함한다.",
        "이미 방문한 식당과 점수 0인 식당은 후보에서 뺀다. query·식당 토큰이 없으면 빈 후보를 반환하되 평가 사용자에서 제외하지 않는다. 최대 100개로 평가하며 후보가 모자라도 임의로 채우지 않는다. 임베딩·외부 API·신경망 학습은 없다.",
    )
    leakage_checks = {"profiles_from_historical_interactions_only": True,
                      "bm25_statistics_from_historical_restaurant_documents_only": True}

    def __init__(self):
        self.query_users: tuple[int, ...] = ()

    def config_type(self):
        from rating_recsys.retrieval.bm25 import BM25Config
        return BM25Config

    def prepare_fit(self, queries):
        self.query_users = tuple(query.user_id for query in queries)

    def fit(self, model_config, interactions, texts, callback=None):
        from rating_recsys.retrieval.bm25 import BM25Retriever
        if texts is None:
            raise ValueError("BM25 needs review texts")
        return BM25Retriever(model_config).fit(interactions, texts, self.query_users, callback)

    def model_summary(self, model):
        return {**model.metadata, "retriever_preprocessing": dict(model.metadata)}

    def diagnostics(self, model, model_config, queries):
        return {"query_profile_coverage": sum(q.user_id in model.user_tokens for q in queries)
                / len(queries) if queries else 0.0}

    def save_artifacts(self, model, run_dir):
        return model.save(run_dir / "bm25_index_test")


CANDIDATE_MODELS: dict[str, CandidateModel] = {
    model.name: model for model in (LightGCNCandidate(), DeepCoNNCandidate(), BM25Candidate())
}


class TwoTowerCandidate(CandidateModel):
    name = "two_tower"
    title = "E5 Two-Tower"
    letter = "T"
    needs_review_texts = True
    requires_embedding_cache = True
    require_prepared_candidates = True
    ranker_training_mode = "window"
    default_max_epochs = 12
    default_patience = 3
    packages = ("numpy", "torch", "tokenizers")
    comparison_sources = ("c5_c1_lightgcn_rrf",)
    fusion_sources = (("c1_item_item",),)
    experiment_defaults = {"satisfaction_mode": "history-aware"}
    evidence_status = "exploratory-reused-holdout"
    loss_description = "cutoff-safe window positives versus sampled eligible negatives (sampled softmax)"
    description = (
        "Frozen multilingual E5-small review vectors + learned ID embeddings feed separate small user/item MLPs. Normalized dot products rank restaurants; E5 is never trained or loaded.",
        "Training rebuilds calendar-aligned historical windows, using only pre-window profiles and catalogs. All eligible positive targets are used, independent of baseline candidate hits.",
        "Seen restaurants and all same-window positive targets are excluded from sampled negatives. Missing text uses a zero vector and an explicit presence feature.",
        "Exactly three conditions: current C5, Two-Tower alone, and equal-weight RRF(C1, Two-Tower). No LTR retraining or hyperparameter grid.",
    )
    leakage_checks = {"historical_window_profiles": True, "frozen_embedding_cache_read_only": True,
                      "training_targets_independent_of_c5": True}

    def __init__(self):
        self.query_users = ()
        self.embedding_run = None

    def config_type(self):
        from rating_recsys.retrieval.two_tower import TwoTowerConfig
        return TwoTowerConfig

    def add_arguments(self, group):
        from rating_recsys.config import PROJECT_ROOT
        super().add_arguments(group)
        group.add_argument("--embedding-run", type=Path,
                           default=PROJECT_ROOT / "artifacts/comparisons/review_ltr/20261007T060016724659Z-e7896add",
                           help="Completed E5 run whose frozen cache, tokenizer and input hashes are reused")

    def grid_from_args(self, args):
        self.embedding_run = args.embedding_run
        return super().grid_from_args(args)

    def prepare_experiment(self, config, snapshot, texts_meta):
        import hashlib
        import json
        from dataclasses import fields
        from tokenizers import Tokenizer
        from rating_recsys.config import PROJECT_ROOT
        from rating_recsys.retrieval.review_embeddings import ReviewEmbeddingConfig, ProfileFormatter

        run = self.embedding_run or PROJECT_ROOT / "artifacts/comparisons/review_ltr/20261007T060016724659Z-e7896add"
        manifest = json.loads((run / "manifest.json").read_text())
        if json.loads((run / "status.json").read_text())["status"] != "complete":
            raise ValueError("E5 source run is incomplete")
        if snapshot["dataset_snapshot_id"] != manifest["snapshot"]["dataset_snapshot_id"]:
            raise ValueError("E5 source snapshot differs")
        if not texts_meta or texts_meta["artifact_sha256"] != manifest["review_texts"]["artifact_sha256"]:
            raise ValueError("E5 source review text hash differs")
        for name in ("train_fraction", "validation_fraction", "candidate_k", "rrf_constant",
                     "satisfaction_mode", "satisfaction_min_history", "satisfaction_mean_weight",
                     "satisfaction_max_shift", "relevance_high_threshold", "relevance_low_threshold"):
            if getattr(config, name) != manifest["config"][name]:
                raise ValueError(f"E5 source experiment differs: {name}")
        self.experiment_config = config
        self.embedding = ReviewEmbeddingConfig(**{
            field.name: manifest["embedding"][field.name]
            for field in fields(ReviewEmbeddingConfig) if field.init
        })
        if self.embedding.backend != "local" or self.embedding.aggregation != "concat":
            raise ValueError("Two-Tower requires local concat E5 profiles")
        tokenizer_path = Path(self.embedding.cache_path).parent / "tokenizers" / f"{self.embedding.tokenizer_revision}.tokenizer.json"
        data = tokenizer_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != manifest["preprocessing"]["tokenizer_sha256"]:
            raise ValueError("Frozen E5 tokenizer hash differs")
        self.formatter = ProfileFormatter(self.embedding, tokenizer_path.parent,
                                         tokenizer=Tokenizer.from_str(data.decode()))
        self.provenance = {"source_run": str(run), "embedding": self.embedding.to_dict(),
                           "tokenizer_sha256": manifest["preprocessing"]["tokenizer_sha256"],
                           "embedding_requests": 0, "encoder_loaded": False}

    def prepare_fit(self, queries):
        self.query_users = tuple(query.user_id for query in queries)

    def fit(self, model_config, interactions, texts, callback=None):
        from contextlib import closing
        from rating_recsys.retrieval.two_tower import CachedProfiles, TwoTower
        if texts is None:
            raise ValueError("Two-Tower needs review texts")
        with closing(CachedProfiles(self.embedding, self.formatter)) as profiles:
            return TwoTower(model_config, embedding_config=self.embedding, formatter=self.formatter,
                            profiles=profiles, eval_user_ids=self.query_users,
                            experiment_config=self.experiment_config).fit(interactions, texts, callback=callback)

    def model_summary(self, model):
        return {"training_interactions": model.edge_count, "users": len(model.user_ids),
                "restaurants": len(model.item_ids), "training_audit": model.training_audit,
                "training_positives": model.metadata["training_positives"],
                "evaluation_bank": model.metadata["evaluation_bank"],
                "negative_policy": model.metadata["negative_policy"],
                "trainable_parameters": sum(p.numel() for tower in (model.user_tower, model.item_tower)
                                            for p in tower.parameters()),
                "frozen_embedding": self.provenance}

    def save_artifacts(self, model, run_dir):
        return model.save(run_dir / "two_tower_test.pt")


CANDIDATE_MODELS["two_tower"] = TwoTowerCandidate()


class ReviewEmbeddingsCandidate(CandidateModel):
    """OpenRouter review-profile retrieval; invoked by the dedicated CLI."""

    name = "review_embeddings"
    title = "OpenRouter 리뷰 임베딩"
    letter = "E"
    needs_review_texts = True
    default_max_epochs = 1
    default_eval_every = 1
    default_patience = 1
    packages = ("numpy", "aiohttp", "tokenizers")
    loss_description = "외부 임베딩 모델을 사용하므로 학습 loss 없음(0으로 표시)"
    description = (
        "사용자는 cutoff 이전의 최근 평점 4점 이상 리뷰 5개, 식당은 같은 시점까지의 최근 평점 4점 이상 리뷰 10개를 선택한다. concat은 본문을 묶어 인코딩하고 review_mean은 리뷰별 벡터를 평균한 뒤 정규화한다. 본문 외의 메뉴·상호·지역 정보는 넣지 않는다.",
        "OpenRouter Liquid LFM2.5-Embedding-350M 무료 모델의 1,024차원 벡터로 코사인 검색한다. own_reviews는 사용자 query: / 식당 document: 접두어를 쓴다. liked_items는 같은 사용자 리뷰 이벤트에서 방문한 식당의 document 벡터를 평균한다. 각 API 입력은 500토큰 이하이며 최신 리뷰부터 선택한다.",
        "여러 입력을 한 요청에 묶고 제한된 동시성·호출 속도로 비동기 실행한다. 성공한 배치는 바로 캐시해 중단 후 재사용한다. 본문·모델·차원·프로필 버전으로 SHA-256 cache key를 만든다.",
        "검증은 T1까지의 리뷰만, 테스트 재구성은 T2까지의 리뷰만 사용한다. 평점·리뷰가 없는 프로필은 후보를 반환하지 않는다.",
    )
    leakage_checks = {"profiles_from_historical_interactions_only": True}

    def __init__(self) -> None:
        self.query_users: tuple[int, ...] = ()

    def config_type(self) -> type:
        from rating_recsys.retrieval.review_embeddings import ReviewEmbeddingConfig

        return ReviewEmbeddingConfig

    def prepare_fit(self, queries: Sequence[WindowQuery]) -> None:
        self.query_users = tuple(query.user_id for query in queries)

    def fit(self, model_config, interactions, texts, callback=None):
        from pathlib import Path
        from time import perf_counter
        from types import SimpleNamespace

        from rating_recsys.config import PROJECT_ROOT
        from rating_recsys.retrieval.review_embeddings import (
            create_embedding_cache, ProfileFormatter, fit_review_profiles,
        )

        if texts is None:
            raise ValueError("Review embeddings need review texts")
        if model_config.epochs != 1:
            raise ValueError("Review embeddings have one fixed fitting step; use --max-epochs 1")
        started = perf_counter()
        cache_path = Path(model_config.cache_path)
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        formatter = ProfileFormatter(model_config, cache_path.parent / "tokenizers")
        import sys
        with create_embedding_cache(
            cache_path, model_config, progress=lambda message: print(message, file=sys.stderr, flush=True),
        ) as cache:
            fitted = fit_review_profiles(
                interactions, texts, self.query_users, model_config, formatter, cache,
            )
            fitted.api_usage = dict(cache.usage)
            fitted.cache_totals = cache.totals()
        fitted.edge_count = len(interactions)
        stats = SimpleNamespace(epoch=1, loss=0.0, seconds=perf_counter() - started)
        fitted.history = [stats]
        if callback:
            callback(stats, fitted)
        return fitted

    def diagnostics(self, model, model_config, queries) -> dict[str, float]:
        return {"query_profile_coverage": len(model.user_ids) / len(queries) if queries else 0.0}

    def model_summary(self, model) -> dict[str, object]:
        return {
            "training_interactions": model.edge_count,
            "users_with_review_profiles": len(model.user_ids),
            "restaurants_with_review_profiles": len(model.item_ids),
            "openrouter_usage": model.api_usage,
            "embedding_cache_totals": model.cache_totals,
            "profile_preprocessing": model.profile_metadata,
        }
