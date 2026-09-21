"""End-to-end E0/E3/E4 offline baseline pipeline."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol

from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.split import build_global_temporal_split, build_seen_user_split
from rating_recsys.evaluation.metrics import (
    candidate_observations,
    evaluate_rankings,
    latency_summary,
    ranked_observations,
    source_contribution,
)
from rating_recsys.experiments.artifacts import (
    candidate_record,
    query_record,
    recommendation_record,
    ranked_record,
    write_json,
    write_jsonl,
)
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.models import (
    Candidate,
    FeatureRow,
    RankedCandidate,
    RecommendationQuery,
)
from rating_recsys.experiments.queries import (
    build_holdout_queries,
    build_prefix_queries,
    interactions_before,
)
from rating_recsys.experiments.snapshot import (
    code_manifest,
    environment_manifest,
    write_snapshot,
)
from rating_recsys.observability.tracking import ExperimentTrackingSession
from rating_recsys.ranking.features import FEATURE_NAMES, build_feature_rows
from rating_recsys.ranking.lambdarank import LightGBMLambdaRanker
from rating_recsys.retrieval.baselines import BaselineCandidateGenerator


class Ranker(Protocol):
    def fit(self, rows: Iterable[FeatureRow]) -> None: ...

    def rank(self, rows: Iterable[FeatureRow]) -> tuple[RankedCandidate, ...]: ...

    def feature_importance(self) -> dict[str, float]: ...

    def save(self, destination: Path) -> None: ...


@dataclass(slots=True)
class PhaseData:
    queries: tuple[RecommendationQuery, ...]
    popularity: dict[str, tuple[Candidate, ...]]
    item_item: dict[str, tuple[Candidate, ...]]
    union: dict[str, tuple[Candidate, ...]]
    feature_rows: tuple[FeatureRow, ...]
    eligible_catalog: set[int]
    target_available: dict[str, bool]
    retrieval_latencies_ms: list[float]
    item_popularity: dict[int, int]
    item_regions: dict[int, str]


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, object]
    manifest: dict[str, object]


def run_baseline_experiment(
    interactions: Iterable[Interaction],
    *,
    project_root: Path,
    artifacts_root: Path,
    config: ExperimentConfig,
    allow_dirty: bool = False,
    enable_mlflow: bool = True,
    ranker_factory: Callable[[int], Ranker] | None = None,
    review_text_by_id: Mapping[int, str | None] | None = None,
) -> ExperimentResult:
    all_interactions = tuple(interactions)
    if not all_interactions:
        raise ValueError("Cannot run an experiment without interactions")

    source = code_manifest(project_root, allow_dirty=allow_dirty)
    dataset_id = _dataset_id(all_interactions)
    now = datetime.now(timezone.utc)
    run_id = now.strftime("%Y%m%dT%H%M%S%fZ") + f"-{dataset_id[:8]}"
    run_dir = artifacts_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    dirty_diff = str(source.pop("git_diff", ""))
    if dirty_diff:
        (run_dir / "source.diff").write_text(dirty_diff, encoding="utf-8")
        source["git_diff_artifact"] = "source.diff"

    snapshot = write_snapshot(all_interactions, run_dir / "dataset.jsonl")
    write_json(run_dir / "config.json", config.to_dict())
    environment = environment_manifest()
    write_json(run_dir / "environment.json", environment)
    (run_dir / "environment.lock.txt").write_text(
        "\n".join(environment["packages"]) + "\n",
        encoding="utf-8",
    )
    (run_dir / "conda-environment.lock.txt").write_text(
        "\n".join(environment["conda_packages"]) + "\n",
        encoding="utf-8",
    )

    with ExperimentTrackingSession(
        enabled=enable_mlflow,
        artifacts_root=artifacts_root,
        run_name=(
            f"Baseline v1 · {now.strftime('%Y-%m-%d %H:%M UTC')} · "
            f"{dataset_id[:8]}"
        ),
        pipeline_run_id=run_id,
        config=config.to_dict(),
        snapshot=snapshot,
        code=source,
    ) as tracking:
        return _execute_pipeline(
            all_interactions=all_interactions,
            run_id=run_id,
            run_dir=run_dir,
            created_at=now,
            config=config,
            source=source,
            snapshot=snapshot,
            ranker_factory=ranker_factory,
            tracking=tracking,
            review_text_by_id=review_text_by_id,
        )


def _execute_pipeline(
    *,
    all_interactions: tuple[Interaction, ...],
    run_id: str,
    run_dir: Path,
    created_at: datetime,
    config: ExperimentConfig,
    source: dict[str, object],
    snapshot: dict[str, object],
    ranker_factory: Callable[[int], Ranker] | None,
    tracking: ExperimentTrackingSession,
    review_text_by_id: Mapping[int, str | None] | None,
) -> ExperimentResult:
    with tracking.span(
        "01_split_dataset",
        inputs={
            "interactions": len(all_interactions),
            "minimum_user_items": config.minimum_user_items,
        },
    ) as span:
        primary = build_seen_user_split(
            all_interactions,
            minimum_user_items=config.minimum_user_items,
        )
        temporal = build_global_temporal_split(all_interactions)
        train_queries = build_prefix_queries(
            primary.train,
            config=config,
            phase="train",
        )
        validation_queries = build_holdout_queries(
            primary.train,
            primary.validation,
            config=config,
            phase="validation",
        )
        test_history = primary.train + primary.validation
        test_queries = build_holdout_queries(
            test_history,
            primary.test,
            config=config,
            phase="test",
        )
        span.set_outputs(
            {
                "train_interactions": len(primary.train),
                "validation_interactions": len(primary.validation),
                "test_interactions": len(primary.test),
                "train_queries": len(train_queries),
                "validation_queries": len(validation_queries),
                "test_queries": len(test_queries),
            }
        )

    generator = BaselineCandidateGenerator(
        candidate_k=config.candidate_k,
        rrf_constant=config.rrf_constant,
    )
    train_data = _build_traced_phase(
        tracking,
        "02_build_train_candidates",
        train_queries,
        primary.train,
        generator,
        inject_training_targets=True,
    )
    validation_data = _build_traced_phase(
        tracking,
        "03_build_validation_candidates",
        validation_queries,
        primary.train,
        generator,
        inject_training_targets=False,
    )
    validation_training_data = _build_traced_phase(
        tracking,
        "04_build_validation_refit_candidates",
        validation_queries,
        primary.train,
        generator,
        inject_training_targets=True,
    )
    test_data = _build_traced_phase(
        tracking,
        "05_build_test_candidates",
        test_queries,
        test_history,
        generator,
        inject_training_targets=False,
    )

    factory = ranker_factory or (
        lambda seed: LightGBMLambdaRanker(
            random_seed=seed,
            ranking_k=config.ranking_k,
        )
    )
    with tracking.span(
        "06_train_validation_lambdamart",
        inputs={
            "feature_rows": len(train_data.feature_rows),
            "queries": len(train_data.queries),
        },
    ) as span:
        validation_ranker = factory(config.random_seed)
        validation_ranker.fit(train_data.feature_rows)
        validation_ranked, validation_ranking_latency = _rank_by_query(
            validation_ranker,
            validation_data.feature_rows,
        )
        validation_ranker.save(run_dir / "validation_lambdarank.txt")
        span.set_outputs(
            {
                "ranked_rows": len(validation_ranked),
                "ranking_queries": len(validation_data.queries),
            }
        )

    final_training_rows = (
        train_data.feature_rows + validation_training_data.feature_rows
    )
    with tracking.span(
        "07_train_final_lambdamart",
        inputs={
            "feature_rows": len(final_training_rows),
            "seed": config.random_seed,
        },
    ) as span:
        final_ranker = factory(config.random_seed)
        final_ranker.fit(final_training_rows)
        test_ranked, test_ranking_latency = _rank_by_query(
            final_ranker,
            test_data.feature_rows,
        )
        final_ranker.save(run_dir / "final_lambdarank.txt")
        feature_importance = final_ranker.feature_importance()
        write_json(run_dir / "feature_importance.json", feature_importance)
        span.set_outputs(
            {
                "ranked_rows": len(test_ranked),
                "ranking_queries": len(test_data.queries),
                "features": len(feature_importance),
            }
        )

    with tracking.span(
        "08_evaluate_offline",
        inputs={
            "validation_queries": len(validation_data.queries),
            "test_queries": len(test_data.queries),
        },
    ) as span:
        metrics = {
            "training": {
                "train_prefix": _training_summary(train_data),
                "validation_refit": _training_summary(validation_training_data),
            },
            "validation": _phase_metrics(
                validation_data,
                validation_ranked,
                validation_ranking_latency,
                config,
            ),
            "test": _phase_metrics(
                test_data,
                test_ranked,
                test_ranking_latency,
                config,
            ),
        }
        test_ranker_metrics = metrics["test"]["e4_lambdarank"]
        final_cutoff = max(
            int(key.removeprefix("ndcg_at_"))
            for key in test_ranker_metrics
            if key.startswith("ndcg_at_")
        )
        span.set_outputs(
            {
                "test_ndcg": test_ranker_metrics[f"ndcg_at_{final_cutoff}"],
                "test_recall": test_ranker_metrics[f"recall_at_{final_cutoff}"],
                "cutoff": final_cutoff,
            }
        )

    with tracking.span("09_write_run_artifacts") as span:
        write_json(run_dir / "metrics.json", metrics)
        write_jsonl(
            run_dir / "queries.jsonl",
            (query_record(query) for query in validation_queries + test_queries),
        )
        review_context_path = run_dir / "review_context.jsonl"
        if review_text_by_id is not None:
            display_review_ids = sorted(
                {
                    interaction.review_id
                    for query in validation_queries + test_queries
                    for interaction in (*query.history, query.target)
                }
            )
            write_jsonl(
                review_context_path,
                (
                    {
                        "review_id": review_id,
                        "review_text": review_text_by_id.get(review_id),
                    }
                    for review_id in display_review_ids
                ),
            )
        _write_phase_artifacts(
            run_dir,
            "validation",
            validation_data,
            validation_ranked,
            ranking_k=config.ranking_k,
        )
        _write_phase_artifacts(
            run_dir,
            "test",
            test_data,
            test_ranked,
            ranking_k=config.ranking_k,
        )
        manifest = {
            "run_id": run_id,
            "created_at": created_at.isoformat(),
            "config": config.to_dict(),
            "snapshot": snapshot,
            "code": source,
            "feature_schema": list(FEATURE_NAMES),
            "primary_split": primary.summary(),
            "secondary_temporal_audit": temporal.summary(),
            "artifacts": {
                "dataset": "dataset.jsonl",
                "python_environment": "environment.lock.txt",
                "conda_environment": "conda-environment.lock.txt",
                "metrics": "metrics.json",
                "queries": "queries.jsonl",
                "validation_recommendations": "recommendations_validation.jsonl",
                "test_recommendations": "recommendations_test.jsonl",
                "feature_importance": "feature_importance.json",
                "validation_model": "validation_lambdarank.txt",
                "final_model": "final_lambdarank.txt",
            },
        }
        if review_context_path.exists():
            manifest["artifacts"]["review_context"] = "review_context.jsonl"
        write_json(run_dir / "manifest.json", manifest)
        span.set_outputs(
            {
                "run_dir": str(run_dir),
                "artifact_files": len(tuple(run_dir.iterdir())),
            }
        )

    with tracking.span("10_log_mlflow_observability") as span:
        tracking_info = tracking.log_results(
            run_dir=run_dir,
            manifest=manifest,
            metrics=metrics,
        )
        manifest.update(tracking_info)
        write_json(run_dir / "manifest.json", manifest)
        span.set_outputs(
            {
                "parent_run_id": tracking_info.get("mlflow_run_id"),
                "child_run_count": len(
                    tracking_info.get("mlflow_child_run_ids", {})
                ),
            }
        )

    return ExperimentResult(
        run_id=run_id,
        run_dir=run_dir,
        metrics=metrics,
        manifest=manifest,
    )


def _dataset_id(interactions: Iterable[Interaction]) -> str:
    from rating_recsys.experiments.snapshot import dataset_digest

    return dataset_digest(interactions)


def _build_traced_phase(
    tracking: ExperimentTrackingSession,
    span_name: str,
    queries: tuple[RecommendationQuery, ...],
    reference_interactions: tuple[Interaction, ...],
    generator: BaselineCandidateGenerator,
    *,
    inject_training_targets: bool,
) -> PhaseData:
    with tracking.span(
        span_name,
        inputs={
            "queries": len(queries),
            "reference_interactions": len(reference_interactions),
            "candidate_k": generator.candidate_k,
            "inject_training_targets": inject_training_targets,
        },
    ) as span:
        data = _build_phase(
            queries,
            reference_interactions,
            generator,
            inject_training_targets=inject_training_targets,
        )
        span.set_outputs(
            {
                "feature_rows": len(data.feature_rows),
                "eligible_catalog": len(data.eligible_catalog),
                "target_available": sum(data.target_available.values()),
                "retrieval_p50_ms": latency_summary(
                    data.retrieval_latencies_ms
                )["p50_ms"],
            }
        )
        return data


def _build_phase(
    queries: tuple[RecommendationQuery, ...],
    reference_interactions: tuple[Interaction, ...],
    generator: BaselineCandidateGenerator,
    *,
    inject_training_targets: bool,
) -> PhaseData:
    popularity: dict[str, tuple[Candidate, ...]] = {}
    item_item: dict[str, tuple[Candidate, ...]] = {}
    union: dict[str, tuple[Candidate, ...]] = {}
    rows: list[FeatureRow] = []
    eligible_catalog: set[int] = set()
    availability: dict[str, bool] = {}
    latencies: list[float] = []

    for query in queries:
        available = interactions_before(reference_interactions, query)
        result, context = generator.retrieve(query, available)
        popularity[query.query_id] = result.popularity
        item_item[query.query_id] = result.item_item
        phase_union = result.union
        if inject_training_targets and query.relevance > 0:
            phase_union = generator.inject_target(query, phase_union, context)
        union[query.query_id] = phase_union
        rows.extend(build_feature_rows(query, phase_union, context))
        eligible_catalog.update(result.eligible_catalog)
        availability[query.query_id] = result.target_available
        latencies.append(result.latency_ms)

    return PhaseData(
        queries=queries,
        popularity=popularity,
        item_item=item_item,
        union=union,
        feature_rows=tuple(rows),
        eligible_catalog=eligible_catalog,
        target_available=availability,
        retrieval_latencies_ms=latencies,
        item_popularity=dict(
            Counter(item.restaurant_id for item in reference_interactions)
        ),
        item_regions={
            item.restaurant_id: item.region for item in reference_interactions
        },
    )


def _rank_by_query(
    ranker: Ranker,
    rows: tuple[FeatureRow, ...],
) -> tuple[tuple[RankedCandidate, ...], list[float]]:
    grouped: dict[str, list[FeatureRow]] = {}
    for row in rows:
        grouped.setdefault(row.query_id, []).append(row)
    ranked: list[RankedCandidate] = []
    latencies: list[float] = []
    for query_id in sorted(grouped):
        started = time.perf_counter()
        ranked.extend(ranker.rank(grouped[query_id]))
        latencies.append((time.perf_counter() - started) * 1000)
    return tuple(ranked), latencies


def _phase_metrics(
    data: PhaseData,
    ranked: tuple[RankedCandidate, ...],
    ranking_latencies_ms: list[float],
    config: ExperimentConfig,
) -> dict[str, object]:
    candidate_cutoffs = tuple(
        cutoff for cutoff in (20, 50, 100) if cutoff <= config.candidate_k
    ) or (config.candidate_k,)
    ranking_cutoffs = tuple(
        cutoff for cutoff in (5, 10) if cutoff <= config.ranking_k
    ) or (config.ranking_k,)
    available_count = sum(data.target_available.values())
    base = {
        "target_availability": {
            "available": available_count,
            "unavailable": len(data.queries) - available_count,
            "rate": available_count / len(data.queries) if data.queries else 0.0,
        },
        "retrieval_latency": latency_summary(data.retrieval_latencies_ms),
        "ranking_latency": latency_summary(ranking_latencies_ms),
    }
    for name, candidates in (
        ("e0_popularity", data.popularity),
        ("item_item_only", data.item_item),
        ("e3_rrf_union", data.union),
    ):
        base[name] = evaluate_rankings(
            candidate_observations(data.queries, candidates),
            cutoffs=candidate_cutoffs,
            catalog_ids=data.eligible_catalog,
            item_popularity=data.item_popularity,
            item_regions=data.item_regions,
        )
    base["e3_rrf_union"]["source_contribution"] = source_contribution(
        data.queries,
        data.union,
    )
    base["e4_lambdarank"] = evaluate_rankings(
        ranked_observations(data.queries, ranked),
        cutoffs=ranking_cutoffs,
        catalog_ids=data.eligible_catalog,
        item_popularity=data.item_popularity,
        item_regions=data.item_regions,
    )
    return base


def _training_summary(data: PhaseData) -> dict[str, int | float]:
    relevant_queries = [query for query in data.queries if query.relevance > 0]
    injected = sum(
        candidate.injected_for_training
        for candidates in data.union.values()
        for candidate in candidates
    )
    available = sum(
        data.target_available[query.query_id] for query in relevant_queries
    )
    retrieved_without_injection = sum(
        any(
            candidate.restaurant_id == query.target.restaurant_id
            and not candidate.injected_for_training
            for candidate in data.union.get(query.query_id, ())
        )
        for query in relevant_queries
    )
    return {
        "queries": len(data.queries),
        "feature_rows": len(data.feature_rows),
        "relevant_queries": len(relevant_queries),
        "zero_relevance_queries": len(data.queries) - len(relevant_queries),
        "available_positive_queries": available,
        "unavailable_positive_queries": len(relevant_queries) - available,
        "retrieved_positive_queries": retrieved_without_injection,
        "injected_positive_queries": injected,
        "injection_rate": injected / len(relevant_queries) if relevant_queries else 0.0,
    }


def _write_phase_artifacts(
    run_dir: Path,
    phase: str,
    data: PhaseData,
    ranked: tuple[RankedCandidate, ...],
    *,
    ranking_k: int,
) -> None:
    write_jsonl(
        run_dir / f"candidates_{phase}.jsonl",
        (
            candidate_record(candidate)
            for query_id in sorted(data.union)
            for candidate in data.union[query_id]
        ),
    )
    write_jsonl(
        run_dir / f"rankings_{phase}.jsonl",
        (ranked_record(item) for item in ranked),
    )
    write_jsonl(
        run_dir / f"recommendations_{phase}.jsonl",
        (
            recommendation_record(item)
            for item in sorted(
                ranked,
                key=lambda value: (value.row.query_id, value.final_rank),
            )
            if item.final_rank <= ranking_k
        ),
    )
