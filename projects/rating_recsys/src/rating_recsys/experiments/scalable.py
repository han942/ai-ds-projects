"""Disk-backed training phases for the full-data two-stage baseline.

The candidate and feature rules are shared with the normal pipeline. Only
training rows are stored as numeric matrices instead of millions of Python
Candidate/FeatureRow objects.
"""

from __future__ import annotations

import gc
import hashlib
import resource
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rating_recsys.evaluation.metrics import latency_summary
from rating_recsys.experiments import pipeline
from rating_recsys.experiments.models import RecommendationQuery
from rating_recsys.ranking.features import build_feature_rows
from rating_recsys.ranking.lambdarank import LightGBMLambdaRanker
from rating_recsys.retrieval.baselines import BaselineCandidateGenerator, IncrementalRetrievalContext


@dataclass(slots=True)
class TrainingMatrix:
    features_path: Path
    labels_path: Path
    counts: np.ndarray
    capacity_per_query: int
    feature_count: int
    feature_rows: int
    summary: dict[str, int | float]

    def __len__(self) -> int:
        return self.feature_rows

    def __add__(self, other: "TrainingMatrix") -> "CombinedTrainingMatrix":
        return CombinedTrainingMatrix((self, other))


@dataclass(slots=True)
class CombinedTrainingMatrix:
    stores: tuple[TrainingMatrix, ...]

    def __len__(self) -> int:
        return sum(len(store) for store in self.stores)


def matrix_arrays(
    value: TrainingMatrix | CombinedTrainingMatrix,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Return the same query-id and restaurant-id row order as the in-memory fit."""
    stores = value.stores if isinstance(value, CombinedTrainingMatrix) else (value,)
    total = sum(int(store.counts.sum()) for store in stores)
    feature_count = stores[0].feature_count
    features = np.empty((total, feature_count), dtype=np.float64)
    labels = np.empty(total, dtype=np.int32)
    groups: list[int] = []
    output_index = 0
    for store in stores:
        source_features = np.load(store.features_path, mmap_mode="r")
        source_labels = np.load(store.labels_path, mmap_mode="r")
        for query_index, count_value in enumerate(store.counts):
            count = int(count_value)
            if not count:
                continue
            start = query_index * store.capacity_per_query
            end = start + count
            features[output_index : output_index + count] = source_features[start:end]
            labels[output_index : output_index + count] = source_labels[start:end]
            groups.append(count)
            output_index += count
        del source_features, source_labels
    if output_index != total:
        raise RuntimeError("Training matrix row count mismatch")
    return features, labels, groups


class MatrixLambdaRanker(LightGBMLambdaRanker):
    def fit(self, rows) -> None:
        if not isinstance(rows, (TrainingMatrix, CombinedTrainingMatrix)):
            super().fit(rows)
            return
        features, labels, groups = matrix_arrays(rows)
        if not groups:
            raise ValueError("LambdaRank requires at least one usable query group")
        self._model.fit(features, labels, group=groups)
        self._fitted = True
        del features, labels
        gc.collect()


def build_training_phase(
    queries: tuple[RecommendationQuery, ...],
    reference_interactions: tuple,
    generator: BaselineCandidateGenerator,
    *,
    run_dir: Path,
    phase_name: str,
    feature_names: tuple[str, ...],
) -> pipeline.PhaseData:
    """Generate cutoff-safe training rows one query at a time to disk."""
    storage_dir = run_dir / "training_cache"
    storage_dir.mkdir(parents=True, exist_ok=True)
    capacity_per_query = generator.candidate_k + 1
    capacity = len(queries) * capacity_per_query
    features_path = storage_dir / f"{phase_name}_features.npy"
    labels_path = storage_dir / f"{phase_name}_labels.npy"
    feature_matrix = np.lib.format.open_memmap(
        features_path, mode="w+", dtype=np.float64,
        shape=(capacity, len(feature_names)),
    )
    label_vector = np.lib.format.open_memmap(
        labels_path, mode="w+", dtype=np.int32, shape=(capacity,),
    )
    counts = np.zeros(len(queries), dtype=np.int16)
    query_indexes = {query.query_id: index for index, query in enumerate(queries)}
    ordered_queries = sorted(
        queries, key=lambda query: pipeline.global_interaction_key(query.target)
    )
    ordered_interactions = sorted(
        reference_interactions, key=pipeline.global_interaction_key
    )
    context_builder = IncrementalRetrievalContext()
    interaction_index = 0
    target_available: dict[str, bool] = {}
    retrieval_latencies: list[float] = []
    eligible_catalog: set[int] = set()
    feature_rows = 0
    relevant_queries = 0
    available_positive = 0
    retrieved_positive = 0
    injected_positive = 0

    for processed, query in enumerate(ordered_queries, start=1):
        if processed % 5000 == 0:
            print(
                f"{phase_name}: {processed}/{len(queries)} queries; "
                f"max_rss_kib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}",
                file=sys.stderr,
                flush=True,
            )
        cutoff = pipeline.global_interaction_key(query.target)
        while (
            interaction_index < len(ordered_interactions)
            and pipeline.global_interaction_key(ordered_interactions[interaction_index])
            < cutoff
        ):
            context_builder.add(ordered_interactions[interaction_index])
            interaction_index += 1
        result, context = generator.retrieve_from_context(
            query, context_builder.context
        )
        candidates = generator.inject_target(query, result.union, context) if query.relevance > 0 else result.union
        rows = build_feature_rows(
            query, candidates, context, include_region=generator.include_region
        )
        feature_rows += len(rows)
        target_available[query.query_id] = result.target_available
        retrieval_latencies.append(result.latency_ms)
        eligible_catalog.update(result.eligible_catalog)

        if query.relevance > 0:
            relevant_queries += 1
            available_positive += int(result.target_available)
            retrieved_positive += int(any(
                candidate.restaurant_id == query.target.restaurant_id
                for candidate in result.union
            ))
            injected_positive += int(any(
                candidate.injected_for_training for candidate in candidates
            ))

        ordered_rows = sorted(rows, key=lambda row: row.restaurant_id)
        usable = (
            any(row.relevance > 0 for row in ordered_rows)
            and any(row.relevance == 0 for row in ordered_rows)
        )
        if not usable:
            continue
        query_index = query_indexes[query.query_id]
        start = query_index * capacity_per_query
        count = len(ordered_rows)
        if count > capacity_per_query:
            raise RuntimeError("Candidate count exceeded matrix capacity")
        for offset, row in enumerate(ordered_rows):
            feature_matrix[start + offset] = [
                float(row.features[name]) for name in feature_names
            ]
            label_vector[start + offset] = row.relevance
        counts[query_index] = count

    feature_matrix.flush()
    label_vector.flush()
    del feature_matrix, label_vector
    summary: dict[str, int | float] = {
        "queries": len(queries),
        "feature_rows": feature_rows,
        "relevant_queries": relevant_queries,
        "zero_relevance_queries": len(queries) - relevant_queries,
        "available_positive_queries": available_positive,
        "unavailable_positive_queries": relevant_queries - available_positive,
        "retrieved_positive_queries": retrieved_positive,
        "injected_positive_queries": injected_positive,
        "injection_rate": injected_positive / relevant_queries if relevant_queries else 0.0,
    }
    store = TrainingMatrix(
        features_path=features_path,
        labels_path=labels_path,
        counts=counts,
        capacity_per_query=capacity_per_query,
        feature_count=len(feature_names),
        feature_rows=feature_rows,
        summary=summary,
    )
    return pipeline.PhaseData(
        queries=queries,
        popularity={},
        item_item={},
        region_popularity={},
        union={},
        feature_rows=store,
        eligible_catalog=eligible_catalog,
        target_available=target_available,
        retrieval_latencies_ms=retrieval_latencies,
        item_popularity={},
        item_regions={},
    )


def run_scalable_baseline(
    interactions,
    *,
    project_root: Path,
    artifacts_root: Path,
    config,
    allow_dirty: bool = True,
):
    """Run the standard baseline with file-backed training rows."""
    original_builder = pipeline._build_traced_phase
    original_summary = pipeline._training_summary
    run_directory: Path | None = None
    feature_names = (
        pipeline.FEATURE_NAMES
        if config.include_region
        else pipeline.NO_REGION_FEATURE_NAMES
    )

    def build_phase(
        tracking, span_name, queries, reference_interactions, generator,
        *, inject_training_targets, retain_source_candidates,
    ):
        nonlocal run_directory
        if not (inject_training_targets and not retain_source_candidates):
            return original_builder(
                tracking, span_name, queries, reference_interactions, generator,
                inject_training_targets=inject_training_targets,
                retain_source_candidates=retain_source_candidates,
            )
        if run_directory is None:
            run_directory = artifacts_root / "runs" / tracking.pipeline_run_id
        with tracking.span(span_name, inputs={
            "queries": len(queries),
            "reference_interactions": len(reference_interactions),
            "candidate_k": generator.candidate_k,
            "inject_training_targets": True,
        }) as span:
            data = build_training_phase(
                queries, reference_interactions, generator,
                run_dir=run_directory, phase_name=span_name,
                feature_names=feature_names,
            )
            span.set_outputs({
                "feature_rows": len(data.feature_rows),
                "eligible_catalog": len(data.eligible_catalog),
                "target_available": sum(data.target_available.values()),
                "retrieval_p50_ms": latency_summary(data.retrieval_latencies_ms)["p50_ms"],
            })
            return data

    def training_summary(data):
        if isinstance(data.feature_rows, TrainingMatrix):
            return data.feature_rows.summary
        return original_summary(data)

    pipeline._build_traced_phase = build_phase
    pipeline._training_summary = training_summary
    try:
        result = pipeline.run_baseline_experiment(
            interactions,
            project_root=project_root,
            artifacts_root=artifacts_root,
            config=config,
            allow_dirty=allow_dirty,
            enable_mlflow=False,
            review_text_by_id=None,
            ranker_factory=lambda seed: MatrixLambdaRanker(
                random_seed=seed,
                ranking_k=config.ranking_k,
                feature_names=feature_names,
            ),
        )
    finally:
        pipeline._build_traced_phase = original_builder
        pipeline._training_summary = original_summary

    source_dir = result.run_dir / "execution_source"
    source_dir.mkdir(exist_ok=True)
    source_files = (
        Path(__file__),
        Path(__file__).with_name("large_cli.py"),
        Path(__file__).parents[1] / "retrieval" / "baselines.py",
    )
    source_bundle = {}
    source_root = Path(__file__).resolve().parents[3]
    for source_path in source_files:
        destination = source_dir / source_path.name
        shutil.copy2(source_path, destination)
        source_bundle[str(source_path.relative_to(source_root))] = {
            "artifact": str(destination.relative_to(result.run_dir)),
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
    shutil.rmtree(result.run_dir / "training_cache")
    result.manifest["execution"] = {
        "mode": "disk-backed-training",
        "mlflow_enabled": False,
        "review_context_included": False,
        "source_bundle": source_bundle,
        "temporary_training_cache_removed": True,
    }
    pipeline.write_json(result.run_dir / "manifest.json", result.manifest)
    return result
