"""Structured MLflow tracking for the two-stage recommendation baseline."""

from __future__ import annotations

import json
import os
import re
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


EXPERIMENT_NAME = "rating-recsys-baseline"
STAGE_RUNS = (
    ("c0_popularity", "01 · C0 Popularity"),
    ("c1_item_item", "02 · C1 Item-item CF"),
    ("c2_region_popularity", "03 · C2 Region popularity"),
    ("c3_rrf_union", "04 · C3 RRF Candidate Union"),
    ("r1_lambdarank", "05 · R1 LambdaMART"),
)
STAGE_DESCRIPTIONS = {
    "c0_popularity": "Global popularity candidate baseline.",
    "c1_item_item": "Item-item collaborative filtering candidate baseline.",
    "c2_region_popularity": "Popularity restricted and weighted by the user's observed regions.",
    "c3_rrf_union": "Popularity, item-item and region candidates fused with RRF.",
    "r1_lambdarank": (
        "LambdaMART reranking over the C3 candidate union."
    ),
}
METRIC_AT_K = re.compile(r"^(?P<metric>.+)_at_(?P<cutoff>\d+)$")


class _NullSpan:
    def set_inputs(self, _inputs: dict[str, object]) -> None:
        return None

    def set_outputs(self, _outputs: dict[str, object]) -> None:
        return None

    def set_attribute(self, _key: str, _value: object) -> None:
        return None


class ExperimentTrackingSession:
    """Own one parent MLflow run and its pipeline trace."""

    def __init__(
        self,
        *,
        enabled: bool,
        artifacts_root: Path,
        run_name: str,
        pipeline_run_id: str,
        config: dict[str, object],
        snapshot: dict[str, object],
        code: dict[str, object],
    ) -> None:
        self.enabled = enabled
        self.artifacts_root = artifacts_root
        self.run_name = run_name
        self.pipeline_run_id = pipeline_run_id
        self.config = config
        self.snapshot = snapshot
        self.code = code
        self.run_id: str | None = None
        self.child_run_ids: dict[str, str] = {}
        self._run_context = None
        self._trace_context = None
        self._root_span = _NullSpan()

    def __enter__(self) -> "ExperimentTrackingSession":
        if not self.enabled:
            return self

        mlflow = _mlflow()
        self.artifacts_root.mkdir(parents=True, exist_ok=True)
        tracking_database = (self.artifacts_root / "mlflow.db").resolve()
        mlflow.set_tracking_uri(f"sqlite:///{tracking_database}")
        mlflow.set_experiment(EXPERIMENT_NAME)
        self._run_context = mlflow.start_run(run_name=self.run_name)
        run = self._run_context.__enter__()
        self.run_id = run.info.run_id
        mlflow.log_params({key: str(value) for key, value in self.config.items()})
        mlflow.set_tags(
            {
                "run_role": "parent",
                "pipeline_run_id": self.pipeline_run_id,
                "local_run_dir": str(
                    (
                        self.artifacts_root / "runs" / self.pipeline_run_id
                    ).resolve()
                ),
                "dataset_snapshot_id": self.snapshot["dataset_snapshot_id"],
                "git_commit": self.code.get("git_commit") or "unknown",
                "git_dirty": str(self.code.get("git_dirty", False)).lower(),
                "protocol": self.config["protocol"],
                "region_mode": self.config["region_mode"],
                "feature_schema_version": self.config["schema_version"],
                "mlflow.note.content": (
                    "Two-stage recommendation baseline. Compare C0-C3 candidate "
                    "components and R1 ranking in the child runs; inspect Dataset inputs, Tables, "
                    "charts and Traces on this parent run."
                ),
            }
        )
        self._trace_context = mlflow.start_span(
            name="baseline_pipeline",
            attributes={"pipeline.type": "two-stage-recommender"},
        )
        self._root_span = self._trace_context.__enter__()
        self._root_span.set_inputs(
            {
                "dataset_snapshot_id": self.snapshot["dataset_snapshot_id"],
                "interactions": self.snapshot["interactions"],
                "users": self.snapshot["users"],
                "restaurants": self.snapshot["restaurants"],
                "candidate_k": self.config["candidate_k"],
                "ranking_k": self.config["ranking_k"],
            }
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if not self.enabled:
            return False
        if self._trace_context is not None:
            self._trace_context.__exit__(exc_type, exc_value, traceback)
        _mlflow().flush_trace_async_logging()
        if self._run_context is not None:
            self._run_context.__exit__(exc_type, exc_value, traceback)
        return False

    @contextmanager
    def span(
        self,
        name: str,
        *,
        inputs: dict[str, object] | None = None,
        attributes: dict[str, object] | None = None,
    ) -> Iterator[object]:
        if not self.enabled:
            yield _NullSpan()
            return

        mlflow = _mlflow()
        with mlflow.start_span(name=name, attributes=attributes) as span:
            if inputs:
                span.set_inputs(inputs)
            yield span

    def log_results(
        self,
        *,
        run_dir: Path,
        manifest: dict[str, object],
        metrics: dict[str, object],
    ) -> dict[str, object]:
        if not self.enabled:
            return {}

        mlflow = _mlflow()
        headline = _headline_metrics(metrics)
        mlflow.log_metrics(headline)
        _log_dataset(run_dir)
        _log_stage_table(metrics)
        _log_recommendation_table(run_dir)
        _log_figures(run_dir, metrics)

        for stage_key, run_name in STAGE_RUNS:
            with mlflow.start_run(run_name=run_name, nested=True) as child:
                self.child_run_ids[stage_key] = child.info.run_id
                mlflow.set_tags(
                    {
                        "run_role": "stage",
                        "stage": stage_key,
                        "parent_pipeline_run": self.run_id or "unknown",
                        "mlflow.note.content": STAGE_DESCRIPTIONS[stage_key],
                        "dataset_snapshot_id": manifest["snapshot"][
                            "dataset_snapshot_id"
                        ],
                    }
                )
                mlflow.log_params(
                    {
                        "candidate_k": str(self.config["candidate_k"]),
                        "ranking_k": str(self.config["ranking_k"]),
                        "rrf_constant": str(self.config["rrf_constant"]),
                    }
                )
                _log_stage_run(stage_key, metrics)

        tracking_info = {
            "mlflow_run_id": self.run_id,
            "mlflow_child_run_ids": dict(self.child_run_ids),
        }
        manifest.update(tracking_info)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _log_small_artifacts(run_dir)
        self._root_span.set_outputs(
            {
                "mlflow_run_id": self.run_id,
                "child_runs": self.child_run_ids,
                "headline_metrics": headline,
                "local_run_dir": str(run_dir),
            }
        )
        return tracking_info


def _mlflow():
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            "MLflow is required for experiment runs; install "
            "`pip install -e '.[experiment]'`"
        ) from exc
    return mlflow


def _headline_metrics(metrics: dict[str, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for phase in ("validation", "test"):
        phase_metrics = metrics[phase]
        availability = phase_metrics["target_availability"]
        result[f"{phase}/target_availability_rate"] = float(availability["rate"])
        for latency_name in ("retrieval_latency", "ranking_latency"):
            latency = phase_metrics[latency_name]
            result[f"{phase}/{latency_name}/p50_ms"] = float(latency["p50_ms"])
            result[f"{phase}/{latency_name}/p95_ms"] = float(latency["p95_ms"])
        for stage in (
            "c0_popularity",
            "c1_item_item",
            "c2_region_popularity",
            "c3_rrf_union",
        ):
            values = phase_metrics[stage]
            cutoff = _maximum_cutoff(values, "recall")
            result[f"{phase}/{stage}/recall_at_{cutoff}"] = float(
                values[f"recall_at_{cutoff}"]
            )
        ranker = phase_metrics["r1_lambdarank"]
        cutoff = _maximum_cutoff(ranker, "recall")
        for name in ("recall", "ndcg", "mrr", "catalog_coverage"):
            key = f"{name}_at_{cutoff}"
            result[f"{phase}/r1_lambdarank/{key}"] = float(ranker[key])
    training = metrics["training"]
    result["training/train_prefix/injection_rate"] = float(
        training["train_prefix"]["injection_rate"]
    )
    result["training/validation_refit/injection_rate"] = float(
        training["validation_refit"]["injection_rate"]
    )
    return result


def _maximum_cutoff(values: dict[str, object], metric: str) -> int:
    prefix = f"{metric}_at_"
    cutoffs = [int(key.removeprefix(prefix)) for key in values if key.startswith(prefix)]
    if not cutoffs:
        raise ValueError(f"No {metric}@K metric found")
    return max(cutoffs)


def _log_stage_run(stage_key: str, metrics: dict[str, object]) -> None:
    mlflow = _mlflow()
    for phase in ("validation", "test"):
        values = metrics[phase][stage_key]
        for key, value in values.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            match = METRIC_AT_K.match(key)
            if match:
                metric_name = match.group("metric")
                cutoff = int(match.group("cutoff"))
                mlflow.log_metric(f"{phase}/{metric_name}", float(value), step=cutoff)
                mlflow.log_metric(f"{phase}/{key}", float(value))
            else:
                mlflow.log_metric(f"{phase}/{key}", float(value))


def _log_dataset(run_dir: Path) -> None:
    mlflow = _mlflow()
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for MLflow dataset tracking") from exc

    frame = pd.read_json(run_dir / "dataset.jsonl", lines=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The specified dataset source can be interpreted in multiple ways",
        )
        warnings.filterwarnings(
            "ignore",
            message="Hint: Inferred schema contains integer column",
        )
        dataset = mlflow.data.from_pandas(
            frame,
            source=str((run_dir / "dataset.jsonl").resolve()),
            name="rating-recsys-first-interactions",
        )
        mlflow.log_input(dataset, context="modeling-snapshot")


def _stage_rows(metrics: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for phase in ("validation", "test"):
        for stage_key, stage_name in STAGE_RUNS:
            values = metrics[phase][stage_key]
            cutoffs = sorted(
                {
                    int(match.group("cutoff"))
                    for key in values
                    if (match := METRIC_AT_K.match(key))
                }
            )
            for cutoff in cutoffs:
                rows.append(
                    {
                        "phase": phase,
                        "stage": stage_name,
                        "stage_key": stage_key,
                        "cutoff": cutoff,
                        "recall": values.get(f"recall_at_{cutoff}"),
                        "ndcg": values.get(f"ndcg_at_{cutoff}"),
                        "mrr": values.get(f"mrr_at_{cutoff}"),
                        "catalog_coverage": values.get(
                            f"catalog_coverage_at_{cutoff}"
                        ),
                        "novelty": values.get(f"novelty_at_{cutoff}"),
                        "region_diversity": values.get(
                            f"intra_list_region_diversity_at_{cutoff}"
                        ),
                        "evaluated_queries": values.get("evaluated_queries"),
                    }
                )
    return rows


def _log_stage_table(metrics: dict[str, object]) -> None:
    mlflow = _mlflow()
    rows = _stage_rows(metrics)
    mlflow.log_table(
        data={
            key: [row.get(key) for row in rows]
            for key in (
                "phase",
                "stage",
                "stage_key",
                "cutoff",
                "recall",
                "ndcg",
                "mrr",
                "catalog_coverage",
                "novelty",
                "region_diversity",
                "evaluated_queries",
            )
        },
        artifact_file="tables/stage_metrics.json",
    )


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _log_recommendation_table(run_dir: Path) -> None:
    mlflow = _mlflow()
    queries = {
        row["query_id"]: row
        for row in _read_jsonl(run_dir / "queries.jsonl")
        if row["phase"] == "test"
    }
    columns = {
        key: []
        for key in (
            "query_id",
            "user_id",
            "history_depth",
            "target_restaurant_id",
            "target_restaurant_name",
            "recommended_restaurant_id",
            "recommended_restaurant_name",
            "region",
            "candidate_sources",
            "candidate_rank",
            "final_rank",
            "ranking_score",
            "is_target",
        )
    }
    for recommendation in _read_jsonl(run_dir / "recommendations_test.jsonl"):
        query = queries[recommendation["query_id"]]
        values = {
            "query_id": recommendation["query_id"],
            "user_id": recommendation["user_id"],
            "history_depth": query["history_depth"],
            "target_restaurant_id": query["target_restaurant_id"],
            "target_restaurant_name": query["target_restaurant_name"],
            "recommended_restaurant_id": recommendation["restaurant_id"],
            "recommended_restaurant_name": recommendation["restaurant_name"],
            "region": recommendation["region"],
            "candidate_sources": ", ".join(recommendation["candidate_sources"]),
            "candidate_rank": recommendation["candidate_rank"],
            "final_rank": recommendation["final_rank"],
            "ranking_score": recommendation["ranking_score"],
            "is_target": (
                recommendation["restaurant_id"] == query["target_restaurant_id"]
            ),
        }
        for key, value in values.items():
            columns[key].append(value)
    mlflow.log_table(
        data=columns,
        artifact_file="tables/test_recommendations.json",
    )


def _log_figures(run_dir: Path, metrics: dict[str, object]) -> None:
    matplotlib_config = run_dir.parent.parent / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    mlflow = _mlflow()
    rows = [row for row in _stage_rows(metrics) if row["phase"] == "test"]
    labels = [f"{row['stage']}@{row['cutoff']}" for row in rows]
    recall = [float(row["recall"]) for row in rows]
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(labels, recall)
    axis.set_title("Test recall by recommendation stage and cutoff")
    axis.set_ylabel("Recall")
    axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    mlflow.log_figure(figure, "charts/test_recall_by_stage.png")
    plt.close(figure)

    importance = json.loads(
        (run_dir / "feature_importance.json").read_text(encoding="utf-8")
    )
    ordered = sorted(importance.items(), key=lambda item: item[1])
    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh([item[0] for item in ordered], [item[1] for item in ordered])
    axis.set_title("LambdaMART feature importance")
    axis.set_xlabel("Split importance")
    figure.tight_layout()
    mlflow.log_figure(figure, "charts/feature_importance.png")
    plt.close(figure)


def _log_small_artifacts(run_dir: Path) -> None:
    mlflow = _mlflow()
    for name in (
        "manifest.json",
        "config.json",
        "metrics.json",
        "feature_importance.json",
        "environment.json",
        "environment.lock.txt",
        "conda-environment.lock.txt",
        "validation_lambdarank.txt",
        "final_lambdarank.txt",
        "source.diff",
    ):
        path = run_dir / name
        if path.exists():
            mlflow.log_artifact(str(path), artifact_path="run")


def _flatten_numeric(
    value: object,
    prefix: str = "",
) -> Iterator[tuple[str, float]]:
    """Retained for loading or comparing legacy flat-metric runs."""

    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_numeric(child, child_prefix)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield prefix, float(value)
