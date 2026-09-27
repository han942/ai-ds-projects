"""Log a finished run to the local MLflow store (``artifacts/mlflow.db``).

Experiment ``rating-recsys``::

    C3→R1 · <date> · <snapshot>        run: params, candidate + LTR metrics
    ├── quota=<value>                   validation C3 Recall@K for one quota
    └── ranker · <name>                 validation metrics + NDCG curve per tree

Only metrics, params and tags are logged. No files are copied, so
``report.md`` and every other artifact exist only in ``artifacts/runs/<run_id>``;
the ``local_run_dir`` tag points there.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from rating_recsys.experiments.artifacts import read_json


EXPERIMENT_NAME = "rating-recsys"
METRIC_AT_K = re.compile(r"^(?P<metric>.+)_at_(?P<cutoff>\d+)$")
STAGES = (
    "c0_popularity",
    "c1_item_item",
    "c2_region_popularity",
    "c3_rrf_union",
    "r0_candidate_order",
    "r1_lambdarank",
)


def tracking_uri(artifacts_root: Path) -> str:
    return f"sqlite:///{(artifacts_root / 'mlflow.db').resolve()}"


class _MetricBuffer:
    """Collect metrics and write them with ``log_batch`` (one call per 1000)."""

    def __init__(self) -> None:
        self.items: list[tuple[str, float, int]] = []

    def add(self, key: str, value: float, step: int = 0) -> None:
        self.items.append((key, float(value), int(step)))

    def add_stage(self, phase: str, stage: str, values: dict) -> None:
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self.add(f"{phase}/{stage}/{key}", value)
            match = METRIC_AT_K.match(key)
            if match:
                self.add(
                    f"{phase}/{stage}/{match.group('metric')}",
                    value,
                    step=int(match.group("cutoff")),
                )

    def flush(self, client, run_id: str) -> None:
        from mlflow.entities import Metric

        timestamp = int(time.time() * 1000)
        metrics = [Metric(key, value, timestamp, step) for key, value, step in self.items]
        for start in range(0, len(metrics), 1000):
            client.log_batch(run_id, metrics=metrics[start : start + 1000])
        self.items.clear()


def _param(value: object) -> str:
    return json.dumps(value) if isinstance(value, (list, dict)) else str(value)


def log_run(run_dir: Path, *, artifacts_root: Path) -> dict[str, object]:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("MLflow is required; install the experiment extra") from exc

    manifest = read_json(run_dir / "manifest.json")
    metrics = read_json(run_dir / "metrics.json")
    config = manifest["config"]
    snapshot_id = manifest["snapshot"]["dataset_snapshot_id"]
    selection = metrics["selection"]

    mlflow.set_tracking_uri(tracking_uri(artifacts_root))
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    # Nothing is uploaded, but pin the location so MLflow never falls back to
    # creating ./mlruns in the current working directory.
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=(artifacts_root / "mlflow_unused").resolve().as_uri(),
        )
    )
    run_name = (
        f"C3→R1 · {manifest['created_at'][:10]} · {snapshot_id[:8]}"
        + (f" · {manifest['label']}" if manifest.get("label") else "")
    )
    report = (run_dir / "report.md").resolve()
    child_ids: list[str] = []
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as parent:
        mlflow.set_tags(
            {
                "pipeline_run_id": manifest["run_id"],
                "local_run_dir": str(run_dir.resolve()),
                "report": str(report),
                "dataset_snapshot_id": snapshot_id,
                "git_commit": manifest["code"].get("git_commit") or "unknown",
                "git_dirty": str(manifest["code"].get("git_dirty", False)).lower(),
                "train_through": manifest["split"]["cutoffs"]["train_through"],
                "validation_through": manifest["split"]["cutoffs"]["validation_through"],
                "label": manifest.get("label") or "",
                "mlflow.note.content": (
                    "Global-cutoff two-stage run. Full report: " + str(report)
                ),
            }
        )
        mlflow.log_params({key: _param(value) for key, value in config.items()})
        mlflow.log_params(
            {
                "chosen_quota": _param(
                    selection["chosen_candidate_policy"]["base_quota_fraction"]
                ),
                "chosen_ranker": selection["chosen_ranker"],
                **{
                    f"final_{key}": _param(value)
                    for key, value in selection["final_ranker_params"].items()
                },
            }
        )
        buffer = _MetricBuffer()
        for phase in ("validation", "test"):
            for stage in STAGES:
                buffer.add_stage(phase, stage, metrics[phase][stage])
            buffer.add(
                f"{phase}/positive_availability",
                metrics[phase]["positive_availability"]["rate"],
            )
        for key, values in (metrics["test"].get("bootstrap_r1_minus_r0") or {}).items():
            if isinstance(values, dict):
                prefix = f"test/r1_minus_r0/{key}"
                buffer.add(f"{prefix}/delta", values["delta"])
                buffer.add(f"{prefix}/ci95_low", values["ci95"][0])
                buffer.add(f"{prefix}/ci95_high", values["ci95"][1])
        buffer.flush(client, parent.info.run_id)

        chosen_quota = selection["chosen_candidate_policy"]["base_quota_fraction"]
        for row in selection["candidate_policy_grid"]:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"quota={row['base_quota_fraction']}",
                nested=True,
            ) as child:
                child_ids.append(child.info.run_id)
                mlflow.set_tags(
                    {
                        "grid": "candidate_policy",
                        "pipeline_run_id": manifest["run_id"],
                        "chosen": str(row["base_quota_fraction"] == chosen_quota).lower(),
                    }
                )
                mlflow.log_params(
                    {
                        "base_quota_fraction": _param(row["base_quota_fraction"]),
                        "base_quota": _param(row["base_quota"]),
                    }
                )
                buffer.add_stage("validation", "c3_rrf_union", row)
                buffer.flush(client, child.info.run_id)

        for row in selection["ranker_grid"]:
            with mlflow.start_run(
                experiment_id=experiment_id,
                run_name=f"ranker · {row['name']}",
                nested=True,
            ) as child:
                child_ids.append(child.info.run_id)
                mlflow.set_tags(
                    {
                        "grid": "ranker",
                        "pipeline_run_id": manifest["run_id"],
                        "chosen": str(row["name"] == selection["chosen_ranker"]).lower(),
                    }
                )
                mlflow.log_params(
                    {
                        key: _param(row[key])
                        for key in (
                            "num_leaves",
                            "min_child_samples",
                            "n_estimators",
                            "early_stopping",
                            "best_iteration",
                        )
                    }
                )
                buffer.add_stage(
                    "validation",
                    "r1_lambdarank",
                    {key: value for key, value in row.items() if METRIC_AT_K.match(key)},
                )
                for step, value in enumerate(
                    row.get("lightgbm_validation_ndcg_curve") or [], start=1
                ):
                    buffer.add("validation/lightgbm_ndcg_curve", value, step=step)
                buffer.flush(client, child.info.run_id)

    return {
        "experiment": EXPERIMENT_NAME,
        "run_id": parent.info.run_id,
        "child_run_ids": child_ids,
        "tracking_uri": tracking_uri(artifacts_root),
    }
