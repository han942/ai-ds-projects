"""Run a paired baseline and no-region ablation on one database snapshot."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from rating_recsys.config import PROJECT_ROOT, get_settings
from rating_recsys.datasets.repository import InteractionRepository
from rating_recsys.db.connection import connect
from rating_recsys.experiments.artifacts import write_json
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import ExperimentResult, run_baseline_experiment


METRICS = {
    "candidate_recall_at_20": ("c3_rrf_union", "recall_at_20"),
    "candidate_recall_at_50": ("c3_rrf_union", "recall_at_50"),
    "candidate_recall_at_100": ("c3_rrf_union", "recall_at_100"),
    "ranker_recall_at_5": ("r1_lambdarank", "recall_at_5"),
    "ranker_recall_at_10": ("r1_lambdarank", "recall_at_10"),
    "ranker_ndcg_at_5": ("r1_lambdarank", "ndcg_at_5"),
    "ranker_ndcg_at_10": ("r1_lambdarank", "ndcg_at_10"),
    "ranker_mrr_at_10": ("r1_lambdarank", "mrr_at_10"),
    "ranker_coverage_at_10": ("r1_lambdarank", "catalog_coverage_at_10"),
    "ranker_novelty_at_10": ("r1_lambdarank", "novelty_at_10"),
    "ranker_region_diversity_at_10": (
        "r1_lambdarank", "intra_list_region_diversity_at_10"
    ),
}


def _summary(baseline: ExperimentResult, ablation: ExperimentResult) -> dict[str, object]:
    baseline_snapshot = baseline.manifest["snapshot"]["dataset_snapshot_id"]
    ablation_snapshot = ablation.manifest["snapshot"]["dataset_snapshot_id"]
    if baseline_snapshot != ablation_snapshot:
        raise ValueError("The two runs used different dataset snapshots")
    if baseline.manifest["primary_split"] != ablation.manifest["primary_split"]:
        raise ValueError("The two runs used different evaluation splits")

    comparisons = {}
    for phase in ("validation", "test"):
        comparisons[phase] = {}
        for name, (stage, metric) in METRICS.items():
            with_region = float(baseline.metrics[phase][stage][metric])
            without_region = float(ablation.metrics[phase][stage][metric])
            comparisons[phase][name] = {
                "with_region": with_region,
                "without_region": without_region,
                "delta_without_minus_with": without_region - with_region,
            }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_snapshot_id": baseline_snapshot,
        "interactions": baseline.manifest["snapshot"]["interactions"],
        "primary_split": baseline.manifest["primary_split"],
        "with_region_run_id": baseline.run_id,
        "without_region_run_id": ablation.run_id,
        "with_region_run_dir": str(baseline.run_dir),
        "without_region_run_dir": str(ablation.run_dir),
        "definition": (
            "The no-region run removes C2 region candidates and the region_affinity "
            "ranker feature. Region metadata remains only for diversity evaluation."
        ),
        "metrics": comparisons,
    }


def main() -> None:
    settings = get_settings(require_database=True, require_user_hash_salt=False)
    with connect(settings.database_url) as connection:
        interactions = InteractionRepository(connection).fetch_first_interactions()
    artifacts_root = PROJECT_ROOT / "artifacts"
    results = []
    for region_mode in ("with_region", "without_region"):
        print(f"Starting {region_mode} on {len(interactions)} interactions", flush=True)
        result = run_baseline_experiment(
            interactions,
            project_root=PROJECT_ROOT,
            artifacts_root=artifacts_root,
            config=ExperimentConfig(region_mode=region_mode),
            allow_dirty=True,
            enable_mlflow=False,
        )
        results.append(result)
        print(f"Completed {region_mode}: {result.run_id}", flush=True)
    summary = _summary(*results)
    destination = artifacts_root / "comparisons" / (
        f"region_ablation_{results[0].run_id}_vs_{results[1].run_id}.json"
    )
    write_json(destination, summary)
    print(json.dumps({"comparison": str(destination), "results": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
