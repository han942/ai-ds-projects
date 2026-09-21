"""Run the reproducible E0/E3/E4 baseline from the Supabase dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rating_recsys.config import PROJECT_ROOT, get_settings
from rating_recsys.datasets.repository import InteractionRepository
from rating_recsys.db.connection import connect
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import run_baseline_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--ranking-k", type=int, default=10)
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--minimum-user-items", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty worktree and save its diff in the run artifacts",
    )
    parser.add_argument(
        "--disable-mlflow",
        action="store_true",
        help="Write local run artifacts without logging them to MLflow",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ExperimentConfig(
        minimum_user_items=args.minimum_user_items,
        candidate_k=args.candidate_k,
        ranking_k=args.ranking_k,
        rrf_constant=args.rrf_constant,
        random_seed=args.seed,
    )
    settings = get_settings(
        require_database=True,
        require_user_hash_salt=False,
    )
    with connect(settings.database_url) as connection:
        repository = InteractionRepository(connection)
        interactions = repository.fetch_first_interactions()
        review_text_by_id = repository.fetch_review_texts(
            [interaction.review_id for interaction in interactions]
        )
    result = run_baseline_experiment(
        interactions,
        project_root=PROJECT_ROOT,
        artifacts_root=args.artifacts_dir,
        config=config,
        allow_dirty=args.allow_dirty,
        enable_mlflow=not args.disable_mlflow,
        review_text_by_id=review_text_by_id,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "run_dir": str(result.run_dir),
                "dataset_snapshot_id": result.manifest["snapshot"][
                    "dataset_snapshot_id"
                ],
                "metrics": result.metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
