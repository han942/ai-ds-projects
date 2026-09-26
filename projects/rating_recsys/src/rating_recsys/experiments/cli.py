"""Run the reproducible C0-C3/R1 baseline from the Supabase dataset."""

from __future__ import annotations

import argparse
import json

from rating_recsys.config import PROJECT_ROOT, get_settings
from rating_recsys.datasets.repository import InteractionRepository
from rating_recsys.db.connection import connect
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import run_baseline_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser


def main() -> None:
    build_parser().parse_args()
    config = ExperimentConfig()
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
        artifacts_root=PROJECT_ROOT / "artifacts",
        config=config,
        allow_dirty=True,
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
