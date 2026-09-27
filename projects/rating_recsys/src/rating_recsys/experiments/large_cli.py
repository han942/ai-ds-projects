"""Run the full C0-C3/R1 baseline with disk-backed training rows."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from rating_recsys.config import PROJECT_ROOT, get_settings
from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.repository import InteractionRepository
from rating_recsys.db.connection import connect
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.scalable import run_scalable_baseline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Existing immutable dataset.jsonl; omit to query the current DB",
    )
    return parser


def load_snapshot(path: Path) -> list[Interaction]:
    interactions = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row["event_date"] = date.fromisoformat(row["event_date"])
            interactions.append(Interaction(**row))
    return interactions


def main() -> None:
    args = build_parser().parse_args()
    if args.snapshot is not None:
        interactions = load_snapshot(args.snapshot)
    else:
        settings = get_settings(
            require_database=True, require_user_hash_salt=False
        )
        with connect(settings.database_url) as connection:
            interactions = InteractionRepository(connection).fetch_first_interactions()
    result = run_scalable_baseline(
        interactions,
        project_root=PROJECT_ROOT,
        artifacts_root=PROJECT_ROOT / "artifacts",
        config=ExperimentConfig(),
    )
    print(json.dumps({
        "run_id": result.run_id,
        "run_dir": str(result.run_dir),
        "dataset_snapshot_id": result.manifest["snapshot"]["dataset_snapshot_id"],
        "metrics": result.metrics,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
