from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import run_baseline_experiment
from rating_recsys.experiments.scalable import run_scalable_baseline


def _interactions() -> list[Interaction]:
    sequences = {
        10: (1, 2, 3, 4),
        20: (2, 3, 4, 1),
        30: (3, 4, 1, 2),
        40: (4, 1, 2, 3),
    }
    rows = []
    review_id = 1
    for position in range(4):
        for user_id, items in sequences.items():
            restaurant_id = items[position]
            rows.append(Interaction(
                review_id=review_id,
                user_id=user_id,
                restaurant_id=restaurant_id,
                event_date=date(2025, 1, review_id),
                rating=5.0 if position != 2 else 4.0,
                reviewed_at_precision="exact",
                restaurant_name=f"restaurant-{restaurant_id}",
                region="서울" if restaurant_id <= 2 else "부산",
            ))
            review_id += 1
    return rows


def test_disk_backed_training_matches_in_memory_baseline() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        rows = _interactions()
        config = ExperimentConfig(candidate_k=4, ranking_k=4)
        standard = run_baseline_experiment(
            rows,
            project_root=root,
            artifacts_root=root / "standard",
            config=config,
            enable_mlflow=False,
        )
        scalable = run_scalable_baseline(
            rows,
            project_root=root,
            artifacts_root=root / "scalable",
            config=config,
        )

        assert standard.manifest["snapshot"] == scalable.manifest["snapshot"]
        assert standard.metrics["training"] == scalable.metrics["training"]
        assert scalable.manifest["execution"]["mode"] == "disk-backed-training"
        assert not (scalable.run_dir / "training_cache").exists()
        assert (scalable.run_dir / "execution_source" / "scalable.py").exists()
        for phase in ("validation", "test"):
            for stage in (
                "c0_popularity",
                "c1_item_item",
                "c2_region_popularity",
                "c3_rrf_union",
                "r1_lambdarank",
            ):
                old = standard.metrics[phase][stage]
                new = scalable.metrics[phase][stage]
                for key in old:
                    if key.startswith(("recall_at_", "ndcg_at_", "mrr_at_")):
                        assert old[key] == new[key], (phase, stage, key)
        assert (
            standard.run_dir / "recommendations_test.jsonl"
        ).read_bytes() == (
            scalable.run_dir / "recommendations_test.jsonl"
        ).read_bytes()
