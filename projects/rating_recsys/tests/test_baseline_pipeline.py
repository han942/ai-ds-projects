from __future__ import annotations

import tempfile
import unittest
from importlib.util import find_spec
from datetime import date
from pathlib import Path

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.models import FeatureRow, RankedCandidate
from rating_recsys.experiments.pipeline import run_baseline_experiment


def interactions() -> list[Interaction]:
    sequences = {
        10: (1, 2, 3, 4),
        20: (2, 3, 4, 1),
        30: (3, 4, 1, 2),
        40: (4, 1, 2, 3),
    }
    rows: list[Interaction] = []
    review_id = 1
    for position in range(4):
        for user_id, sequence in sequences.items():
            restaurant_id = sequence[position]
            rows.append(
                Interaction(
                    review_id=review_id,
                    user_id=user_id,
                    restaurant_id=restaurant_id,
                    event_date=date(2025, 1, review_id),
                    rating=5.0 if position != 2 else 4.0,
                    reviewed_at_precision="exact",
                    restaurant_name=f"restaurant-{restaurant_id}",
                    region="서울" if restaurant_id <= 2 else "부산",
                )
            )
            review_id += 1
    return rows


class FakeRanker:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def fit(self, rows) -> None:
        self.fitted_rows = tuple(rows)
        if not any(row.relevance > 0 for row in self.fitted_rows):
            raise ValueError("fixture must contain a positive")

    def rank(self, rows) -> tuple[RankedCandidate, ...]:
        ordered = sorted(
            rows,
            key=lambda row: (
                -row.features["item_item_sum_similarity"],
                -row.features["popularity_score"],
                row.restaurant_id,
            ),
        )
        return tuple(
            RankedCandidate(row=row, ranking_score=float(len(ordered) - index), final_rank=index + 1)
            for index, row in enumerate(ordered)
        )

    def feature_importance(self) -> dict[str, float]:
        return {"item_item_sum_similarity": 1.0}

    def save(self, destination: Path) -> None:
        destination.write_text("fake-model\n", encoding="utf-8")


class BaselinePipelineTests(unittest.TestCase):
    def test_writes_reproducible_stage_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_baseline_experiment(
                interactions(),
                project_root=root,
                artifacts_root=root / "artifacts",
                config=ExperimentConfig(candidate_k=4, ranking_k=4),
                enable_mlflow=False,
                ranker_factory=FakeRanker,
                review_text_by_id={
                    interaction.review_id: f"review-{interaction.review_id}"
                    for interaction in interactions()
                },
            )

            self.assertTrue((result.run_dir / "dataset.jsonl").exists())
            self.assertTrue((result.run_dir / "environment.lock.txt").exists())
            self.assertTrue((result.run_dir / "candidates_test.jsonl").exists())
            self.assertTrue((result.run_dir / "rankings_test.jsonl").exists())
            self.assertTrue((result.run_dir / "recommendations_test.jsonl").exists())
            self.assertTrue((result.run_dir / "review_context.jsonl").exists())
            self.assertEqual(
                result.manifest["artifacts"]["review_context"],
                "review_context.jsonl",
            )
            self.assertIn("e0_popularity", result.metrics["test"])
            self.assertIn("e3_rrf_union", result.metrics["test"])
            self.assertIn("e4_lambdarank", result.metrics["test"])

            repeated = run_baseline_experiment(
                interactions(),
                project_root=root,
                artifacts_root=root / "repeated-artifacts",
                config=ExperimentConfig(candidate_k=4, ranking_k=4),
                enable_mlflow=False,
                ranker_factory=FakeRanker,
            )
            for artifact in (
                "dataset.jsonl",
                "config.json",
                "candidates_validation.jsonl",
                "candidates_test.jsonl",
                "rankings_validation.jsonl",
                "rankings_test.jsonl",
                "recommendations_validation.jsonl",
                "recommendations_test.jsonl",
                "feature_importance.json",
            ):
                self.assertEqual(
                    (result.run_dir / artifact).read_bytes(),
                    (repeated.run_dir / artifact).read_bytes(),
                    artifact,
                )

    @unittest.skipUnless(find_spec("lightgbm"), "experiment extra is not installed")
    def test_real_lambdarank_and_mlflow_complete_on_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_baseline_experiment(
                interactions(),
                project_root=root,
                artifacts_root=root / "artifacts",
                config=ExperimentConfig(candidate_k=4, ranking_k=4),
                enable_mlflow=True,
            )

            self.assertTrue((result.run_dir / "final_lambdarank.txt").exists())
            self.assertTrue((root / "artifacts" / "mlflow.db").exists())
            self.assertIn("mlflow_run_id", result.manifest)
            self.assertEqual(len(result.manifest["mlflow_child_run_ids"]), 4)

            from mlflow.tracking import MlflowClient

            client = MlflowClient(
                tracking_uri=(
                    f"sqlite:///{(root / 'artifacts' / 'mlflow.db').resolve()}"
                )
            )
            parent_run_id = result.manifest["mlflow_run_id"]
            parent = client.get_run(parent_run_id)
            self.assertEqual(parent.data.tags["run_role"], "parent")
            self.assertIn("test/e4_lambdarank/ndcg_at_4", parent.data.metrics)
            self.assertLess(len(parent.data.metrics), 40)
            self.assertEqual(len(parent.inputs.dataset_inputs), 1)

            table_artifacts = {
                item.path for item in client.list_artifacts(parent_run_id, "tables")
            }
            self.assertIn("tables/stage_metrics.json", table_artifacts)
            self.assertIn("tables/test_recommendations.json", table_artifacts)

            for stage, child_run_id in result.manifest[
                "mlflow_child_run_ids"
            ].items():
                child = client.get_run(child_run_id)
                self.assertEqual(child.data.tags["run_role"], "stage")
                self.assertEqual(child.data.tags["stage"], stage)

            traces = client.search_traces(
                locations=[parent.info.experiment_id],
                run_id=parent_run_id,
                flush=True,
            )
            self.assertEqual(len(traces), 1)
            span_names = {span.name for span in traces[0].data.spans}
            self.assertIn("baseline_pipeline", span_names)
            self.assertIn("02_build_train_candidates", span_names)
            self.assertIn("07_train_final_lambdamart", span_names)


if __name__ == "__main__":
    unittest.main()
