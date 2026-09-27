from __future__ import annotations

import json
import random
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.report import write_report
from rating_recsys.experiments.cli import build_parser, config_from_args
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import run_experiment
from rating_recsys.experiments.queries import build_window_queries


def _interaction(review_id: int, user: int, item: int, day: int, rating: float = 5.0):
    return Interaction(
        review_id=review_id,
        user_id=user,
        restaurant_id=item,
        event_date=date(2025, 1, 1) + timedelta(days=day),
        rating=rating,
        reviewed_at_precision="exact",
        restaurant_name=f"restaurant-{item}",
        region="서울" if item <= 15 else "부산",
    )


def synthetic_interactions() -> list[Interaction]:
    """Three taste clusters, staggered starts, and a few late-joining users."""

    rng = random.Random(7)
    clusters = {0: range(1, 11), 1: range(11, 21), 2: range(21, 31)}
    rows: list[Interaction] = []
    review_id = 1
    for user in range(1, 67):
        items = rng.sample(list(clusters[user % 3]), 6) + rng.sample(range(1, 31), 2)
        items = list(dict.fromkeys(items))
        day = rng.randint(0, 40) if user <= 60 else rng.randint(85, 95)
        for item in items:
            rows.append(
                _interaction(review_id, user, item, day, rng.choice([5, 5, 4, 4, 3, 2]))
            )
            review_id += 1
            day += rng.randint(3, 8)
    return rows


SMALL = ExperimentConfig(
    candidate_k=10,
    ranking_k=5,
    quota_grid=(0.0, 0.5),
    num_leaves_grid=(4,),
    min_child_samples_grid=(1, 5),
    max_estimators=30,
    early_stopping_rounds=5,
    reference_estimators=10,
    reference_num_leaves=4,
    reference_min_child_samples=1,
    bootstrap_samples=50,
    n_jobs=1,
)
RUN_FILES = (
    "manifest.json",
    "metrics.json",
    "report.md",
    "model.txt",
    "queries_validation.jsonl",
    "queries_test.jsonl",
    "recommendations_validation.jsonl",
    "recommendations_test.jsonl",
)


def run_small(rows, root: Path, **kwargs):
    root.mkdir(parents=True, exist_ok=True)
    return run_experiment(
        rows,
        project_root=root,
        artifacts_root=root / "artifacts",
        config=kwargs.pop("config", SMALL),
        log=lambda _: None,
        **kwargs,
    )


def _without_timing(value):
    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if not key.endswith("_seconds")
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


class WindowQueryTests(unittest.TestCase):
    def test_history_window_and_new_user_count(self) -> None:
        rows = [
            _interaction(1, 10, 101, 1),
            _interaction(2, 10, 102, 2),
            _interaction(3, 20, 201, 3),
            _interaction(4, 10, 103, 6, rating=2.0),
            _interaction(5, 10, 104, 7, rating=3.5),
            _interaction(6, 30, 301, 8),
        ]
        cutoff = date(2025, 1, 1) + timedelta(days=3)
        queries, new_users = build_window_queries(
            rows[:3], rows[3:], config=ExperimentConfig(), phase="validation", cutoff=cutoff
        )

        self.assertEqual([q.user_id for q in queries], [10])
        self.assertEqual(new_users, 1)
        self.assertEqual([i.restaurant_id for i in queries[0].history], [101, 102])
        self.assertEqual(queries[0].relevance_by_item, {103: 0, 104: 1})
        self.assertTrue(queries[0].relevant)

    def test_rejects_window_rows_before_cutoff(self) -> None:
        rows = [_interaction(1, 10, 101, 1), _interaction(2, 10, 102, 2)]
        with self.assertRaisesRegex(ValueError, "not after the cutoff"):
            build_window_queries(
                rows[:1], rows[1:], config=ExperimentConfig(), phase="test",
                cutoff=rows[1].event_date,
            )


class PipelineTests(unittest.TestCase):
    def test_run_folder_leakage_boundaries_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_small(synthetic_interactions(), root, label="unit")
            manifest, metrics = result.manifest, result.metrics
            t1 = date.fromisoformat(manifest["split"]["cutoffs"]["train_through"])
            t2 = date.fromisoformat(manifest["split"]["cutoffs"]["validation_through"])
            training = metrics["training"]

            for name in RUN_FILES:
                self.assertTrue((result.run_dir / name).exists(), name)
            snapshot = root / manifest["snapshot"]["path"]
            self.assertEqual(manifest["snapshot"]["path"], f"artifacts/snapshots/{snapshot.name}")
            self.assertEqual(snapshot.parent, root / "artifacts" / "snapshots")
            self.assertTrue(snapshot.exists())
            self.assertLessEqual(
                date.fromisoformat(training["train_prefix"]["latest_target_date"]), t1
            )
            refit = date.fromisoformat(
                training["refit_validation_window_prefix"]["latest_target_date"]
            )
            self.assertTrue(t1 < refit <= t2)
            self.assertEqual(manifest["leakage_checks"]["test_evaluations"], 1)
            self.assertGreater(manifest["windows"]["test"]["new_users_not_evaluated"], 0)
            for phase in ("validation", "test"):
                for stage in (
                    "c0_popularity", "c1_item_item", "c2_region_popularity",
                    "c3_rrf_union", "r0_candidate_order", "r1_lambdarank",
                ):
                    self.assertGreater(metrics[phase][stage]["evaluated_queries"], 0)
                self.assertIn("map_at_5", metrics[phase]["r1_lambdarank"])
            self.assertIn("ndcg_at_5", metrics["test"]["bootstrap_r1_minus_r0"])
            self.assertEqual(
                len(metrics["selection"]["ranker_grid"]),
                1 + len(SMALL.num_leaves_grid) * len(SMALL.min_child_samples_grid),
            )

            report = (result.run_dir / "report.md").read_text(encoding="utf-8")
            for heading in (
                "## 1. 요약", "## 2. 데이터와 조건", "## 3. Validation에서 고른 설정",
                "## 4. 후보 생성 (test)", "## 5. LTR 재정렬 (test, Top-5)",
                "## 6. 해석 (작성자 기입)", "## 7. 재현",
            ):
                self.assertIn(heading, report)
            self.assertIn("`candidate_k` 10 (기본 100)", report)
            self.assertIn("--candidate-k 10", report)
            with self.assertRaises(FileExistsError):
                write_report(result.run_dir)

            recommendation = json.loads(
                (result.run_dir / "recommendations_test.jsonl").read_text().splitlines()[0]
            )
            self.assertLessEqual(len(recommendation["recommendations"]), SMALL.ranking_k)

    def test_selection_and_models_ignore_test_window(self) -> None:
        rows = synthetic_interactions()
        test_ids = {item.review_id for item in build_global_temporal_split(rows).test}
        # Same dates (identical cutoffs) but different test outcomes.
        perturbed = [
            replace(item, restaurant_id=1000 + item.review_id, rating=1.0)
            if item.review_id in test_ids
            else item
            for item in rows
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = run_small(rows, root / "a")
            changed = run_small(perturbed, root / "b")

            for key in ("selection", "validation", "training"):
                self.assertEqual(
                    _without_timing(original.metrics[key]),
                    _without_timing(changed.metrics[key]),
                    key,
                )
            self.assertEqual(
                (original.run_dir / "model.txt").read_bytes(),
                (changed.run_dir / "model.txt").read_bytes(),
            )
            self.assertNotEqual(
                _without_timing(original.metrics["test"]),
                _without_timing(changed.metrics["test"]),
            )

    def test_is_deterministic(self) -> None:
        rows = synthetic_interactions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = run_small(rows, root / "a")
            second = run_small(list(reversed(rows)), root / "b")

            self.assertEqual(
                _without_timing(first.metrics), _without_timing(second.metrics)
            )
            self.assertEqual(
                (first.run_dir / "recommendations_test.jsonl").read_bytes(),
                (second.run_dir / "recommendations_test.jsonl").read_bytes(),
            )


class CliTests(unittest.TestCase):
    def test_flags_map_to_config_and_are_validated(self) -> None:
        args = build_parser().parse_args(
            [
                "--candidate-k", "50",
                "--ranking-k", "5",
                "--region-mode", "without_region",
                "--quota-grid", "0.5,1",
                "--num-leaves-grid", "7,15",
                "--relevance-low", "3.5",
            ]
        )
        config = config_from_args(args)
        self.assertEqual(config.candidate_k, 50)
        self.assertEqual(config.region_mode, "without_region")
        self.assertEqual(config.quota_grid, (0.5, 1.0))
        self.assertEqual(config.num_leaves_grid, (7, 15))
        self.assertEqual(config.relevance(3.4), 0)
        with self.assertRaises(ValueError):
            config_from_args(build_parser().parse_args(["--quota-grid", "2"]))

    def test_cli_writes_one_run_folder(self) -> None:
        from rating_recsys.experiments import cli
        from rating_recsys.experiments.snapshot import write_snapshot

        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            snapshot = artifacts / "input.jsonl"
            write_snapshot(synthetic_interactions(), snapshot)
            summary = cli.main(
                [
                    "--snapshot", str(snapshot),
                    "--artifacts-dir", str(artifacts),
                    "--candidate-k", "10",
                    "--ranking-k", "5",
                    "--quota-grid", "0,0.5",
                    "--num-leaves-grid", "4",
                    "--min-child-samples-grid", "1",
                    "--max-estimators", "20",
                    "--early-stopping-rounds", "5",
                    "--bootstrap-samples", "20",
                    "--n-jobs", "1",
                    "--no-mlflow",
                ]
            )
            runs = list((artifacts / "runs").iterdir())
            self.assertEqual(len(runs), 1)
            self.assertEqual(Path(summary["report"]), runs[0] / "report.md")
            self.assertEqual(
                sorted(path.name for path in artifacts.iterdir()),
                ["input.jsonl", "runs", "snapshots"],
            )


if __name__ == "__main__":
    unittest.main()
