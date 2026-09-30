from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.report import write_report
from rating_recsys.experiments.cli import build_parser, config_from_args
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_window_queries

from support import SMALL, _interaction, _without_timing, run_small, synthetic_interactions


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
                    "c3_rrf_union", "c4_lightgcn", "c5_c1_lightgcn_rrf",
                    "r0_candidate_order", "r1_lambdarank",
                ):
                    self.assertGreater(metrics[phase][stage]["evaluated_queries"], 0)
                self.assertIn("map_at_5", metrics[phase]["r1_lambdarank"])
            self.assertIn("ndcg_at_5", metrics["test"]["bootstrap_r1_minus_r0"])
            self.assertIn("recall_at_10", metrics["test"]["bootstrap_c5_minus_c3"])

            # LightGCN graphs never include the window they score, and each
            # training checkpoint only sees interactions before its block.
            rows = sorted(synthetic_interactions(), key=lambda item: item.event_date)
            graph = metrics["lightgcn"]
            self.assertEqual(graph["window_models"]["validation"]["edges"], len(
                [item for item in rows if item.event_date <= t1]))
            self.assertEqual(graph["window_models"]["test"]["edges"], len(
                [item for item in rows if item.event_date <= t2]))
            checkpoints = graph["training_checkpoints"]
            self.assertGreater(len(checkpoints), 1)
            for checkpoint in checkpoints:
                start = date.fromisoformat(checkpoint["checkpoint"])
                self.assertEqual(start.day, 1)
                self.assertLessEqual(start, t2)
                self.assertEqual(
                    checkpoint["edges"], len([i for i in rows if i.event_date < start])
                )
            self.assertEqual(
                sum(c["queries"] for c in checkpoints),
                training["train_prefix"]["queries"]
                + training["refit_validation_window_prefix"]["queries"],
            )
            self.assertGreater(training["train_prefix"]["lightgcn_scored_queries"], 0)
            self.assertLess(
                training["train_prefix"]["lightgcn_scored_queries"],
                training["train_prefix"]["queries"],
            )
            self.assertEqual(
                len(metrics["selection"]["ranker_grid"]),
                1 + len(SMALL.num_leaves_grid) * len(SMALL.min_child_samples_grid),
            )
            self.assertNotIn("candidate_policy_grid", metrics["selection"])

            report = (result.run_dir / "report.md").read_text(encoding="utf-8")
            for heading in (
                "## 1. 요약", "## 2. 데이터와 조건", "## 3. 설정 선택",
                "## 4. 후보 생성 (test)", "## 5. LTR 재정렬 (test, Top-5)",
                "## 6. 해석 (작성자 기입)", "## 7. 재현",
            ):
                self.assertIn(heading, report)
            self.assertIn("`candidate_k` 10 (기본 100)", report)
            self.assertIn("--candidate-k 10", report)
            self.assertIn("--lightgcn-checkpoint-months 1", report)
            self.assertIn("C5 C1+LightGCN RRF (Stage 1)", report)
            self.assertIn("참고 · C3 quota RRF (이전 기준선)", report)
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


    def test_training_groups_only_hold_real_candidates(self) -> None:
        """No positive is added outside the Stage 1 list (the old injection shortcut)."""

        from rating_recsys.experiments.pipeline import (
            CheckpointedLightGCN,
            _generator,
            prefix_training_arrays,
        )
        from rating_recsys.experiments.queries import build_prefix_queries

        rows = synthetic_interactions()
        train = build_global_temporal_split(rows).train
        queries = build_prefix_queries(train, config=SMALL, phase="train")
        arrays = prefix_training_arrays(
            queries, train, _generator(SMALL),
            CheckpointedLightGCN(SMALL.lightgcn_config, months=1),
            feature_names=SMALL.feature_names,
        )
        summary = arrays.summary
        self.assertGreater(summary["usable_groups"], 0)
        self.assertLess(summary["retrieved_positive_queries"], summary["relevant_queries"])
        self.assertLessEqual(summary["usable_groups"], summary["retrieved_positive_queries"])
        self.assertNotIn("injection_rate", summary)
        rank = arrays.features[:, SMALL.feature_names.index("candidate_rank_inverse")]
        self.assertGreaterEqual(float(rank.min()), 1 / SMALL.candidate_k - 1e-6)
        offset = 0
        for size in arrays.groups:
            self.assertLessEqual(size, SMALL.candidate_k)
            self.assertEqual(int((arrays.labels[offset : offset + size] > 0).sum()), 1)
            offset += size
        self.assertEqual(offset, len(arrays.labels))

    def test_source_diff_includes_untracked_code(self) -> None:
        import subprocess

        from rating_recsys.experiments.snapshot import code_manifest

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            project = repo / "project"
            (project / "src").mkdir(parents=True)
            (project / "artifacts").mkdir()
            (repo / ".gitignore").write_text("*.log\n")
            (project / "src" / "old.py").write_text("x = 1\n")

            def git(*args: str) -> None:
                subprocess.run(
                    ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                    cwd=repo, check=True, capture_output=True,
                )

            git("init", "-q")
            git("add", ".")
            git("commit", "-q", "-m", "init")
            (project / "src" / "old.py").write_text("x = 2\n")
            (project / "src" / "new.py").write_text("y = 1\n")
            (project / "artifacts" / "report.md").write_text("result\n")
            (project / "run.log").write_text("ignored\n")

            diff = code_manifest(project, allow_dirty=True)["git_diff"]
            self.assertIn("diff --git a/project/src/old.py", diff)
            self.assertIn("diff --git a/project/src/new.py b/project/src/new.py", diff)
            self.assertIn("+y = 1", diff)
            self.assertNotIn("artifacts/report.md", diff)
            self.assertNotIn("run.log", diff)
            check = subprocess.run(
                ["git", "apply", "--check", "-R", "-"], cwd=repo, input=diff,
                text=True, capture_output=True,
            )
            self.assertEqual(check.returncode, 0, check.stderr)


class CliTests(unittest.TestCase):
    def test_flags_map_to_config_and_are_validated(self) -> None:
        args = build_parser().parse_args(
            [
                "--candidate-k", "50",
                "--ranking-k", "5",
                "--region-mode", "without_region",
                "--legacy-c3-quota", "0.5",
                "--lightgcn-layers", "2",
                "--lightgcn-checkpoint-months", "6",
                "--num-leaves-grid", "7,15",
                "--relevance-low", "3.5",
            ]
        )
        config = config_from_args(args)
        self.assertEqual(config.candidate_k, 50)
        self.assertEqual(config.region_mode, "without_region")
        self.assertEqual(config.legacy_c3_quota, 0.5)
        self.assertEqual(config.lightgcn_config.layers, 2)
        self.assertEqual(config.lightgcn_checkpoint_months, 6)
        self.assertEqual(config.num_leaves_grid, (7, 15))
        self.assertEqual(config.relevance(3.4), 0)
        with self.assertRaises(ValueError):
            config_from_args(build_parser().parse_args(["--legacy-c3-quota", "2"]))
        with self.assertRaises(ValueError):
            config_from_args(build_parser().parse_args(["--lightgcn-checkpoint-months", "5"]))

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
                    "--lightgcn-dimension", "8",
                    "--lightgcn-epochs", "2",
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

    def test_console_scripts_exit_zero(self) -> None:
        import re
        import subprocess
        import sys

        from rating_recsys.experiments.snapshot import write_snapshot

        project = Path(__file__).resolve().parents[1]
        text = (project / "pyproject.toml").read_text()
        scripts = dict(re.findall(r'^(rating-recsys-[\w-]+) = "([^"]+)"$', text, re.M))
        # Entry points that return a summary dict would make sys.exit print it and exit 1.
        self.assertEqual(scripts["rating-recsys-experiment"], "rating_recsys.experiments.cli:console_main")
        self.assertEqual(scripts["rating-recsys-compare"], "rating_recsys.experiments.compare_cli:console_main")
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            snapshot = artifacts / "input.jsonl"
            write_snapshot(synthetic_interactions(), snapshot)
            code = (
                "import sys; from rating_recsys.experiments.cli import console_main; "
                "sys.exit(console_main())"
            )
            result = subprocess.run(
                [sys.executable, "-c", code, "--snapshot", str(snapshot),
                 "--artifacts-dir", str(artifacts), "--candidate-k", "10", "--ranking-k", "5",
                 "--lightgcn-dimension", "8", "--lightgcn-epochs", "1", "--num-leaves-grid", "4",
                 "--min-child-samples-grid", "1", "--max-estimators", "5",
                 "--bootstrap-samples", "10", "--n-jobs", "1", "--no-mlflow"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
            self.assertNotIn("{'run_id'", result.stderr)


if __name__ == "__main__":
    unittest.main()
