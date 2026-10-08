"""The shared candidate-comparison runner, CLI and report, for every registered model."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass, replace
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments import compare_cli
from rating_recsys.experiments.candidate_models import CANDIDATE_MODELS, CandidateModel, LightGCNCandidate
from rating_recsys.experiments.comparison import fusions, policies, require_prepared_window, run_candidate_comparison
from rating_recsys.experiments.snapshot import write_review_texts, write_snapshot
from support import (
    MODEL_CASES, SMALL, _without_timing, available_models, run_small,
    synthetic_interactions, texts_for, tiny_cli_flags, tiny_grid,
)


HAS_TORCH = find_spec("torch") is not None
LIGHTGCN = CANDIDATE_MODELS["lightgcn"]
def run_tiny(model, rows, root: Path, **kwargs):
    root.mkdir(parents=True, exist_ok=True)
    if model.needs_review_texts:
        kwargs.setdefault("texts", texts_for(rows))
    if "grid" not in kwargs:
        kwargs["grid"] = tiny_grid(model)[:1]
    kwargs.setdefault("eval_every", 1)
    return run_candidate_comparison(
        model,
        rows,
        project_root=root,
        artifacts_root=root / "artifacts",
        config=kwargs.pop("config", SMALL),
        patience=kwargs.pop("patience", 2),
        log=lambda _: None,
        plot=kwargs.pop("plot", False),
        **kwargs,
    )


class RegistryTests(unittest.TestCase):
    def test_prepared_preflight_fails_closed_on_corrupt_files(self):
        from types import SimpleNamespace
        from rating_recsys.experiments.prepared import _digest, _file_digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = {"fixture": "baseline", "kind": "validation"}
            folder = root / f"validation-{_digest(identity)[:16]}"
            folder.mkdir()
            (folder / "data.npz").write_bytes(b"fixture arrays")
            metadata = {"identity": identity, "arrays_sha256": _file_digest(folder / "data.npz")}
            metadata["metadata_sha256"] = _digest(metadata)
            (folder / "manifest.json").write_text(json.dumps(metadata))
            cache = SimpleNamespace(enabled=True, rebuild=False, identity={"fixture": "baseline"}, root=root)
            require_prepared_window(cache, "validation")
            (folder / "data.npz").write_bytes(b"corrupt arrays")
            with self.assertRaisesRegex(ValueError, "integrity"):
                require_prepared_window(cache, "validation")
            cache.rebuild = True
            with self.assertRaisesRegex(ValueError, "existing prepared"):
                require_prepared_window(cache, "validation")

    def test_two_tower_scope_is_exactly_three_conditions(self):
        model = CANDIDATE_MODELS["two_tower"]
        self.assertEqual(model.comparison_sources, ("c5_c1_lightgcn_rrf",))
        self.assertEqual(policies(model), ("two_tower", "rrf_c1_two_tower"))
        self.assertEqual(fusions(model), {"rrf_c1_two_tower": ("c1_item_item", "two_tower")})
        args = compare_cli.build_parser(model).parse_args(["--snapshot", "x.jsonl"])
        config, grid = compare_cli.configs_from_args(model, args)
        self.assertEqual(config.satisfaction_mode, "history-aware")
        self.assertEqual(config.ranker_training_mode, "window")
        self.assertEqual(len(grid), 1)
        self.assertEqual(grid[0].epochs, 12)

    def test_every_registered_model_has_a_small_case(self) -> None:
        self.assertEqual(set(MODEL_CASES), set(CANDIDATE_MODELS))

    def test_every_model_builds_a_parser_and_a_default_grid(self) -> None:
        for model in available_models():
            with self.subTest(model.name):
                args = compare_cli.build_parser(model).parse_args(["--snapshot", "x.jsonl"])
                config, grid = compare_cli.configs_from_args(model, args)
                self.assertTrue(grid)
                self.assertEqual(grid[0].epochs, model.default_max_epochs)
                self.assertTrue(model.default_grid(config.random_seed))
                self.assertEqual(grid, model.default_grid(config.random_seed))
                self.assertEqual(policies(model)[0], model.name)
                self.assertEqual(len({c.name for c in grid}), len(grid))

    def test_lightgcn_parser_does_not_import_torch(self) -> None:
        import subprocess
        import sys

        code = (
            "import sys; from rating_recsys.experiments import compare_cli as c; "
            "c.build_parser(c.CANDIDATE_MODELS['lightgcn']); "
            "import rating_recsys.experiments.comparison; print('torch' in sys.modules)"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), "False")


class ComparisonRunTests(unittest.TestCase):
    def test_limited_sources_produce_only_requested_three_conditions(self):
        class Limited(LightGCNCandidate):
            name, title, letter = "limited", "Limited", "T"
            comparison_sources = ("c5_c1_lightgcn_rrf",)
            fusion_sources = (("c1_item_item",),)
            evidence_status = "exploratory-reused-holdout"

        rows = synthetic_interactions()
        with tempfile.TemporaryDirectory() as directory:
            result = run_tiny(Limited(), rows, Path(directory), grid=tiny_grid(LIGHTGCN)[:1])
            expected = {"c5_c1_lightgcn_rrf", "limited", "rrf_c1_limited"}
            for phase in ("validation", "test"):
                self.assertEqual(set(result.metrics[phase]), expected)
            self.assertEqual(set(result.manifest["fusions"]), {"rrf_c1_limited"})
            self.assertNotIn("RRF C1+C4+T", (result.run_dir / "report.md").read_text())
            self.assertIn("exploratory only", (result.run_dir / "report.md").read_text())
            self.assertEqual(json.loads((result.run_dir / "status.json").read_text())["status"], "complete")

    def test_new_model_reuses_pipeline_baseline_candidates(self):
        rows = synthetic_interactions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = run_small(rows, root)
            with patch("rating_recsys.experiments.pipeline.fit_lightgcn", side_effect=AssertionError("baseline graph rebuilt")), \
                 patch("rating_recsys.experiments.pipeline.window_candidates", side_effect=AssertionError("baseline candidates rebuilt")), \
                 patch.object(LIGHTGCN, "fit", wraps=LIGHTGCN.fit) as new_model_fit:
                compared = run_tiny(LIGHTGCN, rows, root)
            self.assertEqual(new_model_fit.call_count, 2)
            self.assertEqual({e["status"] for e in compared.manifest["prepared_data"]["entries"]}, {"hit"})
            self.assertEqual(compared.metrics["test"]["c5_c1_lightgcn_rrf"]["recall_at_10"],
                             baseline.metrics["test"]["c5_c1_lightgcn_rrf"]["recall_at_10"])

    def test_run_folder_curves_leakage_and_report(self) -> None:
        rows = synthetic_interactions()
        split = build_global_temporal_split(rows)
        for model in available_models():
            with self.subTest(model.name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                extra = {"grid": tiny_grid(model)}
                result = run_tiny(model, rows, root, plot=True, label="unit", **extra)
                metrics, manifest = result.metrics, result.manifest

                self.assertEqual(result.run_dir.parent, root / "artifacts" / "comparisons" / model.name)
                for name in ("report.md", "manifest.json", "metrics.json", "candidates_test.jsonl",
                             "learning_curve.png"):
                    self.assertTrue((result.run_dir / name).exists(), name)
                grid = metrics["selection"]["grid"]
                for row in grid:
                    self.assertEqual(row["training_interactions"], len(split.train))
                    self.assertEqual(len(row["curve"]), row["stopped_epoch"])
                    self.assertIn("loss", row["curve"][0])
                refit = metrics["refit"]
                self.assertEqual(refit["training_interactions"], len(split.train + split.validation))
                self.assertEqual(refit["config"]["epochs"], metrics["selection"]["chosen_epochs"])
                self.assertEqual(len(refit["loss_curve"]), metrics["selection"]["chosen_epochs"])
                self.assertEqual(manifest["leakage_checks"]["test_evaluations"], 1)
                self.assertEqual(manifest["model"]["name"], model.name)
                for phase in ("validation", "test"):
                    for stage in ("c5_c1_lightgcn_rrf", *policies(model)):
                        self.assertGreater(metrics[phase][stage]["evaluated_queries"], 0)
                self.assertIn(metrics["selection"]["chosen_policy"], policies(model))
                self.assertIn("recall_at_10", metrics["test_comparisons"]["vs_stage1"][model.name]["at_10"])

                first = json.loads((result.run_dir / "candidates_test.jsonl").read_text().splitlines()[0])
                history = {
                    item.restaurant_id for item in split.train + split.validation
                    if item.user_id == first["user_id"]
                }
                self.assertFalse(set(first[model.name]) & history)

                report = (result.run_dir / "report.md").read_text(encoding="utf-8")
                self.assertIn(f"# {model.title} 후보 실험", report)
                for heading in ("## 1. 요약", "## 3. 학습 곡선 (validation)", "## 5. 후보 결과",
                                "## 6. Test 비교 (paired bootstrap)", "## 8. 재현"):
                    self.assertIn(heading, report)
                self.assertIn(f"RRF C1+C4+{model.letter}", report)
                self.assertIn(f"rating-recsys-compare {model.name} --snapshot", report)
                if model.name == "deepconn":
                    for candidate in grid:
                        self.assertEqual(
                            "rmse" in candidate["curve"][0], candidate["config"]["objective"] == "mse"
                        )
                    self.assertIn("RMSE", report)

    def test_selection_ignores_test_window_and_is_deterministic(self) -> None:
        rows = synthetic_interactions()
        test_ids = {item.review_id for item in build_global_temporal_split(rows).test}
        perturbed = [
            replace(item, restaurant_id=1000 + item.review_id, rating=1.0)
            if item.review_id in test_ids
            else item
            for item in rows
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = run_tiny(LIGHTGCN, rows, root / "a")
            repeated = run_tiny(LIGHTGCN, list(reversed(rows)), root / "b")
            changed = run_tiny(LIGHTGCN, perturbed, root / "c")

            self.assertEqual(_without_timing(original.metrics), _without_timing(repeated.metrics))
            for key in ("selection", "validation", "refit"):
                self.assertEqual(
                    _without_timing(original.metrics[key]),
                    _without_timing(changed.metrics[key]),
                    key,
                )
            self.assertNotEqual(
                _without_timing(original.metrics["test"]),
                _without_timing(changed.metrics["test"]),
            )

    @unittest.skipUnless(HAS_TORCH, "deepconn extra is not installed")
    def test_window_text_never_reaches_earlier_models(self) -> None:
        model = CANDIDATE_MODELS["deepconn"]
        rows = synthetic_interactions()
        split = build_global_temporal_split(rows)
        texts = texts_for(rows)
        test_changed = {**texts, **{r.review_id: "완전히 다른 글" for r in split.test}}
        validation_changed = {**texts, **{r.review_id: "완전히 다른 글" for r in split.validation}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = run_tiny(model, rows, root / "a")
            repeated = run_tiny(model, list(reversed(rows)), root / "b")
            after_test = run_tiny(model, rows, root / "c", texts=test_changed)
            after_validation = run_tiny(model, rows, root / "d", texts=validation_changed)

            self.assertEqual(_without_timing(original.metrics), _without_timing(repeated.metrics))
            # Test-window text is never read, not even for the test documents.
            self.assertEqual(_without_timing(original.metrics), _without_timing(after_test.metrics))
            # Validation-window text only enters the refit for the test window.
            for key in ("selection", "validation"):
                self.assertEqual(
                    _without_timing(original.metrics[key]),
                    _without_timing(after_validation.metrics[key]),
                    key,
                )
            self.assertNotEqual(
                _without_timing(original.metrics["refit"]),
                _without_timing(after_validation.metrics["refit"]),
            )

    @unittest.skipUnless(HAS_TORCH, "deepconn extra is not installed")
    def test_text_models_reject_missing_text_rows(self) -> None:
        model = CANDIDATE_MODELS["deepconn"]
        rows = synthetic_interactions()
        texts = texts_for(rows)
        del texts[rows[0].review_id]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "No review text"):
                run_tiny(model, rows, Path(directory), texts=texts)
            with self.assertRaisesRegex(ValueError, "needs review texts"):
                run_tiny(model, rows, Path(directory) / "none", texts=None)

    def test_compares_with_a_pipeline_run_on_the_same_snapshot(self) -> None:
        rows = synthetic_interactions()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = run_small(rows, root / "base")
            result = run_tiny(LIGHTGCN, rows, root / "lgcn", baseline_run=baseline.run_dir)
            compared = result.metrics["test_comparisons"]["vs_baseline_run"]
            self.assertEqual(compared["run_id"], baseline.run_id)
            self.assertAlmostEqual(
                compared["metrics"]["r1_lambdarank"]["ndcg_at_5"],
                baseline.metrics["test"]["r1_lambdarank"]["ndcg_at_5"],
            )
            self.assertIn("ndcg_at_5", compared["bootstrap_vs_r1"]["lightgcn"])
            self.assertIn("R1 LambdaRank", (result.run_dir / "report.md").read_text())

            other = [replace(item, rating=1.0) if item.review_id == 1 else item for item in rows]
            with self.assertRaisesRegex(ValueError, "different snapshot"):
                run_tiny(LIGHTGCN, other, root / "other", baseline_run=baseline.run_dir)


class CompareCliTests(unittest.TestCase):
    def test_new_config_fields_become_cli_options_without_model_parser_code(self) -> None:
        @dataclass(frozen=True)
        class Config:
            width: int = 4
            scale: float = 0.5
            strategy: str = "mean"
            normalize: bool = True
            epochs: int = 2
            seed: int = 42

        class Model(CandidateModel):
            name = title = "example"
            grid_parameters = {"width": ("--width-grid", (4, 8))}

            def config_type(self):
                return Config

        model = Model()
        parser = compare_cli.build_parser(model)
        args = parser.parse_args([
            "--snapshot", "x", "--width-grid", "8,16", "--scale", "0.25",
            "--strategy", "max", "--no-normalize", "--max-epochs", "3", "--seed", "7",
        ])
        _, grid = compare_cli.configs_from_args(model, args)
        self.assertEqual(grid, (
            Config(8, 0.25, "max", False, 3, 7), Config(16, 0.25, "max", False, 3, 7),
        ))
        self.assertEqual(
            model.default_grid(),
            compare_cli.configs_from_args(model, parser.parse_args(["--snapshot", "x"]))[1],
        )
        with self.assertRaisesRegex(ValueError, "grid must not be empty"):
            compare_cli.configs_from_args(model, parser.parse_args(["--snapshot", "x", "--width-grid", ""]))

    def test_model_is_the_first_argument(self) -> None:
        with self.assertRaises(SystemExit):
            compare_cli.main([])
        with self.assertRaises(SystemExit):
            compare_cli.main(["unknown", "--snapshot", "x"])

    def test_lightgcn_grid_flags(self) -> None:
        args = compare_cli.build_parser(LIGHTGCN).parse_args(
            ["--snapshot", "x", "--layers-grid", "1,3", "--regularization-grid", "0.01",
             "--max-epochs", "40", "--dimension", "16", "--candidate-k", "50"]
        )
        config, grid = compare_cli.configs_from_args(LIGHTGCN, args)
        self.assertEqual(config.candidate_k, 50)
        self.assertEqual([(g.layers, g.regularization, g.epochs, g.dimension) for g in grid],
                         [(1, 0.01, 40, 16), (3, 0.01, 40, 16)])
        self.assertEqual((args.eval_every, args.patience), (5, 6))

    @unittest.skipUnless(HAS_TORCH, "deepconn extra is not installed")
    def test_deepconn_grid_flags(self) -> None:
        model = CANDIDATE_MODELS["deepconn"]
        parser = compare_cli.build_parser(model)
        args = parser.parse_args(
            ["--snapshot", "x.jsonl", "--variants", "mse,bpr:linear", "--kernel-sizes", "3,5",
             "--doc-length", "100", "--max-epochs", "4"]
        )
        _, grid = compare_cli.configs_from_args(model, args)
        self.assertEqual(
            [(g.objective, g.latent_activation, g.kernel_size, g.doc_length, g.epochs) for g in grid],
            [("mse", "relu", 3, 100, 4), ("mse", "relu", 5, 100, 4),
             ("bpr", "linear", 3, 100, 4), ("bpr", "linear", 5, 100, 4)],
        )
        default = compare_cli.configs_from_args(model, parser.parse_args(["--snapshot", "x"]))[1]
        self.assertEqual(
            [(g.objective, g.latent_activation, g.epochs) for g in default],
            [("bpr", "relu", 12), ("bpr", "linear", 12), ("mse", "relu", 12)],
        )
        for bad in ("ce", "bpr:tanh"):
            with self.assertRaises(ValueError):
                compare_cli.configs_from_args(
                    model, parser.parse_args(["--snapshot", "x", "--variants", bad])
                )

    def test_cli_writes_one_folder_per_model_and_records_the_command(self) -> None:
        rows = synthetic_interactions()
        common = ["--candidate-k", "10", "--ranking-k", "5", "--bootstrap-samples", "20", "--no-plot"]
        for model in available_models():
            with self.subTest(model.name), tempfile.TemporaryDirectory() as directory:
                artifacts = Path(directory)
                snapshot = artifacts / "input.jsonl"
                write_snapshot(rows, snapshot)
                if model.needs_review_texts:
                    write_review_texts(texts_for(rows), artifacts / "input.reviews.jsonl")
                argv = [model.name, "--snapshot", str(snapshot), "--artifacts-dir", str(artifacts),
                        *common, *tiny_cli_flags(model)]
                summary = compare_cli.main(argv)
                runs = list((artifacts / "comparisons" / model.name).iterdir())
                self.assertEqual(len(runs), 1)
                self.assertEqual(Path(summary["report"]), runs[0] / "report.md")
                manifest = json.loads((runs[0] / "manifest.json").read_text())
                self.assertEqual(manifest["command"], argv)
                self.assertIn(" ".join(argv[:3]), (runs[0] / "report.md").read_text())
                if model.needs_review_texts:
                    self.assertEqual(manifest["review_texts"]["artifact"], "input.reviews.jsonl")
                else:
                    self.assertIsNone(manifest["review_texts"])


if __name__ == "__main__":
    unittest.main()
