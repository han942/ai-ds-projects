from __future__ import annotations

import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from rating_recsys.observability.app import (
    bootstrap_rows,
    discover_runs,
    hit_count,
    item_rows,
    item_summary,
    stage_rows,
    user_rows,
    RANKING_STAGES,
)

from support import SMALL, run_small, synthetic_interactions


class _Run:
    """One small run shared by every test in this module."""

    directory: tempfile.TemporaryDirectory | None = None

    @classmethod
    def get(cls):
        if cls.directory is None:
            cls.directory = tempfile.TemporaryDirectory()
            cls.root = Path(cls.directory.name)
            cls.result = run_small(synthetic_interactions(), cls.root)
        return cls


def tearDownModule() -> None:
    if _Run.directory is not None:
        _Run.directory.cleanup()


QUERY = {
    "query_id": "test:u1",
    "user_id": 1,
    "history": [{"restaurant_id": 8}],
    "window": [
        {"restaurant_id": 1, "event_date": "2026-01-01", "relevance": 2},
        {"restaurant_id": 2, "event_date": "2026-01-02", "relevance": 0},
        {"restaurant_id": 3, "event_date": "2026-01-03", "relevance": 1},
    ],
}
RECOMMENDATIONS = [
    {"restaurant_id": 3, "restaurant_name": "c", "final_rank": 1, "candidate_rank": 4},
    {"restaurant_id": 2, "restaurant_name": "b", "final_rank": 2, "candidate_rank": 1},
    {"restaurant_id": 9, "restaurant_name": "i", "final_rank": 3, "candidate_rank": 2},
]


class DashboardDataTests(unittest.TestCase):
    def test_user_rows_mark_hits(self) -> None:
        window, top = user_rows(QUERY, RECOMMENDATIONS)
        self.assertEqual([row["rank_in_top_k"] for row in window], [None, 2, 1])
        self.assertEqual([row["is_positive"] for row in top], [True, False, False])
        self.assertEqual([row["visited_in_window"] for row in top], [True, True, False])
        self.assertEqual(hit_count(QUERY, RECOMMENDATIONS), 1)

    def test_item_views_count_exposures_and_visits(self) -> None:
        recommendations = {"test:u1": RECOMMENDATIONS}
        summary = {row["restaurant_id"]: row for row in item_summary([QUERY], recommendations)}
        self.assertEqual(summary[3]["top_k_exposures"], 1)
        self.assertEqual(summary[3]["visited_by_exposed_users"], 1)
        self.assertEqual(summary[2]["visited_by_exposed_users"], 0)
        rows = item_rows(2, [QUERY], recommendations)
        self.assertEqual(rows[0]["final_rank"], 2)
        self.assertTrue(rows[0]["visited_in_window"])
        self.assertFalse(rows[0]["is_positive"])

    def test_helpers_on_real_run(self) -> None:
        run = _Run.get()
        self.assertEqual(discover_runs(run.root / "artifacts"), [run.result.run_dir])
        rows = stage_rows(run.result.metrics["test"], RANKING_STAGES, ("recall", "map"))
        self.assertEqual({row["K"] for row in rows}, set(SMALL.ranking_cutoffs))
        self.assertTrue(bootstrap_rows(run.result.metrics["test"]["bootstrap_r1_minus_r0"]))


@unittest.skipUnless(find_spec("streamlit"), "experiment extra is not installed")
class DashboardSmokeTests(unittest.TestCase):
    def _app(self, artifacts: Path):
        from streamlit.testing.v1 import AppTest

        import rating_recsys.observability.app as dashboard

        with patch.dict(os.environ, {"RATING_RECSYS_ARTIFACTS_DIR": str(artifacts)}):
            return AppTest.from_file(str(Path(dashboard.__file__))).run(timeout=60)

    def test_empty_artifact_directory_renders_help(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = self._app(Path(directory))
        self.assertFalse(app.exception)
        self.assertEqual(app.info[0].value, "No runs found. Run `rating-recsys-experiment` first.")

    def test_run_renders_report_metrics_and_explorers(self) -> None:
        run = _Run.get()
        app = self._app(run.root / "artifacts")
        self.assertFalse(app.exception)
        self.assertTrue(any("실험 보고서" in item.value for item in app.markdown))
        self.assertGreaterEqual(len(app.dataframe), 5)
        app.sidebar.radio(key="phase").set_value("validation").run(timeout=60)
        self.assertFalse(app.exception)


@unittest.skipUnless(find_spec("mlflow"), "experiment extra is not installed")
class TrackingTests(unittest.TestCase):
    def test_log_run_records_metrics_without_artifacts(self) -> None:
        import mlflow
        from mlflow.tracking import MlflowClient

        from rating_recsys.observability.tracking import EXPERIMENT_NAME, log_run, tracking_uri

        run = _Run.get()
        with tempfile.TemporaryDirectory() as directory:
            artifacts = Path(directory)
            logged = log_run(run.result.run_dir, artifacts_root=artifacts)
            mlflow.set_tracking_uri(tracking_uri(artifacts))
            client = MlflowClient()
            experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
            runs = client.search_runs([experiment.experiment_id])
            parent = client.get_run(logged["run_id"])

            grid = 1 + len(SMALL.num_leaves_grid) * len(SMALL.min_child_samples_grid)
            self.assertEqual(len(runs), 1 + grid)
            self.assertIn("test/r1_lambdarank/ndcg_at_5", parent.data.metrics)
            self.assertIn("test/r1_minus_r0/ndcg_at_5/ci95_low", parent.data.metrics)
            self.assertIn("test/c5_c1_lightgcn_rrf/recall_at_10", parent.data.metrics)
            self.assertIn("test/c5_minus_c3/recall_at_10/delta", parent.data.metrics)
            self.assertTrue(parent.data.tags["mlflow.runName"].startswith("C5→R1"))
            self.assertEqual(
                Path(parent.data.tags["report"]), (run.result.run_dir / "report.md").resolve()
            )
            self.assertEqual(client.list_artifacts(logged["run_id"]), [])
            self.assertFalse((artifacts / "mlflow_unused").exists())
            self.assertFalse(Path("mlruns").exists())
            steps = client.get_metric_history(logged["run_id"], "test/r1_lambdarank/recall")
            self.assertEqual(sorted(m.step for m in steps), list(SMALL.ranking_cutoffs))
            chosen = [
                r for r in runs
                if r.data.tags.get("grid") == "ranker" and r.data.tags.get("chosen") == "true"
            ]
            self.assertEqual(len(chosen), 1)
            self.assertTrue(
                client.get_metric_history(chosen[0].info.run_id, "validation/lightgbm_ndcg_curve")
            )


if __name__ == "__main__":
    unittest.main()
