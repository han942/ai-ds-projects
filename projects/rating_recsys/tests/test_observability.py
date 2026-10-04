from __future__ import annotations

import os
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from rating_recsys.experiments.artifacts import read_jsonl, write_jsonl
from rating_recsys.observability.app import (
    attach_review_text,
    bootstrap_rows,
    diagnostic_index,
    discover_runs,
    hit_count,
    item_rows,
    item_summary,
    rating_threshold_rows,
    review_text_index,
    stage_rows,
    target_rows,
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

    def test_target_rows_separate_retrieval_miss_from_reranking_miss(self) -> None:
        query = {
            "query_id": "test:u2",
            "user_id": 2,
            "history": [],
            "window": [
                {"restaurant_id": 11, "event_date": "2026-02-01", "restaurant_name": "a", "rating": 5, "relevance": 2},
                {"restaurant_id": 12, "event_date": "2026-02-02", "restaurant_name": "b", "rating": 3.5, "relevance": 1},
                {"restaurant_id": 13, "event_date": "2026-02-03", "restaurant_name": "c", "rating": 2, "relevance": 0},
            ],
        }
        audit = {
            11: {"candidate_rank": 4, "final_rank": 14},
            12: {"candidate_rank": None, "final_rank": None},
            13: {"candidate_rank": 1, "final_rank": 1},
        }
        rows = target_rows(
            query,
            [{"restaurant_id": 13, "final_rank": 1}],
            target_diagnostics=audit,
            ranking_k=10,
            low_threshold=3,
            high_threshold=4,
        )
        self.assertEqual(rows[0]["결과"], "후보 검색 성공 · 최종 14위")
        self.assertEqual(rows[1]["결과"], "Retriever가 후보로 찾지 못함")
        self.assertEqual(rows[2]["평가상 분류"], "저평점 방문 · 비관련")
        self.assertEqual(rows[2]["결과"], "최종 Top-K 추천")

    def test_rating_threshold_sensitivity_keeps_low_ratings_out_of_positive_labels(self) -> None:
        query = {
            "query_id": "test:u3",
            "user_id": 3,
            "window": [
                {"restaurant_id": 21, "rating": 5},
                {"restaurant_id": 22, "rating": 3.5},
                {"restaurant_id": 23, "rating": 2},
            ],
        }
        recommendations = {
            "test:u3": [
                {"restaurant_id": 22, "final_rank": 1},
                {"restaurant_id": 23, "final_rank": 2},
            ]
        }
        rows = rating_threshold_rows(
            [query], recommendations, thresholds=(3, 3.5, 4), cutoff=2
        )
        self.assertEqual([row["긍정 방문 수"] for row in rows], [2, 2, 1])
        self.assertEqual([row["긍정 정답 없는 사용자"] for row in rows], [0, 0, 0])
        self.assertEqual([row["Top-K에 든 저평점 방문"] for row in rows], [1, 1, 1])
        self.assertEqual(rows[0]["Recall@2"], 0.5)
        self.assertEqual(rows[2]["Recall@2"], 0.0)
        self.assertEqual(rows[0]["Precision@2"], 0.5)

    def test_review_text_joins_only_to_the_matching_user_restaurant_date(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshot.jsonl"
            write_jsonl(snapshot, [{
                "review_id": 1001,
                "user_id": 4,
                "restaurant_id": 31,
                "event_date": "2026-01-03",
            }])
            write_jsonl(snapshot.with_name("snapshot.reviews.jsonl"), [{
                "review_id": 1001,
                "review_text": "재방문하고 싶은 맛이었어요.",
            }])
            index = review_text_index(snapshot)
        row = attach_review_text(
            {"restaurant_id": 31, "event_date": "2026-01-03"}, 4, index
        )
        self.assertEqual(row["review_text"], "재방문하고 싶은 맛이었어요.")
        self.assertIsNone(
            attach_review_text(
                {"restaurant_id": 31, "event_date": "2026-01-04"}, 4, index
            )["review_text"]
        )

    def test_helpers_on_real_run(self) -> None:
        run = _Run.get()
        self.assertEqual(discover_runs(run.root / "artifacts"), [run.result.run_dir])
        rows = stage_rows(run.result.metrics["test"], RANKING_STAGES, ("recall", "map"))
        self.assertEqual({row["K"] for row in rows}, set(SMALL.ranking_cutoffs))
        self.assertTrue(bootstrap_rows(run.result.metrics["test"]["bootstrap_r1_minus_r0"]))
        target_audit = read_jsonl(run.result.run_dir / "target_diagnostics_test.jsonl")
        query_rows = read_jsonl(run.result.run_dir / "queries_test.jsonl")
        self.assertTrue(target_audit)
        self.assertEqual(
            len(diagnostic_index(target_audit)),
            len(query_rows),
        )


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
        self.assertGreaterEqual(len(app.get("vega_lite_chart")), 2)
        self.assertTrue(any(item.label == "사용자별 진단" for item in app.tabs))
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
