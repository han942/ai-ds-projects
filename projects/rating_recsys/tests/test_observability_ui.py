from __future__ import annotations

import os
import tempfile
import unittest
import json
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

from rating_recsys.observability.app import (
    _dataset_review_index,
    _discover_run_dirs,
    _enrich_rankings,
    _item_summary_rows,
    _metric_cutoff_rows,
    _query_review_ids,
    _read_jsonl_query,
)


class DashboardDataTests(unittest.TestCase):
    def test_metrics_are_flattened_by_stage_and_cutoff(self) -> None:
        rows = _metric_cutoff_rows(
            {
                "e4_lambdarank": {
                    "recall_at_5": 0.2,
                    "ndcg_at_5": 0.1,
                    "mrr_at_5": 0.08,
                    "catalog_coverage_at_5": 0.4,
                    "novelty_at_5": 7.2,
                    "intra_list_region_diversity_at_5": 0.3,
                }
            }
        )
        self.assertEqual(rows[0]["stage"], "E4 LambdaMART")
        self.assertEqual(rows[0]["K"], 5)
        self.assertEqual(rows[0]["recall_hit_rate"], 0.2)

    def test_review_ids_support_new_and_legacy_queries(self) -> None:
        dataset = [
            {"user_id": 1, "restaurant_id": 10, "review_id": 100},
            {"user_id": 1, "restaurant_id": 20, "review_id": 200},
        ]
        index = _dataset_review_index(dataset)
        legacy = {
            "query_id": "test:u1:r200",
            "user_id": 1,
            "history_restaurant_ids": [10],
            "target_restaurant_id": 20,
        }
        self.assertEqual(_query_review_ids(legacy, index), ([100], 200))
        current = {
            **legacy,
            "history_review_ids": [101],
            "target_review_id": 201,
        }
        self.assertEqual(_query_review_ids(current, index), ([101], 201))

    def test_discovery_skips_incomplete_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            complete = root / "runs" / "20260102"
            incomplete = root / "runs" / "20260103"
            complete.mkdir(parents=True)
            incomplete.mkdir()
            for filename in ("manifest.json", "metrics.json", "queries.jsonl"):
                (complete / filename).write_text("{}\n", encoding="utf-8")
            (incomplete / "manifest.json").write_text("{}\n", encoding="utf-8")

            self.assertEqual(_discover_run_dirs(root), [complete])

    def test_query_reader_and_item_summary_use_served_rows(self) -> None:
        queries = [
            {
                "query_id": "test:u1:r9",
                "target_restaurant_id": 20,
                "target_restaurant_name": "target",
                "target_rating": 5.0,
                "history_restaurant_names": ["history"],
            }
        ]
        rankings = [
            {
                "query_id": "test:u1:r9",
                "user_id": 1,
                "restaurant_id": 20,
                "restaurant_name": "target",
                "candidate_rank": 4,
                "final_rank": 2,
            },
            {
                "query_id": "test:u1:r9",
                "user_id": 1,
                "restaurant_id": 30,
                "restaurant_name": "other",
                "candidate_rank": 1,
                "final_rank": 12,
            },
        ]
        enriched = _enrich_rankings(rankings, queries, ranking_k=10)

        self.assertTrue(enriched[0]["is_target"])
        self.assertTrue(enriched[0]["served_top_k"])
        self.assertEqual(enriched[0]["rank_movement"], 2)
        summaries = _item_summary_rows(enriched)
        self.assertEqual(summaries[0]["restaurant_id"], 20)
        self.assertEqual(summaries[0]["target_matches"], 1)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rankings.jsonl"
            path.write_text(
                "\n".join(json.dumps(row, separators=(",", ":")) for row in rankings)
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(
                _read_jsonl_query(path, "test:u1:r9"),
                rankings,
            )


@unittest.skipUnless(find_spec("streamlit"), "experiment extra is not installed")
class DashboardSmokeTests(unittest.TestCase):
    def test_empty_artifact_directory_renders_help(self) -> None:
        from streamlit.testing.v1 import AppTest

        import rating_recsys.observability.app as dashboard

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {"RATING_RECSYS_ARTIFACTS_DIR": directory},
        ):
            app = AppTest.from_file(str(Path(dashboard.__file__))).run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(
            app.info[0].value,
            "No experiment runs found. Run `rating-recsys-experiment` first.",
        )


if __name__ == "__main__":
    unittest.main()
