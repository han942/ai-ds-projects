from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.models import RecommendationQuery
from rating_recsys.experiments.queries import interactions_before
from rating_recsys.experiments.snapshot import (
    code_manifest,
    dataset_digest,
    write_snapshot,
)


def interaction(review_id: int, user: int, restaurant: int) -> Interaction:
    return Interaction(
        review_id=review_id,
        user_id=user,
        restaurant_id=restaurant,
        event_date=date(2025, 1, review_id),
        rating=4.0,
        reviewed_at_precision="exact",
        restaurant_name=f"restaurant-{restaurant}",
        region="서울",
    )


class SnapshotTests(unittest.TestCase):
    def test_digest_and_bytes_are_order_independent(self) -> None:
        rows = [interaction(2, 20, 2), interaction(1, 10, 1)]
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.jsonl"
            second_path = Path(directory) / "second.jsonl"
            first = write_snapshot(rows, first_path)
            second = write_snapshot(reversed(rows), second_path)

            self.assertEqual(dataset_digest(rows), dataset_digest(reversed(rows)))
            self.assertEqual(first["dataset_snapshot_id"], second["dataset_snapshot_id"])
            self.assertEqual(first["artifact_sha256"], second["artifact_sha256"])
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

    def test_snapshot_uses_canonical_json_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.jsonl"
            write_snapshot([interaction(1, 10, 1)], path)
            record = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(record["event_date"], "2025-01-01")
            self.assertEqual(record["restaurant_id"], 1)

    @patch("rating_recsys.experiments.snapshot._git_output")
    def test_strict_reproducibility_rejects_dirty_worktree(self, git_output) -> None:
        git_output.side_effect = ["abc123", " M model.py"]

        with self.assertRaisesRegex(RuntimeError, "dirty worktree"):
            code_manifest(Path("."), allow_dirty=False)


class CutoffTests(unittest.TestCase):
    def test_future_and_target_interactions_are_not_available(self) -> None:
        history = interaction(1, 10, 1)
        target = Interaction(
            review_id=3,
            user_id=10,
            restaurant_id=3,
            event_date=date(2025, 1, 3),
            rating=5.0,
            reviewed_at_precision="exact",
            restaurant_name="restaurant-3",
            region="서울",
        )
        future = Interaction(
            review_id=4,
            user_id=20,
            restaurant_id=4,
            event_date=date(2025, 1, 4),
            rating=5.0,
            reviewed_at_precision="exact",
            restaurant_name="restaurant-4",
            region="서울",
        )
        query = RecommendationQuery(
            query_id="validation:u10:r3",
            phase="validation",
            user_id=10,
            cutoff=target.event_date,
            history=(history,),
            target=target,
            relevance=2,
        )

        available = interactions_before((future, target, history), query)

        self.assertEqual(available, (history,))


if __name__ == "__main__":
    unittest.main()
