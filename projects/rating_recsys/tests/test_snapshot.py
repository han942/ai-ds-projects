from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.snapshot import (
    code_manifest,
    dataset_digest,
    freeze_snapshot,
    load_snapshot,
    load_review_texts,
    write_review_texts,
    write_snapshot,
)

from support import synthetic_interactions


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


class FrozenSnapshotTests(unittest.TestCase):
    def test_freeze_is_content_addressed_and_round_trips(self) -> None:
        rows = [interaction(2, 20, 2), interaction(1, 10, 1)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path, first = freeze_snapshot(rows, root)
            second_path, _ = freeze_snapshot(reversed(rows), root)

            self.assertEqual(first_path, second_path)
            self.assertEqual(first_path.name, f"{first['dataset_snapshot_id'][:16]}.jsonl")
            self.assertEqual(sorted(load_snapshot(first_path), key=lambda i: i.review_id),
                             sorted(rows, key=lambda i: i.review_id))
            self.assertEqual([path.name for path in root.iterdir()], [first_path.name])


class ReviewTextFileTests(unittest.TestCase):
    def test_round_trip_and_coverage_check(self) -> None:
        rows = synthetic_interactions()[:5]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s.reviews.jsonl"
            meta = write_review_texts({3: "c", 1: None, 2: "b", 4: "", 5: "e"}, path)
            self.assertEqual([json.loads(l)["review_id"] for l in path.read_text().splitlines()], [1, 2, 3, 4, 5])
            texts, loaded = load_review_texts(path, rows)
            self.assertEqual(texts[3], "c")
            self.assertEqual(loaded["artifact_sha256"], meta["artifact_sha256"])
            self.assertEqual(meta["non_empty"], 3)
            with self.assertRaisesRegex(ValueError, "no text row"):
                load_review_texts(path, synthetic_interactions()[:6])


if __name__ == "__main__":
    unittest.main()
