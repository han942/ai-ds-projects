"""V2 settings, ingestion, temporal splits and reproducible snapshots."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from rating_recsys.config import get_settings
from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments.snapshot import (
    code_manifest, dataset_digest, freeze_snapshot, load_review_texts,
    load_snapshot, write_review_texts, write_snapshot,
)
from rating_recsys.ingestion.transform import (
    infer_region, infer_region_from_address, parse_reviewed_at,
    parse_scraped_at, transform_row,
)

from support import synthetic_interactions


class SettingsTests(unittest.TestCase):
    @patch("rating_recsys.config.load_dotenv_if_available")
    def test_read_only_database_command_does_not_require_hash_salt(
        self,
        _load_dotenv,
    ) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example.invalid/postgres",
                "USER_HASH_SALT": "replace-with-a-long-random-secret",
            },
            clear=True,
        ):
            settings = get_settings(
                require_database=True,
                require_user_hash_salt=False,
            )

        self.assertEqual(
            settings.database_url,
            "postgresql://example.invalid/postgres",
        )

    @patch("rating_recsys.config.load_dotenv_if_available")
    def test_ingestion_still_rejects_example_hash_salt(self, _load_dotenv) -> None:
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://example.invalid/postgres",
                "USER_HASH_SALT": "replace-with-a-long-random-secret",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be changed"):
                get_settings(require_database=True)


SEOUL = ZoneInfo("Asia/Seoul")


class DateParsingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scraped_at = datetime(2026, 3, 21, 22, 1, tzinfo=SEOUL)

    def test_exact_date(self) -> None:
        parsed = parse_reviewed_at("2025년 12월 13일", self.scraped_at)
        self.assertEqual(parsed.value, date(2025, 12, 13))
        self.assertEqual(parsed.precision, "exact")

    def test_month_day_infers_current_year(self) -> None:
        parsed = parse_reviewed_at("2월 5일", self.scraped_at)
        self.assertEqual(parsed.value, date(2026, 2, 5))
        self.assertEqual(parsed.precision, "inferred_year")

    def test_month_day_rolls_back_future_date(self) -> None:
        parsed = parse_reviewed_at("12월 31일", self.scraped_at)
        self.assertEqual(parsed.value, date(2025, 12, 31))

    def test_relative_date(self) -> None:
        parsed = parse_reviewed_at("3일 전", self.scraped_at)
        self.assertEqual(parsed.value, date(2026, 3, 18))
        self.assertEqual(parsed.precision, "relative")

    def test_filename_metadata(self) -> None:
        path = Path("diningcode_data_crawling_busan_20260321_2201.csv")
        self.assertEqual(infer_region(path), "부산")
        self.assertEqual(parse_scraped_at(path), self.scraped_at)
        national = Path("diningcode_playwright_national_20260924_091004.csv.partial")
        self.assertEqual(
            parse_scraped_at(national),
            datetime(2026, 9, 24, 9, 10, 4, tzinfo=SEOUL),
        )


class RowTransformTests(unittest.TestCase):
    def test_transformation_pseudonymizes_user_and_maps_ordinals(self) -> None:
        source = {
            "item_name": " 테스트 식당 ",
            "item_area": "부산역",
            "item_avg_rating": "4.5",
            "item_spec_area": "부산 동구 테스트로 1",
            "user_name": "\n홍길동 다코미식가\n",
            "user_tot_avg_rating": "4.2",
            "user_tot_rating_num": "10",
            "user_tot_follow_num": "3",
            "user_rating": "5점",
            "user_query": " 아주 맛있어요\n또 갈게요 ",
            "taste": "맛: 맛있음",
            "price": "가격: 만족",
            "service": "응대: 친절함",
            "menu": "국밥",
            "date": "1일 전",
        }
        transformed = transform_row(
            source,
            source_row_number=2,
            source_file="diningcode_data_crawling_busan_20260321_2201.csv",
            region="부산",
            scraped_at=datetime(2026, 3, 21, 22, 1, tzinfo=SEOUL),
            user_hash_salt="test-salt",
        )

        self.assertEqual(transformed.canonical_name, "테스트 식당")
        self.assertEqual(transformed.review_text, "아주 맛있어요 또 갈게요")
        self.assertEqual((transformed.taste, transformed.price, transformed.service), (2, 2, 2))
        self.assertEqual(transformed.reviewed_at, date(2026, 3, 20))
        self.assertTrue(transformed.user_metadata["daco_gourmand"])
        self.assertEqual(len(transformed.user_key), 64)
        self.assertNotIn("user_name", transformed.source_payload)
        self.assertNotIn("홍길동", str(transformed.source_payload))


    def test_national_crawl_uses_row_time_and_address_region(self) -> None:
        source = {
            "item_name": "테스트 식당", "item_spec_area": "서울특별시 중구 테스트로 1",
            "user_name": "테스트 사용자", "user_rating": "4.0",
            "date": "1일 전", "crawl_timestamp": "2026-09-26T15:00:00+09:00",
        }
        transformed = transform_row(
            source, source_row_number=2, source_file="national.csv",
            region="전국", scraped_at=datetime(2026, 9, 24, tzinfo=SEOUL),
            user_hash_salt="test-salt",
        )
        self.assertEqual(transformed.region, "서울")
        self.assertEqual(transformed.reviewed_at, date(2026, 9, 25))
        self.assertEqual(transformed.scraped_at.date(), date(2026, 9, 26))
        self.assertEqual(infer_region_from_address("광주시 테스트로 1"), "전국")


def split_interaction(review_id: int, user_id: int, restaurant_id: int, day: int) -> Interaction:
    return Interaction(
        review_id=review_id,
        user_id=user_id,
        restaurant_id=restaurant_id,
        event_date=date(2025, 1, 1) + timedelta(days=day - 1),
        rating=5.0,
        reviewed_at_precision="exact",
        restaurant_name=f"restaurant-{restaurant_id}",
        region="서울",
    )


class GlobalTemporalSplitTests(unittest.TestCase):
    def test_cutoffs_and_seen_new_users(self) -> None:
        interactions = [
            split_interaction(1, 10, 101, 1),
            split_interaction(2, 10, 102, 2),
            split_interaction(3, 20, 201, 3),
            split_interaction(4, 20, 202, 4),
            split_interaction(5, 20, 203, 5),
            split_interaction(6, 30, 301, 6),
            split_interaction(7, 40, 401, 7),
            split_interaction(8, 10, 103, 8),
            split_interaction(9, 20, 204, 9),
            split_interaction(10, 50, 501, 10),
        ]
        split = build_global_temporal_split(
            interactions, train_fraction=0.5, validation_fraction=0.2
        )

        self.assertEqual(split.train_cutoff, date(2025, 1, 5))
        self.assertEqual(split.validation_cutoff, date(2025, 1, 7))
        self.assertTrue(all(item.event_date <= split.train_cutoff for item in split.train))
        self.assertTrue(
            all(split.train_cutoff < item.event_date <= split.validation_cutoff
                for item in split.validation)
        )
        self.assertTrue(all(item.event_date > split.validation_cutoff for item in split.test))
        self.assertEqual(split.validation_users, {"seen": 0, "new": 2})
        self.assertEqual(split.test_users, {"seen": 2, "new": 1})

    def test_validation_first_user_is_seen_in_test(self) -> None:
        # User 30 first appears in validation (day 6) and returns in test (day 9).
        interactions = [
            split_interaction(1, 10, 101, 1),
            split_interaction(2, 10, 102, 2),
            split_interaction(3, 20, 201, 3),
            split_interaction(4, 20, 202, 4),
            split_interaction(5, 10, 103, 5),
            split_interaction(6, 30, 301, 6),
            split_interaction(7, 20, 203, 7),
            split_interaction(8, 10, 104, 8),
            split_interaction(9, 30, 302, 9),
            split_interaction(10, 50, 501, 10),
        ]
        split = build_global_temporal_split(
            interactions, train_fraction=0.5, validation_fraction=0.2
        )

        self.assertEqual(split.validation_users["new"], 1)
        self.assertEqual(split.test_users, {"seen": 2, "new": 1})
        self.assertEqual(split.summary()["window_users"]["test"], {"seen": 2, "new": 1})

    def test_rejects_duplicate_user_restaurant_pairs(self) -> None:
        interactions = [
            split_interaction(1, 10, 101, 1),
            split_interaction(2, 10, 101, 2),
            split_interaction(3, 10, 102, 3),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate pair"):
            build_global_temporal_split(interactions)

    def test_validates_fraction_configuration(self) -> None:
        interactions = [split_interaction(i, 10, 100 + i, i) for i in range(1, 4)]
        with self.assertRaisesRegex(ValueError, "sum to less than 1"):
            build_global_temporal_split(
                interactions, train_fraction=0.8, validation_fraction=0.2
            )


def snapshot_interaction(review_id: int, user: int, restaurant: int) -> Interaction:
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
        rows = [snapshot_interaction(2, 20, 2), snapshot_interaction(1, 10, 1)]
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
            write_snapshot([snapshot_interaction(1, 10, 1)], path)
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
        rows = [snapshot_interaction(2, 20, 2), snapshot_interaction(1, 10, 1)]
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
