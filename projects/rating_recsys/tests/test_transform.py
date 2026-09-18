from __future__ import annotations

import unittest
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rating_recsys.config import PROJECT_ROOT
from rating_recsys.ingestion.cli import dry_run
from rating_recsys.ingestion.transform import (
    infer_region,
    parse_reviewed_at,
    parse_scraped_at,
    transform_row,
    discover_csv_files,
)


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


class LegacyDatasetSmokeTests(unittest.TestCase):
    def test_all_five_files_can_be_transformed(self) -> None:
        input_dir = (
            PROJECT_ROOT
            / "legacy"
            / "v1_rating_prediction"
            / "crawled_data"
        )
        files = discover_csv_files(input_dir)
        report = dry_run(files, "test-only-salt")

        self.assertEqual(report["summary"]["file_count"], 5)
        self.assertEqual(report["summary"]["total_rows"], 31_085)
        self.assertEqual(report["summary"]["restaurants"], 748)
        self.assertGreater(report["summary"]["unique_reviews"], 20_000)


if __name__ == "__main__":
    unittest.main()
