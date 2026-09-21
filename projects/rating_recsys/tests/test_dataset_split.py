from __future__ import annotations

import unittest
from datetime import date, timedelta

from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.split import (
    build_global_temporal_split,
    build_seen_user_split,
)


def interaction(
    review_id: int,
    user_id: int,
    restaurant_id: int,
    day: int,
) -> Interaction:
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


class SeenUserSplitTests(unittest.TestCase):
    def test_combines_sparse_and_deeper_history_users(self) -> None:
        interactions = [
            interaction(1, 10, 101, 1),
            interaction(2, 10, 102, 2),
            interaction(3, 10, 103, 3),
            interaction(4, 20, 201, 1),
            interaction(5, 20, 202, 2),
            interaction(6, 20, 203, 3),
            interaction(7, 20, 204, 4),
            interaction(8, 20, 205, 5),
            interaction(9, 30, 301, 1),
            interaction(10, 30, 302, 2),
        ]

        split = build_seen_user_split(interactions)

        self.assertEqual(split.source_user_count, 3)
        self.assertEqual(split.eligible_user_count, 2)
        self.assertEqual(split.excluded_user_count, 1)
        self.assertEqual(split.history_1_2_user_count, 1)
        self.assertEqual(split.history_3_plus_user_count, 1)
        self.assertEqual(len(split.train), 4)
        self.assertEqual(len(split.validation), 2)
        self.assertEqual(len(split.test), 2)
        self.assertEqual(
            {item.restaurant_id for item in split.test},
            {103, 205},
        )

    def test_rejects_duplicate_user_restaurant_pairs(self) -> None:
        interactions = [
            interaction(1, 10, 101, 1),
            interaction(2, 10, 101, 2),
            interaction(3, 10, 102, 3),
        ]

        with self.assertRaisesRegex(ValueError, "duplicate pair"):
            build_seen_user_split(interactions)

    def test_rejects_minimum_below_three(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 3"):
            build_seen_user_split([], minimum_user_items=2)


class GlobalTemporalSplitTests(unittest.TestCase):
    def test_reports_seen_new_and_history_diagnostics(self) -> None:
        interactions = [
            interaction(1, 10, 101, 1),
            interaction(2, 10, 102, 2),
            interaction(3, 20, 201, 3),
            interaction(4, 20, 202, 4),
            interaction(5, 20, 203, 5),
            interaction(6, 30, 301, 6),
            interaction(7, 40, 401, 7),
            interaction(8, 10, 103, 8),
            interaction(9, 20, 204, 9),
            interaction(10, 50, 501, 10),
        ]

        split = build_global_temporal_split(
            interactions,
            train_fraction=0.5,
            validation_fraction=0.2,
        )

        self.assertEqual(split.train_cutoff, date(2025, 1, 5))
        self.assertEqual(split.validation_cutoff, date(2025, 1, 7))
        self.assertEqual(
            split.validation_cohorts["new"],
            {"users": 2, "interactions": 2},
        )
        self.assertEqual(
            split.test_cohorts["seen"],
            {"users": 2, "interactions": 2},
        )
        self.assertEqual(
            split.test_cohorts["new"],
            {"users": 1, "interactions": 1},
        )
        self.assertEqual(
            split.test_cohorts["history_1_2"],
            {"users": 1, "interactions": 1},
        )
        self.assertEqual(
            split.test_cohorts["history_3_plus"],
            {"users": 1, "interactions": 1},
        )

    def test_validates_fraction_configuration(self) -> None:
        interactions = [
            interaction(1, 10, 101, 1),
            interaction(2, 10, 102, 2),
            interaction(3, 10, 103, 3),
        ]

        with self.assertRaisesRegex(ValueError, "sum to less than 1"):
            build_global_temporal_split(
                interactions,
                train_fraction=0.8,
                validation_fraction=0.2,
            )


if __name__ == "__main__":
    unittest.main()
