from __future__ import annotations

import unittest
from datetime import date, timedelta

from rating_recsys.datasets.models import Interaction
from rating_recsys.datasets.split import build_global_temporal_split


def interaction(review_id: int, user_id: int, restaurant_id: int, day: int) -> Interaction:
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
            interaction(1, 10, 101, 1),
            interaction(2, 10, 102, 2),
            interaction(3, 20, 201, 3),
            interaction(4, 20, 202, 4),
            interaction(5, 10, 103, 5),
            interaction(6, 30, 301, 6),
            interaction(7, 20, 203, 7),
            interaction(8, 10, 104, 8),
            interaction(9, 30, 302, 9),
            interaction(10, 50, 501, 10),
        ]
        split = build_global_temporal_split(
            interactions, train_fraction=0.5, validation_fraction=0.2
        )

        self.assertEqual(split.validation_users["new"], 1)
        self.assertEqual(split.test_users, {"seen": 2, "new": 1})
        self.assertEqual(split.summary()["window_users"]["test"], {"seen": 2, "new": 1})

    def test_rejects_duplicate_user_restaurant_pairs(self) -> None:
        interactions = [
            interaction(1, 10, 101, 1),
            interaction(2, 10, 101, 2),
            interaction(3, 10, 102, 3),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate pair"):
            build_global_temporal_split(interactions)

    def test_validates_fraction_configuration(self) -> None:
        interactions = [interaction(i, 10, 100 + i, i) for i in range(1, 4)]
        with self.assertRaisesRegex(ValueError, "sum to less than 1"):
            build_global_temporal_split(
                interactions, train_fraction=0.8, validation_fraction=0.2
            )


if __name__ == "__main__":
    unittest.main()
