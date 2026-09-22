from __future__ import annotations

import unittest
from datetime import date

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.models import RecommendationQuery
from rating_recsys.retrieval.baselines import (
    BaselineCandidateGenerator,
    IncrementalRetrievalContext,
    build_context,
)


def interaction(review_id: int, user: int, restaurant: int, day: int) -> Interaction:
    return Interaction(
        review_id=review_id,
        user_id=user,
        restaurant_id=restaurant,
        event_date=date(2025, 1, day),
        rating=5.0,
        reviewed_at_precision="exact",
        restaurant_name=f"restaurant-{restaurant}",
        region="서울" if restaurant < 4 else "부산",
    )


class BaselineCandidateGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = interaction(1, 10, 1, 1)
        self.target = interaction(20, 10, 3, 10)
        self.query = RecommendationQuery(
            query_id="validation:u10:r20",
            phase="validation",
            user_id=10,
            cutoff=self.target.event_date,
            history=(self.history,),
            target=self.target,
            relevance=2,
        )
        self.available = (
            self.history,
            interaction(2, 20, 1, 2),
            interaction(3, 20, 3, 3),
            interaction(4, 30, 1, 4),
            interaction(5, 30, 3, 5),
            interaction(6, 40, 2, 6),
            interaction(7, 40, 4, 7),
        )

    def test_excludes_seen_items_and_preserves_source_attribution(self) -> None:
        result, _ = BaselineCandidateGenerator(candidate_k=3).retrieve(
            self.query,
            self.available,
        )

        self.assertNotIn(1, {item.restaurant_id for item in result.union})
        self.assertEqual(result.item_item[0].restaurant_id, 3)
        target = next(item for item in result.union if item.restaurant_id == 3)
        self.assertEqual(
            target.candidate_sources,
            ("popularity", "item_item", "region_popularity"),
        )
        self.assertTrue(result.target_available)

    def test_incremental_context_matches_batch_context_and_candidates(self) -> None:
        incremental = IncrementalRetrievalContext()
        for item in self.available:
            incremental.add(item)

        batch_context = build_context(self.available)
        self.assertEqual(incremental.context, batch_context)

        generator = BaselineCandidateGenerator(candidate_k=4)
        batch_result, _ = generator.retrieve(self.query, self.available)
        incremental_result, _ = generator.retrieve_from_context(
            self.query,
            incremental.context,
        )
        self.assertEqual(batch_result.popularity, incremental_result.popularity)
        self.assertEqual(batch_result.item_item, incremental_result.item_item)
        self.assertEqual(
            batch_result.region_popularity,
            incremental_result.region_popularity,
        )
        self.assertEqual(batch_result.base_union, incremental_result.base_union)
        self.assertEqual(batch_result.union, incremental_result.union)

    def test_is_deterministic_and_uses_restaurant_id_for_ties(self) -> None:
        generator = BaselineCandidateGenerator(candidate_k=4)
        first, _ = generator.retrieve(self.query, self.available)
        second, _ = generator.retrieve(self.query, reversed(self.available))

        self.assertEqual(first.union, second.union)
        popularity_ids = [item.restaurant_id for item in first.popularity]
        self.assertLess(popularity_ids.index(2), popularity_ids.index(4))

    def test_injects_available_target_only_for_training(self) -> None:
        generator = BaselineCandidateGenerator(candidate_k=1)
        target = interaction(21, 10, 4, 10)
        query = RecommendationQuery(
            query_id="train:u10:r21",
            phase="train",
            user_id=10,
            cutoff=target.event_date,
            history=(self.history,),
            target=target,
            relevance=2,
        )
        result, context = generator.retrieve(query, self.available)
        injected = generator.inject_target(query, result.union, context)

        self.assertEqual(len(injected), 2)
        self.assertTrue(injected[-1].injected_for_training)
        self.assertEqual(injected[-1].restaurant_id, target.restaurant_id)


if __name__ == "__main__":
    unittest.main()
