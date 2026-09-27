from __future__ import annotations

import math
import unittest

from rating_recsys.evaluation.metrics import RankingObservation, evaluate_rankings


class RankingMetricTests(unittest.TestCase):
    def test_hand_computed_multi_positive_metrics(self) -> None:
        # q1 relevant {3: 2, 4: 1}: item 3 at rank 1, item 4 at rank 3.
        # q2 relevant {7: 1}: item 6 was visited but rated low (relevance 0).
        # q3 has only a low-rated visit and is excluded from accuracy.
        observations = (
            RankingObservation("q1", 1, {3: 2, 4: 1}, (3, 9, 4, 8)),
            RankingObservation("q2", 2, {6: 0, 7: 1}, (6, 5, 8, 7)),
            RankingObservation("q3", 3, {5: 0}, (5, 1, 2, 3)),
        )
        metrics = evaluate_rankings(
            observations,
            cutoffs=(3,),
            catalog_ids=range(1, 11),
            item_popularity={i: 1 for i in range(1, 11)},
            item_regions={i: "서울" if i % 2 else "부산" for i in range(1, 11)},
        )

        self.assertEqual(metrics["queries"], 3)
        self.assertEqual(metrics["evaluated_queries"], 2)
        self.assertAlmostEqual(metrics["mean_relevant_items"], 1.5)
        self.assertAlmostEqual(metrics["recall_at_3"], (1.0 + 0.0) / 2)
        self.assertAlmostEqual(metrics["precision_at_3"], (2 / 3) / 2)
        self.assertAlmostEqual(metrics["mrr_at_3"], 0.5)
        self.assertAlmostEqual(metrics["map_at_3"], ((1 + 2 / 3) / 2) / 2)
        dcg = 3 / math.log2(2) + 1 / math.log2(4)
        ideal = 3 / math.log2(2) + 1 / math.log2(3)
        self.assertAlmostEqual(metrics["ndcg_at_3"], (dcg / ideal) / 2)
        # Items 1,2,3,4,5,6,8,9 appear in some top-3 list.
        self.assertAlmostEqual(metrics["catalog_coverage_at_3"], 8 / 10)
        self.assertGreater(metrics["novelty_at_3"], 0.0)
        self.assertGreater(metrics["intra_list_region_diversity_at_3"], 0.0)

    def test_graded_ndcg_prefers_high_relevance_first(self) -> None:
        relevance = {1: 2, 2: 1}
        high = evaluate_rankings(
            (RankingObservation("a", 1, relevance, (1, 2)),), cutoffs=(2,), catalog_ids=(1, 2)
        )
        low = evaluate_rankings(
            (RankingObservation("b", 1, relevance, (2, 1)),), cutoffs=(2,), catalog_ids=(1, 2)
        )

        self.assertAlmostEqual(high["ndcg_at_2"], 1.0)
        self.assertLess(low["ndcg_at_2"], 1.0)
        self.assertEqual(high["recall_at_2"], low["recall_at_2"])


if __name__ == "__main__":
    unittest.main()
