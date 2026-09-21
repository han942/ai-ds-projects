from __future__ import annotations

import unittest

from rating_recsys.evaluation.metrics import RankingObservation, evaluate_rankings


class EvaluationMetricTests(unittest.TestCase):
    def test_single_target_recall_hit_rate_ndcg_and_mrr(self) -> None:
        observations = (
            RankingObservation("q1", 1, 3, 2, "history_1_2", (1, 3, 2)),
            RankingObservation("q2", 2, 4, 1, "history_3_plus", (4, 2, 1)),
            RankingObservation("q3", 3, 5, 0, "history_1_2", (5, 1, 2)),
        )
        metrics = evaluate_rankings(
            observations,
            cutoffs=(1, 2),
            catalog_ids=(1, 2, 3, 4, 5),
            item_popularity={1: 5, 2: 4, 3: 2, 4: 1, 5: 1},
            item_regions={1: "서울", 2: "서울", 3: "부산", 4: "부산", 5: "대구"},
        )

        self.assertEqual(metrics["queries"], 3)
        self.assertEqual(metrics["evaluated_queries"], 2)
        self.assertEqual(metrics["zero_relevance_queries"], 1)
        self.assertEqual(metrics["recall_at_1"], 0.5)
        self.assertEqual(metrics["hit_rate_at_1"], metrics["recall_at_1"])
        self.assertEqual(metrics["mrr_at_2"], 0.75)
        self.assertGreater(metrics["ndcg_at_2"], metrics["ndcg_at_1"])
        self.assertGreater(metrics["novelty_at_2"], 0.0)
        self.assertGreater(metrics["intra_list_region_diversity_at_2"], 0.0)


if __name__ == "__main__":
    unittest.main()
