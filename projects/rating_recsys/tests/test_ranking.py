"""Ranking metrics, window labels and mean-rating shrinkage."""

from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from datetime import date
from unittest.mock import Mock

import numpy as np
import pytest

from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.evaluation.metrics import RankingObservation, evaluate_observed_pairs, evaluate_rankings
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import _generator, build_context, window_training_arrays
from rating_recsys.experiments.queries import build_prefix_queries, build_training_windows
from rating_recsys.experiments.shrinkage import run_shrinkage_comparison, shrink_feature_matrix
from rating_recsys.ranking.features import feature_values, global_rating_prior, shrink_mean
from rating_recsys.ranking.lambdarank import LightGBMLambdaRanker

from support import SMALL, _interaction, run_small, synthetic_interactions


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


def window_config(**kwargs):
    return replace(SMALL, **{"ranker_training_mode": "window", "ranker_label_mode": "rating", **kwargs})


def test_observed_pair_accuracy_macro_grades_and_missing_targets():
    result = evaluate_observed_pairs((
        # Three pairs; two correct (2 > 0, 2 > 1), one inversion (0 < 1).
        RankingObservation("q1", 1, {1: 2, 2: 0, 3: 1}, (1, 99, 2, 3)),
        # One inverted pair; same-grade pair is excluded. Item 7 was not retrieved.
        RankingObservation("q2", 2, {4: 2, 5: 2, 6: 0, 7: 1}, (6, 4)),
        RankingObservation("q3", 3, {8: 0, 9: 0}, (8, 9)),
        RankingObservation("q4", 4, {10: 2, 11: 0}, (10, 90)),
    ))
    assert result == {
        "accuracy": (2 / 3) / 2, "queries": 4, "eligible_queries": 3,
        "evaluated_queries": 2, "eligible_pairs": 9,
        "compared_pairs": 4, "correct_pairs": 2,
    }


def test_observed_pairs_use_delivered_order_when_scores_tie():
    grades = {1: 2, 2: 0}
    assert evaluate_observed_pairs((RankingObservation("a", 1, grades, (1, 2)),))["accuracy"] == 1
    assert evaluate_observed_pairs((RankingObservation("a", 1, grades, (2, 1)),))["accuracy"] == 0


def test_observed_pairs_without_comparable_pairs_are_undefined():
    assert evaluate_observed_pairs(())["accuracy"] is None
    assert evaluate_observed_pairs((RankingObservation("q", 1, {1: 2, 2: 0}, (1,)),))["accuracy"] is None


def test_observed_pairs_reject_duplicate_order_and_invalid_grades():
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_observed_pairs((RankingObservation("q", 1, {1: 2}, (1, 1)),))
    with pytest.raises(ValueError, match="0/1/2"):
        evaluate_observed_pairs((RankingObservation("q", 1, {1: 3}, (1,)),))


def test_raw_rating_labels_preserve_half_stars_without_changing_evaluation():
    config = window_config()
    assert [config.training_label(r) for r in (1, 3, 3.5, 4, 4.5, 5)] == [2, 6, 7, 8, 9, 10]
    assert [config.label_gain[config.training_label(r)] for r in (3.5, 4, 4.5, 5)] == [3.5, 4, 4.5, 5]
    assert config.relevance(4.5) == ExperimentConfig().relevance(4.5) == 2
    for rating in (0, 5.5, 4.1):
        with pytest.raises(ValueError, match="half-star"):
            config.training_label(rating)


def test_windows_freeze_history_and_do_not_duplicate_partial_refit_targets():
    rows = [_interaction(1, 1, 1, 1), _interaction(2, 1, 2, 35),
            _interaction(3, 1, 3, 40), _interaction(4, 1, 4, 50),
            _interaction(5, 1, 5, 65)]
    config = window_config()
    train = build_training_windows(rows, config=config, through=date(2025, 2, 12))
    final = build_training_windows(rows, config=config, through=date(2025, 3, 10))
    assert [v.review_id for _, qs in train for q in qs for v in q.window] == [2, 3]
    assert [v.review_id for _, qs in final for q in qs for v in q.window] == [2, 3, 4, 5]
    for start, queries in final:
        for query in queries:
            assert all(row.event_date < start for row in query.history)
            assert all(start <= row.event_date <= date(2025, 3, 10) for row in query.window)
            assert not {v.review_id for v in query.history} & {v.review_id for v in query.window}
    # February features stay frozen even though the refit adds a February label.
    assert train[0][1][0].history == final[0][1][0].history


def test_multiple_ratings_are_labels_only_and_unretrieved_target_is_not_injected():
    rows = (
        _interaction(1, 1, 1, 1, 5),
        _interaction(2, 2, 1, 1), _interaction(3, 2, 2, 2),
        _interaction(4, 3, 1, 1), _interaction(5, 3, 3, 2, 1),
        _interaction(6, 4, 1, 1), _interaction(7, 4, 4, 2),
        _interaction(8, 1, 2, 35, 3), _interaction(9, 1, 3, 36, 4.5),
        _interaction(10, 1, 99, 37, 5),
    )
    config = window_config()
    graphs = Mock()
    graphs.model_for.return_value = None
    arrays = window_training_arrays(
        rows, _generator(config), graphs, config=config, through=date(2025, 2, 28),
    )
    assert arrays.groups == [3]  # Restaurants 2, 3 and 4; no future restaurant 99.
    assert sorted(arrays.labels.tolist()) == [0, 6, 9]
    assert arrays.summary["multi_positive_groups"] == 1
    assert arrays.summary["distinct_observed_label_groups"] == 1
    assert arrays.summary["observed_preference_pairs"] == 1
    user_mean = config.feature_names.index("user_average_rating")
    assert np.all(arrays.features[:, user_mean] == 5)
    item_mean = config.feature_names.index("item_average_rating")
    assert sorted(arrays.features[:, item_mean].tolist()) == [1, 5, 5]
    assert all(call.args[0] == date(2025, 2, 1) for call in graphs.model_for.call_args_list)


def test_early_stopping_metric_is_independent_of_training_gain():
    results = []
    for gain in ((0.0, 1.0, 3.0), tuple(i / 2 for i in range(11))):
        ranker = LightGBMLambdaRanker(label_gain=gain, ranking_k=2)
        ranker._model = Mock()
        ranker.fit(np.zeros((3, 1)), np.array([0, 1, 2]), [3],
                   eval_set=(np.zeros((3, 1)), np.array([2, 0, 1]), [3]))
        metric = ranker._model.fit.call_args.kwargs["eval_metric"]
        results.append(metric(np.array([2, 0, 1]), np.array([0.2, 0.1, 0.3])))
    assert results[0] == results[1]


def test_window_pipeline_keeps_c5_and_test_selection_boundary(tmp_path):
    rows = synthetic_interactions()
    control = run_small(rows, tmp_path / "control", config=window_config(ranker_label_mode="relevance"))
    treatment = run_small(rows, tmp_path / "rating", config=window_config())
    for phase in ("validation", "test"):
        assert control.metrics[phase]["c5_c1_lightgcn_rrf"] == treatment.metrics[phase]["c5_c1_lightgcn_rrf"]
    summary = treatment.metrics["training"]["refit_all_windows"]
    assert summary["multi_positive_groups"] > 0
    assert summary["observed_preference_pairs"] > 0
    assert summary["latest_target_date"] <= treatment.manifest["split"]["cutoffs"]["validation_through"]
    assert "원래 평점" in (treatment.run_dir / "report.md").read_text()
    assert "--ranker-training-mode window" in (treatment.run_dir / "report.md").read_text()


def test_shrinkage_weights_counts_and_preserves_baseline():
    assert shrink_mean(5, 1, 3, 0) == 5
    assert shrink_mean(5, 1, 3, 10) == pytest.approx(35 / 11)
    assert shrink_mean(500, 100, 3, 10) == pytest.approx(530 / 110)
    assert shrink_mean(0, 0, 3, 10) == 3
    for strength in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ExperimentConfig(rating_shrinkage_strength=strength)


def test_shared_matrix_matches_past_only_canonical_features():
    rows = synthetic_interactions()
    split = build_global_temporal_split(rows)
    query = build_prefix_queries(split.train, config=SMALL)[-1]
    past = [r for r in split.train if (r.event_date, r.review_id) <
            (query.target.event_date, query.target.review_id)]
    context = build_context(past)
    candidates = _generator(SMALL).retrieve(query, context).union
    assert candidates
    prior = global_rating_prior(context)
    assert prior == pytest.approx(sum(r.rating for r in past) / len(past))
    raw = np.array([[feature_values(query, c, context)[n] for n in SMALL.feature_names]
                    for c in candidates], dtype=np.float32)
    expected = np.array([[feature_values(query, c, context, rating_shrinkage_strength=10)[n]
                          for n in SMALL.feature_names] for c in candidates], dtype=np.float32)
    actual = shrink_feature_matrix(raw, SMALL.feature_names, prior, 10)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_array_equal(shrink_feature_matrix(raw, SMALL.feature_names, prior, 0), raw)
    untouched = [i for i, n in enumerate(SMALL.feature_names)
                 if n not in {"user_average_rating", "item_average_rating"}]
    np.testing.assert_array_equal(actual[:, untouched], raw[:, untouched])
    for c in candidates:
        assert c.source_scores["popularity"] == context.item_counts[c.restaurant_id]


def test_paired_runner_reproduces_baseline_and_ignores_test_labels(tmp_path):
    rows = synthetic_interactions()
    baseline = run_small(rows, tmp_path / "baseline")
    def run(data, name):
        root = tmp_path / name
        root.mkdir()
        return run_shrinkage_comparison(data, project_root=root,
                                       artifacts_root=root / "artifacts", config=SMALL, log=lambda _: None)
    result = run(rows, "paired")
    assert result.metrics["test"]["baseline"] == baseline.metrics["test"]["r1_lambdarank"]
    assert result.metrics["validation"]["baseline"]["ndcg_at_5"] == baseline.metrics["validation"]["r1_lambdarank"]["ndcg_at_5"]
    report = (result.run_dir / "report.md").read_text()
    assert "V2" in report and "shrinkage" in report
    saved = [json.loads(line) for line in (result.run_dir / "recommendations_test.jsonl").read_text().splitlines()]
    for query in saved:
        candidates = set(query["c5_candidates"])
        assert set(r["restaurant_id"] for r in query["baseline"]) == candidates
        assert set(r["restaurant_id"] for r in query["shrinkage"]) == candidates
        assert not candidates.intersection(query["history_restaurant_ids"])
    cutoff = build_global_temporal_split(rows).validation_cutoff
    changed = [replace(r, rating=1.0) if r.event_date > cutoff else r for r in rows]
    second = run(changed, "changed_test")
    for name in ("baseline", "shrinkage"):
        assert result.metrics["selection"][name]["final_params"] == second.metrics["selection"][name]["final_params"]
        assert (result.run_dir / f"model_{name}.txt").read_bytes() == (second.run_dir / f"model_{name}.txt").read_bytes()
