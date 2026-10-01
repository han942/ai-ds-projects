"""Verify observed preference labels and temporal isolation in window LTR."""

from dataclasses import replace
from datetime import date
from unittest.mock import Mock

import numpy as np
import pytest

from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import _generator, window_training_arrays
from rating_recsys.experiments.queries import build_training_windows
from rating_recsys.ranking.lambdarank import LightGBMLambdaRanker
from support import SMALL, _interaction, run_small, synthetic_interactions


def window_config(**kwargs):
    return replace(SMALL, **{"ranker_training_mode": "window", "ranker_label_mode": "rating", **kwargs})


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
