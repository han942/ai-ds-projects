"""Evaluation identity and the distinction between added/lost positives."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.review_multivector_cli import (
    candidate_changes, compatible_prepared, paired_change, validate_rankings,
)


def test_prepared_identity_rejects_absolute_labels_and_changed_candidate_conditions():
    config = ExperimentConfig(satisfaction_mode="history-aware", ranker_training_mode="window")
    cutoffs = {"train_through": "2025-12-19", "validation_through": "2026-05-04"}
    split = SimpleNamespace(summary=lambda: {"cutoffs": cutoffs})
    identity = {"kind": "validation", "snapshot_id": "test", "cutoffs": cutoffs,
                "candidate_k": config.candidate_k, "rrf_constant": config.rrf_constant,
                "satisfaction": {key: value for key, value in config.to_dict().items() if key.startswith("satisfaction_")},
                "lightgcn": config.lightgcn_config.to_dict(), "features": list(config.feature_names),
                "region_mode": config.region_mode, "relevance_thresholds": [3., 4.], "rating_shrinkage_strength": 0.}
    assert compatible_prepared({"identity": identity}, "test", split, config)
    assert not compatible_prepared({"identity": identity}, "test", split, replace(config, satisfaction_mode="absolute"))
    assert not compatible_prepared({"identity": identity}, "test", split, replace(config, candidate_k=200))
    assert not compatible_prepared({"identity": identity}, "other", split, config)
    assert not compatible_prepared({"identity": {**identity, "satisfaction": None}}, "test", split, config)


def query(key, truth, seen=()):
    return SimpleNamespace(query_id=key, user_id=1, relevance_by_item=truth,
                           relevant=any(value > 0 for value in truth.values()),
                           history=[SimpleNamespace(restaurant_id=item) for item in seen])


def test_gain_loss_and_bootstrap_use_full_relevant_denominator():
    queries = [query("a", {1: 2, 2: 1, 3: 0}), query("b", {4: 2}), query("c", {8: 0})]
    before = {"a": (1, 9), "b": (9, 10), "c": (9, 10)}
    after = {"a": (2, 9), "b": (4, 9), "c": (9, 10)}
    summary, rows = candidate_changes(queries, before, after)
    assert summary["evaluated_queries"] == 2
    assert summary["gained_relevant_items"] == 2
    assert summary["lost_relevant_items"] == 1
    assert summary["macro_recall_gain"] == .75
    assert summary["macro_recall_loss"] == .25
    paired = paired_change(queries, before, after, 100, "recall", samples=100)
    assert paired["paired_users"] == 2
    assert paired["mean_difference"] == .5
    assert rows[0]["relevant_items"] == 2


@pytest.mark.parametrize("rankings", [{"a": (1, 1)}, {"a": (99,)}, {"a": (3,)}, {"b": (1,)}])
def test_candidates_reject_duplicates_future_catalog_seen_items_and_wrong_query_ids(rankings):
    with pytest.raises(ValueError):
        validate_rankings([query("a", {1: 2}, seen=(3,))], {"text": rankings}, {1, 2, 3}, 100)
