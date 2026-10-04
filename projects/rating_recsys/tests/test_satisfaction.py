"""History-aware labels, cutoff isolation, diagnostics and prepared identity."""

import json
from dataclasses import replace
from datetime import date
from unittest.mock import patch

import pytest

from rating_recsys.datasets.split import build_global_temporal_split
from rating_recsys.experiments.cli import build_parser, config_from_args
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.pipeline import rating_diagnostics
from rating_recsys.experiments.queries import build_window_queries

from support import SMALL, _interaction, run_small, synthetic_interactions


def history(ratings):
    return tuple(_interaction(i, 1, i, i, r) for i, r in enumerate(ratings, 1))


def test_history_count_boundary_and_bounded_personal_threshold():
    config = ExperimentConfig(satisfaction_mode="history-aware")
    strict = history([3.0] * 10)
    generous = history([4.0] * 4 + [5.0] * 6)
    assert config.relevance(3.5, history=strict[:9]) == 1
    assert config.relevance(3.5, history=strict) == 2
    assert config.relevance(4.0, history=generous[:9]) == 2
    assert config.satisfaction_profile(generous)["strong_threshold"] == pytest.approx(4.3)
    assert config.relevance(4.0, history=generous) == 1
    assert config.relevance(4.5, history=generous) == 2
    # Constant histories must not divide by a zero standard deviation. A
    # generous all-five user still has reachable strong labels below five.
    assert config.satisfaction_profile(history([5.0] * 10))["strong_threshold"] == 4.5
    assert config.relevance(4.5, history=history([5.0] * 10)) == 2
    assert config.satisfaction_profile(history([1.0] * 10))["strong_threshold"] == 3.5
    assert config.relevance(2.0, history=history([1.0] * 10)) == 0
    assert config.relevance(3.0, history=strict) == 1
    assert config.training_label(3.5, history=strict) == 2
    assert config.label_gain == (0.0, 1.0, 3.0)
    # Historical experiments retain the original task unless opted in.
    assert ExperimentConfig().relevance(3.5, history=strict) == 1


@pytest.mark.parametrize("changes", [
    {"satisfaction_mode": "unknown"}, {"satisfaction_min_history": 1},
    {"satisfaction_min_history": 2.5}, {"satisfaction_min_history": True},
    {"satisfaction_mean_weight": -0.1}, {"satisfaction_mean_weight": float("nan")},
    {"satisfaction_max_shift": float("inf")}, {"satisfaction_max_shift": -1},
    {"satisfaction_max_shift": 1.0},
])
def test_invalid_personal_label_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        ExperimentConfig(**{"satisfaction_mode": "history-aware", **changes})


def test_window_label_uses_frozen_history_and_preserves_low_ratings_and_same_day_visits():
    config = ExperimentConfig(satisfaction_mode="history-aware")
    past = history([3.0] * 10)
    visits = (_interaction(11, 1, 11, 31, 3.5), _interaction(12, 1, 12, 31, 2.0))
    queries, _ = build_window_queries(past, visits, config=config, phase="test", cutoff=date(2025, 1, 31))
    assert queries[0].relevance_by_item == {11: 2, 12: 0}
    assert len(queries[0].window) == 2
    assert config.satisfaction_profile(queries[0].history)["past_mean"] == 3.0
    altered = (visits[0], replace(visits[1], rating=5.0))
    changed, _ = build_window_queries(past, altered, config=config, phase="test", cutoff=date(2025, 1, 31))
    assert changed[0].relevance_by_item[11] == 2
    assert changed[0].history == queries[0].history


def test_absolute_rating_diagnostics_keep_all_low_rating_queries_in_denominator():
    past = (_interaction(1, 1, 1, 1), _interaction(2, 2, 2, 1))
    visits = (_interaction(3, 1, 3, 32, 4), _interaction(4, 1, 4, 32, 2),
              _interaction(5, 2, 5, 32, 1))
    queries, _ = build_window_queries(past, visits, config=ExperimentConfig(), phase="test", cutoff=date(2025, 1, 31))
    ordered = {queries[0].query_id: (4, 3), queries[1].query_id: (5, 6)}
    diagnostics = rating_diagnostics(queries, ordered, (1, 2))
    assert diagnostics["high_rating_queries"] == 1
    assert diagnostics["low_rating_visits"] == 2
    assert diagnostics["high_rating_recall_at_1"] == 0
    assert diagnostics["high_rating_recall_at_2"] == 1
    assert diagnostics["low_rating_hit_count_at_1"] == 2
    assert diagnostics["low_rating_visit_inclusion_rate_at_1"] == 1


def test_personal_cli_flags_are_captured():
    config = config_from_args(build_parser().parse_args([
        "--ranker-training-mode", "window", "--satisfaction-mode", "history-aware",
        "--satisfaction-min-history", "12", "--satisfaction-mean-weight", "0.4",
    ]))
    assert config.ranker_training_mode == "window"
    assert config.satisfaction_mode == "history-aware"
    assert config.satisfaction_min_history == 12
    assert config.satisfaction_mean_weight == 0.4


def test_personalized_pipeline_cache_and_test_rating_isolation(tmp_path):
    rows = synthetic_interactions()
    absolute = replace(SMALL, ranker_training_mode="window")
    # Small synthetic users have <=8 visits; lower only the fixture's count to
    # exercise the personal path. Production threshold remains 10.
    config = replace(absolute, satisfaction_mode="history-aware", satisfaction_min_history=3)
    old = run_small(rows, tmp_path, config=absolute)
    cold = run_small(rows, tmp_path, config=config)
    assert {e["key"] for e in old.manifest["prepared_data"]["entries"]}.isdisjoint(
        {e["key"] for e in cold.manifest["prepared_data"]["entries"]})
    assert cold.manifest["windows"]["validation"]["satisfaction"]["personalized_users"] > 0
    assert cold.metrics["training"]["train_window"]["personalized_queries"] > 0
    saved = [json.loads(line) for line in (cold.run_dir / "queries_validation.jsonl").read_text().splitlines()]
    assert any(q["satisfaction"]["personalized"] for q in saved)
    report = (cold.run_dir / "report.md").read_text()
    assert "--satisfaction-mode history-aware" in report
    assert "--satisfaction-min-history 3" in report
    assert "개인별 만족도 기준" in report
    with patch("rating_recsys.experiments.pipeline.fit_lightgcn", side_effect=AssertionError("graph rebuilt")), \
         patch("rating_recsys.experiments.pipeline.window_training_arrays", side_effect=AssertionError("rows rebuilt")):
        warm = run_small(rows, tmp_path, config=config)
    assert {e["status"] for e in warm.manifest["prepared_data"]["entries"]} == {"hit"}
    assert (cold.run_dir / "model.txt").read_bytes() == (warm.run_dir / "model.txt").read_bytes()
    cutoff = build_global_temporal_split(rows).validation_cutoff
    changed = [replace(row, rating=1.0) if row.event_date > cutoff else row for row in rows]
    alternate = run_small(changed, tmp_path / "alternate", config=config)
    assert cold.metrics["selection"]["final_ranker_params"] == alternate.metrics["selection"]["final_ranker_params"]
    assert (cold.run_dir / "model.txt").read_bytes() == (alternate.run_dir / "model.txt").read_bytes()
    assert (cold.run_dir / "recommendations_test.jsonl").read_bytes() == (alternate.run_dir / "recommendations_test.jsonl").read_bytes()
    profiles = lambda run: [json.loads(line)["satisfaction"] for line in (run / "queries_test.jsonl").read_text().splitlines()]
    assert profiles(cold.run_dir) == profiles(alternate.run_dir)
