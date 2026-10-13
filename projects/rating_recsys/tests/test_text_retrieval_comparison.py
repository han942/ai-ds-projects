from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from rating_recsys.experiments.text_retrieval_comparison_cli import (
    mean_seed_comparison, per_user_values, ranks_from_scores, sampling_audit,
)
from rating_recsys.retrieval.lightgcn import LightGCNConfig


def query(user=1, truth=None):
    truth = truth or {2: 2, 99: 1}
    return SimpleNamespace(query_id=f"q{user}", user_id=user,
                           relevance_by_item=truth, relevant={i for i, v in truth.items() if v > 0},
                           history=[SimpleNamespace(restaurant_id=1)])


def test_full_catalog_ties_seen_and_unfindable_truth():
    q = query()
    ranks = ranks_from_scores(np.array([[8., 2., 2.]]), [1], np.array([1, 2, 3]), [q], k=2)
    assert ranks == {"q1": (2, 3)}
    # The unavailable future item 99 remains in the denominator.
    assert per_user_values([q], ranks, 100, "recall").tolist() == [.5]


def test_seed_average_bootstraps_users_not_user_seed_rows():
    queries = [query(1, {2: 1}), query(2, {3: 1})]
    base = {"q1": (4,), "q2": (4,)}
    a = {"q1": (2,), "q2": (4,)}
    b = {"q1": (4,), "q2": (3,)}
    result = mean_seed_comparison(queries, [base, base], [a, b], samples=100)
    assert result["paired_users"] == 2
    assert result["seeds"] == 2
    assert result["mean_difference"] == .5
    assert result["bootstrap_95pct"] == [.5, .5]


@pytest.mark.parametrize("scores,ids", [(np.array([[np.nan, 1]]), np.array([1, 2])),
                                        (np.ones((1, 2)), np.array([2, 1]))])
def test_fail_closed_invalid_catalog_scores(scores, ids):
    with pytest.raises(ValueError):
        ranks_from_scores(scores, [1], ids, [query()])


def test_review_budget_can_exceed_graph_budget_without_a_sampling_mismatch():
    rows = [SimpleNamespace(user_id=user, restaurant_id=item)
            for user, item in ((1, 10), (1, 20), (2, 20), (2, 30))]
    graph = LightGCNConfig(epochs=1, batch_size=2, dimension=2)
    original = sampling_audit(rows, graph)
    extended = sampling_audit(rows, graph, review_epochs=2)
    two_epochs = sampling_audit(rows, replace(graph, epochs=2))
    assert extended["epochs"] == two_epochs["epochs"]
    assert extended["epochs"][:1] == original["epochs"]
    assert extended["graph_batches"] == original["graph_batches"]
    assert extended["initial_ego_sha256"] == original["initial_ego_sha256"]
