from types import SimpleNamespace
from dataclasses import replace
from datetime import date
import json
import time

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


def test_resume_reuses_completed_graph_and_then_all_arms_without_refitting(tmp_path, monkeypatch):
    """An interrupted real tiny run resumes past its finished baseline."""
    from rating_recsys.experiments import text_retrieval_comparison_cli as cli
    from rating_recsys.experiments.config import ExperimentConfig
    from rating_recsys.retrieval.learned_review_retrieval import LearnedReviewRetriever
    from rating_recsys.retrieval.lightgcn import LightGCN
    from rating_recsys.retrieval.review_multivector import ReviewMultiVectorConfig, build_review_bank
    from rating_recsys.retrieval.text_item_graph import TextItemGraph

    rows = [SimpleNamespace(review_id=n+1, user_id=user, restaurant_id=item,
                            event_date=date(2025, 1, n+1), rating=5.)
            for n, (user, item) in enumerate(((1, 10), (1, 20), (2, 20),
                                             (2, 30), (3, 10), (3, 30)))]
    vectors = np.random.default_rng(17).normal(size=(7, 384)).astype(np.float32)
    vectors[0] = 0
    bank = build_review_bank(rows, date(2025, 1, 6), {i: i for i in range(1, 7)},
        vectors, vectors, query_users=[1, 2, 3],
        config=ReviewMultiVectorConfig(max_user_reviews=8, max_item_reviews=12))
    queries = [SimpleNamespace(query_id=f"q{user}", user_id=user,
        history=[row for row in rows if row.user_id == user],
        relevance_by_item={item: 2}, relevant={item})
        for user, item in ((1, 30), (2, 10), (3, 20))]
    split = SimpleNamespace(train=rows, train_cutoff=date(2025, 1, 6))
    exp = ExperimentConfig(lightgcn_epochs=20, lightgcn_layers=1)
    args = SimpleNamespace(threads=1, seeds=[42], graph_epochs=1, review_epochs=1,
                           reuse_graph_run=None, baseline_run=tmp_path, balance_text_presence=True)
    ranks = {q.query_id: tuple(q.relevant) for q in queries}
    state = {"arms": {}}
    monkeypatch.setattr(cli, "protect_ranker", lambda _: {})
    monkeypatch.setattr(cli, "code_manifest", lambda *a, **k: {
        "git_commit": "synthetic", "git_dirty": False, "git_status": "", "git_diff": "original diff"})
    original_text_fit = TextItemGraph.fit

    def interrupt(*args, **kwargs):
        raise RuntimeError("simulated interruption after completed baseline")

    monkeypatch.setattr(TextItemGraph, "fit", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        cli.train(args, exp, split, queries, bank, ranks, ranks, tmp_path, state, {}, time.perf_counter())
    state = json.loads((tmp_path/"progress.json").read_text())
    assert set(state["arms"]) == {"graph_baseline_seed42"}
    original_checkpoint = (tmp_path/"graph_baseline_seed42.npz").read_bytes()
    original_source = (tmp_path/"source.diff").read_text()
    baseline_record = state["arms"]["graph_baseline_seed42"]
    original_baseline_fit = LightGCN.fit

    def forbid_baseline_refit(*args, **kwargs):
        raise AssertionError("completed baseline must not be fitted again")

    # TextItemGraph calls super().fit internally, so forbid only plain C4 fits.
    def guarded_graph_fit(self, *fit_args, **fit_kwargs):
        if type(self) is LightGCN:
            return forbid_baseline_refit()
        return original_baseline_fit(self, *fit_args, **fit_kwargs)

    monkeypatch.setattr(LightGCN, "fit", guarded_graph_fit)
    monkeypatch.setattr(TextItemGraph, "fit", original_text_fit)
    cli.train(args, exp, split, queries, bank, ranks, ranks, tmp_path, state, {}, time.perf_counter())
    assert state["status"] == "complete" and len(state["arms"]) == 6
    assert state["arms"]["graph_baseline_seed42"] == baseline_record
    assert (tmp_path/"graph_baseline_seed42.npz").read_bytes() == original_checkpoint
    assert (tmp_path/"source.diff").read_text() == original_source

    # A fully completed run performs no model fit at all on another resume.
    monkeypatch.setattr(LightGCN, "fit", forbid_baseline_refit)
    monkeypatch.setattr(TextItemGraph, "fit", forbid_baseline_refit)
    monkeypatch.setattr(LearnedReviewRetriever, "fit", forbid_baseline_refit)
    finished = json.loads((tmp_path/"metrics.json").read_text())
    cli.train(args, exp, split, queries, bank, ranks, ranks, tmp_path, finished, {}, time.perf_counter())
    assert finished["mean_seed_metrics"] == state["mean_seed_metrics"]
    assert finished["paired_comparisons"] == state["paired_comparisons"]
    assert len(finished["execution_attempts"]) == 3
    assert "active_epoch" not in finished and "active_arm" not in finished

    # Reusing a non-default epoch budget must not require the default C4 ranks.
    reused_run = tmp_path/"reuse-nondefault-budget"
    reused_run.mkdir()
    reused_state = {"arms": {}}
    reuse_args = SimpleNamespace(**{**vars(args), "reuse_graph_run": tmp_path})

    def stop_before_review_fit(*args, **kwargs):
        raise RuntimeError("stop after graph reuse")

    monkeypatch.setattr(LearnedReviewRetriever, "fit", stop_before_review_fit)
    with pytest.raises(RuntimeError, match="stop after graph reuse"):
        cli.train(reuse_args, exp, split, queries, bank, ranks,
                  {q.query_id: () for q in queries}, reused_run, reused_state, {}, time.perf_counter())
    assert set(reused_state["arms"]) == {
        "graph_baseline_seed42", "graph_actual_seed42", "graph_rewired_seed42"}

    finished["arms"]["graph_baseline_seed42"]["metrics"]["recall_at_100"] = -1.
    with pytest.raises(ValueError, match="metrics differ"):
        cli.train(args, exp, split, queries, bank, ranks, ranks, tmp_path, finished, {}, time.perf_counter())

    # Changed training code must never mix new and completed conditions.
    (tmp_path/"source/retrieval/lightgcn.py").write_text("changed training code")
    with pytest.raises(ValueError, match="different training code"):
        cli.record_training_source(tmp_path, finished)


def test_restored_graph_rejects_different_axes_or_invalid_vectors(tmp_path):
    from rating_recsys.experiments.text_retrieval_comparison_cli import restore_graph_backbone

    path = tmp_path/"baseline.npz"
    kwargs = {"user_ids": [1], "item_ids": [10, 20], "ego": np.zeros((3, 2)),
              "user_vectors": np.zeros((1, 2)), "item_vectors": np.zeros((2, 2))}
    np.savez(path, **kwargs)
    config = LightGCNConfig(dimension=2)
    with pytest.raises(ValueError, match="axes differ"):
        restore_graph_backbone(path, config, expected_user_ids=[2], expected_item_ids=[10, 20])
    kwargs["item_vectors"][0, 0] = np.nan
    np.savez(path, **kwargs)
    with pytest.raises(ValueError, match="Invalid completed"):
        restore_graph_backbone(path, config, expected_user_ids=[1], expected_item_ids=[10, 20])
