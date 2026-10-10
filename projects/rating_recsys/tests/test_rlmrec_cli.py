"""Independent temporal, frozen-profile and experiment provenance contracts.

Synthetic vectors and graph stubs suffice; no embeddings or large fits run.
"""

from dataclasses import dataclass
from datetime import date
import json
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments import rlmrec_cli as cli
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.models import WindowQuery
from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.retrieval.lightgcn import LightGCNConfig
from rating_recsys.retrieval.rlmrec import RLMRecConfig


def row(rid, user, item, day, rating=5):
    return Interaction(rid, user, item, day, rating, "exact", "synthetic", "region")


def store_for(values):
    """Each review maps to separate lists of query and passage block vectors."""
    width = 2
    vectors = [[99.0, -99.0]]  # Foreign padding contents must never contribute.
    max_blocks = max(1, *(len(blocks) for roles in values.values() for blocks in roles))
    query = np.zeros((len(values) + 1, max_blocks), np.int64)
    passage = query.copy()
    index = {}
    for review_index, (rid, roles) in enumerate(sorted(values.items()), 1):
        index[rid] = review_index
        for positions, blocks in zip((query, passage), roles, strict=True):
            for position, vector in enumerate(blocks):
                assert len(vector) == width
                positions[review_index, position] = len(vectors)
                vectors.append(vector)
    return SimpleNamespace(
        review_index=index, vectors=torch.tensor(vectors, dtype=torch.float32),
        query_blocks=torch.from_numpy(query), document_blocks=torch.from_numpy(passage),
        manifest={"input_manifest_sha256": "synthetic-input", "cache_identity": {"model": "frozen-test"},
                  "block_config": {"tokens": 512, "max_blocks": max_blocks}, "records": []},
    )


def test_padding_block_mean_equal_review_weight_and_roles():
    store = store_for({1: ([[1, 0], [0, 1]], [[1, 0], [1, 0]]),
                       2: ([[0, 1]], [[0, -1]]), 3: ([], [])})
    before = store.vectors.clone()
    profiles = cli.FrozenReviewProfiles(store)
    rows = [row(1, 10, 100, date(2025, 1, 1)), row(2, 10, 200, date(2025, 1, 2)),
            row(3, 30, 300, date(2025, 1, 3)), row(99, 90, 900, date(2025, 1, 4))]
    users, items, audit = profiles.build(rows, date(2025, 1, 4))
    # Two blocks in r1 still give r1 one review's weight, and do not get
    # normalized individually before the entity mean.
    np.testing.assert_allclose(users[10], np.array([1, 3]) / np.sqrt(10), atol=1e-7)
    np.testing.assert_allclose(items[100], [1, 0])
    np.testing.assert_allclose(items[200], [0, -1])
    np.testing.assert_allclose(profiles.review_vectors["user"][1], [.5, .5])
    assert set(users) == {10} and set(items) == {100, 200}
    assert audit["user"]["review_ids_by_entity"] == {10: [1, 2]}
    assert audit["identity"]["user_role"] == "query"
    assert audit["identity"]["item_role"] == "passage"
    assert torch.equal(store.vectors, before)


def test_caps_choose_latest_valid_reviews_with_deterministic_same_day_order():
    values = {rid: ([[rid, 1]], [[1, rid]]) for rid in range(1, 6)}
    values[6] = ([], [])
    profiles = cli.FrozenReviewProfiles(store_for(values), max_user_reviews=2, max_item_reviews=2)
    rows = [row(1, 1, 10, date(2025, 1, 1)), row(2, 1, 20, date(2025, 1, 2)),
            row(3, 2, 10, date(2025, 1, 3)), row(4, 1, 30, date(2025, 1, 4), rating=1),
            row(5, 3, 10, date(2025, 1, 4)), row(6, 1, 40, date(2025, 1, 5)),
            row(7, 1, 50, date(2025, 1, 6))]
    users, items, audit = profiles.build(tuple(reversed(rows)), date(2025, 1, 6))
    assert audit["user"]["review_ids_by_entity"][1] == [2, 4]
    assert audit["item"]["review_ids_by_entity"][10] == [3, 5]
    np.testing.assert_allclose(users[1], np.array([3, 1]) / np.sqrt(10), atol=1e-7)
    np.testing.assert_allclose(items[10], np.array([1, 4]) / np.sqrt(17), atol=1e-7)
    other_users, other_items, other_audit = profiles.build(rows, date(2025, 1, 6))
    assert audit == other_audit
    assert cli.profile_hash(users) == cli.profile_hash(other_users)
    assert cli.profile_hash(items) == cli.profile_hash(other_items)
    assert audit["identity"]["ratings"] == "all"


def test_same_day_tie_caps_and_inclusive_cutoff():
    profiles = cli.FrozenReviewProfiles(store_for({1: ([[1, 0]], [[1, 0]]),
                                                  2: ([[0, 1]], [[0, 1]])}),
                                       max_user_reviews=1, max_item_reviews=1)
    cutoff = date(2025, 1, 1)
    rows = [row(2, 1, 20, cutoff), row(1, 1, 10, cutoff)]
    users, _, audit = profiles.build(rows, cutoff)
    np.testing.assert_array_equal(users[1], [0, 1])
    assert audit["user"]["review_ids_by_entity"][1] == [2]
    assert audit["maximum_event_date"] == cutoff.isoformat()
    # Future rows are rejected even when they have no frozen text vector.
    with pytest.raises(ValueError, match="future review"):
        profiles.build(rows + [row(99, 1, 99, date(2025, 1, 2))], cutoff)


@pytest.mark.parametrize("caps", [(0, 30), (20, 0), (-1, 30)])
def test_profile_caps_must_be_positive(caps):
    with pytest.raises(ValueError, match="positive"):
        cli.FrozenReviewProfiles(store_for({1: ([], [])}),
                                 max_user_reviews=caps[0], max_item_reviews=caps[1])


def test_empty_or_cancelled_profiles_do_not_create_fake_coverage():
    profiles = cli.FrozenReviewProfiles(store_for({1: ([[1, 0]], [[1, 0]]),
                                                  2: ([[-1, 0]], [[0, 1]])}))
    users, items, audit = profiles.build([], date(2025, 1, 1), shuffle_seed=1701)
    assert users == items == {} and audit["maximum_event_date"] is None
    assert audit["user"]["shuffle_permutation"] == []
    rows = [row(1, 1, 10, date(2025, 1, 1)), row(2, 1, 20, date(2025, 1, 2))]
    users, items, audit = profiles.build(rows, date(2025, 1, 2))
    assert users == {} and set(items) == {10, 20}
    assert audit["user"]["profiles"] == audit["user"]["reviews_selected"] == 0


def test_shuffle_preserves_type_coverage_norms_and_exact_vector_multisets():
    values = {rid: ([[rid, 1]], [[-1, rid]]) for rid in range(1, 7)}
    profiles = cli.FrozenReviewProfiles(store_for(values))
    rows = [row(rid, rid, 10 * rid, date(2025, 1, rid)) for rid in values]
    cutoff = date(2025, 1, 6)
    users, items, original = profiles.build(rows, cutoff)
    shuffled_users, shuffled_items, shuffled = profiles.build(rows, cutoff, shuffle_seed=1701)
    repeat_users, repeat_items, repeated = profiles.build(tuple(reversed(rows)), cutoff, shuffle_seed=1701)
    for side, bank, permuted, repeated_bank in (("user", users, shuffled_users, repeat_users),
                                               ("item", items, shuffled_items, repeat_items)):
        assert set(bank) == set(permuted)
        assert sorted(v.tobytes() for v in bank.values()) == sorted(v.tobytes() for v in permuted.values())
        np.testing.assert_allclose([np.linalg.norm(v) for v in permuted.values()], 1, atol=1e-7)
        assert shuffled[side]["unshuffled_vectors_sha256"] == original[side]["vectors_sha256"]
        assert shuffled[side]["review_ids_by_entity"] == original[side]["review_ids_by_entity"]
        assert cli.profile_hash(permuted) == cli.profile_hash(repeated_bank)
        assert sorted(shuffled[side]["shuffle_permutation"]) == list(range(len(bank)))
    assert shuffled == repeated
    assert cli.profile_hash(users) != cli.profile_hash(shuffled_users)
    assert cli.profile_hash(items) != cli.profile_hash(shuffled_items)


@dataclass
class FakeAlignmentStats:
    epoch: int = 1
    alignment_loss: float = .25


@pytest.fixture
def graph_stub(monkeypatch):
    models = []

    class Graph:
        def __init__(self, config, alignment, users, items):
            self.config, self.alignment, self.profiles = config, alignment, (users, items)
            self.history = [SimpleNamespace(bpr_loss=.5)]
            self.alignment_history = [FakeAlignmentStats()]
            self.profile_coverage = {"profiled_users": len(users), "profiled_items": len(items)}
            self.fitted = None
            self.propagations = 0
            models.append(self)

        def fit(self, rows):
            self.fitted = tuple(rows)
            self.edge_count = len({(r.user_id, r.restaurant_id) for r in rows})
            self.user_ids = sorted({r.user_id for r in rows})
            self.item_ids = sorted({r.restaurant_id for r in rows})
            return self

        @property
        def final_embeddings(self):
            self.propagations += 1
            return np.zeros((len(self.user_ids) + len(self.item_ids), 2))

        def save(self, path):
            path.write_bytes(b"synthetic-graph")

    monkeypatch.setattr(cli, "RLMRecLightGCN", Graph)
    return models


def temporal_graphs(tmp_path, profiles):
    return cli.TemporalRLMGraphs(LightGCNConfig(epochs=1), months=3, profiles=profiles,
                                 alignment=RLMRecConfig(threads=1), directory=tmp_path, log=lambda _: None)


def test_historical_graph_and_profile_strictly_precede_calendar_block(tmp_path, graph_stub):
    rows = (row(1, 1, 10, date(2024, 12, 31)), row(2, 2, 20, date(2025, 3, 31)),
            row(3, 1, 30, date(2025, 4, 1)), row(4, 2, 40, date(2025, 5, 1)))
    values = {r.review_id: ([[1, r.review_id]], [[r.review_id, 1]]) for r in rows}
    profiles = cli.FrozenReviewProfiles(store_for(values))
    graphs = temporal_graphs(tmp_path, profiles)
    model = graphs.model_for(date(2025, 6, 30), rows, [r.event_date for r in rows])
    assert model.fitted == rows[:2]
    assert model.propagations == 1
    audit = json.loads((tmp_path / "2025-04-01.profiles.json").read_text())
    assert audit["cutoff_inclusive"] == "2025-03-31"
    assert audit["maximum_event_date"] == "2025-03-31"
    assert audit["user"]["review_ids_by_entity"] == {"1": [1], "2": [2]}
    assert set(model.profiles[1]) == {10, 20}  # No block-start target restaurant.
    same = graphs.model_for(date(2025, 4, 1), rows, [r.event_date for r in rows])
    assert same is model and len(graph_stub) == 1
    graphs.count_query(date(2025, 5, 1))
    assert graphs.rows()[0]["queries"] == 1
    assert graphs.rows()[0]["input_rows"] == 2
    assert (tmp_path / "2025-04-01.npz").exists()


@pytest.mark.parametrize("pairs", [[], [(1, 10)], [(1, 10), (2, 10)], [(1, 10), (1, 10)]])
def test_tiny_historical_graph_never_fits(tmp_path, graph_stub, pairs):
    rows = tuple(row(rid, user, item, date(2025, 1, rid))
                 for rid, (user, item) in enumerate(pairs, 1))
    values = {r.review_id: ([[1, 0]], [[0, 1]]) for r in rows} or {99: ([], [])}
    graphs = temporal_graphs(tmp_path, cli.FrozenReviewProfiles(store_for(values)))
    assert graphs.model_for(date(2025, 4, 1), rows, [r.event_date for r in rows]) is None
    assert all(model.fitted is None for model in graph_stub)
    assert graphs.rows()[0]["edges"] == len(set(pairs))
    assert graphs.rows()[0]["skipped"]
    assert not list(tmp_path.glob("*.npz"))


@pytest.mark.parametrize("pairs", [[(1, 10), (1, 20)], [(1, 10), (1, 20), (2, 10), (2, 20)]])
def test_saturated_historical_graph_has_no_negative_sampler(tmp_path, graph_stub, pairs):
    rows = tuple(row(rid, user, item, date(2025, 1, rid))
                 for rid, (user, item) in enumerate(pairs, 1))
    values = {r.review_id: ([[1, 0]], [[0, 1]]) for r in rows}
    graphs = temporal_graphs(tmp_path, cli.FrozenReviewProfiles(store_for(values)))
    assert graphs.model_for(date(2025, 4, 1), rows, [r.event_date for r in rows]) is None
    assert all(model.fitted is None for model in graph_stub)
    assert graphs.rows()[0]["skipped"] == "No user has an unvisited restaurant to sample"
    assert not list(tmp_path.glob("*.npz"))


def test_historical_cache_rejects_changed_input_count_before_reuse(tmp_path, graph_stub):
    rows = (row(1, 1, 10, date(2025, 1, 1)), row(2, 2, 20, date(2025, 1, 2)))
    graphs = temporal_graphs(tmp_path, cli.FrozenReviewProfiles(store_for({
        1: ([[1, 0]], [[0, 1]]), 2: ([[0, 1]], [[1, 0]])})))
    graphs.model_for(date(2025, 4, 1), rows, [r.event_date for r in rows])
    changed = rows + (row(3, 3, 30, date(2025, 1, 3)),)
    with pytest.raises(RuntimeError, match="input changed"):
        graphs.model_for(date(2025, 6, 1), changed, [r.event_date for r in changed])
    assert len(graph_stub) == 1


def test_non_tiny_fit_error_is_not_reported_as_a_small_graph(tmp_path, graph_stub, monkeypatch):
    rows = (row(1, 1, 10, date(2025, 1, 1)), row(2, 2, 20, date(2025, 1, 2)))
    graphs = temporal_graphs(tmp_path, cli.FrozenReviewProfiles(store_for({
        1: ([[1, 0]], [[0, 1]]), 2: ([[0, 1]], [[1, 0]])})))
    def failed_fit(self, rows):
        raise ValueError("synthetic training failure")
    monkeypatch.setattr(cli.RLMRecLightGCN, "fit", failed_fit)
    with pytest.raises(ValueError, match="synthetic training failure"):
        graphs.model_for(date(2025, 4, 1), rows, [r.event_date for r in rows])
    assert not graphs.summary


@pytest.fixture
def main_inputs(tmp_path, monkeypatch):
    """A valid tiny validation input; halt all model work at checkpoint loading."""
    exp = ExperimentConfig(ranker_training_mode="window", satisfaction_mode="history-aware")
    train = (row(1, 1, 10, date(2025, 1, 1)), row(2, 2, 20, date(2025, 1, 2)))
    validation = (row(3, 1, 20, date(2025, 2, 1)),)
    split = SimpleNamespace(train=train, validation=validation, train_cutoff=date(2025, 1, 31),
                            validation_cutoff=date(2025, 2, 28))
    queries, _ = build_window_queries(train, validation, config=exp, phase="validation", cutoff=split.train_cutoff)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text(json.dumps({"config": exp.to_dict(), "snapshot": {"dataset_snapshot_id": "same-snapshot"}}))
    (baseline / "recommendations_validation.jsonl").write_text(json.dumps({"query_id": queries[0].query_id, "baseline": [20]}) + "\n")
    prepared_path = tmp_path / "prepared.json"
    prepared = {"identity": {"snapshot_id": "same-snapshot", "lightgcn": exp.lightgcn_config.to_dict(),
                             "candidate_k": exp.candidate_k, "rrf_constant": exp.rrf_constant,
                             "features": list(exp.feature_names), "cutoffs": {"train_through": "2025-01-31", "validation_through": "2025-02-28"}},
                "data": {"ordered": {stage: {queries[0].query_id: [20]} for stage in ("c1_item_item", "c4_lightgcn", cli.STAGE1)}}}
    prepared_path.write_text(json.dumps(prepared))
    checkpoint = tmp_path / "frozen.pt"
    checkpoint.write_bytes(b"not-read-by-torch")
    artifacts = tmp_path / "runs"
    args = ["--snapshot", str(tmp_path / "snapshot.jsonl"), "--baseline-run", str(baseline),
            "--prepared-validation", str(prepared_path), "--frozen-checkpoint", str(checkpoint),
            "--artifacts-dir", str(artifacts), "--threads", "1"]
    monkeypatch.setattr(cli, "load_snapshot", lambda _: train + validation)
    monkeypatch.setattr(cli, "dataset_digest", lambda _: "same-snapshot")
    monkeypatch.setattr(cli, "load_review_texts", lambda *args: ({1: "original", 2: "original", 3: "future"}, {"synthetic": True}))
    monkeypatch.setattr(cli, "build_global_temporal_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(cli, "code_manifest", lambda *args, **kwargs: {"git_diff": "", "synthetic": True})
    def unexpected_load(*args, **kwargs):
        raise AssertionError("Model work reached")
    monkeypatch.setattr(cli.torch, "load", unexpected_load)
    return SimpleNamespace(args=args, prepared=prepared, prepared_path=prepared_path, baseline=baseline,
                           artifacts=artifacts, rows=train, queries=queries)


@pytest.mark.parametrize("args", [["--weights", "0"], ["--weights", "-0.1"], ["--weights", "nan"],
                                  ["--weights", "inf"], ["--weights", ".01", ".01"]])
def test_invalid_weights_fail_before_reading_inputs(args, monkeypatch):
    monkeypatch.setattr(cli, "load_snapshot", lambda _: pytest.fail("Input must not be read"))
    with pytest.raises(ValueError, match="distinct positive"):
        cli.main(args)


@pytest.mark.parametrize("field", ["snapshot_id", "lightgcn", "candidate_k", "rrf_constant", "features", "cutoffs"])
def test_main_rejects_prepared_protocol_mismatch_before_model_work(main_inputs, field):
    inputs = main_inputs
    inputs.prepared["identity"][field] = "different"
    inputs.prepared_path.write_text(json.dumps(inputs.prepared))
    with pytest.raises(ValueError, match="Prepared .* differs"):
        cli.main(inputs.args)
    assert not inputs.artifacts.exists()


def test_main_rejects_baseline_snapshot_mismatch_before_model_work(main_inputs):
    inputs = main_inputs
    path = inputs.baseline / "manifest.json"
    value = json.loads(path.read_text())
    value["snapshot"]["dataset_snapshot_id"] = "other-snapshot"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Baseline snapshot differs"):
        cli.main(inputs.args)
    assert not inputs.artifacts.exists()


@pytest.mark.parametrize("bad_ranks", [{"other-query": [20]}, {"validation:u1": [10]},
                                       {"validation:u1": [20, 20]}, {"validation:u1": [999]}])
def test_main_rejects_invalid_saved_candidate_sets_before_model_work(main_inputs, bad_ranks):
    inputs = main_inputs
    inputs.prepared["data"]["ordered"][cli.STAGE1] = bad_ranks
    inputs.prepared_path.write_text(json.dumps(inputs.prepared))
    with pytest.raises(ValueError, match="Ranking query IDs differ|Invalid catalog/history/ranking"):
        cli.main(inputs.args)
    assert not inputs.artifacts.exists()


@pytest.mark.parametrize("failure", ["future-store", "text-drift"])
def test_main_rejects_future_store_or_changed_retained_text_before_training(main_inputs, monkeypatch, failure):
    inputs = main_inputs
    store = store_for({1: ([[1, 0]], [[0, 1]])})
    if failure == "future-store":
        store.review_index[3] = 1
        expected = "non-training reviews"
    else:
        store.manifest["records"] = [{"review_id": 1, "blocks": [{"start": 0, "end": 8, "text": "modified"}]}]
        expected = "text spans differ"
    monkeypatch.setattr(cli.torch, "load", lambda *args, **kwargs: {"store": store})
    monkeypatch.setattr(cli, "RLMRecLightGCN", lambda *args, **kwargs: pytest.fail("Training must not begin"))
    with pytest.raises(ValueError, match=expected):
        cli.main(inputs.args)
    progress_paths = list(inputs.artifacts.glob("*/progress.json"))
    assert len(progress_paths) == 1
    progress = json.loads(progress_paths[0].read_text())
    assert progress["status"] == "failed" and progress["variants"] == []
    assert progress["test_evaluated"] is False
    assert progress["api_requests"] == progress["new_embedding_inference"] == 0


def test_ranking_contract_rejects_overlong_lists():
    history = row(1, 1, 10, date(2025, 1, 1))
    query = WindowQuery("q", "validation", 1, date(2025, 1, 31), (history,), (), {})
    with pytest.raises(ValueError, match="Invalid catalog/history/ranking"):
        cli.validate_ranks((query,), {"arm": {"q": (20, 30)}}, {10, 20, 30}, 1)
