"""Graph semantics, exact baseline preservation and independent BPR gradients."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from rating_recsys.datasets.models import Interaction
from rating_recsys.retrieval.lightgcn import LightGCN, LightGCNConfig
from rating_recsys.retrieval.text_item_graph import TextItemGraph, build_semantic_graph


CONFIG = LightGCNConfig(dimension=4, layers=3, epochs=3, batch_size=3,
                        learning_rate=.01, regularization=.002)


def rows():
    return [Interaction(index, user, item, date(2025, 1, index), 5.,
                        "exact", "synthetic", "region")
            for index, (user, item) in enumerate([
                (10, 100), (10, 200), (20, 200), (20, 300),
                (30, 300), (30, 400), (40, 400), (40, 500),
            ], 1)]


def profiles():
    return np.array([100, 200, 300, 400, 500], dtype=np.int64), np.array([
        [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=np.float32)


def metrics_without_time(model):
    return [(s.epoch, s.bpr_loss, s.reg_loss) for s in model.history]


@pytest.mark.parametrize("mode", ["binary", "weighted"])
def test_knn_matches_independent_scalar_cosine_and_normalization(mode):
    ids = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    vectors = np.array([[1, 0], [1, 0], [.8, .6], [0, 1], [0, 0], [-1, 0]], dtype=np.float32)
    adjacency, audit = build_semantic_graph(ids, vectors, neighbors=1, graph_mode=mode)
    expected = np.zeros((len(ids), len(ids)), dtype=np.float64)
    for source in range(len(ids)):
        if not np.linalg.norm(vectors[source]):
            continue
        scores = []
        for target in range(len(ids)):
            if source == target or not np.linalg.norm(vectors[target]):
                continue
            a, b = vectors[source].astype(float), vectors[target].astype(float)
            cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if cosine > 0:
                scores.append((target, cosine))
        for target, cosine in sorted(scores, key=lambda pair: (-pair[1], ids[pair[0]]))[:1]:
            expected[source, target] = 1 if mode == "binary" else cosine
    expected = np.maximum(expected, expected.T)
    degree = expected.sum(1)
    for source, target in zip(*np.nonzero(expected)):
        expected[source, target] /= np.sqrt(degree[source] * degree[target])
    np.testing.assert_allclose(adjacency.toarray(), expected, atol=1e-7, rtol=1e-7)
    assert np.array_equal(adjacency.toarray(), adjacency.toarray().T)
    assert not adjacency.diagonal().any()
    assert adjacency[4].nnz == adjacency[5].nnz == 0
    assert audit["missing_text_nodes"] == 1
    assert audit["nonisolated_nodes"] == 4


def test_binary_shuffle_independently_preserves_each_degree_and_changes_edges():
    rng = np.random.default_rng(7)
    vectors = rng.uniform(.1, 1., size=(40, 8)).astype(np.float32)
    vectors[[3, 11]] = 0
    ids = np.arange(100, 140, dtype=np.int64)
    actual, audit = build_semantic_graph(ids, vectors, neighbors=3)
    shuffled, control = build_semantic_graph(ids, vectors, neighbors=3, shuffle_seed=91)
    repeat, repeated = build_semantic_graph(ids, vectors, neighbors=3, shuffle_seed=91)
    assert actual.nnz == shuffled.nnz
    # Nonzero entries count raw binary edges independently of normalization.
    assert np.array_equal(np.diff(actual.indptr), np.diff(shuffled.indptr))
    assert np.array_equal(shuffled.toarray(), shuffled.toarray().T)
    assert not shuffled.diagonal().any()
    assert shuffled[3].nnz == shuffled[11].nnz == 0
    assert (actual != shuffled).nnz > 0
    assert np.array_equal(repeat.toarray(), shuffled.toarray())
    assert control == repeated
    assert audit["sorted_weighted_degrees_sha256"] == control["sorted_weighted_degrees_sha256"]
    assert control["accepted_swaps"] == control["swap_target"]
    assert control["changed_undirected_edges"] > 0
    assert control["control"] == "degree_preserving_edge_swaps"


def test_weighted_shuffle_preserves_raw_weights_and_weighted_degree_multiset():
    ids, vectors = profiles()
    model = TextItemGraph(CONFIG, ids, vectors, neighbors=1, graph_mode="weighted").fit(rows())
    control = TextItemGraph(CONFIG, ids, vectors, neighbors=1, graph_mode="weighted", shuffle_seed=9).fit(rows())
    assert np.array_equal(np.sort(model._semantic_raw.data), np.sort(control._semantic_raw.data))
    actual_degrees = np.asarray(model._semantic_raw.sum(axis=1, dtype=np.float64)).ravel()
    shuffled_degrees = np.asarray(control._semantic_raw.sum(axis=1, dtype=np.float64)).ravel()
    assert np.array_equal(np.sort(actual_degrees), np.sort(shuffled_degrees))
    assert control.graph_audit["control"] == "text_node_permutation"
    assert "Per-restaurant degrees" in control.graph_audit["control_limitations"]


@pytest.mark.parametrize("layers", [0, 1, 3])
@pytest.mark.parametrize("shuffle_seed", [None, 11])
def test_zero_weight_exactly_preserves_baseline(layers, shuffle_seed):
    ids, vectors = profiles()
    config = replace(CONFIG, layers=layers)
    baseline = LightGCN(config).fit(rows())
    model = TextItemGraph(config, ids, vectors, weight=0, neighbors=2,
                          shuffle_seed=shuffle_seed).fit(reversed(rows()))
    assert np.array_equal(model._ego, baseline._ego)
    assert np.array_equal(model.final_embeddings, baseline.final_embeddings)
    assert metrics_without_time(model) == metrics_without_time(baseline)
    users, exclude = [10, 20, 999], {10: [100, 200], 20: [200, 300]}
    assert model.recommend(users, exclude, 100) == baseline.recommend(users, exclude, 100)
    assert model.score(10, [100, 500, 999]) == baseline.score(10, [100, 500, 999])


def test_active_semantic_edges_do_not_change_initialization_or_bpr_samples():
    ids, vectors = profiles()
    captures = []
    models = []
    for weight, shuffle_seed in ((0, None), (.1, None), (.1, 29)):
        model = TextItemGraph(CONFIG, ids, vectors, weight=weight, neighbors=1,
                              shuffle_seed=shuffle_seed)
        original = model._loss_and_gradient
        captured = []

        def capture(u, i, j, original=original, captured=captured):
            captured.append((u.copy(), i.copy(), j.copy()))
            return original(u, i, j)

        model._loss_and_gradient = capture
        model.fit(rows())
        captures.append(captured)
        models.append(model)
    assert models[0].initialization_sha256 == models[1].initialization_sha256 == models[2].initialization_sha256
    assert models[0].sampling_history == models[1].sampling_history == models[2].sampling_history
    for control in captures[1:]:
        assert len(control) == len(captures[0])
        for actual, expected in zip(control, captures[0]):
            assert all(np.array_equal(a, b) for a, b in zip(actual, expected))
    assert not np.array_equal(models[0].final_embeddings, models[1].final_embeddings)
    assert all(item["examples"] == 8 and item["batches"] == 3 for item in models[1].sampling_history)


@pytest.mark.parametrize("mode", ["binary", "weighted"])
def test_gradient_matches_independent_dense_torch_combined_graph(mode):
    torch = pytest.importorskip("torch")
    ids, vectors = profiles()
    model = TextItemGraph(replace(CONFIG, epochs=1), ids, vectors,
                          neighbors=1, weight=.3, graph_mode=mode).fit(rows())
    # Repeated user and positive/negative overlap exercise gradient accumulation.
    u = np.array([0, 0, 1, 2], dtype=np.int64)
    i = np.array([4, 4, 5, 6], dtype=np.int64)
    j = np.array([6, 7, 8, 7], dtype=np.int64)
    ego = torch.tensor(model._ego, requires_grad=True)
    adjacency = torch.tensor(model._adjacency.toarray())
    n_users = len(model.user_ids)
    adjacency[n_users:, n_users:] += model.weight * torch.tensor(model._semantic.toarray())
    layer = ego
    states = [layer]
    for _ in range(model.config.layers):
        layer = adjacency @ layer
        states.append(layer)
    final = torch.stack(states).mean(0)
    margin = (final[u] * (final[i] - final[j])).sum(1)
    bpr = torch.nn.functional.softplus(-margin).sum()
    reg = model.config.regularization / 2 * ego[np.concatenate((u, i, j))].square().sum()
    ((bpr + reg) / len(u)).backward()
    actual_bpr, actual_reg, gradient = model._loss_and_gradient(u, i, j)
    assert actual_bpr == pytest.approx(float(bpr.detach()), rel=2e-6)
    assert actual_reg == pytest.approx(float(reg.detach()), rel=2e-6)
    np.testing.assert_allclose(gradient, ego.grad.numpy(), atol=2e-7, rtol=3e-5)


def test_missing_and_unknown_profile_nodes_use_only_training_catalog():
    ids = np.array([100, 200, 999], dtype=np.int64)
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    model = TextItemGraph(CONFIG, ids, vectors, neighbors=1).fit(rows())
    assert np.array_equal(model.item_keys, [100, 200, 300, 400, 500])
    assert model._semantic[2:].nnz == 0
    assert model.graph_audit["catalog_items_absent_from_bank"] == 3
    assert model.graph_audit["missing_text_nodes"] == 3
    assert 999 not in model.semantic_edges[:, :2]


@pytest.mark.parametrize("mode", ["binary", "weighted"])
def test_all_missing_text_has_exact_baseline_behavior(mode):
    ids, vectors = profiles()
    vectors[:] = 0
    baseline = LightGCN(CONFIG).fit(rows())
    model = TextItemGraph(CONFIG, ids, vectors, graph_mode=mode, shuffle_seed=9).fit(rows())
    assert model._semantic.nnz == 0
    assert np.array_equal(model._ego, baseline._ego)
    assert model.graph_audit["undirected_edges"] == 0


def test_shuffle_handles_complete_graph_with_finite_budget():
    ids = np.arange(1, 5, dtype=np.int64)
    adjacency, audit = build_semantic_graph(ids, np.ones((4, 2)), neighbors=10, shuffle_seed=1)
    assert adjacency.nnz == 12
    assert audit["accepted_swaps"] == 0
    assert not audit["swap_budget_reached"]


def test_repeated_fit_resets_audit_and_early_stop_callback_preserves_contract():
    ids, vectors = profiles()
    model = TextItemGraph(CONFIG, ids, vectors)
    called = []

    def stop(stats, passed_model):
        assert passed_model is model
        called.append(stats.epoch)
        return stats.epoch == 2

    assert model.fit(iter(rows()), callback=stop) is model
    assert called == [1, 2]
    assert len(model.history) == len(model.sampling_history) == 2
    model.fit(rows())
    assert len(model.sampling_history) == 3
    assert [s["epoch"] for s in model.sampling_history] == [1, 2, 3]


@pytest.mark.parametrize("mode,shuffle_seed", [("binary", None), ("binary", 51), ("weighted", 51)])
def test_save_load_restores_exact_scores_and_provenance(tmp_path, mode, shuffle_seed):
    ids, vectors = profiles()
    model = TextItemGraph(CONFIG, ids, vectors, neighbors=1, graph_mode=mode,
                          shuffle_seed=shuffle_seed).fit(rows())
    path = tmp_path / "model.npz"
    model.save(path)
    restored = TextItemGraph.load(path)
    assert np.array_equal(restored.final_embeddings, model.final_embeddings)
    assert np.array_equal(restored.semantic_edges, model.semantic_edges)
    assert restored.graph_audit == model.graph_audit
    assert restored.manifest() == model.manifest()
    assert restored.recommend([10, 20, 999], {10: [100, 200]}, 5) == model.recommend([10, 20, 999], {10: [100, 200]}, 5)
    with pytest.raises(ValueError, match="inference-only"):
        restored.fit(rows())


def test_checkpoint_rejects_semantic_matrix_tampering(tmp_path):
    ids, vectors = profiles()
    model = TextItemGraph(CONFIG, ids, vectors, neighbors=1).fit(rows())
    path = tmp_path / "model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    data["semantic_data"] *= .5
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="provenance hashes"):
        TextItemGraph.load(path)


@pytest.mark.parametrize("kwargs", [
    {"weight": -1}, {"weight": np.nan}, {"weight": np.inf}, {"weight": True},
    {"neighbors": 0}, {"neighbors": 1.5}, {"neighbors": False},
    {"shuffle_seed": -1}, {"shuffle_seed": True}, {"graph_mode": "unknown"},
])
def test_invalid_configuration_is_rejected(kwargs):
    ids, vectors = profiles()
    with pytest.raises(ValueError):
        TextItemGraph(CONFIG, ids, vectors, **kwargs)


@pytest.mark.parametrize("ids,vectors", [
    ([2, 1], [[1], [1]]), ([1, 1], [[1], [1]]), ([0, 1], [[1], [1]]),
    ([1., 2.], [[1], [1]]), ([1, 2], [[1]]), ([1, 2], [[np.nan], [1]]),
    ([1, 2], [[np.inf], [1]]), ([1, 2], [[], []]),
])
def test_invalid_vector_bank_is_rejected(ids, vectors):
    with pytest.raises(ValueError):
        build_semantic_graph(np.asarray(ids), np.asarray(vectors))
