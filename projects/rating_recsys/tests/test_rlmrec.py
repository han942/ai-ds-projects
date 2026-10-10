"""Independent objective, baseline preservation and inference checkpoint checks."""

from copy import deepcopy
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as functional

from rating_recsys.datasets.models import Interaction
from rating_recsys.retrieval.lightgcn import LightGCN, LightGCNConfig
from rating_recsys.retrieval.rlmrec import RLMRecConfig, RLMRecLightGCN


GRAPH = LightGCNConfig(dimension=4, layers=2, epochs=3, batch_size=4, learning_rate=0.01)
ALIGNMENT = RLMRecConfig(threads=1, batch_size=3)


def edges():
    return [Interaction(index, user, item, date(2025, 1, index), 5.0,
                        "exact", "synthetic", "region")
            for index, (user, item) in enumerate([
                (10, 100), (10, 200), (20, 200), (20, 300),
                (30, 300), (30, 400), (40, 400), (40, 500),
            ], 1)]


def profiles():
    rng = np.random.default_rng(9)
    return ({key: rng.normal(size=384).astype(np.float32) for key in (10, 20, 30, 40)},
            {key: rng.normal(size=384).astype(np.float32) for key in (100, 200, 300, 400, 500)})


def stats_without_time(model):
    return [(stats.epoch, stats.bpr_loss, stats.reg_loss) for stats in model.history]


@pytest.mark.parametrize("kwargs", [
    {"weight": -1}, {"weight": float("nan")}, {"weight": float("inf")},
    {"temperature": 0}, {"temperature": float("nan")}, {"batch_size": 0},
    {"batch_size": 2.5}, {"threads": False}, {"seed": -1}, {"seed": 2**63},
])
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RLMRecConfig(**kwargs)


@pytest.mark.parametrize("layers", [0, 2])
def test_zero_weight_exactly_preserves_lightgcn(layers):
    graph = replace(GRAPH, layers=layers)
    users, items = profiles()
    baseline = LightGCN(graph).fit(edges())
    model = RLMRecLightGCN(graph, replace(ALIGNMENT, weight=0), users, items).fit(edges())
    assert np.array_equal(model._ego, baseline._ego)
    assert np.array_equal(model.final_embeddings, baseline.final_embeddings)
    assert stats_without_time(model) == stats_without_time(baseline)
    assert model.projector is None
    assert model.score(10, [100, 300, 500, 999]) == baseline.score(10, [100, 300, 500, 999])
    assert model.top_k([10, 20, 999], [{100, 200}, {200, 300}, set()], 5) == baseline.top_k(
        [10, 20, 999], [{100, 200}, {200, 300}, set()], 5)
    assert all(stats.alignment_loss == 0 for stats in model.alignment_history)


@pytest.mark.parametrize("mode", ["empty", "unknown", "singleton", "zero"])
def test_missing_profiles_have_no_alignment_effect(mode):
    users, items = profiles()
    if mode == "empty":
        users, items = {}, {}
    elif mode == "unknown":
        users, items = {999: users[10]}, {999: items[100]}
    elif mode == "singleton":
        users, items = {10: users[10]}, {100: items[100]}
    else:
        users = {key: np.zeros_like(value) for key, value in users.items()}
        items = {key: np.zeros_like(value) for key, value in items.items()}
    baseline = LightGCN(GRAPH).fit(edges())
    model = RLMRecLightGCN(GRAPH, ALIGNMENT, users, items).fit(edges())
    assert np.array_equal(model._ego, baseline._ego)
    assert stats_without_time(model) == stats_without_time(baseline)
    assert all(stats.aligned_batches == 0 for stats in model.alignment_history)


def test_combined_gradient_matches_independent_torch_graph_objective():
    users, items = profiles()
    model = RLMRecLightGCN(replace(GRAPH, epochs=1), ALIGNMENT, users, items).fit(edges())
    # A repeated user/positive item and an item in both roles exercise dedup and
    # accumulation. The small groups stay below the cap, so no sampling occurs.
    u = np.array([0, 0, 1, 2], dtype=np.int64)
    i = np.array([4, 4, 5, 6], dtype=np.int64)
    j = np.array([6, 7, 8, 7], dtype=np.int64)
    projector = deepcopy(model.projector)
    ego = torch.tensor(model._ego, requires_grad=True)
    adjacency = torch.tensor(model._adjacency.toarray())
    layer = ego
    layers = [layer]
    for _ in range(model.config.layers):
        layer = adjacency @ layer
        layers.append(layer)
    final = torch.stack(layers).mean(dim=0)
    margin = (final[u] * (final[i] - final[j])).sum(dim=1)
    bpr = functional.softplus(-margin).sum()
    regularization = model.config.regularization / 2 * ego[np.concatenate([u, i, j])].square().sum()
    objective = (bpr + regularization) / len(u)
    expected_terms = []
    for group in (np.unique(u), np.unique(i), np.unique(j)):
        text = np.stack([model._node_profiles[int(node)] for node in group])
        semantic = projector(torch.tensor(text))
        cf = final[group]
        cf = cf / (cf.square().sum(dim=1, keepdim=True) + 1e-8).sqrt()
        semantic = semantic / (semantic.square().sum(dim=1, keepdim=True) + 1e-8).sqrt()
        logits = cf @ semantic.T / ALIGNMENT.temperature
        # Explicit equation 18, independently of the implementation's CE call.
        term = (torch.logsumexp(logits, dim=1) - logits.diag()).mean()
        expected_terms.append(term)
        objective = objective + ALIGNMENT.weight * term
    objective.backward()
    old_weights = {key: value.detach().clone() for key, value in model.projector.state_dict().items()}
    bpr_sum, reg_sum, gradient = model._loss_and_gradient(u, i, j)
    assert bpr_sum == pytest.approx(float(bpr.detach()), abs=1e-6)
    assert reg_sum == pytest.approx(float(regularization.detach()), abs=1e-8)
    np.testing.assert_allclose(gradient, ego.grad.numpy(), rtol=3e-5, atol=2e-6)
    for actual, expected in zip(model.projector.parameters(), projector.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=3e-5, atol=2e-6)
    assert any(not torch.equal(old_weights[key], value) for key, value in model.projector.state_dict().items())
    np.testing.assert_allclose(model._epoch_losses / len(u), [float(t.detach()) for t in expected_terms], rtol=2e-6)


def test_alignment_pools_are_unique_capped_and_exclude_missing_profiles():
    users, items = profiles()
    del users[20]
    del items[300]
    model = RLMRecLightGCN(GRAPH, replace(ALIGNMENT, batch_size=2), users, items).fit(edges())
    u, i, j = model._alignment_groups(np.array([0, 0, 1, 2, 3]),
                                    np.array([4, 4, 5, 6, 7, 8]), np.array([4, 5, 6, 7, 8]))
    assert len(u) == len(i) == len(j) == 2
    for group in (u, i, j):
        assert len(group) == len(set(group))
        assert all(int(node) in model._node_profiles for node in group)
    assert 1 not in u and 6 not in i and 6 not in j


def test_active_alignment_preserves_bpr_triple_sampling():
    baseline_triples, aligned_triples = [], []

    class ObservedLightGCN(LightGCN):
        def _loss_and_gradient(self, u, i, j):
            baseline_triples.append((u.copy(), i.copy(), j.copy()))
            return super()._loss_and_gradient(u, i, j)

    class ObservedRLMRec(RLMRecLightGCN):
        def _loss_and_gradient(self, u, i, j):
            aligned_triples.append((u.copy(), i.copy(), j.copy()))
            return super()._loss_and_gradient(u, i, j)

    users, items = profiles()
    ObservedLightGCN(GRAPH).fit(edges())
    model = ObservedRLMRec(GRAPH, replace(ALIGNMENT, batch_size=2), users, items).fit(edges())
    assert any(stats.alignment_loss > 0 for stats in model.alignment_history)
    for before, after in zip(baseline_triples, aligned_triples, strict=True):
        for first, second in zip(before, after, strict=True):
            np.testing.assert_array_equal(first, second)


def test_deterministic_profiles_are_frozen_and_global_torch_state_is_preserved():
    users, items = profiles()
    captured = torch.get_rng_state().clone()
    thread_count = torch.get_num_threads()
    first = RLMRecLightGCN(GRAPH, ALIGNMENT, users, items)
    for value in (*users.values(), *items.values()):
        value.fill(999)
    first.fit(edges())
    fresh_users, fresh_items = profiles()
    second = RLMRecLightGCN(GRAPH, ALIGNMENT, dict(reversed(list(fresh_users.items()))),
                           dict(reversed(list(fresh_items.items())))).fit(reversed(edges()))
    np.testing.assert_array_equal(first._ego, second._ego)
    assert stats_without_time(first) == stats_without_time(second)
    for a, b in zip(first.projector.parameters(), second.projector.parameters(), strict=True):
        assert torch.equal(a, b)
    assert torch.equal(captured, torch.get_rng_state())
    assert torch.get_num_threads() == thread_count
    before = first._ego.copy()
    first.fit(edges())
    np.testing.assert_array_equal(before, first._ego)


def test_callback_receives_committed_alignment_history_and_can_stop():
    users, items = profiles()
    seen = []

    def stop(stats, model):
        assert model.alignment_history[-1].epoch == stats.epoch
        assert len(model.history) == len(model.alignment_history)
        seen.append(stats.epoch)
        return stats.epoch == 2

    model = RLMRecLightGCN(GRAPH, ALIGNMENT, users, items).fit(edges(), callback=stop)
    assert seen == [1, 2]
    assert model.profile_coverage == {"graph_users": 4, "graph_items": 5,
                                      "profiled_users": 4, "profiled_items": 5}
    for stats in model.alignment_history:
        assert stats.batches == 2
        assert stats.weighted_alignment_loss == pytest.approx(ALIGNMENT.weight * stats.alignment_loss)
        assert stats.alignment_gradient_norm > 0 and stats.bpr_gradient_norm > 0


def test_projector_initialization_failure_restores_torch_threads(monkeypatch):
    users, items = profiles()
    model = RLMRecLightGCN(GRAPH, ALIGNMENT, users, items)
    before = torch.get_num_threads()

    def fail():
        raise RuntimeError("injected projector initialization failure")

    monkeypatch.setattr(model, "_new_projector", fail)
    with pytest.raises(RuntimeError, match="injected"):
        model.fit(edges())
    assert torch.get_num_threads() == before


@pytest.mark.parametrize("weight", [0, 0.01])
def test_inference_checkpoint_reload_is_exact_and_cannot_resume_training(tmp_path, weight):
    users, items = profiles()
    model = RLMRecLightGCN(GRAPH, replace(ALIGNMENT, weight=weight), users, items).fit(edges())
    path = tmp_path / "model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
    loaded = RLMRecLightGCN.load(path)
    np.testing.assert_array_equal(loaded._ego, model._ego)
    np.testing.assert_array_equal(loaded.final_embeddings, model.final_embeddings)
    assert loaded.score(10, [100, 300, 500, 999]) == model.score(10, [100, 300, 500, 999])
    assert loaded.recommend([10, 20, 999], {10: {100, 200}, 20: {200, 300}}, 5) == model.recommend(
        [10, 20, 999], {10: {100, 200}, 20: {200, 300}}, 5)
    assert loaded.alignment_history == model.alignment_history
    assert loaded.history == model.history
    assert loaded.profile_coverage == model.profile_coverage
    if model.projector is not None:
        for first, second in zip(model.projector.parameters(), loaded.projector.parameters(), strict=True):
            assert torch.equal(first, second)
    with pytest.raises(ValueError, match="inference-only"):
        loaded.fit(edges())
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_checkpoint_write_preserves_existing_file(tmp_path, monkeypatch):
    users, items = profiles()
    model = RLMRecLightGCN(GRAPH, ALIGNMENT, users, items).fit(edges())
    path = tmp_path / "model.npz"
    model.save(path)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected archive failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="injected"):
        model.save(path)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_rejects_invalid_csr_offsets_and_object_arrays(tmp_path):
    users, items = profiles()
    model = RLMRecLightGCN(GRAPH, ALIGNMENT, users, items).fit(edges())
    path = tmp_path / "model.npz"
    model.save(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    values["adjacency_indptr"][2] = values["adjacency_indptr"][1] - 1
    corrupt = tmp_path / "corrupt.npz"
    np.savez_compressed(corrupt, **values)
    with pytest.raises(ValueError, match="adjacency"):
        RLMRecLightGCN.load(corrupt)
    np.savez_compressed(corrupt, metadata=np.asarray([{}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        RLMRecLightGCN.load(corrupt)


@pytest.mark.parametrize("bad", [np.full(384, np.nan), np.ones((2, 384)), np.ones(0)])
def test_bad_profiles_rejected(bad):
    with pytest.raises(ValueError, match="Profiles"):
        RLMRecLightGCN(GRAPH, ALIGNMENT, {10: bad}, {})


def test_profile_dimension_mismatch_rejected():
    with pytest.raises(ValueError, match="dimension"):
        RLMRecLightGCN(GRAPH, ALIGNMENT, {10: np.ones(384)}, {100: np.ones(383)})
