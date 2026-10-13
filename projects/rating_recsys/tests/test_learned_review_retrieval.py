"""Learned semantic retrieval: controlled inputs, masking, BPR and persistence."""
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rating_recsys.retrieval.learned_review_retrieval import (
    LearnedReviewRetrievalConfig, LearnedReviewRetriever, mean_vectors,
    pair_text_scores, shuffle_review_vectors,
)
from rating_recsys.retrieval.lightgcn import _sample_negatives
from rating_recsys.retrieval.review_multivector import (
    ReviewMultiVectorConfig, array_hash, build_review_bank,
)


def example():
    rows = [SimpleNamespace(review_id=i+1, user_id=user, restaurant_id=item,
                            event_date=date(2025, 1, i+1), rating=1 + i % 5)
            for i, (user, item) in enumerate(((1, 10), (1, 20), (2, 20), (2, 30), (3, 10), (3, 30)))]
    vectors = np.random.default_rng(17).normal(size=(7, 4)).astype(np.float32)
    vectors[0] = 0
    bank = build_review_bank(rows, date(2025, 1, 6), {i: i for i in range(1, 7)}, vectors, vectors,
                             query_users=[1, 2, 3], config=ReviewMultiVectorConfig(max_user_reviews=2, max_item_reviews=3))
    config = LearnedReviewRetrievalConfig(input_dim=4, projected_dim=2, cf_dimension=2,
        max_user_reviews=2, max_item_reviews=3, epochs=2, threads=1, batch_size=2)
    rng = np.random.default_rng(42)
    cf_u = rng.normal(size=(3, 2)).astype(np.float32)
    cf_i = rng.normal(size=(3, 2)).astype(np.float32)
    return rows, bank, config, cf_u, cf_i


def fitted(config_override=None, callback=None):
    rows, bank, config, cf_u, cf_i = example()
    model = LearnedReviewRetriever(config_override or config).fit(rows, bank=bank,
        cf_user_vectors=cf_u, cf_item_vectors=cf_i, cutoff=date(2025, 1, 6), callback=callback)
    return model, rows, bank, cf_u, cf_i


def test_scalar_mean_and_late_definitions_mask_all_padding_including_negative_matches():
    users = torch.tensor([[[1., 0.], [0., 1.], [999., 999.]]], requires_grad=True)
    items = torch.tensor([[[[-1., 0.], [999., 999.], [999., 999.]],
                            [[1., 0.], [0., 1.], [-1., 0.]]]], requires_grad=True)
    umask = torch.tensor([[True, True, False]])
    imask = torch.tensor([[[True, False, False], [True, True, True]]])
    for mode in ("mean", "late"):
        actual = pair_text_scores(users, items, umask, imask, mode, 3)
        u = users.detach().numpy()[0, :2]
        expected = []
        for choice in range(2):
            v = items.detach().numpy()[0, choice][imask[0, choice].numpy()]
            if mode == "mean":
                a, b = u.mean(0), v.mean(0)
                expected.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
            else:
                expected.append(float(np.mean([np.mean(sorted(float(x@y) for y in v)[-min(3, len(v)):]) for x in u])))
        np.testing.assert_allclose(actual.detach().numpy(), [expected], atol=1e-6)
        actual.sum().backward(retain_graph=True)
    assert torch.isfinite(users.grad).all() and torch.isfinite(items.grad).all()
    assert torch.count_nonzero(users.grad[0, 2]) == 0
    assert torch.count_nonzero(items.grad[0, 0, 1:]) == 0
    # With k=1 this is MaxSim; padding cannot beat a negative cosine.
    scores = pair_text_scores(users[:,:1], items[:,:1], umask[:,:1], imask[:,:1], "late", 1)
    assert scores.item() == -1


@pytest.mark.parametrize("mode", ["mean", "late"])
def test_empty_masks_have_zero_semantic_contribution_and_finite_gradients(mode):
    users = torch.randn(2, 2, 3, requires_grad=True)
    items = torch.randn(2, 2, 3, 3, requires_grad=True)
    mask_u = torch.zeros(2, 2, dtype=torch.bool)
    mask_i = torch.zeros(2, 2, 3, dtype=torch.bool)
    scores = pair_text_scores(users, items, mask_u, mask_i, mode, 3)
    assert torch.count_nonzero(scores) == 0
    scores.sum().backward()
    assert torch.isfinite(users.grad).all() and torch.isfinite(items.grad).all()


def test_positive_relationship_reviews_are_removed_on_both_sides():
    model, _, bank, _, _ = fitted()
    umask, imask = model.training_masks([model.user_ids[1]], [model.item_ids[10]], [model.item_ids[30]])
    selected_user = bank.user_review_ids[model.user_ids[1]][umask[0].numpy()].tolist()
    selected_positive = bank.item_review_ids[model.item_ids[10]][imask[0, 0].numpy()].tolist()
    assert selected_user == [2]  # review #1 about positive restaurant 10 is removed
    assert selected_positive == [5]  # positive restaurant review #1 written by user 1 is removed
    assert set(selected_user).isdisjoint(selected_positive)
    # The reduced user mask is shared by positive and negative scoring.
    scores, count = model._batch_scores(torch.tensor([model.user_ids[1]]),
        torch.tensor([model.item_ids[10]]), torch.tensor([model.item_ids[30]]))
    assert count == 1 and scores.shape == (1, 2) and torch.isfinite(scores).all()


@pytest.mark.parametrize("mode", ["mean", "late"])
def test_balanced_presence_keeps_cf_and_has_no_text_loss_gradient_for_coverage_only_triple(mode):
    rows, bank, base, cf_u, cf_i = example()
    config = replace(base, mode=mode, balance_text_presence=True)
    balanced = LearnedReviewRetriever(config).fit(rows, bank=bank, cf_user_vectors=cf_u,
        cf_item_vectors=cf_i, cutoff=date(2025, 1, 6))
    users, positives, negatives = torch.tensor([balanced.user_ids[1]]), torch.tensor([balanced.item_ids[10]]), torch.tensor([balanced.item_ids[30]])
    # Delete review #5 so only user 1's own review remains in positive item 10.
    # Own-review exclusion removes that last review, but the negative pool has text.
    balanced._item_mask[balanced.item_ids[10], 1:] = False
    balanced.scorer.zero_grad(set_to_none=True)
    scores, usable = balanced._batch_scores(users, positives, negatives)
    choices = torch.stack((positives, negatives), dim=1)
    cf = torch.einsum("bd,bcd->bc", balanced._cf_users[users], balanced._cf_items[choices])
    torch.testing.assert_close(scores, cf, rtol=0, atol=0)
    assert usable == 0 and balanced._batch_presence["negative_only"] == 1
    torch.nn.functional.softplus(scores[:, 1] - scores[:, 0]).mean().backward()
    assert balanced.scorer.coefficient.grad.item() == 0
    assert torch.count_nonzero(balanced.scorer.projector.weight.grad) == 0
    # The legacy arm still exposes the asymmetric availability offset.
    balanced.config = replace(config, balance_text_presence=False)
    legacy_scores, legacy_usable = balanced._batch_scores(users, positives, negatives)
    assert legacy_usable == 1 and not torch.equal(legacy_scores, cf)


def test_balancing_changes_only_text_training_policy_not_parameters_sampling_or_cf():
    _, _, config, _, _ = example()
    ordinary, _, _, _, _ = fitted(config)
    balanced, _, _, _, _ = fitted(replace(config, balance_text_presence=True))
    for key in ("initial_parameters_sha256", "sampling_audit", "parameter_count", "frozen_cf_sha256"):
        assert ordinary.metadata[key] == balanced.metadata[key]
    for audit in balanced.metadata["training_text_presence"]:
        assert audit["triples"] == balanced.metadata["trainable_edges"]
        assert audit["positive_only"] + audit["negative_only"] + audit["both_text_present"] + audit["neither"] == audit["triples"]
        assert audit["text_usable_triples"] == audit["both_text_present"]
    assert balanced.metadata["balance_text_presence"] is True


def test_shuffle_keeps_correspondence_masks_and_counts_and_does_not_mutate_original():
    _, bank, _, _, _ = example()
    originals = (bank.user_vectors.copy(), bank.item_vectors.copy())
    first, audit = shuffle_review_vectors(bank, 42)
    second, again = shuffle_review_vectors(bank, 42)
    assert audit == again
    for side, original in zip(("user", "item"), originals):
        refs = getattr(bank, side + "_review_ids")
        mask = refs >= 0
        values = getattr(first, side + "_vectors")
        np.testing.assert_array_equal(refs, getattr(first, side + "_review_ids"))
        np.testing.assert_array_equal(original, getattr(bank, side + "_vectors"))
        np.testing.assert_array_equal(values, getattr(second, side + "_vectors"))
        assert not np.array_equal(values[mask], original[mask])
        assert np.count_nonzero(values[~mask]) == 0
        assert sorted(map(tuple, values[mask])) == sorted(map(tuple, original[mask]))


def test_real_shuffle_mean_late_share_initialization_and_exact_lightgcn_triples():
    rows, bank, base, cf_u, cf_i = example()
    models = []
    for mode, condition in (("mean", "actual"), ("late", "actual"), ("late", "shuffled"), ("mean", "shuffled")):
        model = LearnedReviewRetriever(replace(base, mode=mode, text_condition=condition)).fit(rows, bank=bank,
            cf_user_vectors=cf_u, cf_item_vectors=cf_i, cutoff=date(2025, 1, 6))
        models.append(model)
    for model in models[1:]:
        for key in ("initial_parameters_sha256", "sampling_audit", "edge_users_sha256", "edge_items_sha256", "parameter_count", "frozen_cf_sha256"):
            assert model.metadata[key] == models[0].metadata[key]
    edges_u = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    edges_i = np.array([0, 1, 1, 2, 0, 2], dtype=np.int64)
    rng = np.random.default_rng(base.seed)
    rng.normal(0, base.cf_init_std, size=(6, base.cf_dimension))
    for epoch in range(base.epochs):
        order = rng.permutation(len(edges_u))
        negatives = _sample_negatives(rng, edges_u, 3, np.unique(edges_u * 3 + edges_i))
        audit = models[0].metadata["sampling_audit"][epoch]
        assert array_hash(order) == audit["order_sha256"]
        assert array_hash(negatives) == audit["negative_items_sha256"]
        assert not np.isin(edges_u * 3 + negatives, edges_u * 3 + edges_i).any()
    assert models[0].coefficient != pytest.approx(base.initial_coefficient, abs=1e-7)
    for model in models:
        np.testing.assert_array_equal(model._cf_users.numpy(), cf_u)
        np.testing.assert_array_equal(model._cf_items.numpy(), cf_i)
        assert not model._cf_users.requires_grad and not model._users.requires_grad


@pytest.mark.parametrize("mode", ["mean", "late"])
def test_catalog_scoring_matches_pairwise_semantics_chunking_and_exact_cf_fallback(mode):
    _, _, base, _, _ = example()
    model, _, bank, _, _ = fitted(replace(base, mode=mode, user_batch_size=1, item_batch_size=1))
    scores = model.score_all([1, 2, 3, 999])
    projected_u, projected_i = model._encode()
    for row in range(3):
        for item in range(3):
            text = pair_text_scores(projected_u[row:row+1], projected_i[item:item+1][None],
                model._user_mask[row:row+1], model._item_mask[item:item+1][None], mode, base.matching_k).item()
            cf = float(model._cf_users[row] @ model._cf_items[item])
            assert scores[row, item] == pytest.approx(cf + model.coefficient * text, abs=1e-6)
    assert np.isneginf(scores[3]).all()
    model.config = replace(model.config, user_batch_size=3, item_batch_size=3)
    np.testing.assert_allclose(scores[:3], model.score_all([1, 2, 3]), atol=1e-6)
    # Missing reviews preserve the full CF catalog rather than dropping candidates.
    model._user_mask[0] = False
    model._item_mask[0] = False
    scores = model.score_all([1, 2])
    np.testing.assert_allclose(scores[0], (model._cf_users[0] @ model._cf_items.T).numpy(), atol=1e-7)
    assert scores[1, 0] == pytest.approx(float(model._cf_users[1] @ model._cf_items[0]), abs=1e-7)
    ranks = model.recommend([1, 999], {1: [10, 20]}, 100)
    assert ranks == {1: (30,), 999: ()}


@pytest.mark.parametrize("condition", ["actual", "shuffled"])
@pytest.mark.parametrize("include_banks", [True, False])
def test_checkpoint_restores_exact_full_catalog_scores_and_verifies_bank(tmp_path, condition, include_banks):
    _, _, base, _, _ = example()
    model, _, bank, _, _ = fitted(replace(base, mode="late", text_condition=condition))
    expected = model.score_all()
    path = tmp_path / "model.pt"
    artifact = model.save(path, include_banks=include_banks)
    assert artifact["bytes"] == path.stat().st_size
    restored = LearnedReviewRetriever.load(path, bank=None if include_banks else bank)
    np.testing.assert_array_equal(expected, restored.score_all())
    assert restored.coefficient == model.coefficient
    assert restored.metadata == model.metadata
    if not include_banks:
        with pytest.raises(ValueError, match="requires"):
            LearnedReviewRetriever.load(path)
        bank.user_vectors[0, 0] *= 1.2
        with pytest.raises(ValueError, match="arrays differ"):
            LearnedReviewRetriever.load(path, bank=bank)


def test_future_unknown_review_and_ownership_fail_closed():
    rows, bank, base, cf_u, cf_i = example()
    def fit(r=rows, b=bank, cutoff=date(2025, 1, 6)):
        return LearnedReviewRetriever(base).fit(r, bank=b, cf_user_vectors=cf_u,
            cf_item_vectors=cf_i, cutoff=cutoff)
    with pytest.raises(ValueError, match="before cutoff"):
        fit(cutoff=date(2025, 1, 5))
    bank.user_review_ids[0, 0] = 999
    with pytest.raises(ValueError, match="nonhistorical"):
        fit()
    bank.user_review_ids[0, 0] = 3
    with pytest.raises(ValueError, match="ownership"):
        fit()


def test_foreign_nonfinite_padding_is_ignored_and_checkpoint_cf_tampering_rejected(tmp_path):
    rows, bank, config, cf_u, cf_i = example()
    original, _, _, _, _ = fitted(replace(config, mode="late"))
    bank.user_vectors[bank.user_review_ids < 0] = np.nan
    bank.item_vectors[bank.item_review_ids < 0] = np.inf
    padded = LearnedReviewRetriever(replace(config, mode="late")).fit(rows, bank=bank,
        cf_user_vectors=cf_u, cf_item_vectors=cf_i, cutoff=date(2025, 1, 6))
    np.testing.assert_array_equal(original.score_all(), padded.score_all())
    path = tmp_path / "tampered.pt"
    padded.save(path)
    saved = torch.load(path, weights_only=True)
    saved["cf_user_vectors"][0, 0] += 1
    torch.save(saved, path)
    with pytest.raises(ValueError, match="CF arrays differ"):
        LearnedReviewRetriever.load(path)


def test_callback_and_determinism_restore_torch_settings():
    _, _, base, _, _ = example()
    calls = []
    threads = torch.get_num_threads()
    model, rows, bank, cf_u, cf_i = fitted(callback=lambda stats, model: calls.append(stats) or True)
    assert len(calls) == len(model.history) == 1
    assert calls[0].triples_with_text == 6 and np.isfinite(calls[0].bpr_loss)
    assert torch.get_num_threads() == threads
    again = LearnedReviewRetriever(replace(base, epochs=1)).fit(list(reversed(rows)), bank=bank,
        cf_user_vectors=cf_u, cf_item_vectors=cf_i, cutoff=date(2025, 1, 6))
    np.testing.assert_array_equal(model.score_all(), again.score_all())


@pytest.mark.parametrize("kwargs", [{"mode": "bad"}, {"text_condition": "bad"}, {"matching_k": 0},
    {"epochs": True}, {"seed": -1}, {"weight_decay": -1}, {"learning_rate": float("nan")},
    {"initial_coefficient": float("inf")}, {"max_user_reviews": 1.5}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        LearnedReviewRetrievalConfig(**kwargs)
