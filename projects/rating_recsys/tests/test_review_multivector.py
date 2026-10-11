"""Temporal banks, actual top-k semantics, missing text and candidate budgets."""
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from rating_recsys.retrieval.review_multivector import (
    ReviewMultiVectorConfig, ReviewMultiVectorRetriever, ReviewVectorBank,
    build_review_bank, review_vectors_from_store, supplement_candidates, unit,
)


def row(rid, user, item, day=1, rating=5):
    return SimpleNamespace(review_id=rid, user_id=user, restaurant_id=item,
                           event_date=date(2025, 1, day), rating=rating)


def bank(users, items):
    """Ragged lists, with deliberately nonzero foreign padding."""
    ukeys, ikeys = sorted(users), sorted(items)
    def pack(groups, keys):
        cap = max(1, *(len(groups[key]) for key in keys))
        values = np.full((len(keys), cap, 2), 777., dtype=np.float32)
        refs = np.full((len(keys), cap), -1, dtype=np.int64)
        for index, key in enumerate(keys):
            count = len(groups[key])
            if count:
                values[index, :count] = unit(groups[key])
                refs[index, :count] = np.arange(count) + 1
        return refs, values
    ur, uv = pack(users, ukeys)
    ir, iv = pack(items, ikeys)
    return ReviewVectorBank(np.asarray(ukeys), np.asarray(ikeys), ur, ir, uv, iv, {})


def test_block_padding_never_contributes_and_roles_remain_separate():
    store = SimpleNamespace(
        vectors=np.asarray([[99., -99.], [1., 0.], [0., 1.]], np.float32),
        query_blocks=np.asarray([[0, 0], [1, 2], [0, 0]]),
        document_blocks=np.asarray([[0, 0], [1, 1], [0, 0]]), review_index={10: 1, 20: 2})
    original = store.vectors.copy()
    users, items = review_vectors_from_store(store)
    np.testing.assert_allclose(users[1], [2**-.5, 2**-.5], atol=1e-7)
    np.testing.assert_array_equal(items[1], [1, 0])
    np.testing.assert_array_equal(users[2], [0, 0])
    np.testing.assert_array_equal(original, store.vectors)


def test_bank_caps_are_temporal_deterministic_and_retain_low_rating():
    rows = [row(1, 1, 10), row(2, 1, 20, 2), row(3, 2, 10, 3),
            row(4, 1, 30, 4, rating=1), row(5, 2, 10, 4)]
    vectors = unit([[0, 0], [1, 0], [1, 1], [0, 1], [-1, 0], [1, -1]])
    config = ReviewMultiVectorConfig(max_user_reviews=2, max_item_reviews=2)
    result = build_review_bank(tuple(reversed(rows)), date(2025, 1, 4),
                              {i: i for i in range(1, 6)}, vectors, vectors,
                              query_users=[1, 99], config=config)
    assert result.user_review_ids.tolist() == [[2, 4], [-1, -1]]
    assert result.item_review_ids.tolist() == [[3, 5], [2, -1], [4, -1]]
    assert result.audit["users_with_text"] == 1
    assert result.audit["ratings"] == "all; no new rating gate"
    repeat = build_review_bank(rows, date(2025, 1, 4), {i: i for i in range(1, 6)},
                               vectors, vectors, query_users=[99, 1], config=config)
    assert repeat.audit == result.audit
    with pytest.raises(ValueError, match="future"):
        build_review_bank(rows+[row(99, 99, 99, 5)], date(2025, 1, 4), {}, vectors, vectors, query_users=[1])
    with pytest.raises(ValueError, match="Duplicate"):
        build_review_bank(rows+rows[:1], date(2025, 1, 4), {}, vectors, vectors, query_users=[1])


def test_matching_precedes_pooling_and_can_change_full_catalog_candidates():
    data = bank({1: [[1, 0], [0, 1]]},
                {10: [[1, 0]]*3 + [[0, 1]]*3 + [[-1, 0]]*3, 20: [[1, 1]]})
    retriever = ReviewMultiVectorRetriever(data)
    scores = retriever.score_all()
    assert retriever.rankings(scores["mean"], {}, k=1)[1] == (20,)
    assert retriever.rankings(scores["late"], {}, k=1)[1] == (10,)
    np.testing.assert_allclose(scores["late"], [[1., 2**-.5]], atol=1e-7)


def test_negative_cosines_small_item_banks_and_missing_text_do_not_use_padding():
    data = bank({1: [[1, 0]], 2: []},
                {10: [[-1, 0]], 20: [[-1, 1], [-1, -1]], 30: []})
    retriever = ReviewMultiVectorRetriever(data)
    scores = retriever.score_all()
    np.testing.assert_allclose(scores["late"][0, :2], [-1, -2**-.5], atol=1e-7)
    assert np.isneginf(scores["late"][0, 2])
    assert np.isneginf(scores["late"][1]).all()
    assert retriever.rankings(scores["late"], {}, k=100) == {1: (20, 10), 2: ()}
    assert retriever.rankings(scores["late"], {1: {20}}, k=100)[1] == (10,)
    # Even extremely large foreign padding did not create positive similarities.
    assert scores["late"][0, :2].max() < 0


def test_batching_does_not_change_matching_definition():
    data = bank({1: [[1, 0], [0, 1]], 2: [], 3: [[1, -1]]},
                {10: [[-1, 1], [0, 1]], 20: [[1, 0]], 30: [], 40: [[1, 1]]*4})
    small = ReviewMultiVectorConfig(user_batch_size=1, item_batch_size=1)
    a = ReviewMultiVectorRetriever(data, small).score_all()
    b = ReviewMultiVectorRetriever(data, replace(small, user_batch_size=3, item_batch_size=4)).score_all()
    for name in a:
        np.testing.assert_allclose(a[name], b[name], atol=1e-7)
    # Independent scalar definition, including the min(k, true review count).
    for u in range(len(data.user_ids)):
        uv = data.user_vectors[u][data.user_review_ids[u] >= 0]
        for i in range(len(data.item_ids)):
            iv = data.item_vectors[i][data.item_review_ids[i] >= 0]
            if not len(uv) or not len(iv):
                assert np.isneginf(a["late"][u, i])
            else:
                expected = np.mean([np.mean(sorted(float(x@y) for y in iv)[-min(3, len(iv)):]) for x in uv])
                np.testing.assert_allclose(a["late"][u, i], expected, atol=1e-7)


def test_deterministic_ties_and_no_popularity_in_standalone_text_arm():
    data = bank({1: [[1, 0]], 2: []}, {30: [[1, 0]], 10: [[1, 0]]})
    retriever = ReviewMultiVectorRetriever(data)
    assert retriever.rankings(retriever.score_all()["late"], {}, k=2) == {1: (10, 30), 2: ()}


def test_quota_only_adds_novel_items_with_fixed_budget_and_exact_empty_fallback():
    original = tuple(range(100))
    assert supplement_candidates(original, (), text_slots=10) == original
    assert supplement_candidates(original, range(200), text_slots=0) == original
    result = supplement_candidates(original, [95, 105, 105, 106], text_slots=10)
    assert result == tuple(range(98)) + (105, 106)
    assert len(result) == len(set(result)) == 100
    assert supplement_candidates(original, range(100, 200), text_slots=10) == tuple(range(90))+tuple(range(100, 110))


@pytest.mark.parametrize("field", ["max_user_reviews", "max_item_reviews", "matching_k", "user_batch_size", "item_batch_size"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_config_is_rejected(field, value):
    with pytest.raises(ValueError, match="positive integer"):
        ReviewMultiVectorConfig(**{field: value})
