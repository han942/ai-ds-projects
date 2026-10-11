"""Separate-interest catalog search, clustering, history identity and controls."""
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from rating_recsys.retrieval.multi_interest import (
    MultiInterestConfig, MultiInterestRetriever, build_multi_interest_bank,
    round_robin_candidates, spherical_clusters,
)
from rating_recsys.retrieval.review_multivector import (
    ReviewMultiVectorConfig, ReviewMultiVectorRetriever, build_review_bank, unit,
)


def row(rid, user, item, day=1):
    return SimpleNamespace(review_id=rid, user_id=user, restaurant_id=item,
                           event_date=date(2025, 1, day), rating=1 if rid % 2 else 5)


def example():
    history = [row(rid, 1, 80) for rid in range(1, 10)] + [row(10, 1, 90)]
    history += [row(11, 2, 10), row(12, 2, 20), row(13, 2, 30)]
    vectors = unit([[0, 0]] + [[1, 0]] * 9 + [[0, 1], [1, 0], [0, 1], [0.9, 0.1]])
    source = build_review_bank(history, date(2025, 1, 2),
                              {i: i for i in range(1, 14)}, vectors, vectors,
                              query_users=[1, 99])
    graph_keys = np.asarray([10, 20, 30, 80, 90])
    graph_values = unit([[1, 0], [0, 1], [0.9, 0.1], [1, 0], [0, 1]])
    return source, history, graph_keys, graph_values


def test_clustering_caps_distinct_vectors_and_means_match_returned_groups():
    values = unit([[1, 0], [0, 1], [1, 0], [0, 1]])
    groups, centers, mask = spherical_clusters(values)
    assert groups.tolist() == [0, 1, 0, 1]
    assert mask.tolist() == [True, True]
    np.testing.assert_array_equal(centers, [[1, 0], [0, 1]])
    repeat = spherical_clusters(values)
    for left, right in zip((groups, centers, mask), repeat):
        np.testing.assert_array_equal(left, right)
    rng = np.random.default_rng(42)
    values = unit(rng.normal(size=(20, 8)))
    # Even at one iteration, centroids describe these exact returned groups.
    groups, centers, mask = spherical_clusters(values, config=MultiInterestConfig(max_iterations=1))
    for group in range(len(centers)):
        np.testing.assert_allclose(centers[group], unit(values[groups == group].mean(0)), atol=1e-7)


def test_one_interest_matches_identical_input_mean_and_does_not_fabricate_cancellation():
    source, history, keys, values = example()
    built = build_multi_interest_bank(source, history, keys, values,
                                     config=MultiInterestConfig(interests=1))
    scorer = MultiInterestRetriever(built)
    old = ReviewMultiVectorRetriever(source).score_all()["mean"]
    np.testing.assert_allclose(scorer.score_all(view="text")[:, 0], old, atol=2e-7)
    for index, user in enumerate(source.user_ids):
        refs = source.user_review_ids[index]
        grouped = [values[np.flatnonzero(keys == next(r.restaurant_id for r in history if r.review_id == rid))[0]]
                   for rid in refs[refs >= 0]]
        if grouped:
            np.testing.assert_allclose(built.graph_centroids[index, 0], unit(np.mean(grouped, 0)), atol=1e-7)
    assert built.assignments[0].tolist() == [0] * 10 + [-1] * 10
    assert (built.assignments[1] == -1).all()
    groups, centers, mask = spherical_clusters(np.asarray([[1., 0], [-1., 0]]),
                                               config=MultiInterestConfig(interests=1))
    assert groups.tolist() == [0, 0] and mask.tolist() == [False]
    assert not np.any(centers)


def test_lists_preserve_minority_interest_without_averaging_interest_scores():
    source, history, keys, values = example()
    exclude = {1: {80, 90}}
    one = MultiInterestRetriever(build_multi_interest_bank(source, history, keys, values,
                                config=MultiInterestConfig(interests=1)))
    many_bank = build_multi_interest_bank(source, history, keys, values)
    many = MultiInterestRetriever(many_bank)
    for view in ("text", "graph"):
        one_rank = one.rankings(one.score_all(view=view), exclude, k=2)
        many_rank = many.rankings(many.score_all(view=view), exclude, k=2)
        assert one_rank.merged[1] == (30, 10)
        assert many_rank.merged[1] == (10, 20)
        assert many_rank.per_interest[1] == ((10, 30), (20, 30), ())
        assert one_rank.merged[99] == many_rank.merged[99] == ()
        assert many_rank.per_interest[99] == ((), (), ())
    assert many_bank.assignments[0, :10].tolist() == [0] * 9 + [1]
    assert many_bank.review_ids.tolist() == source.user_review_ids.tolist()
    # No rating gate: both odd low-rated and even high-rated reviews are retained.
    assert many_bank.audit["ratings"].startswith("all;")


def test_negative_scores_id_ties_missing_graph_and_foreign_padding_are_explicit():
    source, history, keys, values = example()
    source.user_vectors[source.user_review_ids < 0] = 777.
    source.item_vectors[source.item_review_ids < 0] = 777.
    keys, values = np.asarray([10, 20, 30]), unit([[-1, 0], [-1, 0], [-1, 0]])
    built = build_multi_interest_bank(source, history, keys, values)
    scorer = MultiInterestRetriever(built)
    # All retained user's visited items are absent from this graph checkpoint.
    assert not built.graph_mask.any()
    assert np.isneginf(scorer.score_all(view="graph")).all()
    assert scorer.rankings(scorer.score_all(view="graph"), {}, k=100).merged == {1: (), 99: ()}
    text = scorer.score_all(view="text")
    np.testing.assert_array_equal(text[0, 0], [1., 0., np.float32(0.9 / np.hypot(.9, .1)), 1., 0.])
    # Arbitrary equal negative scores are valid candidates; ID is the tie break.
    mock = np.full(text.shape, -np.inf, dtype=np.float32)
    mock[0, 0] = -.5
    rankings = scorer.rankings(mock, {1: {20, 80}}, k=100)
    assert rankings.merged[1] == (10, 30, 90)


def test_round_robin_deduplicates_fills_budget_and_one_list_is_exact():
    assert round_robin_candidates([(1, 2, 3), (1, 4, 2), ()], k=4) == (1, 2, 4, 3)
    original = tuple(range(100))
    assert round_robin_candidates([original], k=100) == original
    result = round_robin_candidates([original, tuple(range(50, 150)), tuple(range(100, 200))], k=100)
    assert len(result) == len(set(result)) == 100
    assert round_robin_candidates([], k=100) == ()


def test_wrong_history_cutoff_entity_duplicate_and_malformed_banks_are_rejected():
    source, history, keys, values = example()
    with pytest.raises(ValueError, match="Future"):
        build_multi_interest_bank(source, history+[row(99, 3, 10, 3)], keys, values)
    with pytest.raises(ValueError, match="Duplicate historical"):
        build_multi_interest_bank(source, history+history[:1], keys, values)
    with pytest.raises(ValueError, match="another user"):
        build_multi_interest_bank(source, [row(1, 7, 80)]+history[1:], keys, values)
    with pytest.raises(ValueError, match="missing"):
        build_multi_interest_bank(source, history[1:], keys, values)
    duplicate = replace(source, user_review_ids=source.user_review_ids.copy())
    duplicate.user_review_ids[0, 1] = duplicate.user_review_ids[0, 0]
    with pytest.raises(ValueError, match="Duplicate retained"):
        build_multi_interest_bank(duplicate, history, keys, values)
    with pytest.raises(ValueError, match="duplicate graph"):
        build_multi_interest_bank(source, history, np.asarray([10, 10]), np.ones((2, 2)))
    with pytest.raises(ValueError, match="Invalid graph"):
        build_multi_interest_bank(source, history, keys, values[:-1])
    with pytest.raises(ValueError, match="Nonfinite"):
        build_multi_interest_bank(source, history, keys, values * np.nan)


def test_cancelled_text_mean_can_still_have_valid_graph_prototype():
    history = [row(1, 1, 10), row(2, 1, 10)]
    vectors = np.asarray([[0, 0], [1, 0], [-1, 0]], dtype=np.float32)
    source = build_review_bank(history, date(2025, 1, 2), {1: 1, 2: 2}, vectors, vectors, query_users=[1])
    built = build_multi_interest_bank(source, history, np.asarray([10]), np.asarray([[1., 0]]),
                                     config=MultiInterestConfig(interests=1))
    assert built.text_mask.tolist() == [[False]]
    assert built.graph_mask.tolist() == [[True]]
    scorer = MultiInterestRetriever(built)
    assert scorer.rankings(scorer.score_all(view="text"), {}, k=100).merged[1] == ()
    assert scorer.rankings(scorer.score_all(view="graph"), {}, k=100).merged[1] == (10,)
    assert scorer.rankings(scorer.score_all(view="graph"), {1: {10}}, k=100).merged[1] == ()


@pytest.mark.parametrize("field", ["interests", "max_iterations"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_config(field, value):
    with pytest.raises(ValueError, match="positive integer"):
        MultiInterestConfig(**{field: value})


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_budget(value):
    with pytest.raises(ValueError, match="positive integer"):
        round_robin_candidates([], k=value)
