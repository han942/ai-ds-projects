from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import date
from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.models import RecommendationQuery
from rating_recsys.ranking.features import FEATURE_NAMES
from rating_recsys.ranking.features import LIGHTGCN as FEATURES_LIGHTGCN
from rating_recsys.retrieval.baselines import (
    BaselineCandidateGenerator,
    IncrementalRetrievalContext,
    build_context,
)
from rating_recsys.retrieval.hybrid import HybridCandidateGenerator, rrf
from rating_recsys.retrieval.lightgcn import (
    LIGHTGCN, LightGCN, LightGCNConfig, _is_positive, _sample_negatives,
)
from rating_recsys.experiments.candidate_models import CANDIDATE_MODELS
from support import (
    _interaction, available_models, cuisine, synthetic_interactions, texts_for, tiny_grid,
)

HAS_TORCH = find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from rating_recsys.retrieval.deepconn import (
        PAD, SEP, CharVocabulary, DeepCoNN, DeepCoNNConfig, DeepCoNNNet,
        EntityDocuments, build_documents,
    )


def interaction(review_id: int, user: int, restaurant: int, day: int) -> Interaction:
    return Interaction(
        review_id=review_id,
        user_id=user,
        restaurant_id=restaurant,
        event_date=date(2025, 1, day),
        rating=5.0,
        reviewed_at_precision="exact",
        restaurant_name=f"restaurant-{restaurant}",
        region="서울" if restaurant < 4 else "부산",
    )


class BaselineCandidateGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = interaction(1, 10, 1, 1)
        self.target = interaction(20, 10, 3, 10)
        self.query = RecommendationQuery(
            query_id="validation:u10:r20",
            phase="validation",
            user_id=10,
            cutoff=self.target.event_date,
            history=(self.history,),
            target=self.target,
            relevance=2,
        )
        self.available = (
            self.history,
            interaction(2, 20, 1, 2),
            interaction(3, 20, 3, 3),
            interaction(4, 30, 1, 4),
            interaction(5, 30, 3, 5),
            interaction(6, 40, 2, 6),
            interaction(7, 40, 4, 7),
        )

    def test_excludes_seen_items_and_preserves_source_attribution(self) -> None:
        result, _ = BaselineCandidateGenerator(candidate_k=3).retrieve(
            self.query,
            self.available,
        )

        self.assertNotIn(1, {item.restaurant_id for item in result.union})
        self.assertEqual(result.item_item[0].restaurant_id, 3)
        target = next(item for item in result.union if item.restaurant_id == 3)
        self.assertEqual(
            target.candidate_sources,
            ("popularity", "item_item", "region_popularity"),
        )
        self.assertTrue(result.target_available)

    def test_no_region_ablation_excludes_region_candidate_source(self) -> None:
        result, _ = BaselineCandidateGenerator(
            candidate_k=3, include_region=False
        ).retrieve(self.query, self.available)

        self.assertEqual(result.region_popularity, ())
        self.assertTrue(result.union)
        for candidate in result.union:
            self.assertNotIn("region_popularity", candidate.candidate_sources)
            self.assertNotIn("region_popularity", candidate.source_scores)
            self.assertNotIn("region_popularity", candidate.source_ranks)

    def test_incremental_context_matches_batch_context_and_candidates(self) -> None:
        incremental = IncrementalRetrievalContext()
        for item in self.available:
            incremental.add(item)

        batch_context = build_context(self.available)
        self.assertEqual(incremental.context, batch_context)

        generator = BaselineCandidateGenerator(candidate_k=4)
        batch_result, _ = generator.retrieve(self.query, self.available)
        incremental_result, _ = generator.retrieve_from_context(
            self.query,
            incremental.context,
        )
        self.assertEqual(batch_result.popularity, incremental_result.popularity)
        self.assertEqual(batch_result.item_item, incremental_result.item_item)
        self.assertEqual(
            batch_result.region_popularity,
            incremental_result.region_popularity,
        )
        self.assertEqual(batch_result.union, incremental_result.union)

    def test_sparse_item_scores_match_direct_similarity(self) -> None:
        second_visit = interaction(8, 10, 4, 8)
        query = RecommendationQuery(
            query_id="validation:u10:r20:two-history",
            phase="validation",
            user_id=10,
            cutoff=self.target.event_date,
            history=(self.history, second_visit),
            target=self.target,
            relevance=2,
        )
        result, context = BaselineCandidateGenerator(candidate_k=4).retrieve(
            query, (*self.available, second_visit)
        )
        expected_scores = {}
        for candidate in result.popularity:
            similarities = [
                context.similarity(item.restaurant_id, candidate.restaurant_id)
                for item in query.history
            ]
            expected_scores[candidate.restaurant_id] = sum(similarities)
            self.assertAlmostEqual(
                candidate.source_scores["item_item"], sum(similarities)
            )
            self.assertAlmostEqual(
                candidate.source_scores["item_item_max"], max(similarities)
            )
        expected_order = sorted(
            (item for item in expected_scores if expected_scores[item] > 0),
            key=lambda item: (-expected_scores[item], item),
        )
        self.assertEqual(
            [candidate.restaurant_id for candidate in result.item_item],
            expected_order,
        )

    def test_is_deterministic_and_uses_restaurant_id_for_ties(self) -> None:
        generator = BaselineCandidateGenerator(candidate_k=4)
        first, _ = generator.retrieve(self.query, self.available)
        second, _ = generator.retrieve(self.query, reversed(self.available))

        self.assertEqual(first.union, second.union)
        popularity_ids = [item.restaurant_id for item in first.popularity]
        self.assertLess(popularity_ids.index(2), popularity_ids.index(4))

    def test_base_quota_fraction_controls_c0_c1_reservation(self) -> None:
        generator = BaselineCandidateGenerator(candidate_k=4)
        self.assertEqual(generator.base_quota, 2)
        self.assertEqual(
            BaselineCandidateGenerator(candidate_k=4, base_quota_fraction=0.0).base_quota,
            0,
        )
        self.assertEqual(
            BaselineCandidateGenerator(candidate_k=4, base_quota_fraction=1.0).base_quota,
            4,
        )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            BaselineCandidateGenerator(base_quota_fraction=1.5)

        default, _ = generator.retrieve(self.query, self.available)
        explicit, _ = BaselineCandidateGenerator(
            candidate_k=4, base_quota_fraction=0.5
        ).retrieve(self.query, self.available)
        self.assertEqual(default.union, explicit.union)

        plain, _ = BaselineCandidateGenerator(
            candidate_k=4, base_quota_fraction=0.0
        ).retrieve(self.query, self.available)
        expected = sorted(
            {item.restaurant_id for item in plain.union},
            key=lambda item: (
                -next(c.rrf_score for c in plain.union if c.restaurant_id == item),
                item,
            ),
        )
        self.assertEqual([item.restaurant_id for item in plain.union], expected)


class _FixedScorer:
    def __init__(self, scores: dict[int, float]) -> None:
        self.scores = scores

    def score(self, user_id, restaurant_ids):
        return {item: self.scores[item] for item in restaurant_ids if item in self.scores}


class HybridCandidateGeneratorTests(unittest.TestCase):
    """C5 = RRF(C1, C4 LightGCN); C0/C2/C3 stay as reference lists."""

    setUp = BaselineCandidateGeneratorTests.setUp

    def test_c5_fuses_c1_and_lightgcn_only(self) -> None:
        context = build_context(self.available)
        generator = HybridCandidateGenerator(candidate_k=3, rrf_constant=1)
        scorer = _FixedScorer({2: 3.0, 4: 2.0, 3: 1.0})
        # 1 is in the history and 99 is not in the context: both are dropped.
        result = generator.retrieve(
            self.query, context, graph_ranked=(1, 2, 99, 4, 3), scorer=scorer
        )

        self.assertEqual(result.lightgcn, (2, 4, 3))
        c1 = [c.restaurant_id for c in result.reference.item_item]
        self.assertEqual(c1, [3])
        # RRF with constant 1: 3 -> 1/2 + 1/4, 2 -> 1/2, 4 -> 1/3.
        self.assertEqual([c.restaurant_id for c in result.union], [3, 2, 4])
        by_id = {c.restaurant_id: c for c in result.union}
        self.assertEqual(by_id[3].candidate_sources, ("item_item", "lightgcn"))
        self.assertEqual(by_id[2].candidate_sources, ("lightgcn",))
        self.assertAlmostEqual(by_id[3].rrf_score, 0.75)
        # C0 is not fused but its rank is kept as a feature.
        self.assertIn("popularity", by_id[2].source_ranks)
        self.assertNotIn("popularity", by_id[2].candidate_sources)
        z = [c.source_scores["lightgcn_z"] for c in result.union]
        self.assertAlmostEqual(sum(z), 0.0)
        self.assertGreater(by_id[2].source_scores["lightgcn_z"], 0)
        self.assertEqual(
            [c.restaurant_id for c in result.reference.union],
            [c.restaurant_id for c in BaselineCandidateGenerator(
                candidate_k=3, rrf_constant=1, base_quota_fraction=0.75
            ).retrieve_from_context(self.query, context)[0].union],
        )

    def test_without_graph_c5_is_c1_order(self) -> None:
        context = build_context(self.available)
        result = HybridCandidateGenerator(candidate_k=3).retrieve(self.query, context)
        self.assertEqual(result.lightgcn, ())
        self.assertEqual(
            [c.restaurant_id for c in result.union],
            [c.restaurant_id for c in result.reference.item_item],
        )
        self.assertTrue(all(c.source_scores["lightgcn_z"] == 0 for c in result.union))


class HybridHelpersTests(unittest.TestCase):
    def test_rrf_sums_reciprocal_ranks_and_breaks_ties_by_id(self) -> None:
        self.assertEqual(rrf([(3, 1), (1, 2)], k=3, constant=1), (1, 3, 2))
        self.assertEqual(rrf([(5,), (4,)], k=2, constant=60), (4, 5))

    def test_feature_names_keep_region_last(self) -> None:
        self.assertEqual(FEATURE_NAMES[-1], "region_affinity")
        self.assertEqual(LIGHTGCN, FEATURES_LIGHTGCN)


def two_communities() -> list:
    """Users 1-10 visit restaurants 1-6, users 11-20 visit 7-12 (4 each)."""

    rng = np.random.default_rng(3)
    rows, review_id = [], 1
    for user in range(1, 21):
        pool = np.arange(1, 7) if user <= 10 else np.arange(7, 13)
        for day, item in enumerate(rng.choice(pool, size=4, replace=False)):
            rows.append(_interaction(review_id, user, int(item), day))
            review_id += 1
    return rows


LIGHTGCN_TINY = LightGCNConfig(dimension=8, layers=2, epochs=60, batch_size=16, learning_rate=0.05)


class LightGCNModelTests(unittest.TestCase):
    def test_gradient_matches_finite_differences(self) -> None:
        config = LightGCNConfig(dimension=3, layers=2, epochs=1, regularization=0.1)
        model = LightGCN(config).fit(two_communities()[:12])
        model._ego = model._ego.astype(np.float64)
        n_users = len(model.user_ids)
        u = np.array([0, 1, 1])
        i = n_users + np.array([0, 1, 0])
        j = n_users + np.array([2, 3, 3])
        adjacency = model._adjacency
        model._adjacency = adjacency.astype(np.float64)

        def objective() -> float:
            bpr, reg, _ = model._loss_and_gradient(u, i, j)
            return (bpr + reg) / len(u)

        _, _, analytic = model._loss_and_gradient(u, i, j)
        numeric = np.zeros_like(model._ego)
        epsilon = 1e-6
        for index in np.ndindex(*model._ego.shape):
            original = model._ego[index]
            model._ego[index] = original + epsilon
            upper = objective()
            model._ego[index] = original - epsilon
            lower = objective()
            model._ego[index] = original
            numeric[index] = (upper - lower) / (2 * epsilon)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-7)

    def test_adjacency_is_symmetric_normalized(self) -> None:
        model = LightGCN(replace(LIGHTGCN_TINY, epochs=1)).fit(two_communities())
        dense = model._adjacency.toarray()
        np.testing.assert_allclose(dense, dense.T)
        degree = (dense > 0).sum(axis=1)
        rows, cols = np.nonzero(dense)
        np.testing.assert_allclose(
            dense[rows, cols], 1 / np.sqrt(degree[rows] * degree[cols]), rtol=1e-6
        )

    def test_learns_communities_and_loss_decreases(self) -> None:
        # Long training on this tiny graph memorises each user's own visits and
        # pushes unvisited in-community restaurants down with the rest; 20
        # epochs is before that point.
        model = LightGCN(replace(LIGHTGCN_TINY, epochs=20)).fit(two_communities())
        losses = [stats.bpr_loss for stats in model.history]
        self.assertLess(losses[-1], 0.5 * losses[0])
        seen = {}
        for row in two_communities():
            seen.setdefault(row.user_id, []).append(row.restaurant_id)
        ranked = model.recommend(list(seen), seen, 2)
        for user, items in ranked.items():
            own = set(range(1, 7)) if user <= 10 else set(range(7, 13))
            self.assertTrue(set(items) <= own, (user, items))
            self.assertFalse(set(items) & set(seen[user]))

    def test_is_deterministic_and_order_invariant(self) -> None:
        rows = two_communities()
        first = LightGCN(replace(LIGHTGCN_TINY, epochs=5)).fit(rows)
        second = LightGCN(replace(LIGHTGCN_TINY, epochs=5)).fit(list(reversed(rows)))
        np.testing.assert_array_equal(first.final_embeddings, second.final_embeddings)
        other_seed = LightGCN(replace(LIGHTGCN_TINY, epochs=5, seed=7)).fit(rows)
        self.assertFalse(np.array_equal(first.final_embeddings, other_seed.final_embeddings))

    def test_callback_stops_training(self) -> None:
        epochs = []
        model = LightGCN(LIGHTGCN_TINY).fit(
            two_communities(), callback=lambda stats, _: epochs.append(stats.epoch) or stats.epoch == 3
        )
        self.assertEqual(epochs, [1, 2, 3])
        self.assertEqual(len(model.history), 3)

    def test_top_k_handles_repeated_users_with_different_exclusions(self) -> None:
        model = LightGCN(replace(LIGHTGCN_TINY, epochs=5)).fit(two_communities())
        full = model.top_k([1], [[]], 12)[0]
        ranked = model.top_k([1, 1, 999], [[], [full[0]], []], 3)
        self.assertEqual(ranked[0], full[:3])
        self.assertEqual(ranked[1], full[1:4])
        self.assertEqual(ranked[2], ())
        self.assertEqual(model.recommend([1], {1: [full[0]]}, 3), {1: full[1:4]})

    def test_unknown_users_get_no_candidates(self) -> None:
        model = LightGCN(replace(LIGHTGCN_TINY, epochs=2)).fit(two_communities())
        ranked = model.recommend([1, 999], {}, 5)
        self.assertEqual(ranked[999], ())
        self.assertEqual(len(ranked[1]), 5)
        self.assertEqual(model.score(999, [1, 2]), {})

    def test_negative_sampling_never_returns_a_visit(self) -> None:
        rng = np.random.default_rng(0)
        users = np.repeat(np.arange(5), 3)
        items = np.tile(np.arange(3), 5)
        positives = np.unique(users * 4 + items)
        negatives = _sample_negatives(rng, users, 4, positives)
        self.assertTrue((negatives == 3).all())
        self.assertFalse(_is_positive(users * 4 + negatives, positives).any())

    def test_rejects_degenerate_graphs(self) -> None:
        with self.assertRaises(ValueError):
            LightGCN(LIGHTGCN_TINY).fit(two_communities()[:1])
        with self.assertRaises(ValueError):
            LightGCNConfig(batch_size=0)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_blocks_and_edges_before_block_start(self) -> None:
        from rating_recsys.experiments.pipeline import CheckpointedLightGCN, checkpoint_start

        self.assertEqual(checkpoint_start(date(2025, 12, 19), 3), date(2025, 10, 1))
        self.assertEqual(checkpoint_start(date(2025, 6, 30), 6), date(2025, 1, 1))
        self.assertEqual(checkpoint_start(date(2025, 2, 1), 1), date(2025, 2, 1))

        rows = sorted(synthetic_interactions(), key=lambda item: item.event_date)
        dates = [item.event_date for item in rows]
        graphs = CheckpointedLightGCN(replace(LIGHTGCN_TINY, epochs=1), months=1)
        self.assertIsNone(graphs.model_for(date(2025, 1, 20), rows, dates))
        model = graphs.model_for(date(2025, 3, 5), rows, dates)
        before = [item for item in rows if item.event_date < date(2025, 3, 1)]
        self.assertEqual(model.edge_count, len(before))
        self.assertIs(graphs.model_for(date(2025, 3, 30), rows, dates), model)
        # Asking for March with different history before March is an error.
        with self.assertRaisesRegex(RuntimeError, "fitted on"):
            graphs.model_for(date(2025, 3, 6), rows[1:], dates[1:])


DEEPCONN_TINY = DeepCoNNConfig(
    embedding_dim=8, num_filters=8, latent_dim=4, fm_k=2, doc_length=60,
    max_review_length=20, batch_size=32, epochs=4, threads=1, dropout=0.0,
    learning_rate=0.01,
) if HAS_TORCH else None


@unittest.skipUnless(HAS_TORCH, "deepconn extra is not installed")
class DocumentTests(unittest.TestCase):
    def test_vocabulary_is_deterministic_and_reserves_special_ids(self) -> None:
        vocabulary = CharVocabulary.build(["abca", "ab"], size=10, min_count=2)
        self.assertEqual(vocabulary.characters, ("a", "b"))
        self.assertEqual(vocabulary.encode("abz", 10).tolist(), [3, 4, 1])
        self.assertEqual(len(vocabulary), 5)
        self.assertEqual(vocabulary.encode("aaaa", 2).tolist(), [3, 3])

    def test_documents_are_newest_first_and_drop_a_review_on_request(self) -> None:
        tokens = lambda *ids: np.asarray(ids, dtype=np.int64)
        documents = EntityDocuments(
            {7: [((1,), 10, tokens(5, 5)), ((3,), 30, tokens(7)), ((2,), 20, tokens(6, 6, 6))]},
            doc_length=6,
            max_review_length=3,
        )
        self.assertEqual(documents.tokens(7).tolist(), [7, SEP, 6, 6, 6, SEP])
        self.assertEqual(documents.tokens(7, exclude_review=20).tolist(), [7, SEP, 5, 5, SEP])
        self.assertEqual(documents.tokens(7, exclude_review=99).tolist(), [7, SEP, 6, 6, 6, SEP])
        batch = documents.batch([7, 8], [30, 30])
        self.assertEqual(batch.tolist(), [[6, 6, 6, SEP, 5, 5], [PAD] * 6])

    def test_build_documents_only_reads_given_interactions(self) -> None:
        rows = synthetic_interactions()[:20]
        texts = texts_for(rows)
        vocabulary = CharVocabulary.build(texts.values(), size=100, min_count=1)
        users, items = build_documents(rows, texts, vocabulary, DEEPCONN_TINY)
        extra = {**texts, 10**9: "미래 리뷰"}
        users2, _ = build_documents(rows, extra, vocabulary, DEEPCONN_TINY)
        for user in {r.user_id for r in rows}:
            self.assertEqual(users.tokens(user).tolist(), users2.tokens(user).tolist())
        self.assertIn(rows[0].restaurant_id, items)


@unittest.skipUnless(HAS_TORCH, "deepconn extra is not installed")
class NetworkTests(unittest.TestCase):
    def test_retrieval_parts_reproduce_the_fm_score(self) -> None:
        torch.manual_seed(0)
        net = DeepCoNNNet(20, DEEPCONN_TINY)
        with torch.no_grad():
            net.fm_linear.bias.fill_(0.3)
        users, items = torch.randn(5, 4), torch.randn(5, 4)
        user_bias, user_vec, item_bias, item_vec = net.retrieval_parts(users, items)
        torch.testing.assert_close(
            net.score(users, items), user_bias + item_bias + (user_vec * item_vec).sum(1)
        )

    def test_padding_length_does_not_change_the_encoding(self) -> None:
        torch.manual_seed(0)
        net = DeepCoNNNet(20, DEEPCONN_TINY).eval()
        with torch.no_grad():
            net.user_tower.conv.bias.fill_(0.5)  # PAD positions would pool relu(bias)
            short = net.encode_users(torch.tensor([[4, 5, 6, 0, 0]]))
            long = net.encode_users(torch.tensor([[4, 5, 6, 0, 0, 0, 0, 0, 0]]))
        torch.testing.assert_close(short, long)


@unittest.skipUnless(HAS_TORCH, "deepconn extra is not installed")
class DeepCoNNModelTests(unittest.TestCase):
    def test_bpr_learns_cuisine_from_text_and_loss_decreases(self) -> None:
        rows = synthetic_interactions()
        config = replace(DEEPCONN_TINY, epochs=40, num_filters=16, latent_dim=8, fm_k=4)
        model = DeepCoNN(config).fit(rows, texts_for(rows))
        losses = [stats.loss for stats in model.history]
        self.assertLess(losses[-1], losses[0])
        seen = {}
        for row in rows:
            seen.setdefault(row.user_id, []).append(row.restaurant_id)
        ranked = model.recommend(list(seen), seen, 3)
        hits = sum(
            cuisine(item) == user % 3 for user, items in ranked.items() for item in items
        )
        self.assertGreater(hits / sum(len(items) for items in ranked.values()), 0.8)
        for user, items in ranked.items():
            self.assertFalse(set(items) & set(seen[user]))
        self.assertEqual(model.recommend([10**6], {}, 3), {10**6: ()})

    def test_mse_objective_predicts_ratings(self) -> None:
        rows = synthetic_interactions()
        model = DeepCoNN(replace(DEEPCONN_TINY, objective="mse", epochs=15)).fit(rows, texts_for(rows))
        predicted = model.score_pairs([r.user_id for r in rows], [r.restaurant_id for r in rows])
        ratings = np.array([r.rating for r in rows])
        rmse = float(np.sqrt(np.mean((predicted - ratings) ** 2)))
        self.assertLess(rmse, float(np.std(ratings)) * 1.05)
        self.assertTrue(np.isnan(model.score_pairs([10**6], [1])[0]))

    def test_is_deterministic_order_invariant_and_ignores_unused_text(self) -> None:
        rows = synthetic_interactions()
        texts = texts_for(rows)
        first = DeepCoNN(replace(DEEPCONN_TINY, epochs=2)).fit(rows, texts)
        second = DeepCoNN(replace(DEEPCONN_TINY, epochs=2)).fit(
            list(reversed(rows)), {**texts, 10**9: "쓰지 않는 리뷰"}
        )
        self.assertEqual([s.loss for s in first.history], [s.loss for s in second.history])
        for a, b in zip(first._encoded(), second._encoded()):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def test_training_removes_the_visit_review_from_both_documents(self) -> None:
        rows = synthetic_interactions()
        texts = texts_for(rows)
        model = DeepCoNN(replace(DEEPCONN_TINY, epochs=1)).fit(rows, texts)
        # The newest review of both its user and its restaurant is in both documents.
        newest_user, newest_item = {}, {}
        for item in rows:
            key = (item.event_date, item.review_id)
            newest_user[item.user_id] = max(newest_user.get(item.user_id, key), key)
            newest_item[item.restaurant_id] = max(newest_item.get(item.restaurant_id, key), key)
        row = next(
            item for item in rows
            if newest_user[item.user_id] == newest_item[item.restaurant_id]
            == (item.event_date, item.review_id)
        )
        target = model.vocabulary.encode(texts[row.review_id], DEEPCONN_TINY.max_review_length).tolist()
        for documents, key in ((model._user_docs, row.user_id), (model._item_docs, row.restaurant_id)):
            full = documents._tokens[key]
            start, end = documents._spans[key][row.review_id]
            self.assertEqual(full[start:end].tolist(), target + [SEP])
            expected = np.concatenate([full[:start], full[end:]])[: DEEPCONN_TINY.doc_length]
            self.assertEqual(
                documents.tokens(key, exclude_review=row.review_id).tolist(), expected.tolist()
            )

    def test_linear_output_keeps_negative_latents(self) -> None:
        torch.manual_seed(0)
        documents = torch.tensor([[4, 5, 6, 7, 0]])
        relu = DeepCoNNNet(20, DEEPCONN_TINY).eval()
        linear = DeepCoNNNet(20, replace(DEEPCONN_TINY, latent_activation="linear")).eval()
        linear.load_state_dict(relu.state_dict())
        with torch.no_grad():
            relu.user_tower.fc.bias.fill_(-100.0)
            linear.user_tower.fc.bias.fill_(-100.0)
            self.assertTrue(bool((relu.encode_users(documents) == 0).all()))
            self.assertTrue(bool((linear.encode_users(documents) < 0).all()))
        self.assertIn("_linear_", replace(DEEPCONN_TINY, latent_activation="linear").name)

    def test_config_validation(self) -> None:
        with self.assertRaises(ValueError):
            DeepCoNNConfig(latent_activation="tanh")
        with self.assertRaises(ValueError):
            DeepCoNNConfig(objective="rmse")
        with self.assertRaises(ValueError):
            DeepCoNNConfig(kernel_size=4)


@pytest.mark.parametrize("model", CANDIDATE_MODELS.values(), ids=lambda model: model.name)
def test_candidate_model_contract(model):
    """Candidates support fitting callbacks, exclusions and repeatable ranks."""

    if model not in available_models():
        pytest.skip(f"{model.name} optional dependencies are not installed")
    rows = synthetic_interactions()
    # A one-step index has no training epochs to stop at epoch 2.
    single_step = model.default_max_epochs == 1
    config = replace(tiny_grid(model)[0], epochs=1 if single_step else 4)
    stop_epoch = 1 if single_step else 2
    model.prepare_fit([SimpleNamespace(user_id=uid) for uid in sorted({row.user_id for row in rows})])
    texts = texts_for(rows) if model.needs_review_texts else None
    epochs = []

    def stop(stats, fitted):
        epochs.append(stats.epoch)
        return stats.epoch == stop_epoch

    first = model.fit(config, rows, texts, callback=stop)
    assert epochs == list(range(1, stop_epoch + 1))
    assert len(first.history) == stop_epoch
    second = model.fit(config, list(reversed(rows)), texts, callback=lambda s, _: s.epoch == stop_epoch)
    seen = {}
    for row in rows:
        seen.setdefault(row.user_id, []).append(row.restaurant_id)
    ranked = first.recommend(list(seen), seen, 5)
    assert ranked == second.recommend(list(seen), seen, 5)
    assert set(ranked) == set(seen)
    assert any(ranked.values())
    catalog = {row.restaurant_id for row in rows}
    for user, items in ranked.items():
        assert len(items) <= 5
        assert len(set(items)) == len(items)
        assert set(items) <= catalog
        assert not set(items) & set(seen[user])


if __name__ == "__main__":
    unittest.main()
