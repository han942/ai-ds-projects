"""Cutoff-safe review profiles, candidate scoring and local comparison integration."""

from __future__ import annotations

import asyncio

import pytest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from rating_recsys.experiments.candidate_models import ReviewEmbeddingsCandidate
from rating_recsys.experiments.comparison import run_candidate_comparison
from rating_recsys.retrieval.review_embeddings import (
    OpenRouterEmbeddingCache, ProfileFormatter, ReviewEmbeddingCandidates, ReviewEmbeddingConfig, build_profiles,
)
from support import SMALL, _interaction, synthetic_interactions, texts_for


def test_profiles_use_only_supplied_history_and_positive_reviews():
    rows = [
        _interaction(1, 1, 10, 1, 5),
        _interaction(2, 1, 11, 2, 2),
        _interaction(3, 2, 10, 3, 4),
        _interaction(4, 1, 12, 4, 5),
    ]
    texts = {1: "old favorite", 2: "bad visit", 3: "another diner", 4: "future favorite"}
    config = ReviewEmbeddingConfig(max_user_reviews=1, max_item_reviews=2)
    users, items = build_profiles(rows[:3], texts, [1], config)
    assert users == {1: "old favorite"}
    assert items == {10: "another diner\nold favorite"}
    assert "future favorite" not in str((users, items))
    assert "bad visit" not in str((users, items))


def test_cache_reuses_content_and_recommendation_excludes_history():
    with TemporaryDirectory() as directory:
        config = ReviewEmbeddingConfig(dimensions=2, batch_size=2)
        path = Path(directory) / "vectors.sqlite"
        calls = []

        async def fake_request(self, session, texts, limiter):
            calls.extend(texts)
            return [[1.0, 0.0] if "국밥" in text else [0.0, 1.0] for text in texts]

        with patch.object(OpenRouterEmbeddingCache, "_request", fake_request), \
             patch("rating_recsys.retrieval.review_embeddings.configured_api_key", return_value="local-test"):
            with OpenRouterEmbeddingCache(path, config) as cache:
                model = ReviewEmbeddingCandidates(
                    {1: "국밥 좋아요"}, {10: "국밥 맛집", 11: "초밥 맛집", 12: "국밥 맛집"}, cache,
                )
                assert model.recommend([1, 2], {1: [10]}, 2) == {1: (12, 11), 2: ()}
            with OpenRouterEmbeddingCache(path, config) as cache:
                cache.embed(["국밥 좋아요", "국밥 맛집"])
                assert cache.usage["api_requests"] == 0
                assert cache.usage["cache_hits"] == 2
        assert len(calls) == 3


@pytest.mark.parametrize('aggregation,user_profile', [
    ('concat','own_reviews'), ('review_mean','own_reviews'), ('review_mean','liked_items'),
])
def test_shared_comparison_runs_without_external_api(aggregation,user_profile):
    rows = synthetic_interactions()
    with TemporaryDirectory() as directory:
        root = Path(directory)
        config = replace(ReviewEmbeddingConfig(aggregation=aggregation,user_profile=user_profile), dimensions=2,
                         cache_path=str(root / "vectors.sqlite"))

        async def fake_request(self, session, texts, limiter):
            return [[1.0, 0.0] if "국밥" in text else [0.0, 1.0] for text in texts]

        with patch.object(OpenRouterEmbeddingCache, "_request", fake_request), \
             patch("rating_recsys.retrieval.review_embeddings.configured_api_key", return_value="local-test"), \
             patch("rating_recsys.retrieval.review_embeddings.ProfileFormatter",
                   side_effect=lambda cfg, path: ProfileFormatter(cfg, path, tokenizer=toy_tokenizer())):
            result = run_candidate_comparison(
                ReviewEmbeddingsCandidate(), rows, project_root=root,
                artifacts_root=root / "artifacts", texts=texts_for(rows),
                config=SMALL, grid=(config,), plot=False, log=lambda _: None,
            )
        assert (result.run_dir / "report.md").exists()
        report = (result.run_dir / "report.md").read_text()
        assert "리뷰 임베딩 입력과 API" in report
        assert "모델 학습 loss가 아니다" in report
        assert result.metrics["selection"]["chosen_epochs"] == 1
        assert result.metrics["refit"]["training_interactions"] > 0
        assert "review_embeddings" in result.metrics["test"]



def toy_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer


def test_token_budget_keeps_role_prefix_and_counts_untruncated_source():
    tokenizer = toy_tokenizer()
    tokenizer.enable_truncation(4)
    formatter = ProfileFormatter(ReviewEmbeddingConfig(max_document_tokens=8), Path("unused"),
                                 tokenizer=tokenizer)
    query = formatter.format(" ".join(f"word{i}" for i in range(30)), "query")
    document = formatter.format("short profile", "document")
    assert query.startswith("query: ")
    assert document.startswith("document: ")
    assert len(tokenizer.encode(query).ids) <= 8
    assert formatter.stats["max_original_tokens"] == 31
    assert formatter.stats["truncated_profiles"] == 1


def test_async_batches_are_bounded_and_out_of_order_responses_keep_mapping():
    with TemporaryDirectory() as directory:
        config = ReviewEmbeddingConfig(dimensions=2, batch_size=1, concurrency=2)
        active = 0
        peak = 0

        async def fake_request(self, session, texts, limiter):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.02 if texts[0] == "slow" else 0.001)
            active -= 1
            return [[float(len(texts[0])), 1.0]]

        with patch.object(OpenRouterEmbeddingCache, "_request", fake_request), \
             patch("rating_recsys.retrieval.review_embeddings.configured_api_key", return_value="local-test"):
            with OpenRouterEmbeddingCache(Path(directory) / "cache.sqlite", config) as cache:
                vectors = cache.embed(["slow", "a", "bb", "ccc"])
                assert vectors["slow"] == [4.0, 1.0]
                assert vectors["a"] == [1.0, 1.0]
                assert peak == 2
                assert cache.missing_count(["slow", "a", "bb", "ccc"]) == 0


def test_successful_batches_survive_a_later_failure():
    from rating_recsys.retrieval.review_embeddings import EmbeddingAPIError

    with TemporaryDirectory() as directory:
        config = ReviewEmbeddingConfig(dimensions=2, batch_size=1, concurrency=1)
        path = Path(directory) / "cache.sqlite"

        async def fake_request(self, session, texts, limiter):
            if texts == ["blocked"]:
                raise EmbeddingAPIError(429, daily_limit=True)
            return [[1.0, 0.0]]

        with patch.object(OpenRouterEmbeddingCache, "_request", fake_request), \
             patch("rating_recsys.retrieval.review_embeddings.configured_api_key", return_value="local-test"):
            with OpenRouterEmbeddingCache(path, config) as cache:
                try:
                    cache.embed(["completed", "blocked"])
                except EmbeddingAPIError as error:
                    assert error.daily_limit
                else:
                    raise AssertionError("expected a quota failure")
            with OpenRouterEmbeddingCache(path, config) as cache:
                assert cache.missing_count(["completed", "blocked"]) == 1
                assert cache.embed(["completed"])["completed"] == [1.0, 0.0]


def test_review_mean_keeps_reviews_separate_normalizes_and_reuses_cache():
    import numpy as np
    from rating_recsys.retrieval.review_embeddings import fit_review_profiles

    rows = [_interaction(1, 1, 10, 1, 5), _interaction(2, 1, 11, 2, 4),
            _interaction(3, 2, 12, 3, 5), _interaction(4, 1, 13, 4, 1)]
    texts = {1: '국밥 맛있다', 2: '초밥 맛있다', 3: '국밥 친절하다', 4: 'bad visit'}
    calls = []

    async def fake_request(self, session, documents, limiter):
        calls.extend(documents)
        return [[2.0, 0.0] if '국밥' in doc else [0.0, 4.0] for doc in documents]

    with TemporaryDirectory() as directory:
        config = ReviewEmbeddingConfig(dimensions=2, aggregation='review_mean',
                                       cache_path=str(Path(directory)/'cache.sqlite'))
        formatter = ProfileFormatter(config, Path(directory), tokenizer=toy_tokenizer())
        with patch.object(OpenRouterEmbeddingCache, '_request', fake_request), \
             patch('rating_recsys.retrieval.review_embeddings.configured_api_key', return_value='local-test'):
            with OpenRouterEmbeddingCache(Path(config.cache_path), config) as cache:
                fitted = fit_review_profiles(rows, texts, [1], config, formatter, cache)
            calls_before = len(calls)
            with OpenRouterEmbeddingCache(Path(config.cache_path), config) as cache:
                repeated = fit_review_profiles(rows, texts, [1], config, formatter, cache)
                assert cache.usage['api_requests'] == 0
        assert len(calls) == calls_before == 5  # two query reviews, three document reviews
        assert all('\n' not in doc and 'bad visit' not in doc for doc in calls)
        np.testing.assert_allclose(fitted.users[1], [2**-0.5, 2**-0.5])
        np.testing.assert_allclose(repeated.users[1], fitted.users[1])
        assert fitted.recommend([1, 999], {1:[10,11]}, 100) == {1:(12,), 999:()}
        assert fitted.profile_metadata['input_unit'] == 'review'


def test_liked_items_uses_same_selected_events_and_document_vectors():
    import numpy as np
    from rating_recsys.retrieval.review_embeddings import fit_review_profiles, build_review_inputs

    rows = [_interaction(1,1,10,1,5), _interaction(2,1,11,2,4),
            _interaction(3,2,10,3,5), _interaction(4,1,12,4,1),
            _interaction(5,1,13,5,5)]
    texts = {1:'국밥',2:'초밥',3:'초밥',4:'bad visit',5:'future review'}
    calls = []

    async def fake_request(self, session, documents, limiter):
        calls.extend(documents)
        return [[1.0,0.0] if '국밥' in doc else [0.0,1.0] for doc in documents]

    with TemporaryDirectory() as directory:
        config = ReviewEmbeddingConfig(dimensions=2, aggregation='review_mean',
                                       user_profile='liked_items', max_user_reviews=2,
                                       max_item_reviews=1,cache_path=str(Path(directory)/'cache.sqlite'))
        formatter = ProfileFormatter(config,Path(directory),tokenizer=toy_tokenizer())
        users,items,liked = build_review_inputs(rows[:4],texts,[1],config,formatter)
        assert users == {}
        assert liked == {1:(11,10)}
        assert items == {10:('document: 초밥',),11:('document: 초밥',)}
        with patch.object(OpenRouterEmbeddingCache,'_request',fake_request), \
             patch('rating_recsys.retrieval.review_embeddings.configured_api_key',return_value='local-test'):
            with OpenRouterEmbeddingCache(Path(config.cache_path),config) as cache:
                fitted = fit_review_profiles(rows[:4],texts,[1],config,formatter,cache)
        np.testing.assert_allclose(fitted.users[1],[0,1])
        assert all(doc.startswith('document: ') for doc in calls)
        assert fitted.profile_metadata['user_role'] == 'document'
        assert 12 not in fitted.items and 13 not in fitted.items


def test_invalid_profile_strategy_is_rejected_before_api():
    import pytest
    with pytest.raises(ValueError,match='requires review_mean'):
        ReviewEmbeddingConfig(user_profile='liked_items')
