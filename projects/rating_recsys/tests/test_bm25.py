"""Lexical retrieval correctness, persistence and historical input boundaries."""

import json
from dataclasses import replace

import pytest

pytest.importorskip("kiwipiepy")
bm25s = pytest.importorskip("bm25s")

from rating_recsys.retrieval.bm25 import BM25Config, BM25Retriever, NEGATION_TOKENS
from rating_recsys.retrieval.review_profiles import build_profiles
from rating_recsys.retrieval.review_preprocessing import clean_review_profile
from support import _interaction


def test_korean_morphology_seen_filter_positive_only_and_zero_match(tmp_path):
    rows = [_interaction(i, i, i * 10, i, 2 if i == 5 else 5) for i in range(1, 8)]
    texts = {1: "국밥이", 2: "국밥을", 3: "파스타", 4: "국밥을",
             5: "국밥", 6: None, 7: "12345 !!!"}
    fitted = BM25Retriever(BM25Config(threads=1)).fit(rows, texts, [1, 3, 6, 7, 999])
    ranked = fitted.recommend([1, 3, 6, 7, 999], {1: [10], 3: [30]}, 100)
    # Korean case particles differ, but the shared noun matches.
    assert fitted.user_tokens[1] == ("국밥",)
    assert ranked[1] == (20, 40)
    assert ranked[3] == ranked[6] == ranked[7] == ranked[999] == ()
    assert 50 not in fitted.item_ids  # low-rated restaurant never indexed
    assert 60 not in fitted.item_ids and 70 not in fitted.item_ids
    assert fitted.metadata["users_without_query_tokens"] == 3

    fitted.save(tmp_path / "index")
    saved = json.loads((tmp_path / "index" / "profiles.json").read_text())
    loaded = bm25s.BM25.load(str(tmp_path / "index"))
    scores = loaded.get_scores(["국밥"])
    assert saved["restaurant_ids"] == list(fitted.item_ids)
    assert scores.tolist() == fitted.index.get_scores(["국밥"]).tolist()


def test_future_text_does_not_change_index_or_queries_and_order_is_deterministic():
    history = [_interaction(1, 1, 10, 1), _interaction(2, 2, 20, 2)]
    texts = {1: "국밥", 2: "국밥", 3: "미래 초밥"}
    config = BM25Config(threads=1)
    first = BM25Retriever(config).fit(history, texts, [1])
    altered = BM25Retriever(config).fit(list(reversed(history)), {**texts, 3: "국밥"}, [1])
    assert first.metadata == altered.metadata
    assert first.user_tokens == altered.user_tokens
    assert first.recommend([1], {1: [10]}, 10) == altered.recommend([1], {1: [10]}, 10)
    assert "초밥" not in first.index.vocab_dict


def test_empty_corpus_and_profile_selection_limits():
    config = BM25Config(threads=1, max_user_reviews=1, max_item_reviews=1, max_review_chars=2)
    rows = [_interaction(1, 1, 10, 1), _interaction(2, 1, 20, 2),
            _interaction(3, 1, 30, 3, 2)]
    texts = {1: "국밥 오래된", 2: "초밥 최신", 3: "피자 낮은 평점"}
    users, items = build_profiles(rows, texts, [1], config)
    assert users == {1: "초밥"}
    assert items == {10: "국밥", 20: "초밥"}
    fitted = BM25Retriever(config).fit(rows, {1: None, 2: "!!!", 3: "국밥"}, [1])
    assert fitted.index is None
    assert fitted.recommend([1], {}, 10) == {1: ()}


def test_cleanup_preserves_food_service_price_and_negation():
    cleaned = clean_review_profile("<b>파스타</b> &amp; 국밥 ㅋㅋㅋ 😊 https://example.com 가격은 비싸고 친절하지 않다. 못 먹음 ＡＢＣ")
    assert all(term in cleaned for term in ("파스타", "국밥", "가격", "비싸", "친절", "않다", "못", "ABC"))
    assert all(noise not in cleaned for noise in ("<b>", "&amp;", "ㅋㅋ", "😊", "example.com"))


def test_optional_preprocessing_fixed_profiles_stopwords_and_future_boundary():
    rows = [_interaction(1, 1, 10, 1), _interaction(2, 2, 20, 2)]
    texts = {1: "국밥 맛있어요 https://example.com ㅋㅋ", 2: "국밥 맛있어요", 3: "미래 피자"}
    baseline = BM25Retriever(BM25Config(threads=1)).fit(rows, texts, [1])
    cleaned = BM25Retriever(BM25Config(threads=1, preprocessing="clean_stopwords")).fit(rows, texts, [1])
    future_changed = BM25Retriever(cleaned.config).fit(list(reversed(rows)), {**texts, 3:"국밥"}, [1])
    assert cleaned.user_tokens[1] == ("국밥",)
    assert cleaned.recommend([1], {1: [10]}, 100) == {1: (20,)}
    assert cleaned.metadata == future_changed.metadata
    assert cleaned.metadata["source_profile_sha256"] == baseline.metadata["profile_sha256"]
    assert cleaned.metadata["preprocessing_version"] != baseline.metadata["preprocessing_version"]
    assert "맛있" not in cleaned.index.vocab_dict and "국밥" in cleaned.index.vocab_dict
    assert cleaned.metadata["stopword_tokens_removed_unique_profiles"]["맛있"] > 0


def test_stopword_only_query_is_empty_but_user_is_retained():
    rows = [_interaction(1, 1, 10, 1), _interaction(2, 2, 20, 2)]
    fitted = BM25Retriever(BM25Config(threads=1, preprocessing="clean_stopwords")).fit(
        rows, {1:"맛있어요", 2:"국밥"}, [1])
    assert fitted.metadata["input_users"] == 1
    assert fitted.metadata["users_without_query_tokens"] == 1
    assert fitted.recommend([1], {}, 100) == {1: ()}


@pytest.mark.parametrize("preprocessing", ["baseline", "clean", "clean_stopwords"])
def test_real_kiwi_preserves_negation_in_queries_and_index(preprocessing, tmp_path):
    # Opposite statements must not collapse to the same lexical representation.
    pairs = [
        ("맵지 않아 좋다", "매워 좋다", "않"),
        ("친절하지 않다", "친절하다", "않"),
        ("안 맵다", "맵다", "안"),
        ("못 먹다", "먹다", "못"),
        ("먹지 못하다", "먹다", "못하"),
        ("먹지 말다", "먹다", "말"),
        ("맛이 없다", "맛이 있다", "없"),
        ("고기 없이 먹다", "고기와 먹다", "없이"),
        ("친절한 곳이 아니다", "친절한 곳이다", "아니"),
    ]
    texts = {2 * i + j + 1: text for i, pair in enumerate(pairs)
             for j, text in enumerate(pair[:2])}
    rows = [_interaction(rid, rid, rid * 10, 1) for rid in texts]
    fitted = BM25Retriever(BM25Config(threads=1, preprocessing=preprocessing)).fit(
        rows, texts, list(texts))
    for i, (_, _, negator) in enumerate(pairs):
        negative, positive = 2 * i + 1, 2 * i + 2
        assert negator in fitted.user_tokens[negative]
        assert negator not in fitted.user_tokens[positive]
        assert fitted.user_tokens[negative] != fitted.user_tokens[positive]
        # Verify that the item document retained the cue as well as the query.
        scores = fitted.index.get_scores([negator])
        assert scores[fitted.item_ids.index(negative * 10)] > 0
        assert scores[fitted.item_ids.index(positive * 10)] == 0

    fitted.save(tmp_path / "negation-index")
    saved = json.loads((tmp_path / "negation-index" / "profiles.json").read_text())
    assert saved["config"]["profile_version"].endswith("-v2")
    assert saved["metadata"]["preprocessing_version"] == fitted.config.profile_version
    assert saved["metadata"]["negation_tokens"] == [list(pair) for pair in sorted(NEGATION_TOKENS)]


@pytest.mark.parametrize("changes", [
    {"epochs": 2}, {"threads": 0}, {"max_user_reviews": 0},
    {"k1": float("nan")}, {"k1": 0}, {"b": -1}, {"min_rating": 6},
    {"preprocessing": "unknown"},
])
def test_invalid_bm25_settings(changes):
    with pytest.raises(ValueError):
        replace(BM25Config(), **changes)
