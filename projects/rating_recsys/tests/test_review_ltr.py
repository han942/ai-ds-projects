"""Review LTR alignment/leakage checks; synthetic vectors never call an API."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rating_recsys.experiments.queries import build_window_queries
from rating_recsys.experiments.review_ltr import profile_preflight, run_comparison
from rating_recsys.ranking.review_features import (
    REVIEW_FEATURE_NAMES, ReviewFeatureBuilder, review_feature_matrix,
)
from rating_recsys.retrieval.review_embeddings import ReviewEmbeddingConfig
from support import SMALL, _interaction, synthetic_interactions


class LocalFormatter:
    def format(self, text, role):
        return f"{role}: {text}"

    def metadata(self):
        return {"kind": "synthetic-test-only"}

    def prepare(self, users, items):
        return ({uid: self.format(doc, "query") for uid, doc in users.items()},
                {iid: self.format(doc, "document") for iid, doc in items.items()})


class LocalCache:
    def __init__(self):
        self.values = {}
        self.usage = {"api_requests": 0}

    def missing_count(self, docs):
        return len(set(docs) - self.values.keys())

    def populate(self, docs):
        for doc in docs:
            self.values[doc] = [1 + b / 255 for b in hashlib.sha256(doc.encode()).digest()[:4]]

    def embed(self, docs):
        assert not self.missing_count(docs), "tests must preload all embeddings"
        return {doc: self.values[doc] for doc in docs}

    def totals(self):
        return {"api_requests": 0, "kind": "synthetic-test-only"}


def case():
    history = [_interaction(1, 1, 100, 1), _interaction(2, 2, 200, 1)]
    window = [_interaction(3, 1, 200, 3)]
    queries, _ = build_window_queries(history, window, config=SMALL, phase="test", cutoff=history[0].event_date)
    candidates = SimpleNamespace(row_restaurant_ids=np.array([300, 200, 400]), group_sizes=[3],
                                 features=np.zeros((3, len(SMALL.feature_names)), dtype=np.float32))
    return history, queries, candidates


def test_cosine_aligns_to_row_ids_and_distinguishes_missing_zero_and_negative():
    _, queries, candidates = case()
    values = review_feature_matrix(queries, candidates, {1: [2, 0]},
                                  {200: [-3, 0], 300: [0, 4]}, {1: 2}, {200: 5, 300: 1})
    assert values[:, 0].tolist() == [0, -1, 0]
    assert values[:, 3].tolist() == [1, 1, 0]
    assert values[:, 5].tolist() == [1, 5, 0]
    assert values[:, 4].tolist() == [2, 2, 2]


def test_vector_dimensions_and_groups_must_match():
    _, queries, candidates = case()
    with pytest.raises(ValueError, match="dimensions"):
        review_feature_matrix(queries, candidates, {1: [1, 0]}, {200: [1, 0, 0]}, {}, {})
    candidates.group_sizes = [2]
    with pytest.raises(ValueError, match="groups"):
        review_feature_matrix(queries, candidates, {}, {}, {}, {})


def test_future_reviews_rejected_before_cache_access():
    history, queries, candidates = case()
    future = _interaction(99, 1, 100, 10)
    builder = ReviewFeatureBuilder({r.review_id: "맛있다" for r in history + [future]},
                                   ReviewEmbeddingConfig(), LocalFormatter(), LocalCache())
    with pytest.raises(ValueError, match="cutoff"):
        builder(history + [future], queries, candidates)


def test_missing_vectors_fail_without_network_or_silent_coverage_loss():
    history, queries, candidates = case()
    cache = LocalCache()
    builder = ReviewFeatureBuilder({r.review_id: "맛있다" for r in history},
                                   ReviewEmbeddingConfig(), LocalFormatter(), cache)
    with pytest.raises(RuntimeError, match="Missing"):
        builder(history, queries, candidates)
    assert cache.usage["api_requests"] == 0


def test_interrupted_extraction_records_partial_progress(tmp_path, monkeypatch):
    from rating_recsys.experiments import review_ltr

    class InterruptedCache(LocalCache):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def prefill(self, docs):
            self.populate(docs[:1])
            self.usage["api_requests"] = 1
            raise KeyboardInterrupt

    cache = InterruptedCache()
    monkeypatch.setattr(review_ltr, "load_snapshot", lambda _: [])
    monkeypatch.setattr(review_ltr, "load_review_texts", lambda *args: ({}, {}))
    monkeypatch.setattr(review_ltr, "ProfileFormatter", lambda *args: LocalFormatter())
    monkeypatch.setattr(review_ltr, "create_embedding_cache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(review_ltr, "profile_preflight", lambda *args, **kwargs: (
        ["query: a", "document: b"], {"unique_inputs": 2, "unique_cache_misses": 2},
    ))

    result = review_ltr.main([
        "--snapshot", str(tmp_path / "snapshot.jsonl"),
        "--artifacts-dir", str(tmp_path), "--embed-only",
        "--embedding-model", "liquid/lfm-2.5-embedding-350m:free",
    ])
    diagnostic = json.loads((tmp_path / "diagnostics/review_ltr_preflight.json").read_text())
    assert result == 130
    assert diagnostic["embedding_status"] == "paused"
    assert diagnostic["api_requests_sent"] == 1
    assert diagnostic["remaining_cache_misses"] == 1
    assert cache.missing_count(["query: a"]) == 0


def test_full_synthetic_comparison_shares_candidates_and_never_calls_api(tmp_path):
    rows = synthetic_interactions()
    texts = {r.review_id: f"음식 {r.restaurant_id % 3} 맛있다" for r in rows}
    config = replace(SMALL, ranker_training_mode="window", satisfaction_mode="history-aware")
    embedding = ReviewEmbeddingConfig(dimensions=4)
    cache, formatter = LocalCache(), LocalFormatter()
    docs, preflight = profile_preflight(rows, texts, config, embedding, formatter, cache)
    assert len(preflight["windows"]) > 2
    cache.populate(docs)
    folder, metrics = run_comparison(rows, texts, {"kind": "synthetic"}, config=config,
                                     embedding=embedding, formatter=formatter, cache=cache,
                                     artifacts_root=tmp_path / "artifacts", project_root=tmp_path)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["embedding_usage"]["api_requests"] == 0
    assert manifest["feature_names"]["review_ltr"] == list(config.feature_names + REVIEW_FEATURE_NAMES)
    assert manifest["test_evaluations_per_arm"] == 1
    assert set(metrics["test"]) == {"baseline", "review_ltr"}
    assert len(metrics["feature_importance"]["review_ltr"]) == len(config.feature_names + REVIEW_FEATURE_NAMES)
    for audit in manifest["profile_audit"]:
        assert audit["latest_history_date"] <= audit["cutoff"]
    for row in map(json.loads, (folder / "recommendations_test.jsonl").read_text().splitlines()):
        assert set(row["baseline"]) == set(row["review_ltr"]) == set(row["c5_candidates"])
    assert hashlib.sha256(json.dumps({
        row["query_id"]: row["c5_candidates"] for row in
        map(json.loads, (folder / "recommendations_test.jsonl").read_text().splitlines())
    }, sort_keys=True).encode()).hexdigest() == manifest["shared_test_candidates_sha256"]
