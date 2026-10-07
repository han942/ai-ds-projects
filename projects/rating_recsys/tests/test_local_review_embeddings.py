"""Local E5 persistence/routing tests, without weights or network access."""

import json
from dataclasses import replace

import numpy as np
import pytest

from rating_recsys.retrieval.review_embeddings import (
    E5_MODEL, E5_REVISION, LIQUID_MODEL, LocalEmbeddingCache,
    OpenRouterEmbeddingCache, ProfileFormatter, create_embedding_cache, model_embedding_config,
)
from test_review_embeddings import toy_tokenizer


class Encoder:
    def __init__(self, *, fail=None):
        self.calls = []
        self.fail = fail

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        if self.fail and self.fail in texts:
            raise RuntimeError("synthetic inference failure")
        values = np.zeros((len(texts), 384), dtype=np.float32)
        values[:, 0] = [len(text) for text in texts]
        values[:, 1] = 1
        return values


def config(tmp_path):
    return replace(model_embedding_config(E5_MODEL, str(tmp_path / "e5.sqlite")), batch_size=1)


@pytest.fixture(autouse=True)
def no_api(monkeypatch):
    def forbidden():
        raise AssertionError("Local E5 must never access an API key")
    monkeypatch.setattr("rating_recsys.retrieval.review_embeddings.configured_api_key", forbidden)


def test_e5_defaults_role_budget_and_api_cache_guard(tmp_path):
    cfg = config(tmp_path)
    assert cfg.backend == "local" and cfg.dimensions == 384
    assert cfg.model_revision == cfg.tokenizer_revision == E5_REVISION
    assert cfg.profile_version == "positive-recent-local-e5-prefix-v1"
    formatter = ProfileFormatter(cfg, tmp_path, tokenizer=toy_tokenizer())
    assert formatter.format("profile", "query") == "query: profile"
    assert formatter.format("profile", "document") == "passage: profile"
    with pytest.raises(ValueError, match="local"):
        OpenRouterEmbeddingCache(tmp_path / "wrong.sqlite", cfg)
    with pytest.raises(ValueError, match="384"):
        replace(cfg, dimensions=1024)
    with pytest.raises(ValueError, match="512"):
        replace(cfg, max_document_tokens=513)


def test_prefill_deduplicates_normalizes_and_warm_cache_never_loads_model(tmp_path, monkeypatch):
    encoder = Encoder()
    monkeypatch.setattr(LocalEmbeddingCache, "_load_encoder", lambda _: encoder)
    cfg = config(tmp_path)
    with create_embedding_cache(tmp_path / "e5.sqlite", cfg) as cache:
        assert cache.prefill(["query: first", "passage: second", "query: first"]) is None
        assert cache.usage["local_documents"] == 2
        assert cache.usage["api_requests"] == 0
        assert cache.missing_count(["query: first", "passage: second"]) == 0
    assert len(encoder.calls) == 2
    assert encoder.calls[0][1]["prompt"] == ""
    assert encoder.calls[0][0] == ["query: first"]
    def forbidden(_):
        raise AssertionError("Warm cache must not load E5")
    monkeypatch.setattr(LocalEmbeddingCache, "_load_encoder", forbidden)
    with create_embedding_cache(tmp_path / "e5.sqlite", cfg) as cache:
        cache.prefill(["query: first"])
        vector = cache.embed(["passage: second"])["passage: second"]
        assert len(vector) == 384
        np.testing.assert_allclose(np.linalg.norm(vector), 1)
        assert cache.totals()["local_documents"] == 2
        assert cache.usage["local_documents"] == 0
        assert cache.usage["cost_usd"] == 0


def test_keys_isolate_revision_and_model_but_not_cpu_threads(tmp_path):
    cfg = config(tmp_path)
    with LocalEmbeddingCache(tmp_path / "e5.sqlite", cfg) as cache, \
         LocalEmbeddingCache(tmp_path / "e5.sqlite", replace(cfg, model_revision="another-revision")) as other, \
         LocalEmbeddingCache(tmp_path / "e5.sqlite", replace(cfg, cpu_threads=4)) as tuned, \
         OpenRouterEmbeddingCache(tmp_path / "liquid.sqlite", model_embedding_config(LIQUID_MODEL, "unused")) as liquid:
        assert cache._key("query: first") != other._key("query: first")
        assert cache._key("query: first") != liquid._key("query: first")
        assert cache._key("query: first") == tuned._key("query: first")


def test_length_grouping_preserves_text_vector_alignment(tmp_path, monkeypatch):
    encoder = Encoder()
    monkeypatch.setattr(LocalEmbeddingCache, "_load_encoder", lambda _: encoder)
    cfg = replace(config(tmp_path), batch_size=2)
    texts = ["query: x", "passage: much longer profile", "query: medium"]
    with LocalEmbeddingCache(tmp_path / "e5.sqlite", cfg) as cache:
        vectors = cache.embed(texts + texts[:1])
        assert encoder.calls[0][0] == sorted(texts, key=len, reverse=True)[:2]
        assert cache.totals()["local_documents"] == 3
        for text in texts:
            expected = np.zeros(384)
            expected[:2] = [len(text), 1]
            np.testing.assert_allclose(vectors[text], expected / np.linalg.norm(expected))
        assert cache.embed(texts) == vectors
        assert len(encoder.calls) == 2


def test_batches_survive_failure_and_resume_only_missing(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(LocalEmbeddingCache, "_load_encoder", lambda _: Encoder(fail="passage: later"))
    with LocalEmbeddingCache(tmp_path / "e5.sqlite", cfg) as cache:
        with pytest.raises(RuntimeError, match="synthetic"):
            cache.prefill(["query: saved", "passage: later"])
        assert cache.missing_count(["query: saved", "passage: later"]) == 1
        assert cache.totals()["local_documents"] == 1
    encoder = Encoder()
    monkeypatch.setattr(LocalEmbeddingCache, "_load_encoder", lambda _: encoder)
    with LocalEmbeddingCache(tmp_path / "e5.sqlite", cfg) as cache:
        cache.prefill(["query: saved", "passage: later"])
        assert cache.missing_count(["query: saved", "passage: later"]) == 0
        assert cache.totals()["local_documents"] == 2
    assert encoder.calls[0][0] == ["passage: later"]


@pytest.mark.parametrize("bad", ["dimensions", "nan", "zero"])
def test_invalid_vectors_do_not_commit(tmp_path, monkeypatch, bad):
    class BadEncoder:
        def encode(self, texts, **kwargs):
            return np.full((len(texts), 2 if bad == "dimensions" else 384),
                           np.nan if bad == "nan" else 0, dtype=np.float32)
    monkeypatch.setattr(LocalEmbeddingCache, "_load_encoder", lambda _: BadEncoder())
    with LocalEmbeddingCache(tmp_path / "e5.sqlite", config(tmp_path)) as cache:
        with pytest.raises(RuntimeError):
            cache.prefill(["query: invalid"])
        assert cache.missing_count(["query: invalid"]) == 1
        assert cache.totals()["local_documents"] == 0


def test_ltr_defaults_to_local_e5_and_records_completion(tmp_path, monkeypatch):
    from rating_recsys.experiments import review_ltr
    from test_review_ltr import LocalFormatter

    monkeypatch.setattr(review_ltr, "load_snapshot", lambda _: [])
    monkeypatch.setattr(review_ltr, "load_review_texts", lambda *args: ({}, {}))
    monkeypatch.setattr(review_ltr, "ProfileFormatter", lambda *args: LocalFormatter())
    monkeypatch.setattr(LocalEmbeddingCache, "_load_encoder", lambda _: Encoder())
    monkeypatch.setattr(review_ltr, "profile_preflight", lambda *args, **kwargs: (
        ["query: first", "passage: second"], {"unique_inputs": 2, "unique_cache_misses": 2},
    ))
    assert review_ltr.main(["--snapshot", str(tmp_path / "snapshot.jsonl"),
                            "--artifacts-dir", str(tmp_path), "--embed-only"]) == 0
    diagnostic = json.loads((tmp_path / "diagnostics/review_ltr_e5_preflight.json").read_text())
    assert diagnostic["embedding_status"] == "complete"
    assert diagnostic["api_requests_sent"] == 0
    assert diagnostic["remaining_cache_misses"] == 0
    assert (tmp_path / "e5_review_embedding_cache.sqlite").exists()
    assert not (tmp_path / "review_embedding_cache.sqlite").exists()


def test_initialized_thread_pool_does_not_prevent_local_loading(tmp_path, monkeypatch):
    import torch
    import sentence_transformers

    class LoadedEncoder(Encoder):
        def get_sentence_embedding_dimension(self):
            return 384

    def initialized(_):
        raise RuntimeError("inter-op pool already initialized")

    monkeypatch.setattr(torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 6)
    monkeypatch.setattr(torch, "set_num_interop_threads", initialized)
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", lambda *args, **kwargs: LoadedEncoder())
    with LocalEmbeddingCache(tmp_path / "e5.sqlite", config(tmp_path)) as cache:
        cache.prefill(["query: example"])
        metadata = json.loads(cache.db.execute("SELECT metadata FROM local_encoders").fetchone()[0])
        assert metadata["interop_threads"] == 6
        assert cache.encoder.max_seq_length == 500
        assert cache.missing_count(["query: example"]) == 0
