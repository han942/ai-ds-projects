"""Frozen-cache, cutoff, sampling and CPU retrieval checks with synthetic data."""

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import date

import numpy as np
import pytest
import torch

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_training_windows
from rating_recsys.retrieval.review_embeddings import E5_MODEL, model_embedding_config, profile_api_inputs
from rating_recsys.retrieval.two_tower import CachedProfiles, TwoTower, TwoTowerConfig


class Formatter:
    def prepare(self, users, items):
        return ({key: "query: " + value for key, value in users.items()},
                {key: "passage: " + value for key, value in items.items()})


class Cache:
    def __init__(self):
        self.values = {}
        self.calls = []

    def populate(self, docs):
        for doc in docs:
            seed = int.from_bytes(hashlib.sha256(doc.encode()).digest()[:4], "little")
            vector = np.random.default_rng(seed).normal(size=384).astype(np.float32)
            self.values[doc] = vector / np.linalg.norm(vector)

    def missing_count(self, docs):
        return len(set(docs) - self.values.keys())

    def embed(self, docs):
        assert not self.missing_count(docs)
        self.calls.extend(docs)
        return {doc: self.values[doc] for doc in docs}


def row(rid, user, item, month, day, rating=5):
    return Interaction(rid, user, item, date(2025, month, day), rating,
                       "exact", "synthetic", "region")


def case(tmp_path):
    rows = [row(i, i, 10 * i, 1, i + 1) for i in range(1, 7)]
    rows += [row(7, 1, 20, 4, 10), row(8, 1, 30, 4, 11, 4),
             row(9, 2, 40, 4, 12, 3), row(10, 3, 50, 4, 13, 2),
             row(11, 4, 60, 4, 14), row(12, 5, 10, 4, 15), row(13, 6, 20, 4, 16)]
    rows += [row(14 + i, i + 1, item, 7, 8 + i)
             for i, item in enumerate((50, 60, 10, 20, 30, 40))]
    texts = {r.review_id: f"past-review-{r.review_id}" for r in rows}
    texts[4] = ""
    embedding = model_embedding_config(E5_MODEL, str(tmp_path / "cache.sqlite"))
    exp = ExperimentConfig(ranker_training_mode="window", satisfaction_mode="history-aware")
    formatter, cache = Formatter(), Cache()
    for start, queries in build_training_windows(rows, config=exp, through=max(r.event_date for r in rows)):
        history = [r for r in rows if r.event_date < start]
        cache.populate(profile_api_inputs(history, texts, [q.user_id for q in queries], embedding, formatter)[3])
    cache.populate(profile_api_inputs(rows, texts, range(1, 7), embedding, formatter)[3])
    reader = CachedProfiles(embedding, formatter, cache)
    cfg = TwoTowerConfig(epochs=2, threads=1, batch_size=4, negative_samples=3,
                        hidden_dim=16, output_dim=8, id_dim=4)
    model = TwoTower(cfg, embedding_config=embedding, formatter=formatter,
                     profiles=reader, eval_user_ids=range(1, 7), experiment_config=exp)
    return rows, texts, model, reader, cache


@pytest.mark.parametrize("kwargs", [
    {"negative_samples": 0}, {"temperature": 0}, {"threads": 0},
    {"id_dropout": 1}, {"learning_rate": float("nan")}, {"weight_decay": -1},
])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        TwoTowerConfig(**kwargs)


def test_bank_uses_only_history_and_explicit_missing_masks(tmp_path):
    rows, texts, _, reader, cache = case(tmp_path)
    bank = reader.build(rows[:6], texts, [1, 4])
    assert bank.user_keys == (1, 4)
    assert bank.item_keys == (10, 20, 30, 40, 50, 60)
    assert bank.users.shape == (2, 387) and bank.items.shape == (6, 386)
    assert bank.users[0, 384] == 1 and bank.users[1, 384] == 0
    assert bank.items[3, 384] == 0
    assert torch.count_nonzero(bank.users[1, :384]) == 0
    assert bank.users[1, 386] > 0
    assert bank.audit["latest_history_date"] == "2025-01-07"
    assert all("past-review-7" not in doc for doc in cache.calls)


def test_missing_vectors_fail_without_cache_embed(tmp_path):
    rows, texts, model, _, _ = case(tmp_path)
    empty = Cache()
    reader = CachedProfiles(model.embedding_config, Formatter(), empty)
    with pytest.raises(RuntimeError, match="Missing frozen"):
        reader.build(rows[:6], texts, [1])
    assert empty.calls == []


@pytest.mark.parametrize("bad", [np.zeros(383), np.zeros(384), np.full(384, np.nan)])
def test_bad_cached_vectors_rejected(tmp_path, bad):
    rows, texts, model, _, cache = case(tmp_path)
    docs = profile_api_inputs(rows[:6], texts, [1], model.embedding_config, Formatter())[3]
    cache.values[docs[0]] = bad
    with pytest.raises(RuntimeError, match="Invalid frozen"):
        CachedProfiles(model.embedding_config, Formatter(), cache).build(rows[:6], texts, [1])


def test_sqlite_reader_does_not_modify_database_or_load_encoder(tmp_path, monkeypatch):
    rows, texts, model, _, cache = case(tmp_path)
    reader = CachedProfiles(model.embedding_config, Formatter(), cache)
    path = tmp_path / "cache.sqlite"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE vectors(key TEXT PRIMARY KEY, vector TEXT, model TEXT, dimensions INTEGER)")
    for doc, vector in cache.values.items():
        db.execute("INSERT INTO vectors VALUES (?,?,?,?)",
                   (reader._key(doc), json.dumps(vector.tolist()), E5_MODEL, 384))
    db.commit()
    db.close()
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError("Two-Tower must not load E5 or use API")
    monkeypatch.setattr("rating_recsys.retrieval.review_embeddings.LocalEmbeddingCache._load_encoder", forbidden)
    monkeypatch.setattr("rating_recsys.retrieval.review_embeddings.configured_api_key", forbidden)
    with CachedProfiles(model.embedding_config, Formatter()) as readonly:
        bank = readonly.build(rows[:6], texts, [1])
        assert bank.audit["encoder_loaded"] is False
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            readonly.db.execute("DELETE FROM vectors")
    assert path.read_bytes() == before


def test_training_masks_seen_other_positives_low_grades_and_future_catalog(tmp_path, monkeypatch):
    rows, texts, model, _, _ = case(tmp_path)
    future = row(99, 1, 999, 7, 20)
    texts[99] = "future-body"
    rows.append(future)
    original = __import__("rating_recsys.retrieval.two_tower", fromlist=["_sample_negatives"])._sample_negatives
    sampled = []
    def checked(rng, users, items, forbidden):
        values = original(rng, users, items, forbidden)
        assert not np.isin(users * items + values, forbidden).any()
        sampled.append(values)
        return values
    monkeypatch.setattr("rating_recsys.retrieval.two_tower._sample_negatives", checked)
    # The novel item is not in the July cutoff bank, so no additional text key is needed there.
    # The final evaluation bank does need its past-by-evaluation profile.
    model.profiles.cache.populate(profile_api_inputs(rows, texts, range(1, 7), model.embedding_config, Formatter())[3])
    model.fit(rows, texts)
    assert sampled
    for audit in model.training_audit:
        assert audit["latest_history_date"] <= audit["cutoff"]
        assert audit["items"] == 6
        assert audit["candidate_policy"] == "all cutoff-eligible items; no C5 filtering"
    assert model.metadata["training_positives"] == 12
    assert model.metadata["e5_training"] is False
    assert model.metadata["api_requests"] == 0
    assert all(not parameter.requires_grad for parameter in (model._bank.users, model._bank.items))


def test_fit_deterministic_callback_stop_exclusion_and_saved_schema(tmp_path):
    rows, texts, model, _, _ = case(tmp_path)
    calls = []
    def callback(stats, fitted):
        assert stats.epoch == 1 and np.isfinite(stats.loss) and stats.seconds > 0
        calls.append(fitted.recommend([1], {1: [10, 20, 30]}, 3))
        return True
    threads = torch.get_num_threads()
    model.fit(rows, texts, callback=callback)
    assert len(model.history) == len(calls) == 1
    assert torch.get_num_threads() == threads
    assert model.edge_count == len(rows)
    result = model.recommend([1, 999], {1: [10, 20, 30]}, 100)
    assert set(result[1]) == {40, 50, 60} and result[999] == ()
    assert model.score(999, [10]) == {}
    _, users, items = model._encode()
    np.testing.assert_allclose(np.linalg.norm(users, axis=1), 1, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(items, axis=1), 1, atol=1e-6)
    again = TwoTower(replace(model.config, epochs=1), embedding_config=model.embedding_config,
                     formatter=Formatter(), profiles=model.profiles, eval_user_ids=range(1, 7),
                     experiment_config=model.experiment_config).fit(list(reversed(rows)), texts)
    np.testing.assert_array_equal(model._encode()[1], again._encode()[1])
    assert model.recommend([1], {}, 6) == again.recommend([1], {}, 6)
    destination = tmp_path / "two_tower.pt"
    artifact = model.save(destination)
    assert artifact["path"] == destination.name
    assert artifact["bytes"] == destination.stat().st_size
    assert artifact["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    saved = torch.load(destination, weights_only=True)
    assert saved["schema"] == "frozen-e5-two-tower-v1"
    assert saved["config"]["name"] == "two_tower"
    assert set(saved["bank"]) == {"users", "items", "user_keys", "item_keys"}
    assert saved["metadata"]["training_positives"] == 12


def test_mismatched_window_settings_rejected(tmp_path):
    _, _, model, _, _ = case(tmp_path)
    with pytest.raises(ValueError, match="window months must match"):
        TwoTower(replace(model.config, window_months=6),
                 embedding_config=model.embedding_config, formatter=Formatter(),
                 experiment_config=model.experiment_config)


def test_no_trainable_windows_and_unfitted_inference_fail(tmp_path):
    rows, texts, model, _, _ = case(tmp_path)
    with pytest.raises(RuntimeError, match="fitted"):
        model.recommend([1], {}, 1)
    with pytest.raises(RuntimeError, match="fitted"):
        model.save(tmp_path / "unfitted.pt")
    with pytest.raises(ValueError, match="No cutoff-safe"):
        model.fit(rows[:6], texts)
