"""CPU retrieval towers over frozen, cutoff-specific cached E5 profiles."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.config import ExperimentConfig
from rating_recsys.experiments.queries import build_training_windows
from rating_recsys.retrieval.lightgcn import _sample_negatives
from rating_recsys.retrieval.review_embeddings import E5_MODEL, profile_api_inputs
from rating_recsys.retrieval.review_profiles import _selected_reviews


TWO_TOWER = "two_tower"


@dataclass(frozen=True, slots=True)
class TwoTowerConfig:
    id_dim: int = 32
    hidden_dim: int = 128
    output_dim: int = 64
    epochs: int = 12
    batch_size: int = 512
    negative_samples: int = 16
    learning_rate: float = 0.001
    temperature: float = 0.1
    threads: int = 4
    seed: int = 42
    id_dropout: float = 0.1
    weight_decay: float = 1e-5
    window_months: int = 3

    def __post_init__(self):
        for name in ("id_dim", "hidden_dim", "output_dim", "epochs", "batch_size",
                     "negative_samples", "threads", "window_months"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if (not math.isfinite(self.learning_rate) or self.learning_rate <= 0
                or not math.isfinite(self.temperature) or self.temperature <= 0
                or not math.isfinite(self.weight_decay) or self.weight_decay < 0
                or not 0 <= self.id_dropout < 1):
            raise ValueError("Invalid Two-Tower optimization settings")

    @property
    def name(self):
        return TWO_TOWER

    def to_dict(self):
        return {"name": self.name, **asdict(self)}


@dataclass(frozen=True, slots=True)
class EpochStats:
    epoch: int
    loss: float
    seconds: float


@dataclass(frozen=True, slots=True)
class ProfileBank:
    user_keys: tuple[int, ...]
    item_keys: tuple[int, ...]
    users: torch.Tensor
    items: torch.Tensor
    audit: dict[str, object]


class CachedProfiles:
    """Read-only E5 lookup; no model loading, network, or cache mutation."""

    def __init__(self, config, formatter, cache=None):
        if (config.model != E5_MODEL or config.backend != "local"
                or config.dimensions != 384 or config.aggregation != "concat"
                or config.user_profile != "own_reviews"):
            raise ValueError("Two-Tower requires frozen local E5 concat/own_reviews")
        self.config, self.formatter, self.cache = config, formatter, cache
        self.identity = {
            "model": config.model, "model_revision": config.model_revision,
            "tokenizer_revision": config.tokenizer_revision,
            "profile_version": config.profile_version, "dimensions": config.dimensions,
            "max_tokens": config.max_document_tokens, "dtype": "float32", "device": "cpu",
        }
        self.db = None
        if cache is None:
            uri = Path(config.cache_path).resolve().as_uri() + "?mode=ro"
            self.db = sqlite3.connect(uri, uri=True)
        self._vectors = {}

    def close(self):
        if self.db is not None:
            self.db.close()
            self.db = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _key(self, text):
        return hashlib.sha256(json.dumps([self.identity, text], ensure_ascii=False,
                                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _lookup(self, docs):
        unique = list(dict.fromkeys(docs))
        missing = [doc for doc in unique if self._key(doc) not in self._vectors]
        if self.cache is not None:
            if self.cache.missing_count(missing):
                raise RuntimeError("Missing frozen E5 profiles; generate the required cache separately")
            found = self.cache.embed(missing) if missing else {}
        else:
            found = {}
            for doc in missing:
                row = self.db.execute("SELECT vector,model,dimensions FROM vectors WHERE key=?",
                                      (self._key(doc),)).fetchone()
                if row is None:
                    raise RuntimeError("Missing frozen E5 profiles; generate the required cache separately")
                if row[1] != self.config.model or row[2] != self.config.dimensions:
                    raise RuntimeError("Frozen E5 cache schema mismatch")
                found[doc] = json.loads(row[0])
        for doc, value in found.items():
            vector = np.asarray(value, dtype=np.float32)
            if vector.shape != (384,) or not np.isfinite(vector).all() or np.linalg.norm(vector) <= 0:
                raise RuntimeError("Invalid frozen E5 vector")
            self._vectors[self._key(doc)] = vector / np.linalg.norm(vector)
        return {doc: self._vectors[self._key(doc)] for doc in unique}

    def build(self, history, texts, user_ids):
        history = tuple(history)
        user_keys = tuple(sorted(set(user_ids)))
        item_keys = tuple(sorted({r.restaurant_id for r in history}))
        users, items, _, docs = profile_api_inputs(
            history, texts, user_keys, self.config, self.formatter,
        )
        vectors = self._lookup(docs)
        by_user, by_item = defaultdict(list), defaultdict(list)
        for row in history:
            by_user[row.user_id].append(row)
            by_item[row.restaurant_id].append(row)
        # Missing text has a zero vector and mask; numeric features use bounded log counts.
        def features(keys, documents, grouped, limit, user_side):
            result = np.zeros((len(keys), 387 if user_side else 386), dtype=np.float32)
            for index, key in enumerate(keys):
                doc = documents.get(key)
                if doc is not None:
                    result[index, :384] = vectors[doc]
                    result[index, 384] = 1
                result[index, 385] = math.log1p(len(_selected_reviews(grouped[key], texts, limit, self.config)))
                if user_side:
                    result[index, 386] = math.log1p(len(grouped[key]))
            return torch.from_numpy(result)
        user_values = features(user_keys, users, by_user, self.config.max_user_reviews, True)
        item_values = features(item_keys, items, by_item, self.config.max_item_reviews, False)
        return ProfileBank(user_keys, item_keys, user_values, item_values, {
            "history_rows": len(history), "user_profiles": len(users), "item_profiles": len(items),
            "users": len(user_keys), "items": len(item_keys), "unique_inputs": len(set(docs)),
            "latest_history_date": max((r.event_date for r in history), default=None).isoformat() if history else None,
            "cache_misses": 0, "encoder_loaded": False,
        })


class _Tower(nn.Module):
    def __init__(self, entities, width, config):
        super().__init__()
        self.ids = nn.Embedding(entities + 1, config.id_dim, padding_idx=0)
        self.layers = nn.Sequential(nn.Linear(width + config.id_dim, config.hidden_dim),
                                    nn.ReLU(), nn.Linear(config.hidden_dim, config.output_dim))
        self.id_dropout = config.id_dropout

    def forward(self, ids, features):
        if self.training and self.id_dropout:
            ids = ids.masked_fill(torch.rand(ids.shape) < self.id_dropout, 0)
        return F.normalize(self.layers(torch.cat((self.ids(ids), features), dim=-1)), dim=-1)


@contextmanager
def _torch_settings(config):
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.set_num_threads(config.threads)
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            yield
    finally:
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic)


@dataclass(slots=True)
class _TrainingWindow:
    bank: ProfileBank
    users: np.ndarray
    positives: np.ndarray
    forbidden: np.ndarray
    user_embedding_ids: torch.Tensor
    item_embedding_ids: torch.Tensor


class TwoTower:
    def __init__(self, config=TwoTowerConfig(), *, embedding_config, formatter,
                 profiles=None, eval_user_ids=(), experiment_config=None):
        self.config, self.embedding_config, self.formatter = config, embedding_config, formatter
        self.profiles = profiles
        self.eval_user_ids = tuple(eval_user_ids)
        self.experiment_config = experiment_config or ExperimentConfig(
            ranker_training_mode="window", satisfaction_mode="history-aware",
            lightgcn_checkpoint_months=config.window_months,
        )
        if self.experiment_config.lightgcn_checkpoint_months != config.window_months:
            raise ValueError("Two-Tower and experiment training window months must match")
        self.user_ids, self.item_ids = {}, {}
        self.item_keys = np.empty(0, dtype=np.int64)
        self.edge_count = 0
        self.history = []
        self.training_audit = []
        self.metadata = {}
        self.user_tower = self.item_tower = None
        self._bank = self._encoded = None

    def _windows(self, rows, texts, reader):
        windows = []
        for start, queries in build_training_windows(
            rows, config=self.experiment_config, through=max(r.event_date for r in rows),
        ):
            if not queries:
                continue
            history = tuple(r for r in rows if r.event_date < start)
            bank = reader.build(history, texts, [q.user_id for q in queries])
            uindex = {key: i for i, key in enumerate(bank.user_keys)}
            iindex = {key: i for i, key in enumerate(bank.item_keys)}
            user_rows, positives, forbidden = [], [], []
            for query in queries:
                seen = {r.restaurant_id for r in query.history}
                blocked = seen | {r.restaurant_id for r in query.window}
                blocked_ids = sorted(iindex[i] for i in blocked if i in iindex)
                uid = uindex[query.user_id]
                forbidden.extend(uid * len(iindex) + iid for iid in blocked_ids)
                if len(blocked_ids) == len(iindex):
                    continue
                for iid, grade in sorted(query.relevance_by_item.items()):
                    if grade > 0 and iid in iindex and iid not in seen:
                        user_rows.append(uid)
                        positives.append(iindex[iid])
            audit = {**bank.audit, "cutoff": queries[0].cutoff.isoformat(), "phase": queries[0].phase,
                     "queries": len(queries), "training_positives": len(positives),
                     "candidate_policy": "all cutoff-eligible items; no C5 filtering"}
            self.training_audit.append(audit)
            if len(self.training_audit) % 10 == 0:
                print(f"Two-Tower cached profile banks: {len(self.training_audit)}", file=sys.stderr, flush=True)
            if positives:
                windows.append(_TrainingWindow(
                    bank, np.asarray(user_rows, np.int64), np.asarray(positives, np.int64),
                    np.unique(np.asarray(forbidden, np.int64)),
                    torch.tensor([self.user_ids[key] + 1 for key in bank.user_keys]),
                    torch.tensor([self.item_ids[key] + 1 for key in bank.item_keys]),
                ))
        return windows

    def fit(self, interactions: Iterable[Interaction], texts: Mapping[int, str | None], *, callback=None):
        rows = tuple(sorted(interactions, key=lambda r: (r.event_date, r.review_id)))
        if len(rows) < 2 or len({r.restaurant_id for r in rows}) < 2:
            raise ValueError("Two-Tower requires at least two visits and restaurants")
        self.user_ids = {key: i for i, key in enumerate(sorted({r.user_id for r in rows}))}
        self.item_ids = {key: i for i, key in enumerate(sorted({r.restaurant_id for r in rows}))}
        self.item_keys = np.asarray(list(self.item_ids), dtype=np.int64)
        self.edge_count = len({(r.user_id, r.restaurant_id) for r in rows})
        self.history, self.training_audit = [], []
        self._encoded = None
        owned = self.profiles is None
        reader = self.profiles or CachedProfiles(self.embedding_config, self.formatter)
        try:
            windows = self._windows(rows, texts, reader)
            if not windows:
                raise ValueError("No cutoff-safe eligible positives with an unobserved negative")
            self._bank = reader.build(rows, texts, self.eval_user_ids or self.user_ids)
            print(f"Two-Tower setup: {len(windows)} training banks, "
                  f"{sum(len(w.positives) for w in windows)} positives, "
                  f"{len(self._bank.item_keys)} evaluation items", file=sys.stderr, flush=True)
        finally:
            if owned:
                reader.close()
        self.metadata = {
            "objective": "sampled softmax; history-aware relevance>0 outcomes",
            "negative_policy": "uniform cutoff catalog excluding seen and every window outcome",
            "frozen_encoder": self.embedding_config.to_dict(), "profile_audit": self.training_audit,
            "evaluation_bank": self._bank.audit, "training_positives": sum(len(w.positives) for w in windows),
            "e5_training": False, "api_requests": 0,
        }
        rng = np.random.default_rng(self.config.seed)
        with _torch_settings(self.config):
            self.user_tower = _Tower(len(self.user_ids), 387, self.config)
            self.item_tower = _Tower(len(self.item_ids), 386, self.config)
            optimizer = torch.optim.Adam(
                [*self.user_tower.parameters(), *self.item_tower.parameters()],
                lr=self.config.learning_rate, weight_decay=self.config.weight_decay,
            )
            for epoch in range(1, self.config.epochs + 1):
                started, total, count = time.perf_counter(), 0.0, 0
                self.user_tower.train()
                self.item_tower.train()
                self._encoded = None
                for wi in rng.permutation(len(windows)):
                    window = windows[wi]
                    order = rng.permutation(len(window.positives))
                    negatives = _sample_negatives(
                        rng, np.repeat(window.users, self.config.negative_samples),
                        len(window.bank.item_keys), window.forbidden,
                    ).reshape(-1, self.config.negative_samples)
                    for offset in range(0, len(order), self.config.batch_size):
                        indices = order[offset:offset + self.config.batch_size]
                        users = torch.from_numpy(window.users[indices])
                        choices = torch.from_numpy(np.column_stack((window.positives[indices], negatives[indices])))
                        uvec = self.user_tower(window.user_embedding_ids[users], window.bank.users[users])
                        ivec = self.item_tower(window.item_embedding_ids[choices], window.bank.items[choices])
                        logits = torch.einsum("bd,bnd->bn", uvec, ivec) / self.config.temperature
                        loss = F.cross_entropy(logits, torch.zeros(len(indices), dtype=torch.long))
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()
                        total += float(loss.detach()) * len(indices)
                        count += len(indices)
                self.user_tower.eval()
                self.item_tower.eval()
                stats = EpochStats(epoch, total / count, time.perf_counter() - started)
                self.history.append(stats)
                if callback is not None and callback(stats, self):
                    break
        self._encoded = None
        return self

    def _encode(self):
        if self.user_tower is None or self._bank is None:
            raise RuntimeError("Two-Tower must be fitted first")
        if self._encoded is None:
            with _torch_settings(self.config), torch.no_grad():
                self.user_tower.eval()
                self.item_tower.eval()
                def encode(tower, keys, features, ids):
                    result = []
                    for start in range(0, len(keys), self.config.batch_size):
                        part = keys[start:start + self.config.batch_size]
                        eid = torch.tensor([ids.get(key, -1) + 1 for key in part])
                        result.append(tower(eid, features[start:start + len(part)]).numpy())
                    return np.concatenate(result) if result else np.empty((0, self.config.output_dim), np.float32)
                self._encoded = (
                    {key: i for i, key in enumerate(self._bank.user_keys)},
                    encode(self.user_tower, self._bank.user_keys, self._bank.users, self.user_ids),
                    encode(self.item_tower, self._bank.item_keys, self._bank.items, self.item_ids),
                )
        return self._encoded

    def recommend(self, user_ids: Sequence[int], exclude: Mapping[int, Iterable[int]], k: int):
        if k < 1:
            raise ValueError("k must be positive")
        uindex, users, items = self._encode()
        result = {}
        iindex = {key: i for i, key in enumerate(self._bank.item_keys)}
        valid = [uid for uid in user_ids if uid in uindex and uid in self.user_ids]
        result.update((uid, ()) for uid in user_ids if uid not in uindex or uid not in self.user_ids)
        for start in range(0, len(valid), 512):
            batch = valid[start:start + 512]
            scores = users[[uindex[uid] for uid in batch]] @ items.T
            for row, uid in enumerate(batch):
                scores[row, [iindex[i] for i in exclude.get(uid, ()) if i in iindex]] = -np.inf
            orders = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            for row, uid in enumerate(batch):
                result[uid] = tuple(self._bank.item_keys[i] for i in orders[row] if np.isfinite(scores[row, i]))
        return result

    def score(self, user_id, restaurant_ids):
        uindex, users, items = self._encode()
        if user_id not in uindex or user_id not in self.user_ids:
            return {}
        index = {key: i for i, key in enumerate(self._bank.item_keys)}
        keys = [key for key in restaurant_ids if key in index]
        values = items[[index[key] for key in keys]] @ users[uindex[user_id]]
        return dict(zip(keys, map(float, values)))

    def save(self, destination: Path):
        if self.user_tower is None:
            raise RuntimeError("Two-Tower must be fitted first")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "frozen-e5-two-tower-v1", "config": self.config.to_dict(),
            "user_tower": self.user_tower.state_dict(), "item_tower": self.item_tower.state_dict(),
            "user_ids": self.user_ids, "item_ids": self.item_ids,
            "bank": {"user_keys": self._bank.user_keys, "item_keys": self._bank.item_keys,
                     "users": self._bank.users, "items": self._bank.items},
            "metadata": self.metadata, "history": [asdict(stats) for stats in self.history],
        }, destination)
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return {"path": destination.name, "sha256": digest.hexdigest(),
                "bytes": destination.stat().st_size, "schema": "frozen-e5-two-tower-v1"}
