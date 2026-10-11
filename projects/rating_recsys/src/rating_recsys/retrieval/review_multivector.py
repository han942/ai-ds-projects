"""Frozen review matching over the entire past catalog, without a ranker.

The mean control and late-interaction path share exactly the same review bank.
Late interaction averages each user review's top-k restaurant-review cosines,
then averages over user reviews. These are review vectors, not ColBERT tokens.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class ReviewMultiVectorConfig:
    max_user_reviews: int = 20
    max_item_reviews: int = 30
    matching_k: int = 3
    user_batch_size: int = 16
    item_batch_size: int = 256

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self):
        return asdict(self)


def unit(values):
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite review vectors")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0)


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def review_vectors_from_store(store):
    """Discard learned Transformer weights; collapse only within each review.

    Padding can contain foreign values and is explicitly excluded. Each review
    is normalized before either retrieval path sees it.
    """
    vectors = _numpy(store.vectors)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("Invalid frozen block vectors")
    result = []
    for table in (store.query_blocks, store.document_blocks):
        indices = _numpy(table)
        if indices.ndim != 2 or indices.dtype.kind not in "iu":
            raise ValueError("Invalid review-block indices")
        if np.any(indices < 0) or np.any(indices >= len(vectors)):
            raise ValueError("Block index outside frozen store")
        counts = np.count_nonzero(indices, axis=1)
        pooled = np.zeros((len(indices), vectors.shape[1]), dtype=np.float32)
        for column in indices.T:
            present = column != 0
            pooled[present] += vectors[column[present]]
        pooled /= np.maximum(counts[:, None], 1)
        result.append(unit(pooled))
    if result[0].shape != result[1].shape:
        raise ValueError("User and item review-vector shapes differ")
    if any(type(rid) is not int or not 0 < index < len(result[0])
           for rid, index in store.review_index.items()):
        raise ValueError("Review index outside frozen store")
    return result[0], result[1]


@dataclass
class ReviewVectorBank:
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_review_ids: np.ndarray
    item_review_ids: np.ndarray
    user_vectors: np.ndarray
    item_vectors: np.ndarray
    audit: dict

    def save(self, path):
        np.savez_compressed(path, **{name: getattr(self, name) for name in (
            "user_ids", "item_ids", "user_review_ids", "item_review_ids",
            "user_vectors", "item_vectors")})


def build_review_bank(history, cutoff, review_index, query_vectors, passage_vectors,
                      *, query_users, config=ReviewMultiVectorConfig()):
    history = tuple(sorted(history, key=lambda row: (row.event_date, row.review_id)))
    if any(row.event_date > cutoff for row in history):
        raise ValueError("Review bank contains a future interaction")
    if len({row.review_id for row in history}) != len(history):
        raise ValueError("Duplicate historical review IDs")
    query_vectors, passage_vectors = unit(query_vectors), unit(passage_vectors)
    if query_vectors.shape != passage_vectors.shape or query_vectors.ndim != 2:
        raise ValueError("Review-vector dimensions differ")
    users = tuple(sorted(set(query_users)))
    catalog = tuple(sorted({row.restaurant_id for row in history}))
    selected_users = defaultdict(lambda: deque(maxlen=config.max_user_reviews))
    selected_items = defaultdict(lambda: deque(maxlen=config.max_item_reviews))
    wanted = set(users)
    for row in history:
        index = review_index.get(row.review_id)
        if index is None:
            continue
        if not 0 <= index < len(query_vectors):
            raise ValueError("Historical review index outside vector array")
        if row.user_id in wanted and np.any(query_vectors[index]):
            selected_users[row.user_id].append((row.review_id, index))
        if np.any(passage_vectors[index]):
            selected_items[row.restaurant_id].append((row.review_id, index))

    def pack(keys, groups, vectors, limit):
        references = np.full((len(keys), limit), -1, dtype=np.int64)
        packed = np.zeros((len(keys), limit, vectors.shape[1]), dtype=np.float32)
        for position, key in enumerate(keys):
            chosen = list(groups.get(key, ()))
            if chosen:
                ids, indices = zip(*chosen)
                references[position, :len(chosen)] = ids
                packed[position, :len(chosen)] = vectors[list(indices)]
        return references, packed

    urefs, uvalues = pack(users, selected_users, query_vectors, config.max_user_reviews)
    irefs, ivalues = pack(catalog, selected_items, passage_vectors, config.max_item_reviews)
    arrays = dict(user_ids=np.asarray(users, dtype=np.int64), item_ids=np.asarray(catalog, dtype=np.int64),
                  user_review_ids=urefs, item_review_ids=irefs, user_vectors=uvalues, item_vectors=ivalues)
    audit = {
        "cutoff_inclusive": cutoff.isoformat(), "historical_rows": len(history),
        "maximum_event_date": history[-1].event_date.isoformat() if history else None,
        "ratings": "all; no new rating gate", "user_role": "query", "item_role": "passage",
        "review_aggregation": "nonpadding block mean -> review L2; same bank for mean and late",
        "config": config.to_dict(),
        "users": len(users), "users_with_text": int((urefs >= 0).any(1).sum()),
        "catalog": len(catalog), "items_with_text": int((irefs >= 0).any(1).sum()),
        "arrays_sha256": {name: array_hash(value) for name, value in arrays.items()},
    }
    return ReviewVectorBank(**arrays, audit=audit)


class ReviewMultiVectorRetriever:
    def __init__(self, bank, config=ReviewMultiVectorConfig()):
        self.bank, self.config = bank, config

    def score_all(self, *, progress=None):
        """Exact catalog scoring in bounded blocks; no candidate shortlist.

        Only the user-by-item final score arrays are materialized. Padding is
        -inf during top-k selection, including when every true cosine is negative.
        """
        bank, config = self.bank, self.config
        umask, imask = bank.user_review_ids >= 0, bank.item_review_ids >= 0
        ucounts, icounts = umask.sum(1), imask.sum(1)
        umean = unit((bank.user_vectors * umask[..., None]).sum(1) / np.maximum(ucounts[:, None], 1))
        imean = unit((bank.item_vectors * imask[..., None]).sum(1) / np.maximum(icounts[:, None], 1))
        mean = umean @ imean.T
        mean[(np.linalg.norm(umean, axis=1) == 0), :] = -np.inf
        mean[:, np.linalg.norm(imean, axis=1) == 0] = -np.inf
        late = np.full(mean.shape, -np.inf, dtype=np.float32)
        dimension = bank.user_vectors.shape[-1]
        slots = bank.item_vectors.shape[1]
        take = min(config.matching_k, slots)
        for start in range(0, len(bank.user_ids), config.user_batch_size):
            stop = min(start + config.user_batch_size, len(bank.user_ids))
            mask = umask[start:stop]
            lengths = ucounts[start:stop]
            query = bank.user_vectors[start:stop][mask]
            if len(query):
                present_users = np.flatnonzero(lengths > 0)
                offsets = np.cumsum(np.r_[0, lengths[lengths > 0]])[:-1]
                for left in range(0, len(bank.item_ids), config.item_batch_size):
                    right = min(left + config.item_batch_size, len(bank.item_ids))
                    documents = bank.item_vectors[left:right]
                    pairwise = (query @ documents.reshape(-1, dimension).T).reshape(len(query), right-left, slots)
                    pairwise[:, ~imask[left:right]] = -np.inf
                    best = np.partition(pairwise, slots-take, axis=2)[:, :, -take:]
                    per_review = np.where(np.isfinite(best), best, 0).sum(2)
                    per_review /= np.maximum(np.minimum(config.matching_k, icounts[left:right]), 1)
                    per_user = np.add.reduceat(per_review, offsets, axis=0) / lengths[present_users, None]
                    per_user[:, icounts[left:right] == 0] = -np.inf
                    late[start + present_users, left:right] = per_user
            if progress:
                progress(stop, len(bank.user_ids))
        return {"mean": np.asarray(mean, dtype=np.float32), "late": late}

    def rankings(self, scores, exclude, *, k=100):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        if scores.shape != (len(self.bank.user_ids), len(self.bank.item_ids)) or np.isnan(scores).any():
            raise ValueError("Invalid retrieval scores")
        result = {}
        for index, user_id in enumerate(self.bank.user_ids.tolist()):
            values = scores[index]
            eligible = np.isfinite(values) & ~np.isin(self.bank.item_ids, tuple(exclude.get(user_id, ())))
            positions = np.flatnonzero(eligible)
            ordered = positions[np.lexsort((self.bank.item_ids[positions], -values[positions]))[:k]]
            result[user_id] = tuple(self.bank.item_ids[ordered].tolist())
        return result


def supplement_candidates(baseline, text, *, k=100, text_slots=10):
    """Keep the baseline prefix and add only text items outside its whole Top-k.

    With insufficient new text candidates, fill the tail from the baseline.
    The resulting list has the same length as the original baseline (up to k).
    """
    if type(k) is not int or k < 1 or type(text_slots) is not int or not 0 <= text_slots <= k:
        raise ValueError("Invalid candidate quota")
    baseline = tuple(dict.fromkeys(baseline))[:k]
    target = len(baseline)
    slots = min(text_slots, target)
    original = set(baseline)
    additions = tuple(dict.fromkeys(item for item in text if item not in original))[:slots]
    keep = target - len(additions)
    return baseline[:keep] + additions
