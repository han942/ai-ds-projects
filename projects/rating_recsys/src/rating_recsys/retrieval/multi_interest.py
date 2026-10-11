"""Text-clustered, separate-interest full-catalog candidate retrieval.

This deterministic frozen-vector adapter is not a neural ComiRec reproduction.
Its one-interest control and multiple-interest treatment share the review bank,
catalog, graph checkpoint and score definition. Separate lists are interleaved;
scores are never averaged across interests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import date

import numpy as np

from .review_multivector import array_hash, unit


@dataclass(frozen=True)
class MultiInterestConfig:
    interests: int = 3
    max_iterations: int = 30

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self):
        return asdict(self)


def spherical_clusters(vectors, *, config=MultiInterestConfig()):
    """Return compact assignments, actual group means and validity masks.

    Initialization starts with the first chronological review, then chooses the
    farthest distinct review. All ties follow input order. Empty clusters are
    omitted. A cancelled group mean remains masked, rather than using a seed or
    an invented vector in its place. At the iteration cap, means still describe
    exactly the returned assignments.
    """
    values = unit(vectors)
    if values.ndim != 2 or np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError("Clustering requires nonzero two-dimensional vectors")
    if not len(values):
        return (np.empty(0, np.int64), np.empty((0, values.shape[1]), np.float32),
                np.empty(0, bool))
    unique = np.sort(np.unique(values, axis=0, return_index=True)[1])
    count = min(config.interests, len(unique))
    seeds = [int(unique[0])]
    while len(seeds) < count:
        remaining = unique[~np.isin(unique, seeds)]
        closeness = (values[remaining] @ values[seeds].T).max(axis=1)
        seeds.append(int(remaining[np.argmin(closeness)]))
    centers = values[seeds].copy()
    previous = None
    assignments = np.zeros(len(values), dtype=np.int64)
    for _ in range(config.max_iterations):
        scores = values @ centers.T
        scores[:, np.linalg.norm(centers, axis=1) == 0] = -np.inf
        if not np.isfinite(scores).any():
            break
        assignments = scores.argmax(axis=1).astype(np.int64)
        used = np.unique(assignments)
        remap = np.full(len(centers), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        assignments = remap[assignments]
        centers = unit(np.stack([values[assignments == group].mean(axis=0)
                                 for group in range(len(used))]))
        if previous is not None and np.array_equal(assignments, previous):
            break
        previous = assignments.copy()
    # The latest centers were computed from these assignments. A zero mean is
    # disabled for text retrieval, while its historical graph mean can be valid.
    return assignments, centers, np.linalg.norm(centers, axis=1) > 0


@dataclass
class MultiInterestBank:
    user_ids: np.ndarray
    item_ids: np.ndarray
    review_ids: np.ndarray
    assignments: np.ndarray
    text_centroids: np.ndarray
    graph_centroids: np.ndarray
    text_mask: np.ndarray
    graph_mask: np.ndarray
    text_items: np.ndarray
    graph_items: np.ndarray
    text_item_mask: np.ndarray
    graph_item_mask: np.ndarray
    audit: dict

    def arrays(self):
        return {field.name: getattr(self, field.name) for field in fields(self)
                if field.name != "audit"}

    def save(self, path):
        np.savez_compressed(path, **self.arrays())


def _ids(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu" or len(np.unique(values)) != len(values):
        raise ValueError(f"Invalid or duplicate {name}")
    return values.astype(np.int64, copy=False)


def build_multi_interest_bank(review_bank, history, graph_item_ids,
                              graph_item_vectors, *, config=MultiInterestConfig()):
    """Cluster only retained historical user text; reuse a frozen graph.

    Graph prototypes use the normalized graph vector of each retained review's
    visited restaurant, preserving repeated review weighting. Graph items absent
    from the supplied checkpoint are excluded, without a fallback. For text
    retrieval, item profiles mean all retained passage-role reviews.
    """
    user_ids = _ids(review_bank.user_ids, "user IDs")
    item_ids = _ids(review_bank.item_ids, "item IDs")
    graph_ids = _ids(graph_item_ids, "graph item IDs")
    graph_values = unit(graph_item_vectors)
    if graph_values.ndim != 2 or len(graph_values) != len(graph_ids) or graph_values.shape[1] < 1:
        raise ValueError("Invalid graph item vectors")
    references = np.asarray(review_bank.user_review_ids)
    user_values = np.asarray(review_bank.user_vectors, np.float32)
    item_references = np.asarray(review_bank.item_review_ids)
    item_values = np.asarray(review_bank.item_vectors, np.float32)
    if (references.ndim != 2 or references.dtype.kind not in "iu"
            or references.shape[0] != len(user_ids)
            or user_values.ndim != 3 or user_values.shape[:2] != references.shape
            or item_references.ndim != 2 or item_references.dtype.kind not in "iu"
            or item_references.shape[0] != len(item_ids)
            or item_values.ndim != 3 or item_values.shape[:2] != item_references.shape
            or user_values.shape[2] != item_values.shape[2]
            or user_values.shape[2] < 1
            or not np.isfinite(user_values).all() or not np.isfinite(item_values).all()):
        raise ValueError("Invalid review bank arrays")
    past = {}
    cutoff_text = review_bank.audit.get("cutoff_inclusive")
    cutoff = date.fromisoformat(cutoff_text) if cutoff_text else None
    for row in history:
        if row.review_id in past:
            raise ValueError("Duplicate historical review IDs")
        if cutoff is not None and row.event_date > cutoff:
            raise ValueError("Future history in multi-interest bank")
        past[row.review_id] = row
    catalog = set(item_ids.tolist())
    for index, item in enumerate(item_ids.tolist()):
        chosen_ids = item_references[index, item_references[index] >= 0].tolist()
        if len(set(chosen_ids)) != len(chosen_ids):
            raise ValueError("Duplicate retained item review IDs")
        if any(rid not in past or past[rid].restaurant_id != item for rid in chosen_ids):
            raise ValueError("Retained item review missing or belongs to another item")
    graph_lookup = {int(key): graph_values[index] for index, key in enumerate(graph_ids)}
    graph_items = np.zeros((len(item_ids), graph_values.shape[1]), np.float32)
    for index, key in enumerate(item_ids):
        if int(key) in graph_lookup:
            graph_items[index] = graph_lookup[int(key)]
    imask = item_references >= 0
    text_items = unit((item_values * imask[..., None]).sum(axis=1)
                      / np.maximum(imask.sum(axis=1)[:, None], 1))
    text_centroids = np.zeros((len(user_ids), config.interests, user_values.shape[2]), np.float32)
    graph_centroids = np.zeros((len(user_ids), config.interests, graph_values.shape[1]), np.float32)
    text_mask = np.zeros(text_centroids.shape[:2], bool)
    graph_mask = np.zeros(graph_centroids.shape[:2], bool)
    assignments = np.full(references.shape, -1, dtype=np.int64)
    iterations_groups = []
    for uindex, user in enumerate(user_ids.tolist()):
        selected = np.flatnonzero(references[uindex] >= 0)
        chosen_ids = references[uindex, selected].tolist()
        if len(set(chosen_ids)) != len(chosen_ids):
            raise ValueError("Duplicate retained user review IDs")
        visited = []
        for rid in chosen_ids:
            row = past.get(rid)
            if row is None or row.user_id != user:
                raise ValueError("Retained review missing or belongs to another user")
            if row.restaurant_id not in catalog:
                raise ValueError("Historical reviewed item outside catalog")
            visited.append(row.restaurant_id)
        if not len(selected):
            iterations_groups.append(0)
            continue
        groups, centroids, valid = spherical_clusters(user_values[uindex, selected], config=config)
        count = len(centroids)
        iterations_groups.append(count)
        assignments[uindex, selected] = groups
        text_centroids[uindex, :count] = centroids
        text_mask[uindex, :count] = valid
        for group in range(count):
            values = [graph_lookup[visited[index]] for index in np.flatnonzero(groups == group)
                      if visited[index] in graph_lookup and np.any(graph_lookup[visited[index]])]
            if values:
                graph_centroids[uindex, group] = unit(np.asarray(values).mean(axis=0))
                graph_mask[uindex, group] = np.any(graph_centroids[uindex, group])
    result = MultiInterestBank(
        user_ids.copy(), item_ids.copy(), references.astype(np.int64, copy=True), assignments,
        text_centroids, graph_centroids, text_mask, graph_mask, text_items, graph_items,
        np.linalg.norm(text_items, axis=1) > 0, np.linalg.norm(graph_items, axis=1) > 0, {})
    result.audit = {
        "config": config.to_dict(), "cutoff_inclusive": cutoff_text,
        "ratings": "all; identical retained reviews to the mean control",
        "clustering": "spherical k-means; chronological first + farthest distinct review seeds; deterministic ties",
        "group_count_histogram": {str(count): iterations_groups.count(count)
                                  for count in sorted(set(iterations_groups))},
        "graph_prototype": "mean of normalized frozen graph vectors of each grouped historical review's visited item -> L2",
        "graph_item_score": "cosine; fixed graph item vectors normalized; not the learned C4 user scorer",
        "text_item_profile": "retained passage-role review mean -> L2",
        "retrieval": "full catalog per interest -> individual Top-k -> round-robin deduplicated Top-k",
        "users_with_text_interests": int(text_mask.any(axis=1).sum()),
        "users_with_graph_interests": int(graph_mask.any(axis=1).sum()),
        "text_interest_count_histogram": {str(count): int((text_mask.sum(axis=1) == count).sum())
                                          for count in range(config.interests + 1)},
        "source_review_bank_hashes": dict(review_bank.audit.get("arrays_sha256", {})),
        "arrays_sha256": {name: array_hash(value) for name, value in result.arrays().items()},
    }
    return result


def round_robin_candidates(lists, *, k=100):
    """Interleave equal-rank positions, preserving each interest's ordering."""
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    lists = tuple(tuple(values) for values in lists)
    selected, seen = [], set()
    for index in range(max((len(values) for values in lists), default=0)):
        for values in lists:
            if index < len(values) and values[index] not in seen:
                selected.append(values[index])
                seen.add(values[index])
                if len(selected) == k:
                    return tuple(selected)
    return tuple(selected)


@dataclass
class InterestRankings:
    per_interest: dict
    merged: dict


class MultiInterestRetriever:
    def __init__(self, bank):
        self.bank = bank

    def score_all(self, *, view=None):
        if view is None:
            return {name: self.score_all(view=name) for name in ("text", "graph")}
        if view not in ("text", "graph"):
            raise ValueError("view must be text or graph")
        centers = getattr(self.bank, f"{view}_centroids")
        documents = getattr(self.bank, f"{view}_items")
        scores = (centers.reshape(-1, centers.shape[-1]) @ documents.T).reshape(
            *centers.shape[:2], len(self.bank.item_ids))
        scores[~getattr(self.bank, f"{view}_mask")] = -np.inf
        scores[:, :, ~getattr(self.bank, f"{view}_item_mask")] = -np.inf
        return np.asarray(scores, np.float32)

    def rankings(self, scores, exclude, *, k=100):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        expected = (*self.bank.text_mask.shape, len(self.bank.item_ids))
        if scores.shape != expected or np.isnan(scores).any():
            raise ValueError("Invalid multi-interest scores")
        per_interest, merged = {}, {}
        for index, user in enumerate(self.bank.user_ids.tolist()):
            unseen = ~np.isin(self.bank.item_ids, tuple(exclude.get(user, ())))
            lists = []
            for values in scores[index]:
                positions = np.flatnonzero(unseen & np.isfinite(values))
                ordered = positions[np.lexsort((self.bank.item_ids[positions], -values[positions]))[:k]]
                lists.append(tuple(self.bank.item_ids[ordered].tolist()))
            per_interest[user] = tuple(lists)
            merged[user] = round_robin_candidates(lists, k=k)
        return InterestRankings(per_interest, merged)
