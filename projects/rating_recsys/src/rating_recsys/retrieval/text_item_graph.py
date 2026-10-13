"""Fixed text kNN item edges added to LightGCN's visit propagation.

This is a LATTICE-inspired ablation, not a reproduction: frozen review E5
vectors produce a fixed cosine kNN graph, without learned graph structure or
images. Training remains the inherited LightGCN BPR/Adam loop. Each layer uses
``S_visit + weight * blockdiag(0_users, S_text)`` and final embeddings average
layers 0..L. The combined matrix is symmetric, so the same linear propagation
backpropagates the BPR gradient exactly.

The primary binary graph uses cosine only to select neighbors. Its shuffled
control performs double-edge swaps, preserving every restaurant's degree and
all raw edge weights (one). The optional weighted graph uses node permutation
and only preserves the weighted-degree multiset, not per-restaurant degrees.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, triu

from rating_recsys.retrieval.lightgcn import EpochStats, LightGCN, LightGCNConfig


SCHEMA = "text-item-graph-inference-v1"


def _array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _sparse_hash(matrix):
    return hashlib.sha256(json.dumps([
        matrix.shape, _array_hash(matrix.data), _array_hash(matrix.indices),
        _array_hash(matrix.indptr),
    ], separators=(",", ":")).encode()).hexdigest()


def _inputs(item_ids, item_vectors, neighbors, shuffle_seed, graph_mode):
    ids = np.asarray(item_ids)
    if (ids.ndim != 1 or ids.dtype.kind not in "iu" or np.any(ids <= 0)
            or np.any(ids[1:] <= ids[:-1])
            or (ids.dtype.kind == "u" and np.any(ids > np.iinfo(np.int64).max))):
        raise ValueError("item_ids must be sorted unique positive integers")
    ids = ids.astype(np.int64, copy=True)
    vectors = np.asarray(item_vectors, dtype=np.float32)
    if (vectors.ndim != 2 or vectors.shape[0] != len(ids)
            or vectors.shape[1] < 1 or not np.isfinite(vectors).all()):
        raise ValueError("item_vectors must be a finite item-by-dimension matrix")
    if type(neighbors) is not int or neighbors < 1:
        raise ValueError("neighbors must be a positive integer")
    if (shuffle_seed is not None and
            (type(shuffle_seed) is not int or not 0 <= shuffle_seed < 2**63)):
        raise ValueError("shuffle_seed must be None or an integer in [0, 2**63)")
    if graph_mode not in ("binary", "weighted"):
        raise ValueError("graph_mode must be binary or weighted")
    return ids, vectors.copy()


def _normalize(raw):
    degrees = np.asarray(raw.sum(axis=1, dtype=np.float64)).ravel()
    inverse = np.divide(1.0, np.sqrt(degrees), out=np.zeros_like(degrees),
                        where=degrees > 0)
    edges = raw.tocoo()
    result = csr_matrix(((edges.data * (inverse[edges.row] * inverse[edges.col])).astype(np.float32),
                         (edges.row, edges.col)), shape=raw.shape, dtype=np.float32)
    result.sort_indices()
    return result


def _degree_preserving_shuffle(raw, seed):
    """Swap two disjoint undirected edges without loops or duplicate edges."""
    upper = triu(raw, k=1, format="coo")
    edges = np.column_stack((upper.row, upper.col)).astype(np.int64)
    edge_set = {tuple(map(int, edge)) for edge in edges}
    original = edge_set.copy()
    rng = np.random.default_rng(seed)
    target = 5 * len(edges)
    accepted = attempted = 0
    # Finite budget also handles complete graphs where no swap is possible.
    while len(edges) >= 2 and accepted < target and attempted < 50 * len(edges):
        attempted += 1
        left, right = map(int, rng.integers(len(edges), size=2))
        if left == right:
            continue
        a, b = map(int, edges[left])
        c, d = map(int, edges[right])
        if len({a, b, c, d}) < 4:
            continue
        if rng.integers(2):
            c, d = d, c
        first, second = tuple(sorted((a, d))), tuple(sorted((c, b)))
        if first in edge_set or second in edge_set:
            continue
        edge_set.remove(tuple(map(int, edges[left])))
        edge_set.remove(tuple(map(int, edges[right])))
        edge_set.update((first, second))
        edges[left], edges[right] = first, second
        accepted += 1
    if len(edges):
        sources = np.concatenate((edges[:, 0], edges[:, 1]))
        targets = np.concatenate((edges[:, 1], edges[:, 0]))
        shuffled = csr_matrix((np.ones(len(sources), dtype=np.float32), (sources, targets)), shape=raw.shape)
    else:
        shuffled = raw.copy()
    shuffled.sort_indices()
    before = np.asarray(raw.sum(axis=1)).ravel()
    after = np.asarray(shuffled.sum(axis=1)).ravel()
    if shuffled.nnz != raw.nnz or not np.array_equal(before, after):
        raise RuntimeError("Degree-preserving semantic edge swaps violated invariants")
    return shuffled, {"swap_target": target, "swap_attempts": attempted,
                      "accepted_swaps": accepted,
                      "changed_undirected_edges": len(original - edge_set),
                      "per_node_degree_preserved": True,
                      "swap_budget_reached": accepted == target}


def _semantic_components(item_ids, item_vectors, *, neighbors, shuffle_seed, graph_mode):
    ids, vectors = _inputs(item_ids, item_vectors, neighbors, shuffle_seed, graph_mode)
    # Float64 norms prevent under/overflow for otherwise finite float32 inputs.
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    active = np.flatnonzero(norms > 0)
    normalized = np.divide(vectors, norms[:, None], out=np.zeros_like(vectors),
                           where=norms[:, None] > 0)
    rows, columns, values = [], [], []
    # Stable sorting resolves equal cosine values by external item ID.
    for start in range(0, len(active), 256):
        sources = active[start:start + 256]
        cosine = np.clip(normalized[sources] @ normalized[active].T, -1, 1)
        for local, source in enumerate(sources):
            scores = cosine[local]
            scores[active == source] = -np.inf
            targets = np.argsort(-scores, kind="stable")[:neighbors]
            targets = targets[scores[targets] > 0]
            rows.extend([int(source)] * len(targets))
            columns.extend(active[targets].tolist())
            values.extend(scores[targets].tolist())
    directed = csr_matrix((np.asarray(values, dtype=np.float32), (rows, columns)),
                          shape=(len(ids), len(ids)), dtype=np.float32)
    # Union kNN edges; reciprocal edges retain their maximum cosine.
    raw = directed.maximum(directed.T).tocsr()
    raw.sort_indices()
    if graph_mode == "binary":
        raw.data[:] = 1
    original_hash = _sparse_hash(raw)
    permutation = np.arange(len(ids), dtype=np.int64)
    swaps = {}
    if shuffle_seed is not None:
        if graph_mode == "binary":
            raw, swaps = _degree_preserving_shuffle(raw, shuffle_seed)
        else:
            permutation[active] = np.random.default_rng(shuffle_seed).permutation(active)
            raw = raw[permutation][:, permutation].tocsr()
            raw.sort_indices()
    semantic = _normalize(raw)
    weighted_degrees = np.asarray(raw.sum(axis=1, dtype=np.float64)).ravel()
    audit = {
        "schema": "fixed-e5-cosine-item-knn-v1", "items": len(ids), "graph_mode": graph_mode,
        "dimensions": vectors.shape[1], "text_nodes": len(active),
        "missing_text_nodes": len(ids) - len(active), "neighbors": neighbors,
        "directed_edges": int(directed.nnz), "undirected_edges": int(raw.nnz // 2),
        "nonisolated_nodes": int(np.count_nonzero(weighted_degrees)),
        "symmetrization": "union of positive-cosine kNN edges; unit weights" if graph_mode == "binary" else "union of positive-cosine kNN edges; reciprocal maximum",
        "normalization": "D^-1/2 A D^-1/2 weighted degrees",
        "control": "actual_text" if shuffle_seed is None else ("degree_preserving_edge_swaps" if graph_mode == "binary" else "text_node_permutation"),
        "shuffle_seed": shuffle_seed,
        "control_preserves": (["edge_count", "raw_weight_multiset", "per_node_degree", "per_node_weighted_degree", "text_missingness_by_node"]
                              if graph_mode == "binary" else ["edge_count", "raw_weight_multiset", "topology", "unweighted_degree_multiset", "weighted_degree_multiset", "text_missingness_by_node"]),
        "control_limitations": ("Edge swaps preserve per-node raw degrees, but do not preserve connected components or the normalized-edge-weight multiset. A finite swap budget does not imply uniform random graph sampling."
                                if graph_mode == "binary" else "Per-restaurant degrees and text coverage beyond missingness are not matched; this is a node permutation, not edge swaps."),
        "item_ids_sha256": _array_hash(ids), "vectors_sha256": _array_hash(vectors),
        "actual_raw_adjacency_sha256": original_hash,
        "raw_adjacency_sha256": _sparse_hash(raw),
        "normalized_adjacency_sha256": _sparse_hash(semantic),
        "sorted_weighted_degrees_sha256": _array_hash(np.sort(weighted_degrees)),
        "permutation_sha256": _array_hash(permutation),
        "permuted_nodes": int(np.count_nonzero(permutation != np.arange(len(ids)))),
        "source_item_id_by_target": ids[permutation].tolist(),
        **swaps,
    }
    return raw, semantic, audit


def build_semantic_graph(item_ids, item_vectors, *, neighbors=10, shuffle_seed=None, graph_mode="binary"):
    """Return normalized symmetric item adjacency and inspectable provenance.

    Zero vectors receive no edges. Only positive-cosine neighbors are kept.
    ``shuffle_seed`` rewires binary graphs preserving each degree; weighted
    graphs instead permute identities among entities with nonzero text.
    """
    _, semantic, audit = _semantic_components(item_ids, item_vectors,
                                             neighbors=neighbors, shuffle_seed=shuffle_seed,
                                             graph_mode=graph_mode)
    return semantic, audit


class TextItemGraph(LightGCN):
    def __init__(self, config: LightGCNConfig, item_ids, item_vectors, *,
                 neighbors=10, weight=.1, shuffle_seed=None, graph_mode="binary"):
        super().__init__(config)
        self._source_ids, self._source_vectors = _inputs(item_ids, item_vectors,
                                                        neighbors, shuffle_seed, graph_mode)
        if isinstance(weight, bool) or not math.isfinite(weight) or weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        self.neighbors, self.weight, self.shuffle_seed = neighbors, float(weight), shuffle_seed
        self.graph_mode = graph_mode
        self._semantic_raw = self._semantic = None
        self.graph_audit = {}
        self.sampling_history = []
        self.initialization_sha256 = None
        self._sample_digest = None
        self._sample_batches = self._sample_examples = 0
        self._inference_only = False

    @property
    def graphaudit(self):
        return self.graph_audit

    @property
    def semantic_edges(self):
        """Raw undirected edges: external source ID, target ID, graph weight."""
        if self._semantic_raw is None:
            return np.empty((0, 3), dtype=np.float64)
        upper = triu(self._semantic_raw, k=1, format="coo")
        return np.column_stack((self.item_keys[upper.row], self.item_keys[upper.col], upper.data))

    def fit(self, interactions, *, callback=None):
        if self._inference_only:
            raise ValueError("Loaded inference-only checkpoint has no input vector bank; construct a new model to fit")
        rows = tuple(interactions)
        keys = np.asarray(sorted({row.restaurant_id for row in rows}), dtype=np.int64)
        lookup = {int(key): index for index, key in enumerate(self._source_ids)}
        vectors = np.zeros((len(keys), self._source_vectors.shape[1]), dtype=np.float32)
        for index, key in enumerate(keys):
            if int(key) in lookup:
                vectors[index] = self._source_vectors[lookup[int(key)]]
        self._semantic_raw, self._semantic, self.graph_audit = _semantic_components(
            keys, vectors, neighbors=self.neighbors, shuffle_seed=self.shuffle_seed,
            graph_mode=self.graph_mode)
        self.graph_audit.update({"propagation_weight": self.weight,
                                 "source_bank_items": len(self._source_ids),
                                 "catalog_items_absent_from_bank": int(sum(int(key) not in lookup for key in keys)),
                                 "propagation": "layer_next = S_visit @ layer + weight * blockdiag(0_users, S_text) @ layer; average layers 0..L"})
        self.sampling_history = []
        self.initialization_sha256 = None
        self._sample_digest = hashlib.sha256()
        self._sample_batches = self._sample_examples = 0

        def record_epoch(stats, model):
            self.sampling_history.append({"epoch": stats.epoch,
                                          "batches": self._sample_batches,
                                          "examples": self._sample_examples,
                                          "ordered_bpr_triples_sha256": self._sample_digest.hexdigest()})
            self._sample_digest = hashlib.sha256()
            self._sample_batches = self._sample_examples = 0
            return callback(stats, model) if callback is not None else None

        try:
            return super().fit(rows, callback=record_epoch)
        finally:
            self._sample_digest = None

    def _loss_and_gradient(self, u, i, j):
        if self._sample_digest is not None:
            if self.initialization_sha256 is None:
                self.initialization_sha256 = _array_hash(self._ego)
            for values in (u, i, j):
                self._sample_digest.update(_array_hash(values).encode())
            self._sample_batches += 1
            self._sample_examples += len(u)
        return super()._loss_and_gradient(u, i, j)

    def _propagate(self, embedding):
        # Call the original implementation for exact zero-weight preservation.
        if not self.weight or self.config.layers == 0 or self._semantic is None or not self._semantic.nnz:
            return super()._propagate(embedding)
        assert self._adjacency is not None
        n_users = len(self.user_ids)
        if self._semantic.shape != (len(self.item_ids), len(self.item_ids)):
            raise ValueError("Semantic and visitation item catalog order differ")
        total = embedding.copy()
        layer = embedding
        for _ in range(self.config.layers):
            following = self._adjacency @ layer
            following[n_users:] += np.float32(self.weight) * (self._semantic @ layer[n_users:])
            total += following
            layer = following
        return total / np.float32(self.config.layers + 1)

    def manifest(self):
        return {"schema": SCHEMA, "graph_config": self.config.to_dict(),
                "neighbors": self.neighbors, "weight": self.weight,
                "graph_mode": self.graph_mode,
                "shuffle_seed": self.shuffle_seed, "edge_count": self.edge_count,
                "graph_audit": self.graph_audit,
                "initialization_sha256": self.initialization_sha256,
                "sampling_history": self.sampling_history,
                "history": [asdict(stats) for stats in self.history],
                "inference_only": True}

    def save(self, path):
        """Atomically persist inference state and graph/sampling provenance."""
        if self._adjacency is None or self._semantic is None or not len(self._ego):
            raise ValueError("Fit the model before saving")
        arrays = {"ego": self._ego,
                  "user_keys": np.asarray(sorted(self.user_ids, key=self.user_ids.get), dtype=np.int64),
                  "item_keys": self.item_keys,
                  "metadata": np.frombuffer(json.dumps(self.manifest(), allow_nan=False).encode(), dtype=np.uint8)}
        for name, matrix in (("visit", self._adjacency), ("semantic", self._semantic),
                             ("semantic_raw", self._semantic_raw)):
            arrays.update({name + "_data": matrix.data, name + "_indices": matrix.indices,
                           name + "_indptr": matrix.indptr})
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path):
        """Load a non-pickle, inference-only checkpoint with matrix validation."""
        with np.load(path, allow_pickle=False) as archive:
            metadata_array = archive["metadata"]
            if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                raise ValueError("Invalid checkpoint metadata")
            metadata = json.loads(metadata_array.tobytes())
            if metadata.get("schema") != SCHEMA:
                raise ValueError("Unsupported text item graph checkpoint schema")
            user_keys, item_keys = archive["user_keys"], archive["item_keys"]
            for keys in (user_keys, item_keys):
                if (keys.dtype != np.int64 or keys.ndim != 1 or not len(keys)
                        or np.any(keys <= 0) or np.any(keys[1:] <= keys[:-1])):
                    raise ValueError("Checkpoint keys must be sorted unique positive int64 vectors")
            model = cls(LightGCNConfig(**metadata["graph_config"]), item_keys,
                        np.zeros((len(item_keys), 1), dtype=np.float32),
                        neighbors=metadata["neighbors"], weight=metadata["weight"],
                        shuffle_seed=metadata["shuffle_seed"], graph_mode=metadata["graph_mode"])
            nodes = len(user_keys) + len(item_keys)
            ego = archive["ego"]
            if (ego.dtype != np.float32 or ego.shape != (nodes, model.config.dimension)
                    or not np.isfinite(ego).all()):
                raise ValueError("Invalid ego embeddings in checkpoint")

            def matrix(name, size):
                data, indices, indptr = (archive[name + "_" + suffix] for suffix in ("data", "indices", "indptr"))
                if (data.dtype != np.float32 or data.ndim != 1 or not np.isfinite(data).all()
                        or np.any(data <= 0) or indices.dtype.kind != "i" or indices.ndim != 1
                        or indptr.dtype.kind != "i" or indptr.shape != (size + 1,)
                        or len(data) != len(indices) or indptr[0] != 0 or indptr[-1] != len(data)
                        or np.any(indptr[1:] < indptr[:-1]) or np.any(indptr < 0)
                        or np.any(indptr > len(data)) or np.any(indices < 0) or np.any(indices >= size)):
                    raise ValueError("Invalid checkpoint sparse matrix: " + name)
                result = csr_matrix((data.copy(), indices.copy(), indptr.copy()), shape=(size, size))
                if (not result.has_canonical_format or np.any(result.diagonal() != 0)
                        or (result - result.T).nnz):
                    raise ValueError("Checkpoint graph must be canonical, symmetric and self-edge free: " + name)
                return result

            model._adjacency = matrix("visit", nodes)
            model._semantic = matrix("semantic", len(item_keys))
            model._semantic_raw = matrix("semantic_raw", len(item_keys))
            if model._adjacency[:len(user_keys), :len(user_keys)].nnz or model._adjacency[len(user_keys):, len(user_keys):].nnz:
                raise ValueError("Visit graph must be bipartite")
            model.graph_audit = metadata["graph_audit"]
            if (_sparse_hash(model._semantic) != model.graph_audit["normalized_adjacency_sha256"]
                    or _sparse_hash(model._semantic_raw) != model.graph_audit["raw_adjacency_sha256"]
                    or _array_hash(item_keys) != model.graph_audit["item_ids_sha256"]):
                raise ValueError("Checkpoint semantic graph provenance hashes differ")
            model.user_ids = {int(key): index for index, key in enumerate(user_keys)}
            model.item_ids = {int(key): index for index, key in enumerate(item_keys)}
            model.item_keys, model._ego = item_keys.copy(), ego.copy()
            model.edge_count = int(metadata["edge_count"])
            model.history = [EpochStats(**value) for value in metadata["history"]]
            model.initialization_sha256 = metadata["initialization_sha256"]
            model.sampling_history = metadata["sampling_history"]
            model._inference_only = True
            return model
