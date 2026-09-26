"""Small full-batch LightGCN with BPR loss for temporal retrieval snapshots.

The normalized bipartite adjacency has no learned graph convolution weights. We
average layers 0..L and differentiate BPR through every propagation layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit

from rating_recsys.datasets.models import Interaction
from rating_recsys.experiments.models import Candidate, RecommendationQuery
from rating_recsys.retrieval.baselines import (
    ITEM_ITEM, POPULARITY, RetrievalContext, RetrievalResult,
)

LIGHTGCN = "lightgcn"


@dataclass(frozen=True, slots=True)
class LightGCNConfig:
    dimension: int = 32
    layers: int = 2
    epochs: int = 80
    learning_rate: float = 0.03
    regularization: float = 0.0001
    seed: int = 42


class LightGCN:
    def __init__(self, config: LightGCNConfig = LightGCNConfig()) -> None:
        self.config = config
        self.user_ids: dict[int, int] = {}
        self.item_ids: dict[int, int] = {}
        self.user_embeddings = np.empty((0, config.dimension), dtype=np.float32)
        self.item_embeddings = np.empty((0, config.dimension), dtype=np.float32)
        self.edge_count = 0
        self.final_loss = 0.0

    def fit(self, interactions: Iterable[Interaction]) -> LightGCN:
        edges = sorted({(row.user_id, row.restaurant_id) for row in interactions})
        self.edge_count = len(edges)
        if len(edges) < 2:
            return self
        users = sorted({user for user, _ in edges})
        items = sorted({item for _, item in edges})
        if len(items) < 2:
            return self
        self.user_ids = {key: index for index, key in enumerate(users)}
        self.item_ids = {key: index for index, key in enumerate(items)}
        n_users, n_items = len(users), len(items)
        positives = np.asarray(
            [(self.user_ids[user], self.item_ids[item]) for user, item in edges],
            dtype=np.int32,
        )
        row = np.concatenate((positives[:, 0], n_users + positives[:, 1]))
        col = np.concatenate((n_users + positives[:, 1], positives[:, 0]))
        degree = np.bincount(row, minlength=n_users + n_items).astype(np.float32)
        weight = np.reciprocal(np.sqrt(degree[row] * degree[col]))
        adjacency = csr_matrix(
            (weight, (row, col)), shape=(n_users + n_items,) * 2,
            dtype=np.float32,
        )
        seen = [set() for _ in users]
        for user, item in positives:
            seen[int(user)].add(int(item))
        trainable = np.fromiter(
            (len(seen[int(user)]) < n_items for user in positives[:, 0]),
            count=len(positives), dtype=bool,
        )
        positives = positives[trainable]
        if not len(positives):
            return self
        rng = np.random.default_rng(self.config.seed)
        initial = rng.normal(
            0, 0.1, size=(n_users + n_items, self.config.dimension)
        ).astype(np.float32)
        first_moment = np.zeros_like(initial)
        second_moment = np.zeros_like(initial)
        for epoch in range(1, self.config.epochs + 1):
            sampled = rng.integers(n_items, size=len(positives), dtype=np.int32)
            bad = np.fromiter(
                (int(item) in seen[int(user)] for (user, _), item in zip(positives, sampled)),
                count=len(sampled), dtype=bool,
            )
            while bad.any():
                sampled[bad] = rng.integers(n_items, size=int(bad.sum()), dtype=np.int32)
                bad = np.fromiter(
                    (int(item) in seen[int(user)] for (user, _), item in zip(positives, sampled)),
                    count=len(sampled), dtype=bool,
                )
            embedding = self._propagate(adjacency, initial)
            u = positives[:, 0]
            i = n_users + positives[:, 1]
            j = n_users + sampled
            margin = np.sum(embedding[u] * (embedding[i] - embedding[j]), axis=1)
            self.final_loss = float(np.logaddexp(0, -margin).mean())
            slope = (-expit(-margin) / len(margin)).astype(np.float32)
            gradient = np.zeros_like(embedding)
            np.add.at(gradient, u, slope[:, None] * (embedding[i] - embedding[j]))
            np.add.at(gradient, i, slope[:, None] * embedding[u])
            np.add.at(gradient, j, -slope[:, None] * embedding[u])
            gradient = self._propagate(adjacency, gradient)
            gradient += self.config.regularization * initial
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
            adjusted = (first_moment / (1 - 0.9**epoch)) / (
                np.sqrt(second_moment / (1 - 0.999**epoch)) + 1e-8
            )
            initial -= self.config.learning_rate * adjusted
        final = self._propagate(adjacency, initial)
        self.user_embeddings = final[:n_users]
        self.item_embeddings = final[n_users:]
        return self

    def _propagate(self, adjacency: csr_matrix, embedding: np.ndarray) -> np.ndarray:
        result = embedding.copy()
        layer = embedding
        for _ in range(self.config.layers):
            layer = adjacency @ layer
            result += layer
        return result / (self.config.layers + 1)

    def score(self, user_id: int, eligible_items: Iterable[int]) -> dict[int, float]:
        user_index = self.user_ids.get(user_id)
        if user_index is None:
            return {}
        ids = [item for item in eligible_items if item in self.item_ids]
        if not ids:
            return {}
        item_indices = [self.item_ids[item] for item in ids]
        scores = self.item_embeddings[item_indices] @ self.user_embeddings[user_index]
        return dict(zip(ids, map(float, scores)))


class LightGCNRRF:
    """Fuse the Top-K C0, C1 and LightGCN ranks without source quotas."""

    def __init__(self, *, candidate_k: int = 100, rrf_constant: int = 60) -> None:
        if candidate_k < 1 or rrf_constant < 1:
            raise ValueError("candidate_k and rrf_constant must be positive")
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant

    def fuse(
        self,
        query: RecommendationQuery,
        result: RetrievalResult,
        context: RetrievalContext,
        model: LightGCN | None,
    ) -> tuple[tuple[Candidate, ...], tuple[int, ...]]:
        popularity = {row.restaurant_id: row.candidate_rank for row in result.popularity}
        item_item = {row.restaurant_id: row.candidate_rank for row in result.item_item}
        scores = model.score(query.user_id, result.eligible_catalog) if model else {}
        lightgcn_ids = tuple(
            item for item, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))
            [: self.candidate_k]
        )
        lightgcn = {item: index for index, item in enumerate(lightgcn_ids, 1)}
        sources = ((POPULARITY, popularity), (ITEM_ITEM, item_item), (LIGHTGCN, lightgcn))
        ids = set(popularity) | set(item_item) | set(lightgcn)
        rrf = {
            item: sum(1 / (self.rrf_constant + ranks[item])
                      for _, ranks in sources if item in ranks)
            for item in ids
        }
        ordered = sorted(ids, key=lambda item: (-rrf[item], item))[: self.candidate_k]
        candidates = tuple(
            Candidate(
                query_id=query.query_id, user_id=query.user_id, restaurant_id=item,
                restaurant_name=context.item_names[item], region=context.item_regions[item],
                candidate_sources=tuple(name for name, ranks in sources if item in ranks),
                source_scores={LIGHTGCN: scores.get(item, 0.0)},
                source_ranks={name: ranks[item] for name, ranks in sources if item in ranks},
                rrf_score=rrf[item], candidate_rank=index,
            )
            for index, item in enumerate(ordered, 1)
        )
        return candidates, lightgcn_ids
