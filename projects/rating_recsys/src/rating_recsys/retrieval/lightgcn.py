"""LightGCN (He et al., 2020) with BPR loss in NumPy/SciPy.

The model has no weights besides the layer-0 ("ego") embeddings of users and
restaurants. Final embeddings are the mean of layers 0..L, where each layer is
the previous one multiplied by the symmetric normalized bipartite adjacency
``D^-1/2 A D^-1/2``. Because that matrix is symmetric, the gradient with respect
to the ego embeddings is the same propagation applied to the gradient with
respect to the final embeddings.

Training uses mini-batch BPR over (user, visited restaurant, uniformly sampled
unvisited restaurant) triples, L2 on the batch ego embeddings as in the paper,
and Adam. Initialisation, shuffling and negative sampling are seeded, and
edges are sorted first, so the same edges and config give the same embeddings
regardless of input order.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import expit

from rating_recsys.datasets.models import Interaction


LIGHTGCN = "lightgcn"


@dataclass(frozen=True, slots=True)
class LightGCNConfig:
    dimension: int = 64
    layers: int = 2
    epochs: int = 100
    batch_size: int = 2048
    learning_rate: float = 0.005
    regularization: float = 1e-4
    init_std: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if self.dimension < 1 or self.layers < 0 or self.epochs < 1:
            raise ValueError("dimension and epochs must be positive, layers >= 0")
        if self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("batch_size and learning_rate must be positive")
        if self.regularization < 0 or self.init_std <= 0:
            raise ValueError("regularization must be >= 0 and init_std > 0")

    @property
    def name(self) -> str:
        return (
            f"d{self.dimension}_l{self.layers}_lr{self.learning_rate:g}"
            f"_reg{self.regularization:g}_b{self.batch_size}"
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpochStats:
    epoch: int
    bpr_loss: float
    reg_loss: float
    seconds: float


# ``callback(epoch_stats, model)`` runs after every epoch. Returning True stops
# training after that epoch.
EpochCallback = Callable[[EpochStats, "LightGCN"], bool | None]


class LightGCN:
    def __init__(self, config: LightGCNConfig = LightGCNConfig()) -> None:
        self.config = config
        self.user_ids: dict[int, int] = {}
        self.item_ids: dict[int, int] = {}
        self.item_keys = np.empty(0, dtype=np.int64)
        self.history: list[EpochStats] = []
        self.edge_count = 0
        self._ego = np.empty((0, config.dimension), dtype=np.float32)
        self._adjacency: csr_matrix | None = None
        self._final: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        interactions: Iterable[Interaction],
        *,
        callback: EpochCallback | None = None,
    ) -> LightGCN:
        config = self.config
        edges = sorted({(row.user_id, row.restaurant_id) for row in interactions})
        users = sorted({user for user, _ in edges})
        items = sorted({item for _, item in edges})
        if len(edges) < 2 or len(items) < 2:
            raise ValueError("LightGCN needs at least two edges and two restaurants")
        self.user_ids = {key: index for index, key in enumerate(users)}
        self.item_ids = {key: index for index, key in enumerate(items)}
        self.item_keys = np.asarray(items, dtype=np.int64)
        self.edge_count = len(edges)
        self.history = []
        n_users, n_items = len(users), len(items)
        edge_users = np.fromiter(
            (self.user_ids[u] for u, _ in edges), dtype=np.int64, count=len(edges)
        )
        edge_items = np.fromiter(
            (self.item_ids[i] for _, i in edges), dtype=np.int64, count=len(edges)
        )
        self._adjacency = _normalized_adjacency(edge_users, edge_items, n_users, n_items)
        # Sorted (user, item) keys of every visit, for rejecting sampled negatives.
        positive_keys = np.unique(edge_users * n_items + edge_items)

        # A user who has visited every restaurant has no negative to sample.
        per_user = np.bincount(edge_users, minlength=n_users)
        trainable = per_user[edge_users] < n_items
        edge_users, edge_items = edge_users[trainable], edge_items[trainable]
        if not len(edge_users):
            raise ValueError("No user has an unvisited restaurant to sample")

        rng = np.random.default_rng(config.seed)
        self._ego = rng.normal(
            0.0, config.init_std, size=(n_users + n_items, config.dimension)
        ).astype(np.float32)
        first = np.zeros_like(self._ego)
        second = np.zeros_like(self._ego)
        step = 0
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        batch = config.batch_size
        count = len(edge_users)

        for epoch in range(1, config.epochs + 1):
            started = time.perf_counter()
            order = rng.permutation(count)
            negatives = _sample_negatives(rng, edge_users, n_items, positive_keys)
            bpr_total = reg_total = 0.0
            for offset in range(0, count, batch):
                index = order[offset : offset + batch]
                bpr_sum, reg_sum, gradient = self._loss_and_gradient(
                    edge_users[index],
                    n_users + edge_items[index],
                    n_users + negatives[index],
                )
                bpr_total += bpr_sum
                reg_total += reg_sum

                step += 1
                first *= beta1
                first += (1 - beta1) * gradient
                second *= beta2
                second += (1 - beta2) * np.square(gradient)
                step_size = float(
                    config.learning_rate
                    * np.sqrt(1 - beta2**step)
                    / (1 - beta1**step)
                )
                self._ego -= step_size * first / (np.sqrt(second) + epsilon)

            self._final = None
            stats = EpochStats(
                epoch=epoch,
                bpr_loss=bpr_total / count,
                reg_loss=reg_total / count,
                seconds=time.perf_counter() - started,
            )
            self.history.append(stats)
            if callback is not None and callback(stats, self):
                break
        self._final = None
        return self

    def _loss_and_gradient(
        self, u: np.ndarray, i: np.ndarray, j: np.ndarray
    ) -> tuple[float, float, np.ndarray]:
        """Batch BPR and L2 sums and the gradient of their batch mean.

        ``u``, ``i`` and ``j`` are node indices (items offset by the number of
        users). The minimised batch objective is
        ``(sum softplus(-margin) + reg / 2 * sum ||ego_row||^2) / batch_size``.
        """

        size = len(u)
        final = self._propagate(self._ego)
        eu, ei, ej = final[u], final[i], final[j]
        margin = np.einsum("bd,bd->b", eu, ei - ej)
        bpr_sum = float(np.logaddexp(0.0, -margin).sum())
        slope = (-expit(-margin) / size).astype(final.dtype, copy=False)[:, None]

        grad_final = np.zeros_like(final)
        np.add.at(grad_final, u, slope * (ei - ej))
        np.add.at(grad_final, i, slope * eu)
        np.add.at(grad_final, j, -slope * eu)
        # The normalized adjacency is symmetric, so backpropagating through
        # the layer average is the same propagation.
        gradient = self._propagate(grad_final)

        reg_sum = 0.0
        regularization = self.config.regularization
        if regularization:
            rows = np.concatenate([u, i, j])
            ego_rows = self._ego[rows]
            reg_sum = float(0.5 * regularization * np.square(ego_rows).sum())
            np.add.at(gradient, rows, (regularization / size) * ego_rows)
        return bpr_sum, reg_sum, gradient

    def _propagate(self, embedding: np.ndarray) -> np.ndarray:
        if self.config.layers == 0:
            return embedding.copy()
        assert self._adjacency is not None
        total = embedding.copy()
        layer = embedding
        for _ in range(self.config.layers):
            layer = self._adjacency @ layer
            total += layer
        return total / np.float32(self.config.layers + 1)

    # ------------------------------------------------------------ inference
    @property
    def final_embeddings(self) -> np.ndarray:
        if self._final is None:
            self._final = self._propagate(self._ego)
        return self._final

    @property
    def user_embeddings(self) -> np.ndarray:
        return self.final_embeddings[: len(self.user_ids)]

    @property
    def item_embeddings(self) -> np.ndarray:
        return self.final_embeddings[len(self.user_ids) :]

    def recommend(
        self,
        user_ids: Sequence[int],
        exclude: Mapping[int, Iterable[int]],
        k: int,
        *,
        chunk_size: int = 1024,
    ) -> dict[int, tuple[int, ...]]:
        """Top-``k`` restaurant ids per user, excluding ``exclude[user]``.

        Ties are broken by restaurant id. Users that were not in the training
        graph get an empty tuple.
        """

        ranked = self.top_k(
            user_ids, [exclude.get(user, ()) for user in user_ids], k, chunk_size=chunk_size
        )
        return dict(zip(user_ids, ranked))

    def top_k(
        self,
        user_ids: Sequence[int],
        excludes: Sequence[Iterable[int]],
        k: int,
        *,
        chunk_size: int = 1024,
    ) -> list[tuple[int, ...]]:
        """Top-``k`` restaurant ids for each (user, excluded ids) pair, in order.

        The same user may appear several times with different exclusions.
        Ties are broken by restaurant id; unknown users get an empty tuple.
        """

        if len(user_ids) != len(excludes):
            raise ValueError("user_ids and excludes must have the same length")
        result: list[tuple[int, ...]] = [()] * len(user_ids)
        known = [index for index, user in enumerate(user_ids) if user in self.user_ids]
        users, items = self.user_embeddings, self.item_embeddings
        for start in range(0, len(known), chunk_size):
            chunk = known[start : start + chunk_size]
            scores = users[[self.user_ids[user_ids[index]] for index in chunk]] @ items.T
            for row, index in enumerate(chunk):
                seen = [
                    self.item_ids[item] for item in excludes[index] if item in self.item_ids
                ]
                scores[row, seen] = -np.inf
            # Items are indexed in restaurant-id order, so a stable sort on the
            # negated score breaks ties by restaurant id.
            order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            for row, index in enumerate(chunk):
                valid = order[row][np.isfinite(scores[row, order[row]])]
                result[index] = tuple(int(x) for x in self.item_keys[valid])
        return result

    def score(self, user_id: int, restaurant_ids: Iterable[int]) -> dict[int, float]:
        user = self.user_ids.get(user_id)
        if user is None:
            return {}
        ids = [item for item in restaurant_ids if item in self.item_ids]
        if not ids:
            return {}
        values = self.item_embeddings[[self.item_ids[i] for i in ids]] @ (
            self.user_embeddings[user]
        )
        return dict(zip(ids, map(float, values)))


def _normalized_adjacency(
    edge_users: np.ndarray, edge_items: np.ndarray, n_users: int, n_items: int
) -> csr_matrix:
    rows = np.concatenate([edge_users, n_users + edge_items])
    cols = np.concatenate([n_users + edge_items, edge_users])
    degree = np.bincount(rows, minlength=n_users + n_items).astype(np.float64)
    weights = 1.0 / np.sqrt(degree[rows] * degree[cols])
    size = n_users + n_items
    return csr_matrix(
        (weights.astype(np.float32), (rows, cols)), shape=(size, size), dtype=np.float32
    )


def _sample_negatives(
    rng: np.random.Generator,
    edge_users: np.ndarray,
    n_items: int,
    positive_keys: np.ndarray,
) -> np.ndarray:
    """One uniformly sampled unvisited restaurant per training edge."""

    negatives = rng.integers(n_items, size=len(edge_users), dtype=np.int64)
    bad = _is_positive(edge_users * n_items + negatives, positive_keys)
    while bad.any():
        negatives[bad] = rng.integers(n_items, size=int(bad.sum()), dtype=np.int64)
        bad[bad] = _is_positive(edge_users[bad] * n_items + negatives[bad], positive_keys)
    return negatives


def _is_positive(keys: np.ndarray, sorted_positive_keys: np.ndarray) -> np.ndarray:
    position = np.searchsorted(sorted_positive_keys, keys)
    position[position == len(sorted_positive_keys)] = 0
    return sorted_positive_keys[position] == keys
