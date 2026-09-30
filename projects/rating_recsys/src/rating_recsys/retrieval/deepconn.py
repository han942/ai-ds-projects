"""DeepCoNN (Zheng et al., 2017) as a Stage 1 candidate source, in PyTorch.

Two CNN towers read review documents: a user document (the user's own past
reviews) and a restaurant document (every past review of the restaurant). Each
tower is token embedding -> 1D convolution -> ReLU -> max over time -> fully
connected layer. A factorization machine on the two concatenated latent
vectors gives the score.

Differences from the paper, all because of this data or the retrieval task:

- Tokens are characters (mostly Hangul syllables) with embeddings learned from
  scratch. There is no Korean tokenizer or pretrained vector in the project.
- Documents list reviews newest first, each cut to ``max_review_length`` and
  followed by a SEP token, then cut to ``doc_length``. Max pooling skips
  padding, so a short document carries no length signal.
- When training on a visit (u, i), that visit's own review is removed from both
  documents. At evaluation the target visit is in the future and never in a
  document; keeping it in training would teach the model to match identical
  text (Catherine & Cohen, 2017, "TransNets").
- ``objective="mse"`` is the paper's rating regression on observed visits.
  ``objective="bpr"`` ranks a visited restaurant above a uniformly sampled
  unvisited one, the usual adaptation for top-K retrieval.
- ``latent_activation="relu"`` is the paper's tower output. On this data most
  user vectors die at all zeros within two BPR epochs (the retrieval then
  ranks by the restaurant term alone), so ``"linear"`` drops that last ReLU.
  ``DeepCoNN.dead_latent_rates`` reports the share of all-zero vectors.
- For retrieval the FM is split into a user term, a restaurant term and a dot
  product (see ``DeepCoNNNet.retrieval_parts``), so scoring every restaurant is
  one matrix product. The user term does not change a user's order.

Only reviews of the interactions passed to ``fit`` are read, so a model fitted
on interactions up to a cutoff never sees later review text.
"""

from __future__ import annotations

import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rating_recsys.datasets.models import Interaction
from rating_recsys.retrieval.lightgcn import _sample_negatives


DEEPCONN = "deepconn"
PAD, UNK, SEP = 0, 1, 2
OBJECTIVES = ("bpr", "mse")
ACTIVATIONS = ("relu", "linear")


@dataclass(frozen=True, slots=True)
class DeepCoNNConfig:
    objective: str = "bpr"
    # Activation after each tower's FC layer. "relu" is the paper; "linear"
    # keeps negative values so a latent vector cannot die at all zeros.
    latent_activation: str = "relu"
    vocab_size: int = 3000
    min_token_count: int = 3
    embedding_dim: int = 64
    num_filters: int = 100
    kernel_size: int = 3
    latent_dim: int = 32
    fm_k: int = 8
    doc_length: int = 400
    max_review_length: int = 150
    dropout: float = 0.5
    learning_rate: float = 0.002
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 15
    seed: int = 42
    threads: int = 6

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.latent_activation not in ACTIVATIONS:
            raise ValueError(f"latent_activation must be one of {ACTIVATIONS}")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        positive = (
            self.vocab_size, self.embedding_dim, self.num_filters, self.latent_dim,
            self.fm_k, self.doc_length, self.max_review_length, self.batch_size,
            self.epochs, self.threads,
        )
        if min(positive) < 1 or self.vocab_size <= SEP + 1:
            raise ValueError("sizes, epochs and threads must be positive")
        if not 0 <= self.dropout < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid dropout, learning_rate or weight_decay")

    @property
    def name(self) -> str:
        return (
            f"{self.objective}_{self.latent_activation}_k{self.kernel_size}"
            f"_f{self.num_filters}_d{self.latent_dim}_doc{self.doc_length}"
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpochStats:
    epoch: int
    loss: float
    seconds: float


EpochCallback = Callable[[EpochStats, "DeepCoNN"], bool | None]


# ---------------------------------------------------------------------------
# Text -> token documents
# ---------------------------------------------------------------------------


def normalize(text: str | None) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").lower().split())


class CharVocabulary:
    """Character ids; 0 = PAD, 1 = unknown, 2 = review separator."""

    def __init__(self, characters: Sequence[str]) -> None:
        self.characters = tuple(characters)
        self._index = {char: index for index, char in enumerate(self.characters, SEP + 1)}

    @classmethod
    def build(cls, texts: Iterable[str], *, size: int, min_count: int) -> CharVocabulary:
        counts = Counter(char for text in texts for char in text)
        kept = sorted(
            (item for item in counts.items() if item[1] >= min_count),
            key=lambda item: (-item[1], item[0]),
        )[: size - SEP - 1]
        return cls([char for char, _ in kept])

    def __len__(self) -> int:
        return len(self.characters) + SEP + 1

    def encode(self, text: str, limit: int) -> np.ndarray:
        return np.fromiter(
            (self._index.get(char, UNK) for char in text[:limit]), dtype=np.int64
        )


class EntityDocuments:
    """Token documents of users or restaurants, newest review first.

    Each document keeps a little more than ``doc_length`` tokens and the span
    of every review in it, so a review can be removed and the gap refilled by
    older reviews.
    """

    def __init__(
        self,
        grouped: Mapping[int, list[tuple[tuple, int, np.ndarray]]],
        *,
        doc_length: int,
        max_review_length: int,
    ) -> None:
        self.doc_length = doc_length
        capacity = doc_length + max_review_length + 1
        self._tokens: dict[int, np.ndarray] = {}
        self._spans: dict[int, dict[int, tuple[int, int]]] = {}
        for key, reviews in grouped.items():
            parts: list[np.ndarray] = []
            spans: dict[int, tuple[int, int]] = {}
            length = 0
            for _, review_id, tokens in sorted(reviews, key=lambda row: row[0], reverse=True):
                if length >= capacity:
                    break
                if not len(tokens):
                    continue
                segment = np.append(tokens, SEP)
                spans[review_id] = (length, length + len(segment))
                parts.append(segment)
                length += len(segment)
            self._tokens[key] = (
                np.concatenate(parts)[:capacity] if parts else np.empty(0, np.int64)
            )
            self._spans[key] = spans

    def __contains__(self, key: int) -> bool:
        return key in self._tokens

    def tokens(self, key: int, exclude_review: int | None = None) -> np.ndarray:
        tokens = self._tokens.get(key, np.empty(0, np.int64))
        span = self._spans.get(key, {}).get(exclude_review) if exclude_review is not None else None
        if span is not None:
            tokens = np.concatenate([tokens[: span[0]], tokens[span[1] :]])
        return tokens[: self.doc_length]

    def batch(
        self, keys: Sequence[int], exclude_reviews: Sequence[int] | None = None
    ) -> torch.Tensor:
        out = np.zeros((len(keys), self.doc_length), dtype=np.int64)
        for row, key in enumerate(keys):
            tokens = self.tokens(key, exclude_reviews[row] if exclude_reviews is not None else None)
            out[row, : len(tokens)] = tokens
        return torch.from_numpy(out)


def build_documents(
    interactions: Sequence[Interaction],
    texts: Mapping[int, str | None],
    vocabulary: CharVocabulary,
    config: DeepCoNNConfig,
) -> tuple[EntityDocuments, EntityDocuments]:
    by_user: dict[int, list] = defaultdict(list)
    by_item: dict[int, list] = defaultdict(list)
    for row in interactions:
        tokens = vocabulary.encode(normalize(texts.get(row.review_id)), config.max_review_length)
        entry = ((row.event_date, row.review_id), row.review_id, tokens)
        by_user[row.user_id].append(entry)
        by_item[row.restaurant_id].append(entry)
    options = {"doc_length": config.doc_length, "max_review_length": config.max_review_length}
    return EntityDocuments(by_user, **options), EntityDocuments(by_item, **options)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class _Tower(nn.Module):
    def __init__(self, config: DeepCoNNConfig) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            config.embedding_dim, config.num_filters, config.kernel_size,
            padding=config.kernel_size // 2,
        )
        self.fc = nn.Linear(config.num_filters, config.latent_dim)
        self.dropout = nn.Dropout(config.dropout)
        self.relu_latent = config.latent_activation == "relu"

    def forward(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.conv(embedded.transpose(1, 2)))
        # ReLU output is >= 0, so zeroing padded positions leaves the max over
        # real tokens unchanged; an empty document pools to zero.
        pooled = hidden.masked_fill(~mask[:, None, :], 0.0).max(dim=2).values
        latent = self.fc(pooled)
        return self.dropout(F.relu(latent) if self.relu_latent else latent)


class DeepCoNNNet(nn.Module):
    def __init__(self, vocab_size: int, config: DeepCoNNConfig) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=PAD)
        self.user_tower = _Tower(config)
        self.item_tower = _Tower(config)
        self.fm_linear = nn.Linear(2 * config.latent_dim, 1)
        self.fm_v = nn.Parameter(torch.randn(2 * config.latent_dim, config.fm_k) * 0.05)

    def encode_users(self, documents: torch.Tensor) -> torch.Tensor:
        return self.user_tower(self.embedding(documents), documents != PAD)

    def encode_items(self, documents: torch.Tensor) -> torch.Tensor:
        return self.item_tower(self.embedding(documents), documents != PAD)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Second-order FM on [user latent; item latent], one score per row."""

        z = torch.cat([users, items], dim=1)
        interactions = z @ self.fm_v
        squared = (z * z) @ (self.fm_v * self.fm_v)
        return self.fm_linear(z).squeeze(1) + 0.5 * (interactions**2 - squared).sum(1)

    def retrieval_parts(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``score(u, i) = user_bias[u] + item_bias[i] + user_vec[u] · item_vec[i]``."""

        d = self.latent_dim
        weight, bias = self.fm_linear.weight[0], self.fm_linear.bias[0]
        v_user, v_item = self.fm_v[:d], self.fm_v[d:]

        def side(x, w, v):
            projected = x @ v
            return x @ w + 0.5 * ((projected**2).sum(1) - ((x * x) @ (v * v)).sum(1)), projected

        user_bias, user_vec = side(users, weight[:d], v_user)
        item_bias, item_vec = side(items, weight[d:], v_item)
        return user_bias + bias, user_vec, item_bias, item_vec


# ---------------------------------------------------------------------------
# Training and retrieval
# ---------------------------------------------------------------------------


class DeepCoNN:
    def __init__(self, config: DeepCoNNConfig = DeepCoNNConfig()) -> None:
        self.config = config
        self.history: list[EpochStats] = []
        self.user_ids: dict[int, int] = {}
        self.item_ids: dict[int, int] = {}
        self.item_keys = np.empty(0, dtype=np.int64)
        self.edge_count = 0
        self.vocabulary: CharVocabulary | None = None
        self.net: DeepCoNNNet | None = None
        self._user_docs: EntityDocuments | None = None
        self._item_docs: EntityDocuments | None = None
        self._cache: tuple | None = None

    def fit(
        self,
        interactions: Iterable[Interaction],
        texts: Mapping[int, str | None],
        *,
        callback: EpochCallback | None = None,
    ) -> DeepCoNN:
        config = self.config
        rows = sorted(
            {(r.user_id, r.restaurant_id): r for r in interactions}.values(),
            key=lambda r: (r.user_id, r.restaurant_id),
        )
        users = sorted({r.user_id for r in rows})
        items = sorted({r.restaurant_id for r in rows})
        if len(rows) < 2 or len(items) < 2:
            raise ValueError("DeepCoNN needs at least two visits and two restaurants")
        self.user_ids = {key: index for index, key in enumerate(users)}
        self.item_ids = {key: index for index, key in enumerate(items)}
        self.item_keys = np.asarray(items, dtype=np.int64)
        self.edge_count = len(rows)
        self.history = []
        self.vocabulary = CharVocabulary.build(
            (normalize(texts.get(r.review_id)) for r in rows),
            size=config.vocab_size,
            min_count=config.min_token_count,
        )
        self._user_docs, self._item_docs = build_documents(rows, texts, self.vocabulary, config)

        edge_users = np.asarray([self.user_ids[r.user_id] for r in rows], dtype=np.int64)
        edge_items = np.asarray([self.item_ids[r.restaurant_id] for r in rows], dtype=np.int64)
        user_keys = np.asarray([r.user_id for r in rows], dtype=np.int64)
        review_ids = np.asarray([r.review_id for r in rows], dtype=np.int64)
        ratings = torch.tensor([r.rating for r in rows], dtype=torch.float32)
        n_items = len(items)
        positive_keys = np.unique(edge_users * n_items + edge_items)
        trainable = np.bincount(edge_users)[edge_users] < n_items
        if config.objective == "bpr" and not trainable.any():
            raise ValueError("No user has an unvisited restaurant to sample")

        rng = np.random.default_rng(config.seed)
        with _torch_settings(config):
            torch.manual_seed(config.seed)
            self.net = DeepCoNNNet(len(self.vocabulary), config)
            optimizer = torch.optim.Adam(
                self.net.parameters(), lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            indices = np.flatnonzero(trainable) if config.objective == "bpr" else np.arange(len(rows))
            for epoch in range(1, config.epochs + 1):
                started = time.perf_counter()
                self.net.train()
                self._cache = None
                order = indices[rng.permutation(len(indices))]
                negatives = None
                if config.objective == "bpr":
                    # Only users with an unvisited restaurant are sampled.
                    negatives = np.zeros(len(rows), dtype=np.int64)
                    negatives[indices] = _sample_negatives(
                        rng, edge_users[indices], n_items, positive_keys
                    )
                total = 0.0
                for offset in range(0, len(order), config.batch_size):
                    batch = order[offset : offset + config.batch_size]
                    targets = review_ids[batch]
                    user_vectors = self.net.encode_users(
                        self._user_docs.batch(user_keys[batch], targets)
                    )
                    item_vectors = self.net.encode_items(
                        self._item_docs.batch(self.item_keys[edge_items[batch]], targets)
                    )
                    positive = self.net.score(user_vectors, item_vectors)
                    if negatives is None:
                        loss = F.mse_loss(positive, ratings[batch])
                    else:
                        negative = self.net.score(
                            user_vectors,
                            self.net.encode_items(
                                self._item_docs.batch(self.item_keys[negatives[batch]])
                            ),
                        )
                        loss = -F.logsigmoid(positive - negative).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach()) * len(batch)
                self.net.eval()
                stats = EpochStats(epoch, total / len(order), time.perf_counter() - started)
                self.history.append(stats)
                if callback is not None and callback(stats, self):
                    break
        self._cache = None
        return self

    # ------------------------------------------------------------ inference
    def _encoded(self) -> tuple:
        """Retrieval parts for every training user and restaurant (cached)."""

        if self._cache is None:
            if self.net is None or self._user_docs is None or self._item_docs is None:
                raise RuntimeError("DeepCoNN must be fitted first")
            users = list(self.user_ids)
            items = [int(key) for key in self.item_keys]
            with _torch_settings(self.config), torch.no_grad():
                self.net.eval()
                user_latent = _encode(self.net.encode_users, self._user_docs, users)
                item_latent = _encode(self.net.encode_items, self._item_docs, items)
                user_bias, user_vec, item_bias, item_vec = self.net.retrieval_parts(
                    user_latent, item_latent
                )
            self._cache = (
                user_bias.numpy(), user_vec.numpy(), item_bias.numpy(), item_vec.numpy()
            )
        return self._cache

    def dead_latent_rates(self) -> dict[str, float]:
        """Share of users / restaurants whose latent vector is all zero.

        The tower ends in ReLU (as in the paper). A zero user vector ranks
        restaurants by the restaurant term alone, the same for every such user.
        """

        _, user_vec, _, item_vec = self._encoded()
        return {
            "zero_user_latent_rate": float((np.abs(user_vec).sum(1) == 0).mean()),
            "zero_item_latent_rate": float((np.abs(item_vec).sum(1) == 0).mean()),
        }

    def score_pairs(self, user_ids: Sequence[int], restaurant_ids: Sequence[int]) -> np.ndarray:
        """Full FM output (a rating for ``objective="mse"``); NaN if unknown."""

        user_bias, user_vec, item_bias, item_vec = self._encoded()
        out = np.full(len(user_ids), np.nan, dtype=np.float64)
        for row, (user, item) in enumerate(zip(user_ids, restaurant_ids, strict=True)):
            u, i = self.user_ids.get(user), self.item_ids.get(item)
            if u is not None and i is not None:
                out[row] = user_bias[u] + item_bias[i] + float(user_vec[u] @ item_vec[i])
        return out

    def recommend(
        self,
        user_ids: Sequence[int],
        exclude: Mapping[int, Iterable[int]],
        k: int,
        *,
        chunk_size: int = 1024,
    ) -> dict[int, tuple[int, ...]]:
        """Top-``k`` restaurant ids per user, excluding ``exclude[user]``.

        Ties are broken by restaurant id; users not seen in ``fit`` get ``()``.
        """

        _, user_vec, item_bias, item_vec = self._encoded()
        result: dict[int, tuple[int, ...]] = {
            user: () for user in user_ids if user not in self.user_ids
        }
        known = [user for user in dict.fromkeys(user_ids) if user in self.user_ids]
        for start in range(0, len(known), chunk_size):
            chunk = known[start : start + chunk_size]
            scores = (
                user_vec[[self.user_ids[user] for user in chunk]] @ item_vec.T
                + item_bias[None, :]
            ).astype(np.float64)
            for row, user in enumerate(chunk):
                seen = [self.item_ids[i] for i in exclude.get(user, ()) if i in self.item_ids]
                scores[row, seen] = -np.inf
            # Items are indexed in restaurant-id order: a stable sort on the
            # negated score breaks ties by restaurant id.
            order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
            for row, user in enumerate(chunk):
                valid = order[row][np.isfinite(scores[row, order[row]])]
                result[user] = tuple(int(x) for x in self.item_keys[valid])
        return result


def _encode(encoder, documents: EntityDocuments, keys: list[int], batch: int = 512) -> torch.Tensor:
    return torch.cat(
        [encoder(documents.batch(keys[start : start + batch])) for start in range(0, len(keys), batch)]
    )


class _torch_settings:
    """Seeded, deterministic, fixed-thread torch for one fit or inference."""

    def __init__(self, config: DeepCoNNConfig) -> None:
        self.threads = config.threads

    def __enter__(self) -> None:
        self._threads = torch.get_num_threads()
        self._deterministic = torch.are_deterministic_algorithms_enabled()
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(True)

    def __exit__(self, *exc) -> None:
        torch.set_num_threads(self._threads)
        torch.use_deterministic_algorithms(self._deterministic)
