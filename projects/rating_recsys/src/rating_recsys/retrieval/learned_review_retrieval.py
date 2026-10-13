"""BPR-trained review retrieval on a shared, frozen LightGCN backbone.

Single-vector mean and review-level late interaction share a bias-free projector,
the frozen review bank, initialization, and static BPR triples. Scores are the
raw CF dot product plus a learned scalar times the review score, over the whole
past catalog. No candidate filter or ranker is used. This is a final-cutoff,
static training experiment, not a reconstruction of historical-window features.

Training leaves the positive relationship's reviews out of both semantic sides:
the user's reviews about that restaurant and that restaurant's reviews written
by that user are masked. The same reduced user pool scores positive and negative.
Uniform review weighting is intentional; no additional attention parameters are
introduced in the late arm. Inputs are already frozen E5 review-level vectors.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rating_recsys.retrieval.lightgcn import _sample_negatives
from rating_recsys.retrieval.review_multivector import ReviewVectorBank, array_hash


@dataclass(frozen=True, slots=True)
class LearnedReviewRetrievalConfig:
    mode: str = "mean"
    text_condition: str = "actual"
    balance_text_presence: bool = False
    input_dim: int = 384
    projected_dim: int = 32
    cf_dimension: int = 64
    max_user_reviews: int = 8
    max_item_reviews: int = 12
    matching_k: int = 3
    epochs: int = 12
    seed: int = 42
    threads: int = 4
    batch_size: int = 512
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    initial_coefficient: float = 0.1
    cf_init_std: float = 0.1
    user_batch_size: int = 16
    item_batch_size: int = 256

    def __post_init__(self):
        if self.mode not in ("mean", "late") or self.text_condition not in ("actual", "shuffled"):
            raise ValueError("Require mean/late mode and actual/shuffled text condition")
        if type(self.balance_text_presence) is not bool:
            raise ValueError("balance_text_presence must be a boolean")
        for name in ("input_dim", "projected_dim", "cf_dimension", "max_user_reviews", "max_item_reviews",
                     "matching_k", "epochs", "threads", "batch_size", "user_batch_size", "item_batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be a nonnegative integer below 2**63")
        for name in ("learning_rate", "cf_init_std"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0 or not math.isfinite(self.initial_coefficient):
            raise ValueError("Invalid weight decay or text coefficient")

    @property
    def name(self):
        return "learned_review_retrieval"

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpochStats:
    epoch: int
    bpr_loss: float
    seconds: float
    coefficient: float
    triples_with_text: int


@contextmanager
def _settings(config):
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


def _state_hash(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def shuffle_review_vectors(bank, seed):
    """Permute valid vectors within each role; references/masks/counts stay fixed.

    Separate RNG streams for query and passage slots never consume the BPR RNG.
    Foreign padding is replaced with zeros in both the real and shuffled banks.
    The original bank is never mutated.
    """
    copied = {}
    audit = {"policy": "independent within-role valid-slot permutations; fixed review IDs and counts", "seed": seed}
    for index, side in enumerate(("user", "item")):
        refs = np.asarray(getattr(bank, side + "_review_ids"), dtype=np.int64)
        vectors = np.asarray(getattr(bank, side + "_vectors"), dtype=np.float32)
        mask = refs >= 0
        values = np.zeros_like(vectors)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7301, index]))
        order = rng.permutation(int(mask.sum()))
        values[mask] = vectors[mask][order]
        copied[side + "_vectors"] = values
        audit[side + "_permutation_sha256"] = array_hash(order)
    result = ReviewVectorBank(np.array(bank.user_ids, copy=True), np.array(bank.item_ids, copy=True),
                             np.array(bank.user_review_ids, copy=True), np.array(bank.item_review_ids, copy=True),
                             copied["user_vectors"], copied["item_vectors"], dict(bank.audit))
    return result, audit


class _Scorer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.projector = nn.Linear(config.input_dim, config.projected_dim, bias=False)
        self.coefficient = nn.Parameter(torch.tensor(config.initial_coefficient, dtype=torch.float32))

    def project(self, values):
        return F.normalize(self.projector(values), p=2, dim=-1)


def mean_vectors(vectors, mask):
    """Masked, uniformly weighted review mean followed by unit normalization."""
    mean = (vectors * mask.unsqueeze(-1)).sum(-2) / mask.sum(-1).clamp_min(1).unsqueeze(-1)
    return F.normalize(mean, p=2, dim=-1)


def pair_text_scores(users, items, user_mask, item_mask, mode, matching_k):
    """users[B,R,D], items[B,C,S,D] -> scores[B,C]; missing text adds zero."""
    if mode == "mean":
        return torch.einsum("bd,bcd->bc", mean_vectors(users, user_mask), mean_vectors(items, item_mask))
    similarities = torch.einsum("brd,bcsd->bcrs", users, items)
    similarities = similarities.masked_fill(~item_mask[:, :, None, :], -torch.inf)
    take = min(matching_k, items.shape[-2])
    best = similarities.topk(take, dim=-1).values
    totals = torch.where(torch.isfinite(best), best, 0).sum(-1)
    per_review = totals / item_mask.sum(-1).clamp(min=1, max=matching_k).unsqueeze(-1)
    return (per_review * user_mask[:, None, :]).sum(-1) / user_mask.sum(-1).clamp_min(1).unsqueeze(-1)


class LearnedReviewRetriever:
    def __init__(self, config=LearnedReviewRetrievalConfig()):
        self.config = config
        self.scorer = None
        self.bank = None
        self.history = []
        self.metadata = {}
        self.user_ids, self.item_ids = {}, {}
        self.item_keys = np.empty(0, dtype=np.int64)
        self._projected = None

    @property
    def coefficient(self):
        if self.scorer is None:
            raise RuntimeError("Model must be fitted first")
        return float(self.scorer.coefficient.detach())

    def _prepare_bank(self, bank, *, apply_shuffle=True):
        config = self.config
        arrays = {}
        for side, cap in (("user", config.max_user_reviews), ("item", config.max_item_reviews)):
            ids = np.asarray(getattr(bank, side + "_ids"))
            refs = np.asarray(getattr(bank, side + "_review_ids"))
            vectors = np.asarray(getattr(bank, side + "_vectors"), dtype=np.float32)
            if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(set(ids.tolist())) != len(ids)
                    or refs.ndim != 2 or refs.dtype.kind not in "iu"
                    or vectors.shape != (*refs.shape, config.input_dim)
                    or len(ids) != len(refs) or refs.shape[1] != cap or np.any(refs < -1)):
                raise ValueError("Invalid review bank IDs, caps, references or dimensions")
            valid = refs >= 0
            if not np.isfinite(vectors[valid]).all() or np.any(np.linalg.norm(vectors[valid], axis=-1) == 0):
                raise ValueError("Valid reviews need finite nonzero vectors")
            clean = np.zeros_like(vectors)
            clean[valid] = vectors[valid]
            arrays.update({side + "_ids": np.array(ids, dtype=np.int64, copy=True),
                           side + "_review_ids": np.array(refs, dtype=np.int64, copy=True),
                           side + "_vectors": clean})
        self.bank = ReviewVectorBank(**arrays, audit=dict(bank.audit))
        self.metadata["shuffle"] = None
        if config.text_condition == "shuffled" and apply_shuffle:
            self.bank, self.metadata["shuffle"] = shuffle_review_vectors(self.bank, config.seed)
        self.user_ids = {int(key): i for i, key in enumerate(self.bank.user_ids)}
        self.item_ids = {int(key): i for i, key in enumerate(self.bank.item_ids)}
        self.item_keys = self.bank.item_ids.copy()
        self._users = torch.from_numpy(self.bank.user_vectors)
        self._items = torch.from_numpy(self.bank.item_vectors)
        self._user_mask = torch.from_numpy(self.bank.user_review_ids >= 0)
        self._item_mask = torch.from_numpy(self.bank.item_review_ids >= 0)
        self.metadata["bank_arrays_sha256"] = {name: array_hash(getattr(self.bank, name)) for name in arrays}

    def _align_cf(self, cf_user_ids, cf_item_ids, cf_user_vectors, cf_item_vectors):
        aligned = []
        for keys, requested, values in ((cf_user_ids, self.bank.user_ids, cf_user_vectors),
                                       (cf_item_ids, self.bank.item_ids, cf_item_vectors)):
            keys, values = np.asarray(keys), np.asarray(values, dtype=np.float32)
            if (keys.ndim != 1 or keys.dtype.kind not in "iu" or len(set(keys.tolist())) != len(keys)
                    or values.shape != (len(keys), self.config.cf_dimension) or not np.isfinite(values).all()):
                raise ValueError("Invalid frozen CF IDs or vectors")
            positions = {int(key): i for i, key in enumerate(keys)}
            if any(int(key) not in positions for key in requested):
                raise ValueError("A review-bank entity has no frozen CF vector")
            aligned.append(torch.from_numpy(np.array(values[[positions[int(key)] for key in requested]], copy=True)))
        self._cf_users, self._cf_items = aligned
        self.metadata["frozen_cf_sha256"] = {"users": array_hash(self._cf_users.numpy()),
                                             "items": array_hash(self._cf_items.numpy())}

    def _ownership(self, rows):
        ownership = {}
        for row in rows:
            if row.review_id in ownership:
                raise ValueError("Duplicate historical review ID")
            ownership[row.review_id] = (row.user_id, row.restaurant_id)
        result = []
        for side, keys, refs in (("user", self.bank.user_ids, self.bank.user_review_ids),
                                 ("item", self.bank.item_ids, self.bank.item_review_ids)):
            owners = np.full(refs.shape, -1, dtype=np.int64)
            for index, entity in enumerate(keys):
                for slot, review_id in enumerate(refs[index]):
                    if review_id < 0:
                        continue
                    if int(review_id) not in ownership:
                        raise ValueError("Bank references a nonhistorical review")
                    user, item = ownership[int(review_id)]
                    if (user if side == "user" else item) != entity:
                        raise ValueError("Review ownership differs from its bank entity")
                    owners[index, slot] = item if side == "user" else user
            result.append(torch.from_numpy(owners))
        self._user_review_items, self._item_review_users = result

    def training_masks(self, user_positions, positive_positions, negative_positions):
        """The positive pair's own reviews cannot supply its semantic score."""
        users = torch.as_tensor(user_positions, dtype=torch.long)
        positives = torch.as_tensor(positive_positions, dtype=torch.long)
        negatives = torch.as_tensor(negative_positions, dtype=torch.long)
        choices = torch.stack((positives, negatives), dim=1)
        positive_keys = torch.from_numpy(self.bank.item_ids)[positives]
        user_keys = torch.from_numpy(self.bank.user_ids)[users]
        user_mask = self._user_mask[users] & (self._user_review_items[users] != positive_keys[:, None])
        item_mask = self._item_mask[choices] & (self._item_review_users[choices] != user_keys[:, None, None])
        return user_mask, item_mask

    def _batch_scores(self, users, positives, negatives):
        choices = torch.stack((positives, negatives), dim=1)
        user_mask, item_mask = self.training_masks(users, positives, negatives)
        projected_users = self.scorer.project(self._users[users])
        projected_items = self.scorer.project(self._items[choices])
        text = pair_text_scores(projected_users, projected_items, user_mask, item_mask,
                                self.config.mode, self.config.matching_k)
        user_present = user_mask.any(-1)
        item_present = item_mask.any(-1)
        positive_present = user_present & item_present[:, 0]
        negative_present = user_present & item_present[:, 1]
        pair_usable = positive_present & negative_present
        self._batch_presence = {
            "triples": len(users), "positive_text_present": int(positive_present.sum()),
            "negative_text_present": int(negative_present.sum()), "both_text_present": int(pair_usable.sum()),
            "positive_only": int((positive_present & ~negative_present).sum()),
            "negative_only": int((negative_present & ~positive_present).sum()),
            "neither": int((~positive_present & ~negative_present).sum()),
        }
        if self.config.balance_text_presence:
            text = text * pair_usable[:, None]
        cf = torch.einsum("bd,bcd->bc", self._cf_users[users], self._cf_items[choices])
        scores = cf + self.scorer.coefficient * text
        with_text = int((pair_usable if self.config.balance_text_presence
                         else positive_present | negative_present).sum())
        return scores, with_text

    def fit(self, rows, *, bank, cf_user_vectors, cf_item_vectors, cutoff: date,
            cf_user_ids=None, cf_item_ids=None, callback=None):
        rows = tuple(sorted(rows, key=lambda row: (row.event_date, row.review_id)))
        if not rows or any(row.event_date > cutoff for row in rows):
            raise ValueError("Training requires nonempty history at or before cutoff")
        if bank.audit.get("cutoff_inclusive", cutoff.isoformat()) != cutoff.isoformat():
            raise ValueError("Review bank cutoff differs")
        self.history, self.metadata, self._projected = [], {}, None
        self._prepare_bank(bank)
        self._align_cf(self.bank.user_ids if cf_user_ids is None else cf_user_ids,
                       self.bank.item_ids if cf_item_ids is None else cf_item_ids,
                       cf_user_vectors, cf_item_vectors)
        self._ownership(rows)
        edges = sorted({(row.user_id, row.restaurant_id) for row in rows})
        graph_users, graph_items = sorted({u for u, _ in edges}), sorted({i for _, i in edges})
        if len(edges) < 2 or len(graph_items) < 2:
            raise ValueError("Require two historical edges and restaurants")
        if (any(key not in self.user_ids for key in graph_users)
                or set(self.item_ids) != set(graph_items)):
            raise ValueError("Review bank must cover every training user and exactly the past catalog")
        graph_um = {key: i for i, key in enumerate(graph_users)}
        graph_im = {key: i for i, key in enumerate(graph_items)}
        edge_users = np.asarray([graph_um[u] for u, _ in edges], dtype=np.int64)
        edge_items = np.asarray([graph_im[i] for _, i in edges], dtype=np.int64)
        forbidden = np.unique(edge_users * len(graph_items) + edge_items)
        trainable = np.bincount(edge_users, minlength=len(graph_users))[edge_users] < len(graph_items)
        edge_users, edge_items = edge_users[trainable], edge_items[trainable]
        if not len(edge_users):
            raise ValueError("No user has an unseen negative restaurant")
        self.edge_count = len(edges)
        user_map = np.asarray([self.user_ids[key] for key in graph_users], dtype=np.int64)
        item_map = np.asarray([self.item_ids[key] for key in graph_items], dtype=np.int64)
        self.metadata.update({
            "objective": "static BPR: softplus(negative total score - positive total score)",
            "training_protocol": "all supplied final-cutoff history; frozen CF; not historical-window reconstruction",
            "cutoff_inclusive": cutoff.isoformat(), "maximum_event_date": rows[-1].event_date.isoformat(),
            "training_edges": len(edges), "trainable_edges": len(edge_users),
            "negative_policy": "uniform unseen past-catalog item; all historical edges forbidden",
            "sampling_rng": "LightGCN ego-normal RNG consumption, then epoch permutation and unseen negatives",
            "sampling_ego_shape": [len(graph_users) + len(graph_items), self.config.cf_dimension],
            "edge_users_sha256": array_hash(edge_users), "edge_items_sha256": array_hash(edge_items),
            "positive_review_exclusion": "user: all positive restaurant reviews; positive/negative item: all current-user reviews; same reduced user pool for both scores",
            "aggregation": "unit projected reviews -> masked unit mean dot" if self.config.mode == "mean"
                           else "unit projected reviews -> per-user-review masked item top-k cosine mean -> uniform user-review mean",
            "review_weights": "uniform", "cf_scale": 1.0,
            "balance_text_presence": self.config.balance_text_presence,
            "training_text_policy": ("both scores receive text only when the reduced user and both item pools have text; all CF triples retained"
                                     if self.config.balance_text_presence else "each semantic score independently uses available text; missing text contributes zero"),
            "coefficient": "unconstrained learned scalar; initial value in config",
            "frozen_cf": True, "frozen_e5": True, "encoder_loaded": False, "api_requests": 0,
            "sampling_audit": [], "parameter_count": self.config.input_dim * self.config.projected_dim + 1,
            "training_text_presence": [],
        })
        rng = np.random.default_rng(self.config.seed)
        rng.normal(0.0, self.config.cf_init_std,
                   size=(len(graph_users) + len(graph_items), self.config.cf_dimension))
        with _settings(self.config):
            self.scorer = _Scorer(self.config)
            self.metadata["initial_parameters_sha256"] = _state_hash(self.scorer)
            optimizer = torch.optim.Adam(self.scorer.parameters(), lr=self.config.learning_rate,
                                         weight_decay=self.config.weight_decay)
            for epoch in range(1, self.config.epochs + 1):
                started = time.perf_counter()
                order = rng.permutation(len(edge_users))
                negatives = _sample_negatives(rng, edge_users, len(graph_items), forbidden)
                self.metadata["sampling_audit"].append({"epoch": epoch, "order_sha256": array_hash(order),
                    "negative_items_sha256": array_hash(negatives),
                    "sequence_sha256": hashlib.sha256(order.tobytes() + negatives.tobytes()).hexdigest()})
                total, count, with_text = 0.0, 0, 0
                presence = {key: 0 for key in ("triples", "positive_text_present", "negative_text_present",
                    "both_text_present", "positive_only", "negative_only", "neither")}
                self.scorer.train()
                self._projected = None
                for offset in range(0, len(order), self.config.batch_size):
                    indices = order[offset:offset + self.config.batch_size]
                    users = torch.from_numpy(user_map[edge_users[indices]])
                    positives = torch.from_numpy(item_map[edge_items[indices]])
                    negative_positions = torch.from_numpy(item_map[negatives[indices]])
                    scores, profiled = self._batch_scores(users, positives, negative_positions)
                    for key, value in self._batch_presence.items():
                        presence[key] += value
                    loss = F.softplus(scores[:, 1] - scores[:, 0]).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach()) * len(indices)
                    count += len(indices)
                    with_text += profiled
                self.scorer.eval()
                stats = EpochStats(epoch, total / count, time.perf_counter() - started,
                                   self.coefficient, with_text)
                self.history.append(stats)
                self.metadata["completed_epochs"] = epoch
                self.metadata["final_coefficient"] = self.coefficient
                self.metadata["training_text_presence"].append({"epoch": epoch, **presence,
                    "text_usable_triples": with_text,
                    "text_usable_fraction": with_text / count})
                if callback is not None and callback(stats, self):
                    break
        self._projected = None
        self.metadata["final_parameters_sha256"] = _state_hash(self.scorer)
        return self

    def _encode(self):
        if self.scorer is None or self.bank is None:
            raise RuntimeError("Model must be fitted first")
        if self._projected is None:
            with _settings(self.config), torch.no_grad():
                self.scorer.eval()
                outputs = []
                for values in (self._users, self._items):
                    blocks = [self.scorer.project(values[start:start + self.config.batch_size])
                              for start in range(0, len(values), self.config.batch_size)]
                    outputs.append(torch.cat(blocks))
                self._projected = tuple(outputs)
        return self._projected

    def score_all(self, user_ids=None, *, progress=None):
        """Return exact full-catalog total scores, in supplied user-ID order.

        Intermediate matching tensors are at most user_batch*item_batch*8*12;
        the final user-by-item output is the only full score matrix allocated.
        Unknown users get -inf, whereas missing text retains the CF score.
        """
        projected_users, projected_items = self._encode()
        requested = self.bank.user_ids.tolist() if user_ids is None else list(user_ids)
        output = np.full((len(requested), len(self.bank.item_ids)), -np.inf, dtype=np.float32)
        positions = [(row, self.user_ids[int(uid)]) for row, uid in enumerate(requested) if int(uid) in self.user_ids]
        with _settings(self.config), torch.no_grad():
            if self.config.mode == "mean":
                means_u = mean_vectors(projected_users, self._user_mask)
                means_i = mean_vectors(projected_items, self._item_mask)
            for offset in range(0, len(positions), self.config.user_batch_size):
                batch = positions[offset:offset + self.config.user_batch_size]
                rows = [row for row, _ in batch]
                upos = torch.tensor([index for _, index in batch])
                for start in range(0, len(self.bank.item_ids), self.config.item_batch_size):
                    stop = min(start + self.config.item_batch_size, len(self.bank.item_ids))
                    cf = self._cf_users[upos] @ self._cf_items[start:stop].T
                    if self.config.mode == "mean":
                        text = means_u[upos] @ means_i[start:stop].T
                    else:
                        uvec, ivec = projected_users[upos], projected_items[start:stop]
                        similarities = (uvec.reshape(-1, self.config.projected_dim)
                                        @ ivec.reshape(-1, self.config.projected_dim).T)
                        similarities = similarities.reshape(len(batch), uvec.shape[1], stop-start, ivec.shape[1]).permute(0, 2, 1, 3)
                        imask = self._item_mask[start:stop]
                        similarities = similarities.masked_fill(~imask[None, :, None, :], -torch.inf)
                        best = similarities.topk(min(self.config.matching_k, ivec.shape[1]), dim=-1).values
                        per_review = torch.where(torch.isfinite(best), best, 0).sum(-1)
                        per_review /= imask.sum(-1).clamp(min=1, max=self.config.matching_k)[None, :, None]
                        text = (per_review * self._user_mask[upos, None, :]).sum(-1)
                        text /= self._user_mask[upos].sum(-1).clamp_min(1)[:, None]
                    output[np.asarray(rows), start:stop] = (cf + self.scorer.coefficient * text).numpy()
                if progress is not None:
                    progress(offset + len(batch), len(positions))
        return output

    def rankings(self, scores, user_ids, exclude, *, k=100):
        if type(k) is not int or k < 1:
            raise ValueError("k must be a positive integer")
        scores = np.asarray(scores)
        user_ids = list(user_ids)
        if scores.shape != (len(user_ids), len(self.item_keys)) or np.isnan(scores).any():
            raise ValueError("Invalid full-catalog scores")
        result = {}
        for row, uid in enumerate(user_ids):
            valid = np.isfinite(scores[row]) & ~np.isin(self.item_keys, tuple(exclude.get(uid, ())))
            positions = np.flatnonzero(valid)
            order = np.lexsort((self.item_keys[positions], -scores[row, positions]))[:k]
            result[int(uid)] = tuple(self.item_keys[positions[order]].tolist())
        self.metadata["last_rankings_sha256"] = hashlib.sha256(repr(sorted(result.items())).encode()).hexdigest()
        return result

    def recommend(self, user_ids, exclude, k=100):
        user_ids = list(user_ids)
        return self.rankings(self.score_all(user_ids), user_ids, exclude, k=k)

    def save(self, destination, *, include_banks=True):
        if self.scorer is None:
            raise RuntimeError("Model must be fitted first")
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        saved = {"schema": "learned-review-retrieval-v1", "config": self.config.to_dict(),
                 "scorer": self.scorer.state_dict(), "metadata": self.metadata,
                 "history": [asdict(stats) for stats in self.history],
                 "cf_user_ids": torch.from_numpy(self.bank.user_ids), "cf_item_ids": torch.from_numpy(self.bank.item_ids),
                 "cf_user_vectors": self._cf_users, "cf_item_vectors": self._cf_items,
                 "bank_audit": self.bank.audit, "bank": None,
                 "bank_is_conditioned": bool(include_banks)}
        if include_banks:
            saved["bank"] = {name: torch.from_numpy(getattr(self.bank, name)) for name in (
                "user_ids", "item_ids", "user_review_ids", "item_review_ids", "user_vectors", "item_vectors")}
        torch.save(saved, destination)
        return {"path": destination.name, "sha256": _file_hash(destination),
                "bytes": destination.stat().st_size, "includes_bank": bool(include_banks)}

    @classmethod
    def load(cls, path, *, bank=None):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["schema"] != "learned-review-retrieval-v1":
            raise ValueError("Unknown learned-review checkpoint schema")
        model = cls(LearnedReviewRetrievalConfig(**saved["config"]))
        if saved["bank"] is not None:
            bank = ReviewVectorBank(**{name: value.numpy() for name, value in saved["bank"].items()},
                                    audit=saved["bank_audit"])
        elif bank is None:
            raise ValueError("Compact checkpoint requires the original frozen review bank")
        model._prepare_bank(bank, apply_shuffle=not saved["bank_is_conditioned"])
        expected_arrays = saved["metadata"]["bank_arrays_sha256"]
        if model.metadata["bank_arrays_sha256"] != expected_arrays:
            raise ValueError("Checkpoint frozen bank arrays differ")
        model._align_cf(saved["cf_user_ids"].numpy(), saved["cf_item_ids"].numpy(),
                        saved["cf_user_vectors"].numpy(), saved["cf_item_vectors"].numpy())
        if model.metadata["frozen_cf_sha256"] != saved["metadata"]["frozen_cf_sha256"]:
            raise ValueError("Checkpoint frozen CF arrays differ")
        with _settings(model.config):
            model.scorer = _Scorer(model.config)
            model.scorer.load_state_dict(saved["scorer"])
            model.scorer.eval()
        model.metadata = saved["metadata"]
        model.history = [EpochStats(**stats) for stats in saved["history"]]
        model.edge_count = model.metadata["training_edges"]
        if _state_hash(model.scorer) != model.metadata["final_parameters_sha256"]:
            raise ValueError("Checkpoint scorer checksum differs")
        return model
