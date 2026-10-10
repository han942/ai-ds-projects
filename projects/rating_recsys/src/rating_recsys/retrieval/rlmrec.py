"""RLMRec-Con adaptation: frozen E5 review profiles regularize LightGCN.

This implements the contrastive alignment in RLMRec equations 16/18:
https://arxiv.org/html/2310.15950v5. The shared semantic MLP follows
https://github.com/HKUDS/RLMRec/blob/main/encoder/models/general_cf/lightgcn_plus.py.

Unlike the paper, inputs are existing cutoff-safe E5 review-block averages,
not LLM-generated profiles. Unique, profiled users/positive items/negative
items form three capped in-batch contrastive pools. Their mean InfoNCE losses
are summed. The caller must construct each frozen profile before the graph's
cutoff. This module never reads reviews, embeds text, or calls an API.

LightGCN's original fit, BPR, ego initialization, sampling, regularization,
Adam and inference are reused. Torch differentiates only the small semantic
projector and selected final CF rows. CF gradients pass back through the
original symmetric layer-average propagation and are added to the BPR
gradient. Checkpoints are inference-only NumPy archives, without pickle.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
from scipy.sparse import csr_matrix

from rating_recsys.datasets.models import Interaction
from rating_recsys.retrieval.lightgcn import (
    EpochCallback, EpochStats, LightGCN, LightGCNConfig,
)


@dataclass(frozen=True, slots=True)
class RLMRecConfig:
    weight: float = 0.01
    temperature: float = 0.2
    batch_size: int = 128
    seed: int = 42
    threads: int = 4

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be finite and nonnegative")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        for name in ("batch_size", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer in [0, 2**63)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AlignmentEpochStats:
    epoch: int
    alignment_loss: float
    weighted_alignment_loss: float
    user_loss: float
    positive_item_loss: float
    negative_item_loss: float
    batches: int
    aligned_batches: int
    user_entities: int
    positive_item_entities: int
    negative_item_entities: int
    bpr_gradient_norm: float
    alignment_gradient_norm: float
    seconds: float


class RLMRecLightGCN(LightGCN):
    """Drop-in LightGCN with a shared text-to-CF alignment projector.

    Profile arrays are copied and frozen. Missing and all-zero profiles are
    omitted from alignment; graph nodes and BPR triples remain unchanged.
    At the default dimensions the MLP is 384 -> 224 -> 64 (LeakyReLU).
    Its separate Adam uses the graph learning rate, without weight decay.
    ``history`` retains the original BPR/L2 stats; ``alignment_history`` is
    committed before the existing epoch callback runs. Alignment losses and
    gradient norms are weighted by BPR batch size in epoch summaries. Entity
    counts are exposures across batches, rather than distinct epoch IDs.
    """

    def __init__(
        self,
        graph_config: LightGCNConfig = LightGCNConfig(),
        alignment_config: RLMRecConfig = RLMRecConfig(),
        user_profiles: Mapping[int, np.ndarray] | None = None,
        item_profiles: Mapping[int, np.ndarray] | None = None,
    ) -> None:
        super().__init__(graph_config)
        self.alignment_config = alignment_config
        self.semantic_dimension = 384
        dimensions: set[int] = set()
        banks = []
        for profiles in (user_profiles or {}, item_profiles or {}):
            bank = {}
            for key, value in profiles.items():
                if isinstance(key, bool) or not isinstance(key, (int, np.integer)):
                    raise ValueError("Profile IDs must be integers")
                vector = np.array(value, dtype=np.float32, copy=True)
                if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                    raise ValueError("Profiles must be finite, nonempty one-dimensional vectors")
                dimensions.add(len(vector))
                if np.any(vector):
                    vector.flags.writeable = False
                    bank[int(key)] = vector
            banks.append(MappingProxyType(bank))
        if len(dimensions) > 1:
            raise ValueError("User and item profiles must share one semantic dimension")
        if dimensions:
            self.semantic_dimension = dimensions.pop()
        self.user_profiles, self.item_profiles = banks
        self.projector = None
        self._projector_optimizer = None
        self._node_profiles: dict[int, np.ndarray] | None = None
        self._alignment_rng = np.random.default_rng(alignment_config.seed)
        self.alignment_history: list[AlignmentEpochStats] = []
        self._saved_profile_coverage = None
        self._inference_only = False
        self._reset_epoch_totals()

    @property
    def profile_coverage(self) -> dict[str, int]:
        if self._saved_profile_coverage is not None:
            return dict(self._saved_profile_coverage)
        return {
            "graph_users": len(self.user_ids), "graph_items": len(self.item_ids),
            "profiled_users": sum(key in self.user_profiles for key in self.user_ids),
            "profiled_items": sum(key in self.item_profiles for key in self.item_ids),
        }

    @property
    def projector_parameter_count(self) -> int:
        if self.projector is None:
            return 0
        return sum(parameter.numel() for parameter in self.projector.parameters())

    def _new_projector(self):
        # fork_rng preserves the caller's global Torch state. Neither projector
        # initialization nor capped alignment sampling touches the BPR RNG.
        import torch
        from torch import nn

        hidden = (self.semantic_dimension + self.config.dimension) // 2
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.alignment_config.seed)
            projector = nn.Sequential(
                nn.Linear(self.semantic_dimension, hidden), nn.LeakyReLU(),
                nn.Linear(hidden, self.config.dimension),
            )
            for layer in projector:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
        return projector

    def _reset_epoch_totals(self) -> None:
        self._epoch_examples = self._epoch_batches = self._epoch_aligned_batches = 0
        self._epoch_losses = np.zeros(3, dtype=np.float64)
        self._epoch_entities = np.zeros(3, dtype=np.int64)
        self._epoch_bpr_norm = self._epoch_alignment_norm = self._epoch_alignment_seconds = 0.0

    def fit(self, interactions: Iterable[Interaction], *, callback: EpochCallback | None = None) -> RLMRecLightGCN:
        if self._inference_only:
            raise ValueError("This inference-only checkpoint cannot resume training; create a new model")
        self.alignment_history = []
        self._node_profiles = None
        self._alignment_rng = np.random.default_rng(self.alignment_config.seed)
        self._reset_epoch_totals()
        self.projector = self._projector_optimizer = None
        old_threads = None

        def after_epoch(stats: EpochStats, model: LightGCN) -> bool | None:
            denominator = max(self._epoch_examples, 1)
            losses = self._epoch_losses / denominator
            self.alignment_history.append(AlignmentEpochStats(
                epoch=stats.epoch, alignment_loss=float(losses.sum()),
                weighted_alignment_loss=float(self.alignment_config.weight * losses.sum()),
                user_loss=float(losses[0]), positive_item_loss=float(losses[1]),
                negative_item_loss=float(losses[2]), batches=self._epoch_batches,
                aligned_batches=self._epoch_aligned_batches,
                user_entities=int(self._epoch_entities[0]),
                positive_item_entities=int(self._epoch_entities[1]),
                negative_item_entities=int(self._epoch_entities[2]),
                bpr_gradient_norm=self._epoch_bpr_norm / denominator,
                alignment_gradient_norm=self._epoch_alignment_norm / denominator,
                seconds=self._epoch_alignment_seconds,
            ))
            self._reset_epoch_totals()
            return callback(stats, model) if callback is not None else None

        try:
            if self.alignment_config.weight and (self.user_profiles or self.item_profiles):
                import torch

                old_threads = torch.get_num_threads()
                torch.set_num_threads(self.alignment_config.threads)
                self.projector = self._new_projector()
                self._projector_optimizer = torch.optim.Adam(
                    self.projector.parameters(), lr=self.config.learning_rate,
                )
            super().fit(interactions, callback=after_epoch)
        finally:
            if old_threads is not None:
                torch.set_num_threads(old_threads)
        return self

    def _alignment_groups(self, u: np.ndarray, i: np.ndarray, j: np.ndarray) -> tuple[np.ndarray, ...]:
        if self._node_profiles is None:
            offset = len(self.user_ids)
            self._node_profiles = {
                index: self.user_profiles[key] for key, index in self.user_ids.items()
                if key in self.user_profiles
            }
            self._node_profiles.update({
                offset + index: self.item_profiles[key] for key, index in self.item_ids.items()
                if key in self.item_profiles
            })
        groups = []
        for nodes in (u, i, j):
            selected = np.asarray([
                int(node) for node in np.unique(nodes) if int(node) in self._node_profiles
            ], dtype=np.int64)
            if len(selected) > self.alignment_config.batch_size:
                selected = np.sort(self._alignment_rng.choice(
                    selected, self.alignment_config.batch_size, replace=False,
                ))
            groups.append(selected)
        return tuple(groups)

    def _loss_and_gradient(self, u: np.ndarray, i: np.ndarray, j: np.ndarray) -> tuple[float, float, np.ndarray]:
        # Keep these exact baseline sums; alignment has a separate diagnostics
        # history, so bpr_loss/reg_loss keep their original interpretation.
        bpr_sum, reg_sum, gradient = super()._loss_and_gradient(u, i, j)
        self._epoch_examples += len(u)
        self._epoch_batches += 1
        self._epoch_bpr_norm += len(u) * float(np.linalg.norm(gradient))
        if not self.alignment_config.weight or self.projector is None:
            return bpr_sum, reg_sum, gradient
        started = time.perf_counter()
        groups = self._alignment_groups(u, i, j)
        # A singleton has no distinct negative and contributes zero InfoNCE.
        if not any(len(nodes) > 1 for nodes in groups):
            self._epoch_alignment_seconds += time.perf_counter() - started
            return bpr_sum, reg_sum, gradient

        import torch
        import torch.nn.functional as functional

        final = self._propagate(self._ego)
        grad_final = np.zeros_like(final)
        self._projector_optimizer.zero_grad(set_to_none=True)
        leaves, terms, term_indexes = [], [], []
        losses = np.zeros(3, dtype=np.float64)
        entities = np.zeros(3, dtype=np.int64)
        for role, nodes in enumerate(groups):
            if len(nodes) < 2:
                continue
            cf = torch.tensor(final[nodes], dtype=torch.float32, requires_grad=True)
            text = torch.from_numpy(np.stack([self._node_profiles[int(node)] for node in nodes]))
            semantic = self.projector(text)
            # Match the official stabilised cosine denominator. Cross-entropy
            # computes logsumexp directly instead of exp/log of large logits.
            cf_normal = cf / torch.sqrt(1e-8 + cf.square().sum(dim=1, keepdim=True))
            text_normal = semantic / torch.sqrt(1e-8 + semantic.square().sum(dim=1, keepdim=True))
            logits = cf_normal @ text_normal.T / self.alignment_config.temperature
            term = functional.cross_entropy(logits, torch.arange(len(nodes)))
            losses[role] = float(term.detach())
            entities[role] = len(nodes)
            terms.append(term)
            leaves.append(cf)
            term_indexes.append(nodes)
        (sum(terms) * self.alignment_config.weight).backward()
        for nodes, cf in zip(term_indexes, leaves, strict=True):
            np.add.at(grad_final, nodes, cf.grad.detach().numpy())
        alignment_gradient = self._propagate(grad_final)
        self._projector_optimizer.step()
        self._epoch_losses += len(u) * losses
        self._epoch_entities += entities
        self._epoch_aligned_batches += 1
        self._epoch_alignment_norm += len(u) * float(np.linalg.norm(alignment_gradient))
        self._epoch_alignment_seconds += time.perf_counter() - started
        return bpr_sum, reg_sum, gradient + alignment_gradient

    def save(self, path: str | Path) -> None:
        """Atomically save inference state; profiles and optimizer are omitted."""
        if self._adjacency is None or not len(self._ego):
            raise ValueError("Fit the model before saving")
        arrays = {
            "ego": self._ego,
            "user_keys": np.asarray(sorted(self.user_ids, key=self.user_ids.get), dtype=np.int64),
            "item_keys": self.item_keys,
            "adjacency_data": self._adjacency.data,
            "adjacency_indices": self._adjacency.indices,
            "adjacency_indptr": self._adjacency.indptr,
        }
        if self.projector is not None:
            arrays.update({"projector_" + key: value.detach().cpu().numpy()
                           for key, value in self.projector.state_dict().items()})
        metadata = {
            "schema": "rlmrec-inference-v1", "graph_config": self.config.to_dict(),
            "alignment_config": self.alignment_config.to_dict(),
            "semantic_dimension": self.semantic_dimension, "edge_count": self.edge_count,
            "has_projector": self.projector is not None,
            "profile_coverage": self.profile_coverage,
            "history": [asdict(stats) for stats in self.history],
            "alignment_history": [asdict(stats) for stats in self.alignment_history],
        }
        arrays["metadata"] = np.frombuffer(json.dumps(metadata, allow_nan=False).encode(), dtype=np.uint8)
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
    def load(cls, path: str | Path) -> RLMRecLightGCN:
        """Read an inference-only checkpoint with ``allow_pickle=False``."""
        with np.load(path, allow_pickle=False) as archive:
            metadata_array = archive["metadata"]
            if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                raise ValueError("Invalid RLMRec checkpoint metadata")
            metadata = json.loads(metadata_array.tobytes())
            if metadata.get("schema") != "rlmrec-inference-v1":
                raise ValueError("Unsupported RLMRec checkpoint schema")
            model = cls(LightGCNConfig(**metadata["graph_config"]),
                        RLMRecConfig(**metadata["alignment_config"]))
            semantic_dimension = metadata["semantic_dimension"]
            if isinstance(semantic_dimension, bool) or not isinstance(semantic_dimension, int) or semantic_dimension < 1:
                raise ValueError("Invalid semantic dimension in RLMRec checkpoint")
            model.semantic_dimension = semantic_dimension
            user_keys, item_keys, ego = (archive[key] for key in ("user_keys", "item_keys", "ego"))
            for keys in (user_keys, item_keys):
                if keys.dtype != np.int64 or keys.ndim != 1 or not len(keys) or np.any(keys[1:] <= keys[:-1]):
                    raise ValueError("Checkpoint entity IDs must be sorted unique int64 vectors")
            nodes = len(user_keys) + len(item_keys)
            if ego.dtype != np.float32 or ego.shape != (nodes, model.config.dimension) or not np.isfinite(ego).all():
                raise ValueError("Invalid ego embeddings in RLMRec checkpoint")
            data, indices, indptr = (archive["adjacency_" + key] for key in ("data", "indices", "indptr"))
            if (data.dtype != np.float32 or data.ndim != 1 or not np.isfinite(data).all()
                    or np.any(data <= 0) or indices.dtype.kind != "i" or indices.ndim != 1
                    or indptr.dtype.kind != "i" or indptr.shape != (nodes + 1,)
                    or len(data) != len(indices) or indptr[0] != 0 or indptr[-1] != len(data)
                    or np.any(indptr < 0) or np.any(indptr > len(data))
                    or np.any(indptr[1:] < indptr[:-1]) or np.any(indices < 0) or np.any(indices >= nodes)):
                raise ValueError("Invalid normalized adjacency in RLMRec checkpoint")
            model.user_ids = {int(key): index for index, key in enumerate(user_keys)}
            model.item_ids = {int(key): index for index, key in enumerate(item_keys)}
            model.item_keys = item_keys.copy()
            model._ego = ego.copy()
            model._adjacency = csr_matrix((data.copy(), indices.copy(), indptr.copy()), shape=(nodes, nodes))
            model.edge_count = int(metadata["edge_count"])
            model.history = [EpochStats(**value) for value in metadata["history"]]
            model.alignment_history = [AlignmentEpochStats(**value) for value in metadata["alignment_history"]]
            model._saved_profile_coverage = metadata["profile_coverage"]
            if metadata["has_projector"]:
                import torch

                model.projector = model._new_projector()
                state = {}
                for key, expected in model.projector.state_dict().items():
                    value = archive["projector_" + key]
                    if value.dtype != np.float32 or value.shape != tuple(expected.shape) or not np.isfinite(value).all():
                        raise ValueError("Invalid semantic projector in RLMRec checkpoint")
                    state[key] = torch.from_numpy(value.copy())
                model.projector.load_state_dict(state, strict=True)
                model.projector.eval()
            model._inference_only = True
            return model
