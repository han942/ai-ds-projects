"""Opinion channels regularize LightGCN without collapsing polarity channels.

The original graph/BPR sampler and optimizer are unchanged. A shared text MLP
and a separate graph head for each aspect/polarity align within that channel.
Channel losses are averaged within each of the three user/positive/negative
pools, so adding channels does not multiply the auxiliary loss weight.
Inference remains the learned graph's raw inner product.
"""
from collections import defaultdict, deque
import time

import numpy as np

from rating_recsys.retrieval.lightgcn import LightGCN
from rating_recsys.retrieval.rlmrec import RLMRecConfig, RLMRecLightGCN
from rating_recsys.retrieval.review_multivector import unit

CATEGORIES = ("taste", "value", "service", "atmosphere", "portion")
CHANNELS = tuple((category, polarity) for category in CATEGORIES for polarity in ("positive", "negative"))


def opinion_input(opinion, role):
    if role not in ("query", "passage"):
        raise ValueError("Unknown opinion role")
    return (f"{role}: 대상: {opinion['target']}. 의견: {opinion['opinion']}. "
            f"근거: {opinion['evidence']}")


def build_opinion_profiles(history, cutoff, texts, opinions, vectors, *, user_cap=20, item_cap=30):
    """Equal review weights within each observed channel, latest text caps.

    Neutral/unknown opinions remain in the extraction artifact but do not get
    an invented positive or negative channel. Missing channels remain zero.
    """
    if min(user_cap, item_cap) < 1:
        raise ValueError("Review caps must be positive")
    rows = sorted(history, key=lambda r: (r.event_date, r.review_id))
    if any(row.event_date > cutoff for row in rows):
        raise ValueError("Future opinion history")
    if len({row.review_id for row in rows}) != len(rows):
        raise ValueError("Duplicate opinion review")
    groups = {"user": defaultdict(lambda: deque(maxlen=user_cap)),
              "item": defaultdict(lambda: deque(maxlen=item_cap))}
    for row in rows:
        if texts.get(row.review_id):
            groups["user"][row.user_id].append(row.review_id)
            groups["item"][row.restaurant_id].append(row.review_id)
    banks, audits = {}, {}
    positions = {channel: i for i, channel in enumerate(CHANNELS)}
    for side, role in (("user", "query"), ("item", "passage")):
        bank, refs, counts = {}, {}, {}
        for entity, review_ids in sorted(groups[side].items()):
            by_channel = defaultdict(list)
            evidence = defaultdict(list)
            for rid in review_ids:
                per_review = defaultdict(list)
                for record in opinions.get(rid, ()):
                    channel = (record["category"], record["polarity"])
                    if channel not in positions:
                        continue
                    value = np.asarray(vectors[opinion_input(record, role)], dtype=np.float32)
                    if value.shape != (384,) or not np.isfinite(value).all():
                        raise ValueError("Invalid opinion embedding")
                    per_review[positions[channel]].append(value)
                for channel, values in per_review.items():
                    pooled = unit(np.mean(values, axis=0))
                    if np.any(pooled):
                        by_channel[channel].append(pooled)
                        evidence[channel].append(rid)
            packed = np.zeros((len(CHANNELS), 384), dtype=np.float32)
            for channel, values in by_channel.items():
                packed[channel] = unit(np.mean(values, axis=0))
            if np.any(packed):
                bank[entity] = packed.reshape(-1)
                refs[entity] = {str(channel): ids for channel, ids in evidence.items()}
                counts[entity] = len(review_ids)
        banks[side] = bank
        audits[side] = {"profiled_entities": len(bank), "review_ids_by_entity_channel": refs,
                        "selected_review_counts": counts,
                        "channel_coverage": {str(channel): int(sum(np.any(value.reshape(len(CHANNELS), 384)[channel])
                                                                   for value in bank.values())) for channel in range(len(CHANNELS))}}
    return banks["user"], banks["item"], {"cutoff_inclusive": cutoff.isoformat(),
            "channels": CHANNELS, "user_cap": user_cap, "item_cap": item_cap,
            "aggregation": "opinion mean within review/channel -> equal review channel mean -> L2",
            "missing": "zero masked; no rating-derived sentiment; neutral/unknown omitted from alignment",
            **audits}


def mean_opinion_profiles(bank):
    result = {}
    for key, value in bank.items():
        slots = np.asarray(value).reshape(len(CHANNELS), 384)
        mask = np.any(slots != 0, axis=1)
        if mask.any():
            mean = unit(slots[mask].mean(0))
            if np.any(mean):
                result[key] = mean
    return result


class AspectGraphLightGCN(RLMRecLightGCN):
    def __init__(self, graph_config=None, alignment_config=RLMRecConfig(weight=.001),
                 user_profiles=None, item_profiles=None):
        from rating_recsys.retrieval.lightgcn import LightGCNConfig
        for bank in (user_profiles or {}, item_profiles or {}):
            if any(np.asarray(value).shape != (len(CHANNELS)*384,) for value in bank.values()):
                raise ValueError("Expected flattened aspect/polarity channels")
        super().__init__(graph_config or LightGCNConfig(), alignment_config, user_profiles, item_profiles)
        self.semantic_dimension = 384
        self.channel_steps = np.zeros((3, len(CHANNELS)), dtype=np.int64)

    def _new_projector(self):
        import torch
        from torch import nn
        dimension = self.config.dimension
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.alignment_config.seed)
            model = nn.ModuleDict({"text": nn.Sequential(nn.Linear(384, (384+dimension)//2),
                                      nn.LeakyReLU(), nn.Linear((384+dimension)//2, dimension)),
                                   "graph_heads": nn.ModuleList([nn.Linear(dimension, dimension, bias=False)
                                                                for _ in CHANNELS])})
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
        return model

    def fit(self, interactions, *, callback=None):
        self.channel_steps[:] = 0
        return super().fit(interactions, callback=callback)

    def _loss_and_gradient(self, u, i, j):
        bpr, reg, gradient = LightGCN._loss_and_gradient(self, u, i, j)
        self._epoch_examples += len(u)
        self._epoch_batches += 1
        self._epoch_bpr_norm += len(u)*float(np.linalg.norm(gradient))
        if not self.alignment_config.weight or self.projector is None:
            return bpr, reg, gradient
        import torch
        import torch.nn.functional as F
        started = time.perf_counter()
        groups = self._alignment_groups(u, i, j)
        final = self._propagate(self._ego)
        grad_final = np.zeros_like(final)
        self._projector_optimizer.zero_grad(set_to_none=True)
        leaves, roles, terms = [], [], []
        losses = np.zeros(3, dtype=np.float64)
        entities = np.zeros(3, dtype=np.int64)
        for role, nodes in enumerate(groups):
            if len(nodes) < 2:
                continue
            cf = torch.tensor(final[nodes], requires_grad=True)
            text = torch.from_numpy(np.stack([self._node_profiles[int(node)] for node in nodes])).reshape(len(nodes), len(CHANNELS), 384)
            per_channel = []
            for channel in range(len(CHANNELS)):
                mask = text[:, channel].square().sum(1) > 0
                count = int(mask.sum())
                if count < 2:
                    continue
                semantic = self.projector["text"](text[mask, channel])
                graph = self.projector["graph_heads"][channel](cf[mask])
                logits = F.normalize(graph, dim=1, eps=1e-4) @ F.normalize(semantic, dim=1, eps=1e-4).T
                per_channel.append(F.cross_entropy(logits/self.alignment_config.temperature, torch.arange(count)))
                self.channel_steps[role, channel] += 1
            if not per_channel:
                continue
            term = torch.stack(per_channel).mean()
            losses[role] = float(term.detach())
            entities[role] = len(nodes)
            terms.append(term)
            leaves.append(cf)
            roles.append(nodes)
        if terms:
            (sum(terms)*self.alignment_config.weight).backward()
            for nodes, cf in zip(roles, leaves):
                np.add.at(grad_final, nodes, cf.grad.detach().numpy())
            auxiliary = self._propagate(grad_final)
            self._projector_optimizer.step()
            self._epoch_losses += len(u)*losses
            self._epoch_entities += entities
            self._epoch_aligned_batches += 1
            self._epoch_alignment_norm += len(u)*float(np.linalg.norm(auxiliary))
            gradient += auxiliary
        self._epoch_alignment_seconds += time.perf_counter()-started
        return bpr, reg, gradient
