"""Cutoff-safe frozen review blocks, review attention and candidate matching.

E5 is used only while explicitly preparing the separate block cache. Training
and inference consume immutable vectors; no LLM/API is part of this model.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rating_recsys.experiments.queries import build_training_windows
from rating_recsys.experiments.review_evidence import source_sentences
from rating_recsys.retrieval.lightgcn import _sample_negatives
from rating_recsys.retrieval.two_tower import _torch_settings
from rating_recsys.retrieval.review_embeddings import LocalEmbeddingCache

VERSION = "review-block-transformer-v1"


class ReviewBlockCache(LocalEmbeddingCache):
    """Separate namespace, including the deterministic block policy."""
    def __init__(self, path, embedding_config, config, progress=None):
        if embedding_config.max_document_tokens != config.block_tokens:
            raise ValueError("Formatter/encoder token budget must match block_tokens")
        super().__init__(path, embedding_config, progress)
        self.identity.update(profile_version=VERSION, max_blocks=config.max_blocks,
                             block_tokens=config.block_tokens, sentences_per_block=3)
        self.identity_hash = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class ReviewTransformerConfig:
    width: int = 128
    heads: int = 4
    layers: int = 2
    max_user_reviews: int = 20
    max_item_reviews: int = 30
    max_blocks: int = 4
    block_tokens: int = 256
    epochs: int = 3
    batch_size: int = 32
    negative_samples: int = 4
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    dropout: float = 0.1
    id_dropout: float = 0.1
    temperature: float = 0.1
    observed_weight: float = 1.0
    unobserved_weight: float = 0.25
    retrieval_weight: float = 1.0
    candidate_k: int = 200
    threads: int = 4
    seed: int = 42
    window_months: int = 3

    def __post_init__(self):
        for field in ("width", "heads", "layers", "max_user_reviews", "max_item_reviews",
                      "max_blocks", "block_tokens", "epochs", "batch_size", "negative_samples",
                      "candidate_k", "threads"):
            if type(getattr(self, field)) is not int or getattr(self, field) < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.width % self.heads or self.block_tokens > 512:
            raise ValueError("width must divide heads and E5 blocks must fit 512 tokens")
        if self.window_months not in (1, 2, 3, 4, 6, 12):
            raise ValueError("window_months must divide 12")
        for field in ("learning_rate", "temperature"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f"Invalid {field}")
        for field in ("weight_decay", "observed_weight", "unobserved_weight", "retrieval_weight"):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) < 0:
                raise ValueError(f"Invalid {field}")
        if not 0 <= self.dropout < 1 or not 0 <= self.id_dropout < 1:
            raise ValueError("Invalid dropout")

    def to_dict(self):
        return {"name": "review_transformer", **asdict(self)}


def review_blocks(text, formatter, config):
    """Raw-text spans; greedy 1-3 sentence blocks, token fallback for long spans.

    No 240-character clipping or high-rating gate. When the block cap is hit,
    evenly spaced blocks retain the beginning/end; omitted characters are audited.
    """
    if not text or not text.strip():
        return [], {"source_chars": 0, "retained_chars": 0, "available_blocks": 0}
    tokenizer = formatter.tokenizer
    prefix = max(len(tokenizer.encode(f"{p}: ").ids)
                 for p in (formatter.config.query_prefix, formatter.config.document_prefix))
    budget = config.block_tokens - prefix - 2
    if budget < 8:
        raise ValueError("Block token budget too small for prefixes/special tokens")
    def count(value):
        return len(tokenizer.encode(value, add_special_tokens=False).ids)
    parts = []
    for sentence in source_sentences(text):
        start, end = sentence["start"], sentence["end"]
        encoded = tokenizer.encode(text[start:end], add_special_tokens=False)
        if len(encoded.ids) <= budget:
            parts.append((start, end))
            continue
        # Only overlong single sentences use token-boundary cuts with overlap.
        step = max(1, budget - min(32, budget // 4))
        for offset in range(0, len(encoded.ids), step):
            offsets = encoded.offsets[offset:offset + budget]
            left, right = min(x[0] for x in offsets), max(x[1] for x in offsets)
            if right > left:
                parts.append((start + left, start + right))
            if offset + budget >= len(encoded.ids):
                break
    spans, pending = [], []
    for part in parts:
        if pending and (len(pending) == 3 or count(text[pending[0][0]:part[1]]) > budget):
            spans.append((pending[0][0], pending[-1][1]))
            pending = []
        pending.append(part)
    if pending:
        spans.append((pending[0][0], pending[-1][1]))
    available = len(spans)
    if available > config.max_blocks:
        picks = np.linspace(0, available - 1, config.max_blocks).astype(int)
        spans = [spans[i] for i in picks]
    blocks = [{"start": left, "end": right, "text": text[left:right]} for left, right in spans]
    covered = set()
    for left, right in spans:
        covered.update(range(left, right))
    return blocks, {"source_chars": len(text), "retained_chars": len(covered),
                    "available_blocks": available}


@dataclass
class ReviewStore:
    review_index: dict[int, int]
    vectors: torch.Tensor
    query_blocks: torch.Tensor
    document_blocks: torch.Tensor
    manifest: dict

    @classmethod
    def prepare(cls, rows, texts, formatter, cache, config, *, progress=None):
        """Explicit cache preparation; callers pass only permitted history rows."""
        if formatter.config.max_document_tokens != config.block_tokens:
            raise ValueError("Formatter token budget must match block_tokens")
        ids = sorted({r.review_id for r in rows if texts.get(r.review_id, "") and
                      texts[r.review_id].strip()})
        inputs, records = [], []
        for rid in ids:
            blocks, audit = review_blocks(texts[rid], formatter, config)
            docs = [(formatter.format(b["text"], "query"),
                     formatter.format(b["text"], "document")) for b in blocks]
            inputs.extend(doc for pair in docs for doc in pair)
            records.append({"review_id": rid, "blocks": blocks, "documents": docs, **audit})
        unique = list(dict.fromkeys(inputs))
        missing = cache.missing_count(unique)
        if progress:
            progress(f"review blocks: {len(ids)} reviews, {len(unique)} unique role inputs, {missing} misses")
        values = cache.embed(unique)
        vectors = np.zeros((len(unique) + 1, 384), np.float32)
        key = {doc: i + 1 for i, doc in enumerate(unique)}
        for doc, index in key.items():
            value = np.asarray(values[doc], np.float32)
            if value.shape != (384,) or not np.isfinite(value).all() or np.linalg.norm(value) <= 0:
                raise ValueError("Invalid frozen review vector")
            vectors[index] = value / np.linalg.norm(value)
        qblocks = np.zeros((len(ids) + 1, config.max_blocks), np.int64)
        dblocks = qblocks.copy()
        for i, record in enumerate(records, 1):
            for j, (qdoc, ddoc) in enumerate(record["documents"]):
                qblocks[i, j], dblocks[i, j] = key[qdoc], key[ddoc]
        canonical = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
        manifest = {"version": VERSION, "review_ids": ids, "reviews": len(ids),
                    "unique_inputs": len(unique), "cache_misses_before": missing,
                    "input_manifest_sha256": hashlib.sha256(canonical).hexdigest(),
                    "source_chars": sum(r["source_chars"] for r in records),
                    "retained_chars": sum(r["retained_chars"] for r in records),
                    "capped_reviews": sum(r["available_blocks"] > config.max_blocks for r in records),
                    "embedding": formatter.config.to_dict(), "formatter": formatter.metadata(),
                    "block_config": {"tokens": config.block_tokens, "max_blocks": config.max_blocks,
                                     "sentences_per_block": 3, "long_sentence_overlap_tokens": 32},
                    "cache_usage": dict(cache.usage), "cache_identity": getattr(cache, "identity", {}),
                    "api_requests": 0,
                    "records": records}
        return cls({rid: i + 1 for i, rid in enumerate(ids)}, torch.from_numpy(vectors),
                   torch.from_numpy(qblocks), torch.from_numpy(dblocks), manifest)


@dataclass
class ReviewBank:
    user_keys: tuple[int, ...]
    item_keys: tuple[int, ...]
    user_reviews: torch.Tensor
    item_reviews: torch.Tensor
    user_metadata: torch.Tensor
    item_metadata: torch.Tensor
    audit: dict

    @classmethod
    def build(cls, rows, texts, store, users, cutoff, config):
        rows = tuple(r for r in rows if r.event_date <= cutoff)
        ukeys = tuple(sorted(set(users)))
        ikeys = tuple(sorted({r.restaurant_id for r in rows}))
        by_user, by_item = defaultdict(list), defaultdict(list)
        for row in rows:
            if row.review_id in store.review_index and texts.get(row.review_id):
                by_user[row.user_id].append(row)
                by_item[row.restaurant_id].append(row)
        def arrays(keys, grouped, limit):
            refs = np.zeros((len(keys), limit), np.int64)
            metadata = np.zeros((len(keys), limit, 3), np.float32)
            selected_ids = []
            for i, key in enumerate(keys):
                selected = sorted(grouped[key], key=lambda r: (r.event_date, r.review_id), reverse=True)[:limit]
                for j, row in enumerate(selected):
                    refs[i, j] = store.review_index[row.review_id]
                    metadata[i, j] = ((row.rating - 3) / 2,
                                      math.log1p((cutoff - row.event_date).days) / 10, 1)
                    selected_ids.append(row.review_id)
            return torch.from_numpy(refs), torch.from_numpy(metadata), selected_ids
        ur, um, uids = arrays(ukeys, by_user, config.max_user_reviews)
        ir, im, iids = arrays(ikeys, by_item, config.max_item_reviews)
        selected = sorted(set(uids + iids))
        return cls(ukeys, ikeys, ur, ir, um, im, {
            "cutoff": cutoff.isoformat(), "history_rows": len(rows),
            "latest_history_date": max((r.event_date for r in rows), default=cutoff).isoformat(),
            "users": len(ukeys), "items": len(ikeys), "selected_review_ids": selected,
            "selected_reviews": len(selected), "all_rating_reviews": True,
            "selection": "latest text-bearing reviews; no rating threshold", "api_requests": 0})


def safe_padding(mask):
    """MHA requires a nonempty key set; original mask still controls outputs."""
    padding = ~mask.clone()
    padding[~mask.any(-1), 0] = False
    return padding


def attentive_pool(values, mask, query):
    logits = values @ query
    weights = torch.softmax(logits.masked_fill(safe_padding(mask), -torch.inf), dim=-1)
    weights = weights * mask
    return torch.einsum("...l,...ld->...d", weights, values)


class ReviewEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        w = config.width
        self.project = nn.Linear(384, w)
        self.block_query = nn.Parameter(torch.randn(w) / math.sqrt(w))
        self.metadata = nn.Linear(3, w)
        layer = nn.TransformerEncoderLayer(w, config.heads, w * 4, config.dropout,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.review_query = nn.Parameter(torch.randn(w) / math.sqrt(w))

    def forward(self, refs, metadata, vectors, block_ids):
        ids = block_ids[refs]
        blocks = self.project(vectors[ids])
        # Collapse blocks inside each review, never into extra review slots.
        b, r, c, d = blocks.shape
        reviews = attentive_pool(blocks.reshape(b * r, c, d), (ids != 0).reshape(b * r, c),
                                 self.block_query).reshape(b, r, d)
        mask = refs != 0
        tokens = (reviews + self.metadata(metadata)) * mask.unsqueeze(-1)
        tokens = self.transformer(tokens, src_key_padding_mask=safe_padding(mask))
        tokens = tokens * mask.unsqueeze(-1)
        pooled = attentive_pool(tokens, mask, self.review_query)
        return tokens, mask, pooled


class ReviewAttentionNetwork(nn.Module):
    def __init__(self, users, items, config):
        super().__init__()
        self.config = config
        w = config.width
        self.user_ids = nn.Embedding(users + 1, w, padding_idx=0)
        self.item_ids = nn.Embedding(items + 1, w, padding_idx=0)
        self.user_encoder = ReviewEncoder(config)
        self.item_encoder = ReviewEncoder(config)
        self.user_search = nn.Sequential(nn.Linear(w * 2 + 2, w), nn.GELU(), nn.Linear(w, w))
        self.item_search = nn.Sequential(nn.Linear(w * 2 + 2, w), nn.GELU(), nn.Linear(w, w))
        self.user_cross = nn.MultiheadAttention(w, config.heads, config.dropout, batch_first=True)
        self.item_cross = nn.MultiheadAttention(w, config.heads, config.dropout, batch_first=True)
        self.user_match_query = nn.Parameter(torch.randn(w) / math.sqrt(w))
        self.item_match_query = nn.Parameter(torch.randn(w) / math.sqrt(w))
        self.scorer = nn.Sequential(nn.Linear(w * 6 + 4, w), nn.GELU(),
                                    nn.Dropout(config.dropout), nn.Linear(w, 1))

    def encode(self, side, ids, refs, metadata, store):
        blocks = store.query_blocks if side == "user" else store.document_blocks
        tokens, mask, pooled = getattr(self, side + "_encoder")(refs, metadata, store.vectors, blocks)
        if self.training and self.config.id_dropout:
            ids = ids.masked_fill(torch.rand(ids.shape) < self.config.id_dropout, 0)
        identity = getattr(self, side + "_ids")(ids)
        numeric = torch.stack((mask.any(-1).float(), torch.log1p(mask.sum(-1).float())), -1)
        search = F.normalize(getattr(self, side + "_search")(
            torch.cat((identity, pooled, numeric), -1)), dim=-1)
        return tokens, mask, identity, numeric, search

    def match(self, user, item):
        ut, um, uid, un, _ = user
        it, im, iid, inn, _ = item
        uc, _ = self.user_cross(ut, it, it, key_padding_mask=safe_padding(im), need_weights=False)
        ic, _ = self.item_cross(it, ut, ut, key_padding_mask=safe_padding(um), need_weights=False)
        # Missing counterpart contributes no arbitrary cross-attention bias.
        uc = uc * im.any(-1)[:, None, None]
        ic = ic * um.any(-1)[:, None, None]
        u = attentive_pool(ut + uc, um, self.user_match_query)
        i = attentive_pool(it + ic, im, self.item_match_query)
        features = torch.cat((uid, iid, u, i, u * i, torch.abs(u - i), un, inn), -1)
        return self.scorer(features).squeeze(-1)


@dataclass
class TrainingWindow:
    bank: ReviewBank
    users: np.ndarray
    positives: np.ndarray
    grades: np.ndarray
    lower_items: np.ndarray
    lower_grades: np.ndarray
    forbidden: np.ndarray


class ReviewTransformer:
    def __init__(self, config, store, experiment_config):
        if experiment_config.lightgcn_checkpoint_months != config.window_months:
            raise ValueError("Training window definitions must match")
        self.config, self.store, self.experiment_config = config, store, experiment_config
        self.user_ids, self.item_ids = {}, {}
        self.trained_users, self.trained_items = set(), set()
        self.network = None
        self.bank = None
        self.history, self.training_audit = [], []
        self._encoded = None

    def build_windows(self, rows, texts):
        windows = []
        for start, queries in build_training_windows(
                rows, config=self.experiment_config, through=max(r.event_date for r in rows)):
            cutoff = queries[0].cutoff
            bank = ReviewBank.build(rows, texts, self.store, [q.user_id for q in queries], cutoff, self.config)
            ui = {key: i for i, key in enumerate(bank.user_keys)}
            ii = {key: i for i, key in enumerate(bank.item_keys)}
            users, positives, grades, lower, lower_grades, forbidden = [], [], [], [], [], []
            for q in queries:
                seen = {r.restaurant_id for r in q.history}
                blocked = seen | {r.restaurant_id for r in q.window}
                blocked_indices = [ii[key] for key in blocked if key in ii]
                u = ui[q.user_id]
                forbidden.extend(u * len(ii) + i for i in blocked_indices)
                if len(blocked_indices) == len(ii):
                    continue
                eligible = {key: grade for key, grade in q.relevance_by_item.items()
                            if key in ii and key not in seen}
                for key, grade in sorted(eligible.items()):
                    if grade <= 0:
                        continue
                    less = sorted((g, k) for k, g in eligible.items() if g < grade)
                    # Nearest lower grade yields 2>1 and 1>0 when all grades exist.
                    g, k = less[-1] if less else (-1, key)
                    users.append(u); positives.append(ii[key]); grades.append(grade)
                    lower.append(ii[k] if less else -1); lower_grades.append(g)
            self.training_audit.append({**bank.audit, "queries": len(queries),
                                        "positive_examples": len(positives),
                                        "observed_comparison_examples": sum(i >= 0 for i in lower)})
            if positives:
                windows.append(TrainingWindow(bank, *[np.asarray(x, np.int64) for x in
                    (users, positives, grades, lower, lower_grades)], np.unique(np.asarray(forbidden, np.int64))))
        return windows

    def _embedding_ids(self, keys, side):
        mapping, trained = getattr(self, side + "_ids"), getattr(self, "trained_" + side + "s")
        return torch.tensor([mapping[key] if key in trained else 0 for key in keys])

    def fit(self, rows, texts, *, cutoff, eval_users, callback=None, max_examples=None, progress=None,
            training_checkpoint=None, resume_from=None, checkpoint_context=None):
        rows = tuple(r for r in rows if r.event_date <= cutoff)
        if not rows:
            raise ValueError("No training rows")
        self.user_ids = {key: i + 1 for i, key in enumerate(sorted({r.user_id for r in rows}))}
        self.item_ids = {key: i + 1 for i, key in enumerate(sorted({r.restaurant_id for r in rows}))}
        self.training_audit, self.history = [], []
        windows = self.build_windows(rows, texts)
        if not windows:
            raise ValueError("No eligible training windows")
        rng = np.random.default_rng(self.config.seed)
        if max_examples is not None:
            if max_examples < 1:
                raise ValueError("max_examples must be positive")
            # Random prespecified pilot sampling across all windows, never validation outcomes.
            sizes = [len(w.users) for w in windows]
            total = sum(sizes)
            picks = set(rng.choice(total, min(total, max_examples), replace=False).tolist())
            offset = 0
            for w, size in zip(windows, sizes):
                indices = [i for i in range(size) if i + offset in picks]
                for field in ("users", "positives", "grades", "lower_items", "lower_grades"):
                    setattr(w, field, getattr(w, field)[indices])
                offset += size
            windows = [w for w in windows if len(w.users)]
        self.trained_users = {w.bank.user_keys[u] for w in windows for u in w.users}
        # All eligible sampled negative items get an ID signal; exact participated IDs recorded below.
        self.trained_items = set()
        self.bank = ReviewBank.build(rows, texts, self.store, eval_users, cutoff, self.config)
        self.metadata = {"training_examples": sum(len(w.users) for w in windows),
                         "available_training_examples": sum(a["positive_examples"] for a in self.training_audit),
                         "pilot_max_examples": max_examples, "evaluation_bank": self.bank.audit,
                         "training_audit": self.training_audit,
                         "negative_policy": "cutoff catalog minus seen and all window outcomes",
                         "objective": "gain-weighted retrieval CE + weak unobserved BPR + observed graded BPR",
                         "e5_trainable": False, "api_requests": 0}
        contract = {"config": {k:v for k,v in self.config.to_dict().items() if k != "epochs"},
                    "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "experiment": self.experiment_config.to_dict(), "cutoff": cutoff.isoformat(),
                    "input_sha256": self.store.manifest.get("input_manifest_sha256"),
                    "frozen_tensors_sha256": {name:hashlib.sha256(memoryview(
                        getattr(self.store,name).numpy()).cast("B")).hexdigest()
                        for name in ("vectors","query_blocks","document_blocks")},
                    "rows_sha256": hashlib.sha256(repr(rows).encode()).hexdigest(),
                    "max_examples": max_examples, "eval_users": list(eval_users),
                    "user_ids": self.user_ids, "item_ids": self.item_ids}
        saved = torch.load(resume_from, map_location="cpu", weights_only=False) if resume_from else None
        if saved is not None and (saved.get("version") != VERSION or saved.get("contract") != contract):
            raise ValueError("Training checkpoint input/config contract differs")
        first_epoch = saved["epoch"] + 1 if saved else 1
        if first_epoch > self.config.epochs:
            raise ValueError("Training checkpoint has exhausted the epoch budget")
        with _torch_settings(self.config):
            self.network = ReviewAttentionNetwork(len(self.user_ids), len(self.item_ids), self.config)
            optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay)
            if saved:
                self.network.load_state_dict(saved["network"])
                optimizer.load_state_dict(saved["optimizer"])
                self.trained_items = saved["trained_items"]
                self.history = saved["history"]
                rng.bit_generator.state = saved["numpy_rng"]
                torch.set_rng_state(saved["torch_rng"])
                del saved
            for epoch in range(first_epoch, self.config.epochs + 1):
                started, totals, count = time.perf_counter(), np.zeros(4), 0
                last_log = started
                self.network.train(); self._encoded = None
                for wi in rng.permutation(len(windows)):
                    w = windows[wi]
                    negatives = _sample_negatives(rng, np.repeat(w.users, self.config.negative_samples),
                                                  len(w.bank.item_keys), w.forbidden).reshape(-1, self.config.negative_samples)
                    order = rng.permutation(len(w.users))
                    for start in range(0, len(order), self.config.batch_size):
                        ix = order[start:start + self.config.batch_size]
                        user_rows = torch.from_numpy(w.users[ix])
                        choices = np.column_stack((w.positives[ix], negatives[ix],
                            np.where(w.lower_items[ix] >= 0, w.lower_items[ix], w.positives[ix])))
                        flat, inverse = np.unique(choices, return_inverse=True)
                        inverse = torch.from_numpy(inverse.reshape(choices.shape))
                        keys = [w.bank.item_keys[i] for i in flat]
                        self.trained_items.update(keys)
                        u = self.network.encode("user", self._embedding_ids([w.bank.user_keys[i] for i in user_rows], "user"),
                                                w.bank.user_reviews[user_rows], w.bank.user_metadata[user_rows], self.store)
                        item_rows = torch.from_numpy(flat)
                        item = self.network.encode("item", self._embedding_ids(keys, "item"),
                                                   w.bank.item_reviews[item_rows], w.bank.item_metadata[item_rows], self.store)
                        pick = tuple(value[inverse] for value in item)
                        search = torch.einsum("bd,bnd->bn", u[-1], pick[-1][:, :-1]) / self.config.temperature
                        gains = torch.from_numpy((2.0 ** w.grades[ix] - 1).astype(np.float32))
                        retrieval = (F.cross_entropy(search, torch.zeros(len(ix), dtype=torch.long), reduction="none") * gains).sum() / gains.sum()
                        repeated = tuple(value.repeat_interleave(choices.shape[1], 0) for value in u)
                        paired = tuple(value.reshape((-1,) + value.shape[2:]) for value in pick)
                        scores = self.network.match(repeated, paired).reshape(choices.shape)
                        weak = (F.softplus(scores[:, 1:-1] - scores[:, :1]).mean(-1) * gains).sum() / gains.sum()
                        observed_mask = torch.from_numpy(w.lower_items[ix] >= 0)
                        if observed_mask.any():
                            differences = torch.from_numpy((2.0 ** w.grades[ix] - 2.0 ** w.lower_grades[ix]).astype(np.float32))
                            observed = (F.softplus(scores[observed_mask, -1] - scores[observed_mask, 0]) * differences[observed_mask]).sum() / differences[observed_mask].sum()
                        else:
                            observed = scores.sum() * 0
                        loss = (self.config.retrieval_weight * retrieval + self.config.unobserved_weight * weak
                                + self.config.observed_weight * observed)
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite review Transformer loss")
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        nn.utils.clip_grad_norm_(self.network.parameters(), 5.0)
                        optimizer.step()
                        totals += np.array([loss.item(), retrieval.item(), weak.item(), observed.item()]) * len(ix)
                        count += len(ix)
                        if progress and time.perf_counter() - last_log >= 30:
                            progress(f"epoch {epoch}: {count}/{self.metadata['training_examples']} examples; loss={totals[0]/count:.4f}")
                            last_log = time.perf_counter()
                self.network.eval(); self._encoded = None
                stats = {"epoch": epoch, "loss": totals[0] / count, "retrieval_loss": totals[1] / count,
                         "unobserved_loss": totals[2] / count, "observed_loss": totals[3] / count,
                         "seconds": time.perf_counter() - started}
                self.history.append(stats)
                stop = callback(stats, self) if callback else False
                if training_checkpoint:
                    path = Path(training_checkpoint)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    staging = path.with_suffix(path.suffix + ".tmp")
                    torch.save({"version": VERSION, "contract": contract, "epoch": epoch,
                                "network": self.network.state_dict(), "optimizer": optimizer.state_dict(),
                                "trained_items": self.trained_items, "history": self.history,
                                "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                                "context": checkpoint_context() if checkpoint_context else None}, staging)
                    staging.replace(path)
                if stop:
                    break
        self._encoded = None
        return self

    def _encode(self):
        if self.network is None or self.bank is None:
            raise RuntimeError("Model must be fitted first")
        if self._encoded is None:
            self.network.eval()
            with _torch_settings(self.config), torch.no_grad():
                def encode(side, keys, refs, meta):
                    chunks = []
                    for start in range(0, len(keys), self.config.batch_size):
                        part = keys[start:start + self.config.batch_size]
                        chunks.append(self.network.encode(side, self._embedding_ids(part, side),
                                      refs[start:start + len(part)], meta[start:start + len(part)], self.store))
                    if not chunks:
                        return None
                    return tuple(torch.cat([c[i] for c in chunks]) for i in range(5))
                self._encoded = (encode("user", self.bank.user_keys, self.bank.user_reviews, self.bank.user_metadata),
                                 encode("item", self.bank.item_keys, self.bank.item_reviews, self.bank.item_metadata))
        return self._encoded

    def rankings(self, users, exclude, k):
        """Return independent retrieval order and cross-attention reranked order."""
        if k < 1 or k > self.config.candidate_k:
            raise ValueError("k must be within configured candidate_k")
        uvalues, ivalues = self._encode()
        result, retrieval = {u: () for u in users}, {u: () for u in users}
        if uvalues is None or ivalues is None:
            return retrieval, result
        ui = {key: i for i, key in enumerate(self.bank.user_keys)}
        ii = {key: i for i, key in enumerate(self.bank.item_keys)}
        started = time.perf_counter()
        with _torch_settings(self.config), torch.no_grad():
            for uid in users:
                if uid not in ui:
                    continue
                row = ui[uid]
                values = (ivalues[-1] @ uvalues[-1][row]).numpy()
                values[[ii[key] for key in exclude.get(uid, ()) if key in ii]] = -np.inf
                order = np.argsort(-values, kind="stable")[:self.config.candidate_k]
                order = [i for i in order if np.isfinite(values[i])]
                retrieval[uid] = tuple(self.bank.item_keys[i] for i in order)
                scores = []
                for start in range(0, len(order), self.config.batch_size):
                    part = torch.tensor(order[start:start + self.config.batch_size], dtype=torch.long)
                    u = tuple(v[row:row + 1].expand((len(part),) + v.shape[1:]) for v in uvalues)
                    item = tuple(v[part] for v in ivalues)
                    scores.extend(self.network.match(u, item).tolist())
                ranked = sorted(zip(order, scores), key=lambda p: (-p[1], self.bank.item_keys[p[0]]))
                result[uid] = tuple(self.bank.item_keys[i] for i, _ in ranked[:k])
        self.metadata["last_ranking_seconds"] = time.perf_counter() - started
        return retrieval, result

    def recommend(self, users, exclude, k):
        return self.rankings(users, exclude, k)[1]

    def save(self, path):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"version": VERSION, "config": asdict(self.config),
                    "experiment": self.experiment_config.to_dict(), "network": self.network.state_dict(),
                    "store": self.store, "bank": self.bank, "user_ids": self.user_ids, "item_ids": self.item_ids,
                    "trained_users": self.trained_users, "trained_items": self.trained_items,
                    "metadata": self.metadata, "history": self.history}, path)
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}

    @classmethod
    def load(cls, path):
        from rating_recsys.experiments.config import ExperimentConfig
        # Trusted local checkpoints only: torch 2.5 pickle includes bank dataclasses.
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["version"] != VERSION:
            raise ValueError("Unsupported review Transformer checkpoint")
        result = cls(ReviewTransformerConfig(**saved["config"]), saved["store"], ExperimentConfig(**saved["experiment"]))
        for name in ("bank", "user_ids", "item_ids", "trained_users", "trained_items", "metadata", "history"):
            setattr(result, name, saved[name])
        with _torch_settings(result.config):
            result.network = ReviewAttentionNetwork(len(result.user_ids), len(result.item_ids), result.config)
            result.network.load_state_dict(saved["network"]); result.network.eval()
        return result
