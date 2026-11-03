"""Neural metrics: Korean sentence similarity, multilingual BERTScore, and NLI-based faithfulness.

Models load one at a time and are released afterward, so the pipeline fits in a few GB of RAM on CPU.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch

from .korean import sentences

SBERT_MODEL = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
BERTSCORE_MODEL = "bert-base-multilingual-cased"
BERTSCORE_LAYERS = 9
NLI_MODEL = "Huffon/klue-roberta-base-nli"


def sbert_similarity(pairs: dict[str, tuple[list[str], list[str]]], batch_size: int = 64) -> dict[str, np.ndarray]:
    """Cosine similarity for each named list of (a, b) pairs."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(SBERT_MODEL, device="cpu")
    cache: dict[str, np.ndarray] = {}

    def encode(texts: list[str]) -> np.ndarray:
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        if missing:
            vecs = model.encode(missing, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
            cache.update(zip(missing, vecs))
        return np.stack([cache[t] for t in texts])

    out = {name: np.sum(encode(a) * encode(b), axis=1) for name, (a, b) in pairs.items()}
    del model, cache
    gc.collect()
    return out


def bertscore(pairs: dict[str, tuple[list[str], list[str]]], batch_size: int = 32) -> dict[str, dict[str, np.ndarray]]:
    """BERTScore P/R/F1 with candidates `a` scored against references `b`."""
    from bert_score import BERTScorer

    scorer = BERTScorer(model_type=BERTSCORE_MODEL, num_layers=BERTSCORE_LAYERS,
                       batch_size=batch_size, lang="ko", device="cpu")
    # A single call lets BERTScore reuse each sentence embedding across all six
    # comparisons, including the shuffled baseline, rather than encoding it repeatedly.
    candidates = [text for cands, _ in pairs.values() for text in cands]
    references = [text for _, refs in pairs.values() for text in refs]
    p, r, f = scorer.score(candidates, references, batch_size=batch_size, verbose=True)
    out = {}
    start = 0
    for name, (cands, refs) in pairs.items():
        stop = start + len(cands)
        out[name] = {"p": p[start:stop].numpy(), "r": r[start:stop].numpy(), "f": f[start:stop].numpy()}
        start = stop
    del scorer
    gc.collect()
    return out


class _NLI:
    def __init__(self, batch_size: int):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(NLI_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).eval()
        # RoBERTa reserves position IDs through padding_idx + 1. This checkpoint
        # has 512 position embeddings and padding_idx=1, so 510 input tokens fit.
        self.max_length = min(self.tok.model_max_length,
                              self.model.config.max_position_embeddings - self.model.config.pad_token_id - 1)
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.entail, self.contra = labels["entailment"], labels["contradiction"]
        self.batch_size = batch_size

    @torch.no_grad()
    def probs(self, premises: list[str], hypotheses: list[str]) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort([len(p) + len(h) for p, h in zip(premises, hypotheses)])
        entail, contra = np.zeros(len(order)), np.zeros(len(order))
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tok([premises[i] for i in idx], [hypotheses[i] for i in idx], padding=True,
                           truncation=True, max_length=self.max_length, return_token_type_ids=False, return_tensors="pt")
            pr = torch.softmax(self.model(**enc).logits, dim=-1).numpy()
            entail[idx], contra[idx] = pr[:, self.entail], pr[:, self.contra]
            if start % (self.batch_size * 20) == 0:
                print(f"[nli] {min(start + self.batch_size, len(order))}/{len(order)} sentence pairs", flush=True)
        return entail, contra


def nli_faithfulness(systems: dict[str, tuple[list[str], list[str]]], batch_size: int = 32,
                     checkpoint_dir: Path | None = None) -> dict[str, dict[str, np.ndarray]]:
    """SummaC-style sentence-level NLI.

    Support: each output sentence checked against the whole source (does the output say anything unsupported?).
    Coverage: each source sentence checked against the whole output (how much of the source survives?).
    """
    nli = _NLI(batch_size)
    def score_pairs(name, premises, hypotheses):
        path = checkpoint_dir / f"{name}.npz" if checkpoint_dir else None
        if path and path.exists():
            print(f"[nli] reuse {name}", flush=True)
            with np.load(path, allow_pickle=False) as saved:
                return saved["entail"], saved["contra"]
        print(f"[nli] {name}: {len(premises)} pairs", flush=True)
        entail, contra = nli.probs(premises, hypotheses)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with tmp.open("wb") as stream:
                np.savez(stream, entail=entail, contra=contra)
            tmp.replace(path)
        return entail, contra

    out = {}
    for name, (srcs, outs) in systems.items():
        prem, hyp, owner = [], [], []
        for i, (s, o) in enumerate(zip(srcs, outs)):
            for sent in sentences(o):
                prem.append(s), hyp.append(sent), owner.append(i)
        e, c = score_pairs(f"{name}_support", prem, hyp)
        owner = np.array(owner)
        n = len(srcs)
        entail_min = np.full(n, np.nan)
        contra_max = np.full(n, np.nan)
        for i in range(n):
            mask = owner == i
            if mask.any():
                entail_min[i], contra_max[i] = e[mask].min(), c[mask].max()

        prem, hyp, owner = [], [], []
        for i, (s, o) in enumerate(zip(srcs, outs)):
            for sent in sentences(s):
                prem.append(o), hyp.append(sent), owner.append(i)
        e2, _ = score_pairs(f"{name}_coverage", prem, hyp)
        owner = np.array(owner)
        coverage = np.array([e2[owner == i].mean() if (owner == i).any() else np.nan for i in range(n)])
        out[name] = {
            "nli_entail_min": entail_min,
            "nli_contra_max": contra_max,
            "nli_supported": (entail_min > 0.5).astype(float),
            "nli_contradiction": (contra_max > 0.5).astype(float),
            "nli_coverage": coverage,
        }
    del nli
    gc.collect()
    return out
