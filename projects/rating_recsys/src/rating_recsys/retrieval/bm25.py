"""Local Kiwi morphology + Lucene-style BM25 restaurant retrieval."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

from rating_recsys.datasets.models import Interaction
from rating_recsys.retrieval.review_profiles import build_profiles
from rating_recsys.retrieval.review_preprocessing import (
    PREPROCESSING_VERSIONS, REVIEW_STOPWORDS, clean_review_profile,
)


CONTENT_POS = frozenset(("NNG", "NNP", "VV", "VA", "XR", "SL"))


@dataclass(frozen=True, slots=True)
class BM25Config:
    name: str = field(default="bm25", init=False)
    profile_version: str = field(default="positive-recent-kiwi-content-v1", init=False)
    method: str = field(default="lucene", init=False)
    epochs: int = 1
    seed: int = 42
    k1: float = 1.2
    b: float = 0.75
    threads: int = 6
    max_user_reviews: int = 5
    max_item_reviews: int = 10
    max_review_chars: int = 240
    min_rating: float = 4.0
    preprocessing: str = "baseline"

    def __post_init__(self):
        if self.preprocessing not in PREPROCESSING_VERSIONS:
            raise ValueError("preprocessing must be baseline, clean or clean_stopwords")
        object.__setattr__(self, "profile_version", PREPROCESSING_VERSIONS[self.preprocessing])
        if self.epochs != 1:
            raise ValueError("BM25 has one indexing step; epochs must be 1")
        for name in ("threads", "max_user_reviews", "max_item_reviews", "max_review_chars"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.k1) or self.k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(self.b) or not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1")
        if not math.isfinite(self.min_rating) or not 1 <= self.min_rating <= 5:
            raise ValueError("min_rating must be between 1 and 5")

    def to_dict(self):
        return asdict(self)


class BM25Retriever:
    def __init__(self, config: BM25Config):
        self.config = config
        self.history = []
        self.index = None
        self.item_ids: tuple[int, ...] = ()
        self.user_tokens: dict[int, tuple[str, ...]] = {}
        self.metadata: dict[str, object] = {}

    def fit(self, interactions: Sequence[Interaction], texts: Mapping[int, str | None],
            query_users: Sequence[int], callback=None):
        import bm25s
        from kiwipiepy import Kiwi

        started = perf_counter()
        users, items = build_profiles(interactions, texts, query_users, self.config)
        source_profile_sha256 = hashlib.sha256(json.dumps(
            {"users": sorted(users.items()), "items": sorted(items.items())},
            ensure_ascii=False).encode()).hexdigest()
        raw_profiles = sorted(set((*users.values(), *items.values())))
        changed_profiles = 0
        if self.config.preprocessing != "baseline":
            changed_profiles = sum(clean_review_profile(doc) != doc for doc in raw_profiles)
            users = {uid: clean_review_profile(doc) for uid, doc in users.items()}
            items = {iid: clean_review_profile(doc) for iid, doc in items.items()}
        self.edge_count = len(interactions)
        self.history = []
        self.index = None
        # Sorting fixes corpus positions and exact-score tie order.
        documents = sorted(set((*users.values(), *items.values())))
        kiwi = Kiwi(num_workers=self.config.threads, model_type="cong")
        tokens = {
            document: tuple(token.form.lower() for token in parsed
                            if token.tag.split("-")[0] in CONTENT_POS)
            for document, parsed in zip(documents, kiwi.tokenize(documents, normalize_coda=True))
        }
        del kiwi
        removed = Counter()
        if self.config.preprocessing == "clean_stopwords":
            removed.update(token for doc in tokens.values() for token in doc if token in REVIEW_STOPWORDS)
            tokens = {doc: tuple(token for token in terms if token not in REVIEW_STOPWORDS)
                      for doc, terms in tokens.items()}
        self.user_tokens = {
            uid: tuple(sorted(set(tokens[doc]))) for uid, doc in sorted(users.items())
            if tokens[doc]
        }
        self.item_ids = tuple(iid for iid, doc in sorted(items.items()) if tokens[doc])
        corpus = [list(tokens[items[iid]]) for iid in self.item_ids]
        if corpus:
            self.index = bm25s.BM25(k1=self.config.k1, b=self.config.b,
                                    method=self.config.method, backend="numpy")
            self.index.index(corpus, show_progress=False)
        self.metadata = {
            "training_interactions": self.edge_count,
            "input_users": len(set(query_users)), "users_with_profile": len(users),
            "users_with_query_tokens": len(self.user_tokens),
            "users_without_query_tokens": len(set(query_users) - set(self.user_tokens)),
            "restaurants_with_profile": len(items), "indexed_restaurants": len(self.item_ids),
            "indexed_tokens": sum(map(len, corpus)),
            "vocabulary_size": len({token for doc in corpus for token in doc}),
            "unique_tokenized_profiles": len(documents),
            "source_profile_sha256": source_profile_sha256,
            "preprocessing": self.config.preprocessing,
            "preprocessing_version": self.config.profile_version,
            "cleanup_after_review_selection_and_char_truncation": True,
            "source_unique_profiles": len(raw_profiles),
            "profiles_changed_by_cleanup": changed_profiles,
            "stopwords": sorted(REVIEW_STOPWORDS) if self.config.preprocessing == "clean_stopwords" else [],
            "stopword_tokens_removed_unique_profiles": dict(sorted(removed.items())),
            "content_pos": sorted(CONTENT_POS), "kiwi_model_type": "cong",
            "normalize_coda": True, "query_term_frequency": "deduplicated",
            "document_term_frequency": "retained", "max_document_tokens": None,
            "input_through": max((row.event_date.isoformat() for row in interactions), default=None),
            "index_token_sha256": hashlib.sha256(json.dumps(
                list(zip(self.item_ids, corpus)), ensure_ascii=False).encode()).hexdigest(),
            "profile_sha256": hashlib.sha256(json.dumps(
                {"users": sorted(users.items()), "items": sorted(items.items())},
                ensure_ascii=False).encode()).hexdigest(),
        }
        stats = SimpleNamespace(epoch=1, loss=0.0, seconds=perf_counter() - started)
        self.history.append(stats)
        if callback is not None:
            callback(stats, self)
        return self

    def recommend(self, user_ids, exclude, k):
        if k < 1:
            raise ValueError("k must be positive")
        result = {}
        for uid in user_ids:
            query = self.user_tokens.get(uid, ())
            if self.index is None or not query:
                result[uid] = ()
                continue
            query = [term for term in query if term in self.index.vocab_dict]
            if not query:
                result[uid] = ()
                continue
            scores = self.index.get_scores(query)
            blocked = set(exclude.get(uid, ()))
            positions = [int(pos) for pos in np.flatnonzero(scores > 0)
                         if self.item_ids[pos] not in blocked]
            positions.sort(key=lambda pos: (-float(scores[pos]), self.item_ids[pos]))
            result[uid] = tuple(self.item_ids[pos] for pos in positions[:k])
        return result

    def save(self, folder: Path):
        """Persist the test index, ID map and query tokens for inspection/reuse."""
        folder.mkdir(parents=True, exist_ok=False)
        if self.index is not None:
            self.index.save(str(folder))
        payload = {"config": self.config.to_dict(), "metadata": self.metadata,
                   "restaurant_ids": self.item_ids, "user_query_tokens": self.user_tokens}
        (folder / "profiles.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return {"test_index": folder.name, "profiles": f"{folder.name}/profiles.json"}
