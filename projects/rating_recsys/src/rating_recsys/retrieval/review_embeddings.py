"""Cutoff-specific review profiles and resumable async embedding batches."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Callable, Mapping, Sequence

from rating_recsys.datasets.models import Interaction
from rating_recsys.retrieval.review_profiles import _selected_reviews, build_profiles


PROFILE_VERSION = "positive-recent-token-budget-prefix-v2"
LIQUID_MODEL = "liquid/lfm-2.5-embedding-350m:free"
TOKENIZER_REVISION = "bf1712f052040a2af193db0fe1e98c6f0df2da0f"
NEMOTRON_MODEL = "nvidia/nemotron-3-embed-1b:free"
NEMOTRON_TOKENIZER_REVISION = "c0c9fea93ea424587517f2c59e20db9f1d6bf615"


@dataclass(frozen=True, slots=True)
class ReviewEmbeddingConfig:
    name: str = field(default="review_embeddings", init=False)
    profile_version: str = field(default=PROFILE_VERSION, init=False)
    epochs: int = 1
    seed: int = 42
    model: str = LIQUID_MODEL
    dimensions: int = 1024
    send_dimensions: bool = False
    batch_size: int = 128
    concurrency: int = 2
    requests_per_minute: float = 18.0
    max_retries: int = 3
    request_timeout_seconds: float = 120.0
    aggregation: str = "concat"
    user_profile: str = "own_reviews"
    max_user_reviews: int = 5
    max_item_reviews: int = 10
    max_review_chars: int = 240
    max_document_tokens: int = 500
    min_rating: float = 4.0
    query_prefix: str = "query"
    document_prefix: str = "document"
    api_role_mode: bool = False
    tokenizer_repo: str = "LiquidAI/LFM2.5-Embedding-350M"
    tokenizer_revision: str = TOKENIZER_REVISION
    cache_path: str = "artifacts/review_embedding_cache.sqlite"

    def __post_init__(self) -> None:
        if self.api_role_mode:
            # Invalidate prefix-only probes: API roles change the representation.
            object.__setattr__(self, "profile_version", "positive-recent-token-budget-api-roles-v3")
        if not self.query_prefix.strip() or not self.document_prefix.strip():
            raise ValueError("embedding role prefixes must be nonempty")
        if self.aggregation not in ("concat", "review_mean"):
            raise ValueError("aggregation must be concat or review_mean")
        if self.user_profile not in ("own_reviews", "liked_items"):
            raise ValueError("user_profile must be own_reviews or liked_items")
        if self.user_profile == "liked_items" and self.aggregation != "review_mean":
            raise ValueError("liked_items requires review_mean aggregation")
        for name in ("dimensions", "batch_size", "concurrency", "max_user_reviews",
                     "max_item_reviews", "max_review_chars", "max_document_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.requests_per_minute <= 0 or self.max_retries < 0:
            raise ValueError("requests_per_minute must be positive and max_retries nonnegative")
        if not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be finite and positive")
        if self.model == LIQUID_MODEL and self.max_document_tokens > 500:
            raise ValueError("Liquid inputs use at most 500 tokens, reserving space below its 512-token limit")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def model_embedding_config(model: str, cache_path: str) -> ReviewEmbeddingConfig:
    """Verified model defaults; preserve Liquid's historical input/cache keys."""
    if model == LIQUID_MODEL:
        return ReviewEmbeddingConfig(cache_path=cache_path)
    if model == NEMOTRON_MODEL:
        return ReviewEmbeddingConfig(
            model=model, dimensions=2048, document_prefix="passage", api_role_mode=True,
            tokenizer_repo="nvidia/Nemotron-3-Embed-1B-BF16",
            tokenizer_revision=NEMOTRON_TOKENIZER_REVISION, cache_path=cache_path,
        )
    raise ValueError(f"Unsupported review embedding model: {model}")


class ProfileFormatter:
    """Pinned tokenizer, explicit query/document prefixes, and a token budget.

    Only tokenizer.json is downloaded; no model weights or remote Python code
    are loaded. Truncation keeps an exact character prefix of the newest-first
    profile, so it cannot introduce replacement characters at token boundaries.
    """

    def __init__(self, config: ReviewEmbeddingConfig, directory: Path, *, tokenizer=None):
        self.config = config
        self.tokenizer_sha256: str | None = None
        if tokenizer is None:
            from tokenizers import Tokenizer

            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{config.tokenizer_revision}.tokenizer.json"
            if not path.exists():
                url = (f"https://huggingface.co/{config.tokenizer_repo}/resolve/"
                       f"{config.tokenizer_revision}/tokenizer.json")
                with urllib.request.urlopen(url, timeout=60) as response:
                    data = response.read()
                staging = path.with_suffix(".staging")
                staging.write_bytes(data)
                staging.replace(path)
            self.tokenizer_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            tokenizer = Tokenizer.from_file(str(path))
        # Exported tokenizer.json enables model-side truncation/padding; disable
        # them here so audits count the full source and our own budget is explicit.
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self.tokenizer = tokenizer
        self.stats = {"profiles": 0, "truncated_profiles": 0, "original_tokens": 0,
                      "sent_tokens": 0, "max_original_tokens": 0, "max_sent_tokens": 0}

    def format(self, text: str, role: str) -> str:
        if role not in ("query", "document"):
            raise ValueError("role must be query or document")
        prefix = self.config.query_prefix if role == "query" else self.config.document_prefix
        value = f"{prefix}: {text}"
        encoded = self.tokenizer.encode(value)
        original_count = len(encoded.ids)
        limit = self.config.max_document_tokens
        if original_count > limit:
            end = max(end for _, end in encoded.offsets[:limit])
            value = value[:end]
            while len(self.tokenizer.encode(value).ids) > limit:
                value = value[:-1]
        sent_count = len(self.tokenizer.encode(value).ids)
        self.stats["profiles"] += 1
        self.stats["truncated_profiles"] += int(original_count > limit)
        self.stats["original_tokens"] += original_count
        self.stats["sent_tokens"] += sent_count
        self.stats["max_original_tokens"] = max(self.stats["max_original_tokens"], original_count)
        self.stats["max_sent_tokens"] = max(self.stats["max_sent_tokens"], sent_count)
        return value

    def prepare(self, user_docs: Mapping[int, str], item_docs: Mapping[int, str]):
        return (
            {uid: self.format(doc, "query") for uid, doc in user_docs.items()},
            {iid: self.format(doc, "document") for iid, doc in item_docs.items()},
        )

    def metadata(self) -> dict[str, object]:
        return {"tokenizer_repo": self.config.tokenizer_repo,
                "tokenizer_revision": self.config.tokenizer_revision,
                "query_prefix": self.config.query_prefix,
                "document_prefix": self.config.document_prefix,
                "api_role_mode": self.config.api_role_mode,
                "tokenizer_sha256": self.tokenizer_sha256, **self.stats}


def build_review_inputs(
    history: Sequence[Interaction], texts: Mapping[int, str | None],
    query_users: Sequence[int], config: ReviewEmbeddingConfig, formatter: ProfileFormatter,
) -> tuple[dict[int, tuple[str, ...]], dict[int, tuple[str, ...]], dict[int, tuple[int, ...]]]:
    """Select the same events as concat, but encode each review independently.

    liked_items uses the restaurants of these same latest positive, nonempty
    user review events. Repeated visits retain event weighting. It consumes
    document-side restaurant vectors instead of query-side own-review vectors.
    """
    by_user: dict[int, list[Interaction]] = defaultdict(list)
    by_item: dict[int, list[Interaction]] = defaultdict(list)
    users = set(query_users)
    for row in history:
        if row.user_id in users:
            by_user[row.user_id].append(row)
        by_item[row.restaurant_id].append(row)
    selected_users = {uid: values for uid, rows in by_user.items()
                      if (values := _selected_reviews(rows, texts, config.max_user_reviews, config))}
    selected_items = {iid: values for iid, rows in by_item.items()
                      if (values := _selected_reviews(rows, texts, config.max_item_reviews, config))}
    user_inputs = ({uid: tuple(formatter.format(text, "query") for _, text in values)
                    for uid, values in selected_users.items()}
                   if config.user_profile == "own_reviews" else {})
    item_inputs = {iid: tuple(formatter.format(text, "document") for _, text in values)
                   for iid, values in selected_items.items()}
    liked_items = {uid: tuple(row.restaurant_id for row, _ in values)
                   for uid, values in selected_users.items()}
    return user_inputs, item_inputs, liked_items


def profile_api_inputs(history, texts, query_users, config, formatter):
    """Input preparation shared by dry-run and fitting; no API access."""
    if config.aggregation == "concat":
        users, items = build_profiles(history, texts, query_users, config)
        users, items = formatter.prepare(users, items)
        return users, items, {}, [*users.values(), *items.values()]
    users, items, liked = build_review_inputs(history, texts, query_users, config, formatter)
    docs = [doc for values in (*users.values(), *items.values()) for doc in values]
    return users, items, liked, docs


def fit_review_profiles(history, texts, query_users, config, formatter, cache):
    users, items, liked, docs = profile_api_inputs(history, texts, query_users, config, formatter)
    if config.aggregation == "concat":
        fitted = ReviewEmbeddingCandidates(users, items, cache)
    else:
        embeddings = cache.embed(docs)

        def mean_unit(vectors):
            # Normalize each review, average coordinates, normalize the result.
            # A zero mean is an error, never a silently invalid cosine vector.
            import numpy as np
            return _unit(np.mean([_unit(vector) for vector in vectors], axis=0))

        item_vectors = {iid: mean_unit([embeddings[doc] for doc in values])
                        for iid, values in items.items()}
        if config.user_profile == "own_reviews":
            user_vectors = {uid: mean_unit([embeddings[doc] for doc in values])
                            for uid, values in users.items()}
        else:
            user_vectors = {uid: mean_unit([item_vectors[iid] for iid in values])
                            for uid, values in liked.items()}
        fitted = ReviewEmbeddingCandidates.from_vectors(user_vectors, item_vectors)
    fitted.profile_metadata = {
        **formatter.metadata(), "aggregation": config.aggregation,
        "user_profile": config.user_profile,
        "input_unit": "profile" if config.aggregation == "concat" else "review",
        "unique_role_inputs": len(set(docs)),
        "user_role": "document" if config.user_profile == "liked_items" else "query",
        "event_selection": "latest positive nonempty review events; repeated visits keep event weights",
        "normalization": "L2 review vectors -> unweighted mean -> L2 profile vectors"
                         if config.aggregation == "review_mean" else "L2 profile vectors",
    }
    return fitted


def _unit(vector: Sequence[float]) -> list[float]:
    values = [float(value) for value in vector]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("OpenRouter returned an empty or nonfinite embedding")
    norm = math.sqrt(sum(value ** 2 for value in values))
    if not norm:
        raise ValueError("OpenRouter returned a zero embedding")
    return [value / norm for value in values]


def configured_api_key() -> str:
    from rating_recsys.config import PROJECT_ROOT
    from dotenv import dotenv_values

    # A project-local key takes precedence over an older shell setting.
    project_key = dotenv_values(PROJECT_ROOT / ".env").get("OPENROUTER_API_KEY")
    key = (project_key or os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is missing")
    return key


class EmbeddingAPIError(RuntimeError):
    def __init__(self, status: int, *, daily_limit: bool = False):
        self.status = status
        self.daily_limit = daily_limit
        suffix = "; daily quota exhausted, rerun later to reuse cached batches" if daily_limit else ""
        super().__init__(f"OpenRouter embeddings HTTP {status}{suffix}")


class AsyncRequestLimiter:
    """Space every attempt globally, including retries across concurrent tasks."""

    def __init__(self, requests_per_minute: float):
        self.interval = 60.0 / requests_per_minute
        self.next_at = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            delay = self.next_at - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_at = time.monotonic() + self.interval


class OpenRouterEmbeddingCache:
    """Content-addressed cache with bounded async batches and durable progress."""

    def __init__(self, path: Path, config: ReviewEmbeddingConfig,
                 progress: Callable[[str], None] | None = None):
        self.path = path
        self.config = config
        self.progress = progress
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS vectors "
            "(key TEXT PRIMARY KEY, vector TEXT NOT NULL, model TEXT NOT NULL, dimensions INTEGER NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS api_calls "
            "(id INTEGER PRIMARY KEY, created_at TEXT, model TEXT, response_model TEXT, "
            "documents INTEGER, prompt_tokens INTEGER, total_tokens INTEGER, cost REAL)"
        )
        self.db.commit()
        self.usage = {"api_requests": 0, "successful_requests": 0, "api_documents": 0,
                      "cache_hits": 0, "retries": 0, "prompt_tokens": 0,
                      "total_tokens": 0, "cost_usd": 0.0}

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> OpenRouterEmbeddingCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _key(self, text: str) -> str:
        payload = json.dumps([self.config.profile_version, self.config.model,
                              self.config.dimensions, text], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def missing_count(self, texts: Sequence[str]) -> int:
        keys = {self._key(text) for text in texts}
        return sum(self.db.execute("SELECT 1 FROM vectors WHERE key=?", (key,)).fetchone() is None
                   for key in keys)

    def totals(self) -> dict[str, object]:
        row = self.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(documents),0), COALESCE(SUM(prompt_tokens),0), "
            "COALESCE(SUM(cost),0) FROM api_calls WHERE model=?", (self.config.model,),
        ).fetchone()
        return dict(zip(("successful_requests", "documents", "prompt_tokens", "cost_usd"), row))

    async def _request(self, session, texts: Sequence[str], limiter: AsyncRequestLimiter):
        import aiohttp

        payload = {"model": self.config.model, "input": list(texts)}
        if self.config.api_role_mode:
            roles = {self._api_input(text)[0] for text in texts}
            if len(roles) != 1:
                raise ValueError("An embedding request must have one API input role")
            payload["input_type"] = roles.pop()
            # API adds its model-specific role prompt; do not double-prefix.
            payload["input"] = [self._api_input(text)[1] for text in texts]
        if self.config.send_dimensions:
            payload["dimensions"] = self.config.dimensions
        for attempt in range(self.config.max_retries + 1):
            await limiter.acquire()
            self.usage["api_requests"] += 1
            retry_delay = min(30.0, 2.0 ** attempt)
            try:
                async with session.post("https://openrouter.ai/api/v1/embeddings", json=payload) as response:
                    try:
                        body = await response.json(content_type=None)
                    except (ValueError, UnicodeDecodeError):
                        if response.status < 400:
                            raise RuntimeError("OpenRouter returned an invalid JSON embedding response") from None
                        body = {}
                    if response.status >= 400:
                        issue = body.get("error") or {}
                        # Inspect classification only; do not echo review-bearing error bodies.
                        message = str(issue.get("message", "")).lower()
                        daily = response.status == 429 and ("daily" in message or "per day" in message or "per-day" in message)
                        if daily or response.status not in (429, 500, 502, 503, 504):
                            raise EmbeddingAPIError(response.status, daily_limit=daily)
                        if attempt == self.config.max_retries:
                            raise EmbeddingAPIError(response.status)
                        try:
                            retry_delay = min(60.0, max(retry_delay, float(response.headers.get("Retry-After", 0))))
                        except ValueError:
                            pass
                    else:
                        rows = sorted(body["data"], key=lambda row: row["index"])
                        if [row["index"] for row in rows] != list(range(len(texts))):
                            raise ValueError("OpenRouter returned missing or duplicate embedding indices")
                        vectors = [_unit(row["embedding"]) for row in rows]
                        if any(len(vector) != self.config.dimensions for vector in vectors):
                            raise ValueError("OpenRouter returned an unexpected embedding dimension")
                        usage = body.get("usage") or {}
                        self.usage["successful_requests"] += 1
                        self.usage["api_documents"] += len(texts)
                        for name in ("prompt_tokens", "total_tokens"):
                            self.usage[name] += int(usage.get(name) or 0)
                        self.usage["cost_usd"] += float(usage.get("cost") or 0)
                        self.db.execute(
                            "INSERT INTO api_calls(created_at,model,response_model,documents,prompt_tokens,total_tokens,cost) "
                            "VALUES (?,?,?,?,?,?,?)",
                            (datetime.now(timezone.utc).isoformat(), self.config.model, body.get("model"),
                             len(texts), int(usage.get("prompt_tokens") or 0), int(usage.get("total_tokens") or 0),
                             float(usage.get("cost") or 0)),
                        )
                        return vectors
            except (asyncio.TimeoutError, aiohttp.ClientError):
                if attempt == self.config.max_retries:
                    raise RuntimeError("OpenRouter embedding request timed out or lost its connection") from None
            self.usage["retries"] += 1
            await asyncio.sleep(retry_delay)
        raise RuntimeError("Embedding request exhausted retries")

    def embed(self, texts: Sequence[str]) -> dict[str, list[float]]:
        return asyncio.run(self.embed_async(texts))

    def prefill(self, texts: Sequence[str]) -> None:
        """Persist missing batches without holding the whole corpus in RAM."""
        asyncio.run(self.embed_async(texts, materialize=False))

    def _api_input(self, text: str) -> tuple[str, str]:
        for prefix, role in ((self.config.query_prefix, "query"),
                             (self.config.document_prefix, "passage")):
            marker = f"{prefix}: "
            if text.startswith(marker):
                return role, text[len(marker):]
        raise ValueError("Embedding input has no recognized role prefix")

    async def embed_async(self, texts: Sequence[str], *, materialize: bool = True) -> dict[str, list[float]]:
        import aiohttp

        unique = {self._key(text): text for text in texts}
        found: dict[str, list[float]] = {}
        missing: list[tuple[str, str]] = []
        for digest, text in unique.items():
            column = "vector" if materialize else "1"
            row = self.db.execute(f"SELECT {column} FROM vectors WHERE key=?", (digest,)).fetchone()
            if row:
                if materialize:
                    found[digest] = json.loads(row[0])
                self.usage["cache_hits"] += 1
            else:
                missing.append((digest, text))
        role_groups = ([missing] if not self.config.api_role_mode else [
            [pair for pair in missing if self._api_input(pair[1])[0] == role]
            for role in ("query", "passage")
        ])
        grouped_batches = [[group[start:start + self.config.batch_size]
                            for start in range(0, len(group), self.config.batch_size)]
                           for group in role_groups]
        # A quota interruption should leave both user and item roles cached.
        batches = [batch for pair in zip_longest(*grouped_batches) for batch in pair if batch]
        if batches:
            limiter = AsyncRequestLimiter(self.config.requests_per_minute)
            semaphore = asyncio.Semaphore(self.config.concurrency)
            completed = 0
            async with aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {configured_api_key()}", "X-Title": "rating_recsys"},
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_seconds),
            ) as session:
                async def process(batch):
                    nonlocal completed
                    async with semaphore:
                        vectors = await self._request(session, [text for _, text in batch], limiter)
                        self.db.executemany(
                            "INSERT OR REPLACE INTO vectors(key,vector,model,dimensions) VALUES (?,?,?,?)",
                            [(digest, json.dumps(vector), self.config.model, self.config.dimensions)
                             for (digest, _), vector in zip(batch, vectors)],
                        )
                        self.db.commit()
                        if materialize:
                            found.update((digest, vector) for (digest, _), vector in zip(batch, vectors))
                        completed += 1
                        if self.progress:
                            self.progress(f"[embeddings] {completed}/{len(batches)} batches cached; "
                                          f"{self.usage['api_documents']} API inputs")

                tasks = [asyncio.create_task(process(batch)) for batch in batches]
                try:
                    await asyncio.gather(*tasks)
                except BaseException:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise
        return {text: found[self._key(text)] for text in texts} if materialize else {}


class ReviewEmbeddingCandidates:
    def __init__(self, user_docs: Mapping[int, str], item_docs: Mapping[int, str],
                 cache: OpenRouterEmbeddingCache):
        self.user_ids = tuple(sorted(user_docs))
        self.item_ids = tuple(sorted(item_docs))
        all_vectors = cache.embed([*user_docs.values(), *item_docs.values()])
        self.users = {uid: all_vectors[doc] for uid, doc in user_docs.items()}
        self.items = {iid: all_vectors[doc] for iid, doc in item_docs.items()}

    @classmethod
    def from_vectors(cls, users, items):
        fitted = cls.__new__(cls)
        fitted.users = {uid: _unit(vector) for uid, vector in users.items()}
        fitted.items = {iid: _unit(vector) for iid, vector in items.items()}
        fitted.user_ids = tuple(sorted(fitted.users))
        fitted.item_ids = tuple(sorted(fitted.items))
        return fitted

    def recommend(self, user_ids: Sequence[int], exclude: Mapping[int, Sequence[int]],
                  k: int) -> dict[int, tuple[int, ...]]:
        import numpy as np

        item_ids = np.asarray(self.item_ids, dtype=np.int64)
        item_matrix = np.asarray([self.items[iid] for iid in self.item_ids], dtype=np.float32)
        result = {uid: () for uid in user_ids}
        known = [uid for uid in user_ids if uid in self.users]
        if not known or not len(item_ids):
            return result
        # A matrix multiply avoids thousands of small BLAS calls.
        scores = np.asarray([self.users[uid] for uid in known], dtype=np.float32) @ item_matrix.T
        for row, uid in zip(scores, known):
            seen = set(exclude.get(uid, ()))
            eligible = [index for index, iid in enumerate(self.item_ids) if iid not in seen]
            best = sorted(eligible, key=lambda index: (-float(row[index]), int(item_ids[index])))[:k]
            result[uid] = tuple(int(item_ids[index]) for index in best)
        return result
