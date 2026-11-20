"""Checkpointed OpenRouter / OpenAI single-shot (C0) experiment runner."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage

from . import prompts, tracing
from .config import PROJECT_ROOT, Settings
from .llm import build_chat, extract_usage, message_text, strip_thinking

DEFAULT_SPLIT_FILE = PROJECT_ROOT / "evaluation" / "splits" / "stratified_v1.json"
DEFAULT_CSV_PATH = PROJECT_ROOT / "result.csv"


class _EmptyOutputError(RuntimeError):
    """Raised when the model returns empty text after stripping thinking blocks."""


@dataclass
class C0Record:
    cache_key: str
    idx: int
    splits: list[str]
    original: str
    simplified_c0: str
    raw_output: str
    had_thinking: bool
    model: str
    provider: str
    prompt_version: str
    prompt_sha256: str
    temperature: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    latency_sec: float
    attempts: int
    timestamp: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_slug(model: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip()).strip("_")
    return slug or "model"


def make_cache_key(
    model: str,
    prompt_version: str,
    prompt_sha256: str,
    temperature: float,
    idx: int,
    original: str,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "temperature": round(float(temperature), 6),
            "idx": int(idx),
            "original": original,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        )
        return out.strip() or None
    except Exception:
        return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


class _RateLimiter:
    """Thread-safe minimum-interval pacing between outbound requests."""

    def __init__(self, min_interval: float):
        self.min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self, sleep_fn: Callable[[float], None] = time.sleep) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = self._next_allowed - now
            target = max(now, self._next_allowed) + self.min_interval
            self._next_allowed = target
        if delay > 0:
            sleep_fn(delay)


def _extract_retry_after(exc: Exception) -> float | None:
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        val = float(raw)
        return max(0.0, val)
    except Exception:
        return None


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, _EmptyOutputError):
        return True
    status = getattr(exc, "status_code", None)
    if status is None:
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None) if resp is not None else None
    if isinstance(status, int):
        return status in {408, 409, 425, 429, 500, 502, 503, 504}
    name = type(exc).__name__.lower()
    if any(k in name for k in ("ratelimit", "timeout", "connection", "internalserver", "apierror")):
        return True
    msg = str(exc).lower()
    if any(k in msg for k in ("429", "rate limit", "timed out", "timeout", "temporarily unavailable", "502", "503", "504")):
        return True
    if any(k in msg for k in ("401", "403", "invalid api key", "unauthorized")):
        return False
    return True


def _load_cached_predictions(path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cached
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        key = rec.get("cache_key")
        if (
            key
            and not rec.get("error")
            and isinstance(rec.get("simplified_c0"), str)
            and rec["simplified_c0"].strip()
        ):
            cached[key] = rec
    return cached


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {csv_path}")
    for col in ("original", "simplified_human", "simplified_by_model"):
        if col not in rows[0]:
            raise ValueError(f"Missing column {col!r} in {csv_path}")
    return rows


def list_openrouter_free_models(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai/api/v1",
    timeout: float = 20.0,
) -> list[dict[str, Any]]:
    """Query OpenRouter's model catalog and return currently free models."""
    url = base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    models: list[dict[str, Any]] = []
    for item in payload.get("data", []):
        model_id = str(item.get("id", ""))
        pricing = item.get("pricing") or {}
        p_prompt = str(pricing.get("prompt", ""))
        p_comp = str(pricing.get("completion", ""))
        is_zero_price = p_prompt in {"0", "0.0"} and p_comp in {"0", "0.0"}
        if model_id.endswith(":free") or is_zero_price:
            models.append(
                {
                    "id": model_id,
                    "name": item.get("name", model_id),
                    "context_length": int(item.get("context_length") or 0),
                }
            )
    models.sort(key=lambda m: (-m["context_length"], m["id"]))
    return models


def run_c0_experiment(
    settings: Settings,
    model: str | None = None,
    csv_path: Path = DEFAULT_CSV_PATH,
    split_file: Path = DEFAULT_SPLIT_FILE,
    split_names: tuple[str, ...] = ("eval", "stress"),
    out_dir: Path | None = None,
    limit: int | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    concurrency: int = 1,
    min_interval: float = 1.5,
    max_retries: int = 5,
    chat_factory: Callable[..., Any] | None = None,
    retry_base_delay: float = 1.5,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Run checkpointed C0 single-shot simplification over selected splits."""
    resolved_model = (model if model is not None else settings.model or "").strip()
    if chat_factory is None:
        settings.require_llm(require_model=False)
    if not resolved_model:
        raise RuntimeError(
            "No model configured. Set OPENROUTER_MODEL (or OPENAI_MODEL) "
            "in .env or pass --model."
        )

    csv_path = Path(csv_path)
    split_file = Path(split_file)
    resolved_out = Path(
        out_dir
        if out_dir is not None
        else PROJECT_ROOT / "evaluation" / "runs" / f"c0_{model_slug(resolved_model)}"
    )
    resolved_out.mkdir(parents=True, exist_ok=True)

    rows = _read_csv_rows(csv_path)
    split_bytes = split_file.read_bytes()
    split_sha256 = hashlib.sha256(split_bytes).hexdigest()
    split_payload = json.loads(split_bytes.decode("utf-8"))
    splits_map = split_payload.get("splits", {})

    index_splits: dict[int, list[str]] = {}
    for sname in split_names:
        if sname not in splits_map:
            raise KeyError(
                f"Split {sname!r} not found in {split_file}. Available: {list(splits_map)}"
            )
        for raw_idx in splits_map[sname]:
            idx = int(raw_idx)
            if not 0 <= idx < len(rows):
                raise IndexError(f"Split index {idx} out of range for {csv_path} ({len(rows)} rows)")
            tags = index_splits.setdefault(idx, [])
            if sname not in tags:
                tags.append(sname)

    ordered_indices = sorted(index_splits.keys())
    if limit is not None:
        if limit < 0:
            raise ValueError("--limit must be non-negative")
        ordered_indices = ordered_indices[:limit]

    prompt_name = "news-simplify-c0"
    prompt_obj = prompts.get_prompt(prompt_name, settings.prompt_label)
    prompt_version = prompts.C0_PROMPT_VERSION
    p_sha256 = prompts.prompt_sha256(prompt_name, settings.prompt_label)
    resolved_max_tokens = max_tokens if max_tokens is not None else settings.max_tokens

    pred_jsonl = resolved_out / "predictions.jsonl"
    err_jsonl = resolved_out / "errors.jsonl"
    cached = _load_cached_predictions(pred_jsonl)

    row_keys: dict[int, str] = {
        idx: make_cache_key(
            model=resolved_model,
            prompt_version=prompt_version,
            prompt_sha256=p_sha256,
            temperature=temperature,
            idx=idx,
            original=rows[idx]["original"],
        )
        for idx in ordered_indices
    }

    pending_indices = [idx for idx in ordered_indices if row_keys[idx] not in cached]
    cached_reused = len(ordered_indices) - len(pending_indices)

    limiter = _RateLimiter(min_interval)
    write_lock = threading.Lock()
    thread_local = threading.local()

    def _get_chat():
        chat = getattr(thread_local, "chat", None)
        if chat is None:
            if chat_factory is not None:
                chat = chat_factory(
                    settings=settings,
                    model=resolved_model,
                    temperature=temperature,
                    max_tokens=resolved_max_tokens,
                )
            else:
                chat = build_chat(
                    settings,
                    model=resolved_model,
                    temperature=temperature,
                    max_tokens=resolved_max_tokens,
                    max_retries=0,
                )
            thread_local.chat = chat
        return chat

    def _run_one(idx: int) -> C0Record:
        original = rows[idx]["original"]
        key = row_keys[idx]
        compiled = prompt_obj.compile(
            article=original, target_reader=settings.target_reader
        )
        invoke_config: dict[str, Any] = {"run_name": prompt_name}
        handler = tracing.get_callback_handler()
        if handler is not None:
            invoke_config["callbacks"] = [handler]
        if prompt_obj.langfuse_obj is not None:
            invoke_config["metadata"] = {"langfuse_prompt": prompt_obj.langfuse_obj}

        last_err: str | None = None
        for attempt in range(1, max_retries + 1):
            limiter.wait(sleep_fn)
            t0 = time.perf_counter()
            try:
                response = _get_chat().invoke(
                    [SystemMessage(prompts.SYSTEM_PERSONA), HumanMessage(compiled)],
                    config=invoke_config,
                )
                latency = round(time.perf_counter() - t0, 4)
                raw_text = message_text(getattr(response, "content", response))
                cleaned, had_thinking = strip_thinking(raw_text)
                if not cleaned:
                    raise _EmptyOutputError("Empty output after stripping thinking/fences")
                usage = extract_usage(response)
                rec = C0Record(
                    cache_key=key,
                    idx=idx,
                    splits=list(index_splits[idx]),
                    original=original,
                    simplified_c0=cleaned,
                    raw_output=raw_text,
                    had_thinking=had_thinking,
                    model=resolved_model,
                    provider=settings.provider,
                    prompt_version=prompt_version,
                    prompt_sha256=p_sha256,
                    temperature=float(temperature),
                    prompt_tokens=usage["prompt_tokens"],
                    completion_tokens=usage["completion_tokens"],
                    total_tokens=usage["total_tokens"],
                    cost_usd=usage["cost_usd"],
                    latency_sec=latency,
                    attempts=attempt,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    error=None,
                )
                with write_lock:
                    with pred_jsonl.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                        stream.flush()
                return rec
            except Exception as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt >= max_retries or not _is_retryable(exc):
                    latency = round(time.perf_counter() - t0, 4)
                    rec = C0Record(
                        cache_key=key,
                        idx=idx,
                        splits=list(index_splits[idx]),
                        original=original,
                        simplified_c0="",
                        raw_output="",
                        had_thinking=False,
                        model=resolved_model,
                        provider=settings.provider,
                        prompt_version=prompt_version,
                        prompt_sha256=p_sha256,
                        temperature=float(temperature),
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        cost_usd=None,
                        latency_sec=latency,
                        attempts=attempt,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        error=last_err[:500],
                    )
                    with write_lock:
                        with err_jsonl.open("a", encoding="utf-8") as stream:
                            stream.write(
                                json.dumps(rec.to_dict(), ensure_ascii=False) + "\n"
                            )
                            stream.flush()
                    return rec

                retry_after = _extract_retry_after(exc)
                delay = (
                    retry_after
                    if retry_after is not None
                    else (retry_base_delay * (2 ** (attempt - 1)) + random.uniform(0.0, 0.35))
                )
                if delay > 0:
                    sleep_fn(delay)

        raise AssertionError("unreachable")  # pragma: no cover

    new_records: dict[int, C0Record] = {}
    if pending_indices:
        workers = max(1, int(concurrency))
        if workers == 1:
            for done_i, idx in enumerate(pending_indices, 1):
                rec = _run_one(idx)
                new_records[idx] = rec
                if done_i % 10 == 0 or done_i == len(pending_indices):
                    print(
                        f"[c0] {done_i}/{len(pending_indices)} new calls done "
                        f"(cached={cached_reused})",
                        flush=True,
                    )
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_run_one, idx): idx for idx in pending_indices}
                for done_i, future in enumerate(as_completed(futures), 1):
                    rec = future.result()
                    new_records[rec.idx] = rec
                    if done_i % 10 == 0 or done_i == len(pending_indices):
                        print(
                            f"[c0] {done_i}/{len(pending_indices)} new calls done "
                            f"(cached={cached_reused})",
                            flush=True,
                        )

    succeeded_rows: list[dict[str, Any]] = []
    failed_indices: list[int] = []
    for idx in ordered_indices:
        key = row_keys[idx]
        if key in cached:
            rec_dict = dict(cached[key])
            rec_dict["splits"] = list(index_splits[idx])
            succeeded_rows.append(rec_dict)
        else:
            rec = new_records[idx]
            if rec.error:
                failed_indices.append(idx)
            else:
                succeeded_rows.append(rec.to_dict())

    pred_csv = resolved_out / "predictions.csv"
    tmp_csv = resolved_out / "predictions.csv.tmp"
    fieldnames = [
        "idx",
        "splits",
        "original",
        "simplified_human",
        "simplified_by_model",
        "simplified_c0",
        "latency_sec",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cost_usd",
    ]
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec_dict in succeeded_rows:
            idx = int(rec_dict["idx"])
            writer.writerow(
                {
                    "idx": idx,
                    "splits": "|".join(index_splits[idx]),
                    "original": rows[idx]["original"],
                    "simplified_human": rows[idx]["simplified_human"],
                    "simplified_by_model": rows[idx]["simplified_by_model"],
                    "simplified_c0": rec_dict["simplified_c0"],
                    "latency_sec": rec_dict.get("latency_sec"),
                    "prompt_tokens": rec_dict.get("prompt_tokens"),
                    "completion_tokens": rec_dict.get("completion_tokens"),
                    "total_tokens": rec_dict.get("total_tokens"),
                    "cost_usd": rec_dict.get("cost_usd"),
                }
            )
    tmp_csv.replace(pred_csv)

    latencies = [
        float(r["latency_sec"])
        for r in succeeded_rows
        if isinstance(r.get("latency_sec"), (int, float))
    ]
    p_toks = [
        int(r["prompt_tokens"])
        for r in succeeded_rows
        if isinstance(r.get("prompt_tokens"), (int, float))
    ]
    c_toks = [
        int(r["completion_tokens"])
        for r in succeeded_rows
        if isinstance(r.get("completion_tokens"), (int, float))
    ]
    t_toks = [
        int(r["total_tokens"])
        for r in succeeded_rows
        if isinstance(r.get("total_tokens"), (int, float))
    ]
    costs = [
        float(r["cost_usd"])
        for r in succeeded_rows
        if isinstance(r.get("cost_usd"), (int, float))
    ]
    had_thinking_count = sum(1 for r in succeeded_rows if r.get("had_thinking"))

    new_succeeded = sum(1 for r in new_records.values() if not r.error)
    manifest = {
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": settings.provider,
        "model": resolved_model,
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "prompt_sha256": p_sha256,
        "temperature": float(temperature),
        "max_tokens": resolved_max_tokens,
        "split_file": str(split_file),
        "split_file_sha256": split_sha256,
        "split_names": list(split_names),
        "limit": limit,
        "counts": {
            "requested": len(ordered_indices),
            "cached_reused": cached_reused,
            "new_succeeded": new_succeeded,
            "total_succeeded": len(succeeded_rows),
            "failed": len(failed_indices),
        },
        "failed_indices": failed_indices,
        "operational": {
            "had_thinking_count": had_thinking_count,
            "latency_mean_sec": round(sum(latencies) / len(latencies), 4)
            if latencies
            else None,
            "latency_p50_sec": round(_percentile(latencies, 0.50), 4)
            if latencies
            else None,
            "latency_p95_sec": round(_percentile(latencies, 0.95), 4)
            if latencies
            else None,
            "total_prompt_tokens": sum(p_toks) if p_toks else None,
            "total_completion_tokens": sum(c_toks) if c_toks else None,
            "total_tokens": sum(t_toks) if t_toks else None,
            "total_cost_usd": round(sum(costs), 6) if costs else None,
        },
        "out_dir": str(resolved_out),
    }
    (resolved_out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
