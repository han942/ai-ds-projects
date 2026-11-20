"""LangChain (``ChatOpenAI``) model factory + response parsing helpers.

Supports OpenAI and OpenAI-compatible providers such as OpenRouter. Langfuse
traces calls via its LangChain ``CallbackHandler`` (see ``tracing.py``).
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI

from .config import Settings

# OpenAI reasoning models take ``reasoning_effort`` instead of ``temperature``.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:[a-zA-Z0-9_-]+)?\s*\n?(.*?)\n?```\s*$", re.DOTALL
)


def _is_reasoning_model(model: str) -> bool:
    short = model.rsplit("/", 1)[-1].lower()
    return any(short.startswith(p) for p in _REASONING_PREFIXES)


def build_chat(
    settings: Settings,
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    max_retries: int | None = None,
) -> ChatOpenAI:
    """Create a ``ChatOpenAI`` configured for OpenRouter or OpenAI.

    Standard chat models use ``temperature``; OpenAI reasoning models
    (o-series / gpt-5) use ``reasoning_effort`` instead.
    """
    resolved_model = (model if model is not None else settings.model or "").strip()
    if not resolved_model:
        raise ValueError(
            "No model specified. Set OPENROUTER_MODEL (or OPENAI_MODEL) or pass --model."
        )

    kwargs: dict[str, Any] = {
        "model": resolved_model,
        "max_tokens": max_tokens if max_tokens is not None else settings.max_tokens,
        "api_key": settings.api_key or None,
        "timeout": settings.request_timeout,
    }
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.provider == "openrouter":
        headers = {"X-Title": settings.openrouter_app_name}
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        kwargs["default_headers"] = headers

    if _is_reasoning_model(resolved_model):
        kwargs["reasoning_effort"] = effort if effort is not None else settings.effort
    else:
        kwargs["temperature"] = (
            temperature if temperature is not None else settings.temperature
        )
    return ChatOpenAI(**kwargs)


def message_text(content: Any) -> str:
    """Extract plain text from an AIMessage content (str or content blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts).strip()
    return str(content).strip()


def strip_thinking(text: str) -> tuple[str, bool]:
    """Remove ``<think>...</think>`` blocks and outer markdown code fences."""
    raw = (text or "").strip()
    if not raw:
        return "", False

    had_thinking = False
    if _THINK_BLOCK_RE.search(raw):
        raw = _THINK_BLOCK_RE.sub("", raw).strip()
        had_thinking = True
    elif "</think>" in raw.lower() and "<think>" not in raw.lower():
        # Some reasoning templates inject <think> in the prompt prefix so only </think> appears.
        parts = re.split(r"</think>", raw, flags=re.IGNORECASE)
        raw = parts[-1].strip()
        had_thinking = True

    fence = _CODE_FENCE_RE.match(raw)
    if fence:
        raw = fence.group(1).strip()

    return raw.strip(), had_thinking


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def extract_usage(response: Any) -> dict[str, Any]:
    """Safely extract token counts and USD cost from a LangChain AIMessage."""
    usage_meta = getattr(response, "usage_metadata", None)
    resp_meta = getattr(response, "response_metadata", None)
    usage_meta = usage_meta if isinstance(usage_meta, dict) else {}
    resp_meta = resp_meta if isinstance(resp_meta, dict) else {}

    token_usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
    if not isinstance(token_usage, dict):
        token_usage = {}

    prompt_tokens = _to_int(
        usage_meta.get("input_tokens")
        if usage_meta.get("input_tokens") is not None
        else token_usage.get("prompt_tokens")
    )
    completion_tokens = _to_int(
        usage_meta.get("output_tokens")
        if usage_meta.get("output_tokens") is not None
        else token_usage.get("completion_tokens")
    )
    total_tokens = _to_int(
        usage_meta.get("total_tokens")
        if usage_meta.get("total_tokens") is not None
        else token_usage.get("total_tokens")
    )
    if (
        total_tokens is None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        total_tokens = prompt_tokens + completion_tokens

    cost_usd = _to_float(
        token_usage.get("cost")
        if token_usage.get("cost") is not None
        else usage_meta.get("total_cost", resp_meta.get("cost"))
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating thinking tags and prose."""
    cleaned, _ = strip_thinking(text)
    for candidate in (cleaned, text):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        match = re.search(r"\{.*\}", candidate, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
    return {}
