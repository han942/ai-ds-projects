"""LangChain (``ChatOpenAI``) model factory + helpers.

The agent is built on LangChain / LangGraph; Langfuse is wired in only for
tracing via its LangChain ``CallbackHandler`` (see ``tracing.py``). This module
just builds configured chat models and parses JSON from their output.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_openai import ChatOpenAI

from .config import Settings

# OpenAI reasoning models take ``reasoning_effort`` instead of ``temperature``.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    return any(model.startswith(p) for p in _REASONING_PREFIXES)


def build_chat(
    settings: Settings,
    model: str | None = None,
    effort: str | None = None,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    """Create a ``ChatOpenAI`` configured for the news task.

    Standard chat models (gpt-4o, ...) use ``temperature``; reasoning models
    (o-series / gpt-5) use ``reasoning_effort`` instead.
    """
    model = model or settings.model
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens or settings.max_tokens,
        "api_key": settings.openai_api_key or None,
    }
    if _is_reasoning_model(model):
        kwargs["reasoning_effort"] = effort or settings.effort
    else:
        kwargs["temperature"] = settings.temperature
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


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences / prose."""
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}
