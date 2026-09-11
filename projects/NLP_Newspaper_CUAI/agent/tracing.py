"""Thin, defensive wrapper around the Langfuse SDK.

Design goal: the rest of the codebase can call ``observe``, ``update_trace``,
``score``, ``get_langfuse`` and ``get_callback_handler`` unconditionally. When
Langfuse is not installed or not configured, every call becomes a no-op and
``observe`` returns the function unchanged. This keeps the agent runnable with
nothing but an OpenAI key, while giving full observability the moment Langfuse
keys are present.

Model calls are traced through the LangChain ``CallbackHandler``
(``get_callback_handler``); this module adds trace-level naming/scores on top.
Targets the Langfuse Python SDK v4 (OpenTelemetry-based).
"""

from __future__ import annotations

import functools
import os
from typing import Any, Callable

_ENABLED_ENV = bool(
    os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")
)

try:
    from langfuse import Langfuse, get_client
    from langfuse import observe as _lf_observe

    _IMPORT_OK = True
except Exception:  # pragma: no cover - langfuse is optional
    _IMPORT_OK = False


def is_enabled() -> bool:
    return _ENABLED_ENV and _IMPORT_OK


_client: "Langfuse | None" = None


def get_langfuse() -> "Langfuse | None":
    """Return a cached Langfuse client, or ``None`` when disabled."""
    global _client
    if not is_enabled():
        return None
    if _client is None:
        try:
            _client = get_client()
        except Exception:
            return None
    return _client


_handler: Any = None


def get_callback_handler() -> Any:
    """Return a cached Langfuse LangChain ``CallbackHandler``, or ``None``.

    Pass it in a LangChain/LangGraph ``config={"callbacks": [handler]}`` to get
    automatic tracing of every model call (tokens, cost, latency).
    """
    global _handler
    if not is_enabled():
        return None
    if _handler is None:
        try:
            from langfuse.langchain import CallbackHandler

            _handler = CallbackHandler()
        except Exception:
            return None
    return _handler


def observe(*d_args: Any, **d_kwargs: Any) -> Callable:
    """``@observe`` that degrades to an identity decorator when disabled.

    Supports both bare ``@observe`` and parameterised
    ``@observe(name="...", as_type="generation")`` usage.
    """

    def _wrap(fn: Callable) -> Callable:
        if is_enabled():
            return _lf_observe(**d_kwargs)(fn)
        return fn

    # Bare @observe usage: observe(fn)
    if len(d_args) == 1 and callable(d_args[0]) and not d_kwargs:
        fn = d_args[0]
        return _lf_observe()(fn) if is_enabled() else fn

    return _wrap


def _safe(method: str, **kwargs: Any) -> None:
    client = get_langfuse()
    if client is None:
        return
    try:
        getattr(client, method)(**kwargs)
    except Exception:
        # Observability must never take down the main flow.
        pass


def update_span(**kwargs: Any) -> None:
    _safe("update_current_span", **kwargs)


def update_trace(**kwargs: Any) -> None:
    """Set trace-level attributes, tolerant of SDK v3 vs v4 differences.

    v3 exposed ``update_current_trace``; v4 splits this into
    ``set_current_trace_io`` (input/output) plus span-level name/metadata.
    """
    client = get_langfuse()
    if client is None:
        return
    if hasattr(client, "update_current_trace"):  # SDK v3
        try:
            client.update_current_trace(**kwargs)
            return
        except Exception:
            pass
    # SDK v4 fallback
    io = {k: kwargs[k] for k in ("input", "output") if k in kwargs}
    if io and hasattr(client, "set_current_trace_io"):
        try:
            client.set_current_trace_io(**io)
        except Exception:
            pass
    span_attrs = {k: kwargs[k] for k in ("name", "metadata") if k in kwargs}
    if span_attrs:
        try:
            client.update_current_span(**span_attrs)
        except Exception:
            pass


def score(name: str, value: float | str, comment: str | None = None, **kwargs: Any) -> None:
    """Attach a score to the current trace."""
    payload: dict[str, Any] = {"name": name, "value": value}
    if comment is not None:
        payload["comment"] = comment
    payload.update(kwargs)
    _safe("score_current_trace", **payload)


def flush() -> None:
    client = get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass
