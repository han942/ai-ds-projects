"""Central configuration, loaded from environment / ``.env``.

Nothing here fails hard when a credential is missing. The agent runs with just
``OPENAI_API_KEY``; Langfuse keys are optional and, when absent, all tracing
and prompt-management calls degrade to no-ops (see ``tracing.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

try:  # optional, but recommended for local dev
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional
    pass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    """Runtime settings for the agent."""

    # --- LLM (OpenAI) ---
    openai_api_key: str | None = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    # Main model that does the rewriting.
    model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )
    # A cheaper model is enough for the analyze/critique/judge steps.
    judge_model: str = field(
        default_factory=lambda: os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOKENS", "4000"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.3"))
    )
    # Only used by reasoning models (o-series / gpt-5); ignored otherwise.
    effort: str = field(default_factory=lambda: os.getenv("AGENT_EFFORT", "medium"))

    # --- Agent behaviour ---
    # Target reader described to the model everywhere.
    target_reader: str = field(
        default_factory=lambda: os.getenv(
            "TARGET_READER", "초등학교 고학년~중학생 (upper-elementary to middle-school students)"
        )
    )
    max_refine_iters: int = field(
        default_factory=lambda: int(os.getenv("MAX_REFINE_ITERS", "2"))
    )

    # --- Langfuse (optional) ---
    langfuse_public_key: str | None = field(
        default_factory=lambda: os.getenv("LANGFUSE_PUBLIC_KEY")
    )
    langfuse_secret_key: str | None = field(
        default_factory=lambda: os.getenv("LANGFUSE_SECRET_KEY")
    )
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    prompt_label: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PROMPT_LABEL", "production")
    )

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def require_llm(self) -> None:
        """Raise a friendly error if the LLM cannot be reached."""
        if not self.openai_api_key:
            raise RuntimeError(
                "No OpenAI credentials found. Set OPENAI_API_KEY in your .env."
            )


def load_settings() -> Settings:
    return Settings()
