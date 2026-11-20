"""Central configuration, loaded from environment / ``.env``.

Nothing here fails hard when a credential is missing. The agent runs with
``OPENROUTER_API_KEY`` or ``OPENAI_API_KEY``; Langfuse keys are optional and,
when absent, all tracing and prompt-management calls degrade to no-ops (see
``tracing.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_opt_env(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


@dataclass
class Settings:
    """Runtime settings for the agent."""

    # --- LLM (OpenRouter preferred when configured, otherwise OpenAI) ---
    openrouter_api_key: str | None = field(
        default_factory=lambda: _get_opt_env("OPENROUTER_API_KEY")
    )
    openrouter_base_url: str = field(
        default_factory=lambda: (
            _get_opt_env("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        )
    )
    openrouter_site_url: str | None = field(
        default_factory=lambda: _get_opt_env("OPENROUTER_SITE_URL")
    )
    openrouter_app_name: str = field(
        default_factory=lambda: (
            _get_opt_env("OPENROUTER_APP_NAME") or "NLP-Newspaper-CUAI"
        )
    )
    openai_api_key: str | None = field(
        default_factory=lambda: _get_opt_env("OPENAI_API_KEY")
    )
    # Main model that does the rewriting. When OPENROUTER_API_KEY is set and no
    # explicit model is provided, stays empty to require explicit selection.
    model: str | None = None
    # A cheaper model is enough for the analyze/critique/judge steps.
    judge_model: str | None = None
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_TOKENS", "4000"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.3"))
    )
    request_timeout: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TIMEOUT", "90.0"))
    )
    # Only used by reasoning models (o-series / gpt-5); ignored otherwise.
    effort: str = field(default_factory=lambda: os.getenv("AGENT_EFFORT", "medium"))

    # --- Agent behaviour ---
    # Target reader described to the model everywhere.
    target_reader: str = field(
        default_factory=lambda: os.getenv(
            "TARGET_READER",
            "초등학교 고학년~중학생 (upper-elementary to middle-school students)",
        )
    )
    max_refine_iters: int = field(
        default_factory=lambda: int(os.getenv("MAX_REFINE_ITERS", "2"))
    )

    # --- Langfuse (optional) ---
    langfuse_public_key: str | None = field(
        default_factory=lambda: _get_opt_env("LANGFUSE_PUBLIC_KEY")
    )
    langfuse_secret_key: str | None = field(
        default_factory=lambda: _get_opt_env("LANGFUSE_SECRET_KEY")
    )
    langfuse_host: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )
    prompt_label: str = field(
        default_factory=lambda: os.getenv("LANGFUSE_PROMPT_LABEL", "production")
    )

    def __post_init__(self) -> None:
        self.openrouter_api_key = (self.openrouter_api_key or "").strip() or None
        self.openai_api_key = (self.openai_api_key or "").strip() or None
        self.openrouter_base_url = (
            self.openrouter_base_url or "https://openrouter.ai/api/v1"
        ).strip()
        self.openrouter_site_url = (self.openrouter_site_url or "").strip() or None
        self.openrouter_app_name = (
            self.openrouter_app_name or "NLP-Newspaper-CUAI"
        ).strip()

        if self.model is None:
            env_model = _get_opt_env("OPENROUTER_MODEL") or _get_opt_env("OPENAI_MODEL")
            if env_model:
                self.model = env_model
            elif self.openrouter_api_key:
                self.model = ""
            else:
                self.model = "gpt-4o"
        else:
            self.model = self.model.strip()

        if self.judge_model is None:
            env_judge = _get_opt_env("JUDGE_MODEL")
            if env_judge:
                self.judge_model = env_judge
            elif self.openrouter_api_key:
                self.judge_model = ""
            else:
                self.judge_model = "gpt-4o-mini"
        else:
            self.judge_model = self.judge_model.strip()

    @property
    def provider(self) -> str:
        return "openrouter" if self.openrouter_api_key else "openai"

    @property
    def api_key(self) -> str | None:
        return (
            self.openrouter_api_key
            if self.provider == "openrouter"
            else self.openai_api_key
        )

    @property
    def base_url(self) -> str | None:
        return self.openrouter_base_url if self.provider == "openrouter" else None

    @property
    def resolved_judge_model(self) -> str:
        return self.judge_model or self.model or ""

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def require_llm(self, require_model: bool = False) -> None:
        """Raise a friendly error if the LLM or required model is missing."""
        if not self.api_key:
            raise RuntimeError(
                "No LLM credentials found. Set OPENROUTER_API_KEY or "
                "OPENAI_API_KEY in your .env."
            )
        if require_model and not self.model:
            raise RuntimeError(
                "No model configured. Set OPENROUTER_MODEL (or OPENAI_MODEL) "
                "in .env or pass --model."
            )


def load_settings() -> Settings:
    return Settings()
