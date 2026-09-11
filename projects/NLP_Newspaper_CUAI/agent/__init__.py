"""Langfuse-instrumented AI agent for simplifying Korean hard-news articles.

This package replaces the earlier QLoRA fine-tuning approach (see
``Final_NLP_Newspaper.ipynb``) with a LangChain agent built on OpenAI that is fully
observable through Langfuse: versioned prompts, per-step tracing with token/cost
accounting, LLM-as-judge evaluation, and dataset-backed experiments.
"""

from .config import Settings, load_settings
from .agent import NewsSimplifierAgent, SimplificationResult

__all__ = [
    "Settings",
    "load_settings",
    "NewsSimplifierAgent",
    "SimplificationResult",
]
