"""LangSmith Studio entrypoint for the newspaper simplification graph."""

from __future__ import annotations

from agent.agent import NewsSimplifierAgent
from agent.config import load_settings


settings = load_settings()
settings.require_llm()

# LangGraph CLI imports this module-level compiled graph via langgraph.json.
graph = NewsSimplifierAgent(settings).graph
