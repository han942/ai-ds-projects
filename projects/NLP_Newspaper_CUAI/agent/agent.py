"""The news-simplification agent, built with LangGraph.

Graph (a real cycle — the reason to use LangGraph over a plain chain):

    START -> analyze -> simplify -> critique --(approved / max iters)--> END
                                       ^
                                       └──────────── refine <──(issues)────┘

Every model call is a LangChain ``ChatOpenAI`` invocation; Langfuse traces
them automatically through its LangChain ``CallbackHandler`` (attached in
``run``). Prompts are pulled from Langfuse (with local fallbacks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from . import prompts, tracing
from .config import Settings
from .llm import build_chat, extract_json, message_text


class AgentState(TypedDict):
    article: str
    analysis: str
    draft: str
    critiques: list[dict[str, Any]]
    iterations: int
    approved: bool


@dataclass
class SimplificationResult:
    original: str
    simplified: str
    analysis: str
    iterations: int
    critiques: list[dict[str, Any]] = field(default_factory=list)
    approved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "simplified": self.simplified,
            "analysis": self.analysis,
            "iterations": self.iterations,
            "approved": self.approved,
            "critiques": self.critiques,
        }


class NewsSimplifierAgent:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._chats: dict[tuple[str, str], Any] = {}
        self.graph = self._build_graph()

    # -- LLM plumbing -----------------------------------------------------
    def _chat(self, model: str, effort: str):
        key = (model, effort)
        if key not in self._chats:
            self._chats[key] = build_chat(self.settings, model=model, effort=effort)
        return self._chats[key]

    def _complete(self, prompt_name: str, model: str, effort: str, **variables: Any) -> str:
        prompt = prompts.get_prompt(prompt_name, self.settings.prompt_label)
        text = prompt.compile(**variables)
        config: dict[str, Any] = {"run_name": prompt_name}
        if prompt.langfuse_obj is not None:
            config["metadata"] = {"langfuse_prompt": prompt.langfuse_obj}
        response = self._chat(model, effort).invoke(
            [SystemMessage(prompts.SYSTEM_PERSONA), HumanMessage(text)],
            config=config,
        )
        return message_text(response.content)

    def _complete_json(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return extract_json(self._complete(*args, **kwargs))

    def _cfg(self, name: str, model_key: str) -> tuple[str, str]:
        cfg = prompts.PROMPT_CONFIG.get(name, {})
        default_model = self.settings.model if model_key == "main" else self.settings.judge_model
        default_effort = self.settings.effort if model_key == "main" else "medium"
        return cfg.get("model", default_model), cfg.get("effort", default_effort)

    # -- graph nodes ------------------------------------------------------
    def _node_analyze(self, state: AgentState) -> dict[str, Any]:
        model, effort = self._cfg("news-analyze", "judge")
        analysis = self._complete(
            "news-analyze", model, effort,
            article=state["article"], target_reader=self.settings.target_reader,
        )
        return {"analysis": analysis}

    def _node_simplify(self, state: AgentState) -> dict[str, Any]:
        model, effort = self._cfg("news-simplify", "main")
        draft = self._complete(
            "news-simplify", model, effort,
            article=state["article"], analysis=state["analysis"],
            target_reader=self.settings.target_reader,
        )
        return {"draft": draft}

    def _node_critique(self, state: AgentState) -> dict[str, Any]:
        model, effort = self._cfg("news-critique", "judge")
        critique = self._complete_json(
            "news-critique", model, effort,
            article=state["article"], draft=state["draft"],
            target_reader=self.settings.target_reader,
        )
        return {
            "critiques": state["critiques"] + [critique],
            "approved": critique.get("approved") is True,
        }

    def _node_refine(self, state: AgentState) -> dict[str, Any]:
        model, effort = self._cfg("news-refine", "main")
        last = state["critiques"][-1] if state["critiques"] else {}
        critique_text = "\n".join(
            "- " + str(item)
            for item in (last.get("issues", []) + last.get("suggestions", []))
        ) or "특이사항 없음"
        draft = self._complete(
            "news-refine", model, effort,
            article=state["article"], draft=state["draft"],
            critique=critique_text, target_reader=self.settings.target_reader,
        )
        return {"draft": draft, "iterations": state["iterations"] + 1}

    def _route(self, state: AgentState) -> str:
        if state["approved"] or state["iterations"] >= self.settings.max_refine_iters:
            return "end"
        return "refine"

    def _build_graph(self):
        g = StateGraph(AgentState)
        g.add_node("analyze", self._node_analyze)
        g.add_node("simplify", self._node_simplify)
        g.add_node("critique", self._node_critique)
        g.add_node("refine", self._node_refine)
        g.add_edge(START, "analyze")
        g.add_edge("analyze", "simplify")
        g.add_edge("simplify", "critique")
        g.add_conditional_edges("critique", self._route, {"refine": "refine", "end": END})
        g.add_edge("refine", "critique")
        return g.compile()

    # -- public entrypoint ------------------------------------------------
    @tracing.observe(name="simplify-article")
    def run(self, article: str) -> SimplificationResult:
        article = article.strip()
        config: dict[str, Any] = {"run_name": "simplify-article"}
        handler = tracing.get_callback_handler()
        if handler is not None:
            config["callbacks"] = [handler]

        final = self.graph.invoke(
            {
                "article": article,
                "analysis": "",
                "draft": "",
                "critiques": [],
                "iterations": 0,
                "approved": False,
            },
            config=config,
        )

        tracing.update_trace(
            input=article,
            output=final["draft"],
            metadata={
                "iterations": final["iterations"],
                "approved": final["approved"],
                "model": self.settings.model,
            },
        )
        return SimplificationResult(
            original=article,
            simplified=final["draft"],
            analysis=final["analysis"],
            iterations=final["iterations"],
            critiques=final["critiques"],
            approved=final["approved"],
        )
