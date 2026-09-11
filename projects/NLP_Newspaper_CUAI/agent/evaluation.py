"""Evaluation: LLM-as-judge (via LangChain) + a Korean readability heuristic.

The judge call is a LangChain ``ChatOpenAI`` invocation traced by Langfuse's
callback handler. Standalone evaluations also log scores to the current Langfuse
trace; inside a dataset experiment they are returned as ``Evaluation`` objects
instead (see ``dataset.py``).

Note on readability: the original project used FKGL, which is English-oriented.
For Korean we use a transparent proxy (sentence length, token length, long-token
ratio). ROUGE-L against the human reference is computed when ``rouge_score`` is
installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from . import prompts, tracing
from .config import Settings
from .llm import build_chat, extract_json, message_text

_JUDGE_AXES = ("meaning_preservation", "simplicity", "age_appropriateness", "fluency")


@dataclass
class EvalResult:
    judge: dict[str, Any] = field(default_factory=dict)
    readability: dict[str, float] = field(default_factory=dict)
    rouge_l: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"judge": self.judge, "readability": self.readability, "rouge_l": self.rouge_l}


def korean_readability(text: str) -> dict[str, float]:
    """Transparent readability proxy for Korean text (lower = easier)."""
    text = (text or "").strip()
    if not text:
        return {"sentences": 0, "avg_sentence_len": 0.0, "avg_token_len": 0.0,
                "long_token_ratio": 0.0}
    sentences = [s for s in re.split(r"[.!?…。\n]", text) if s.strip()]
    tokens = re.findall(r"\S+", text)
    n_sent = max(len(sentences), 1)
    n_tok = max(len(tokens), 1)
    long_tokens = sum(1 for t in tokens if len(t) >= 7)
    return {
        "sentences": float(len(sentences)),
        "avg_sentence_len": round(len(tokens) / n_sent, 2),
        "avg_token_len": round(sum(len(t) for t in tokens) / n_tok, 2),
        "long_token_ratio": round(long_tokens / n_tok, 3),
    }


def _rouge_l(reference: str, hypothesis: str) -> float | None:
    try:
        from rouge_score import rouge_scorer
    except Exception:
        return None
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        return round(scorer.score(reference, hypothesis)["rougeL"].fmeasure, 4)
    except Exception:
        return None


class Evaluator:
    def __init__(self, settings: Settings):
        self.settings = settings
        cfg = prompts.PROMPT_CONFIG.get("news-judge", {})
        self._chat = build_chat(
            settings,
            model=cfg.get("model", settings.judge_model),
            effort=cfg.get("effort", "medium"),
        )

    def judge(self, original: str, simplified: str) -> dict[str, Any]:
        prompt = prompts.get_prompt("news-judge", self.settings.prompt_label)
        text = prompt.compile(
            original=original, simplified=simplified,
            target_reader=self.settings.target_reader,
        )
        config: dict[str, Any] = {"run_name": "news-judge"}
        handler = tracing.get_callback_handler()
        if handler is not None:
            config["callbacks"] = [handler]
        if prompt.langfuse_obj is not None:
            config["metadata"] = {"langfuse_prompt": prompt.langfuse_obj}
        response = self._chat.invoke(
            [SystemMessage(prompts.SYSTEM_PERSONA), HumanMessage(text)], config=config
        )
        return extract_json(message_text(response.content))

    @tracing.observe(name="evaluate")
    def evaluate(
        self,
        original: str,
        simplified: str,
        reference: str | None = None,
        log_scores: bool = True,
    ) -> EvalResult:
        judge = self.judge(original, simplified)
        read_out = korean_readability(simplified)
        read_src = korean_readability(original)
        rouge = _rouge_l(reference, simplified) if reference else None

        if log_scores:
            for axis in _JUDGE_AXES + ("overall",):
                value = judge.get(axis)
                if isinstance(value, (int, float)):
                    tracing.score(name=f"judge_{axis}", value=float(value),
                                  comment=judge.get("rationale"))
            tracing.score(
                name="sentence_len_reduction",
                value=round(read_src["avg_sentence_len"] - read_out["avg_sentence_len"], 2),
            )
            if rouge is not None:
                tracing.score(name="rouge_l", value=rouge)

        return EvalResult(judge=judge, readability=read_out, rouge_l=rouge)
