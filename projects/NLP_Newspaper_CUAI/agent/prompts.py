"""Prompt management.

Prompts live in Langfuse (versioned, editable in the UI without code changes)
and fall back to the local defaults below when Langfuse is unavailable. Use
``seed_prompts()`` (or ``python -m agent.cli seed-prompts``) to push the
defaults into your Langfuse project the first time.

Template variables use Langfuse mustache syntax: ``{{variable}}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import tracing

# A single persona shared by every step; the per-step user prompts below carry
# the task-specific instructions so they can be tuned independently in Langfuse.
SYSTEM_PERSONA = (
    "당신은 어려운 시사 뉴스를 초등학교 고학년~중학생이 이해할 수 있는 쉬운 한국어로 "
    "바꿔주는 전문 에디터입니다. 원문의 사실과 핵심 의미는 반드시 유지하고, 새로운 "
    "정보를 지어내지 않으며, 어려운 개념은 쉬운 말로 풀어서 설명합니다."
)

DEFAULT_PROMPTS: dict[str, str] = {
    # 1) Analyze the source article.
    "news-analyze": (
        "다음 뉴스 문장을 분석하세요. 대상 독자는 {{target_reader}}입니다.\n"
        "아래 항목을 간단한 목록으로 정리하세요:\n"
        "- 어려운 단어/한자어/전문용어\n"
        "- 배경지식이 필요한 개념\n"
        "- 문장이 너무 길거나 복잡한 부분\n"
        "- 반드시 보존해야 하는 핵심 사실\n\n"
        "뉴스: {{article}}"
    ),
    # 2) Produce the simplified rewrite.
    "news-simplify": (
        "다음 뉴스를 {{target_reader}}가 쉽게 이해할 수 있도록 다시 써주세요.\n"
        "규칙:\n"
        "1. 원래 의미와 사실을 그대로 유지하세요.\n"
        "2. 쉬운 단어를 쓰고, 어려운 개념은 풀어서 설명하세요.\n"
        "3. 한 문장을 짧게, 필요하면 여러 문장으로 나누세요.\n"
        "4. 존댓말(~해요체)로 친근하게 쓰세요.\n"
        "5. 없는 내용을 새로 지어내지 마세요.\n"
        "설명이나 머리말 없이 바꾼 결과 문장만 출력하세요.\n\n"
        "[분석]\n{{analysis}}\n\n"
        "[원문]\n{{article}}"
    ),
    # 3) Self-critique the draft (returns JSON).
    "news-critique": (
        "당신은 엄격한 편집 검수자입니다. 대상 독자는 {{target_reader}}입니다.\n"
        "아래 '쉬운 버전'이 '원문'을 잘 옮겼는지 평가하세요.\n"
        "다음 JSON 형식으로만 답하세요(다른 텍스트 금지):\n"
        '{"approved": true/false, "issues": ["문제점", ...], '
        '"suggestions": ["개선 방법", ...]}\n'
        "- 의미가 바뀌었거나, 사실이 빠졌거나, 지어낸 내용이 있거나, "
        "여전히 어려우면 approved=false 로 하세요.\n\n"
        "[원문]\n{{article}}\n\n"
        "[쉬운 버전]\n{{draft}}"
    ),
    # 4) Refine the draft using the critique.
    "news-refine": (
        "아래 '쉬운 버전'을 검수 의견을 반영해 개선하세요. 대상 독자는 "
        "{{target_reader}}입니다.\n"
        "원문의 의미와 사실은 유지하고, 지적된 문제를 고치세요.\n"
        "설명 없이 개선된 결과 문장만 출력하세요.\n\n"
        "[원문]\n{{article}}\n\n"
        "[현재 쉬운 버전]\n{{draft}}\n\n"
        "[검수 의견]\n{{critique}}"
    ),
    # 5) LLM-as-judge for evaluation (returns JSON).
    "news-judge": (
        "당신은 텍스트 간소화 품질 평가자입니다. 대상 독자는 {{target_reader}}입니다.\n"
        "'원문'과 이를 쉽게 바꾼 '결과'를 비교해 1~5점으로 평가하세요"
        "(5=매우 좋음, 1=매우 나쁨).\n"
        "평가 항목:\n"
        "- meaning_preservation: 원문의 의미/사실을 얼마나 잘 보존했는가\n"
        "- simplicity: 대상 독자에게 얼마나 쉬운가\n"
        "- age_appropriateness: 대상 연령대에 얼마나 적합한가\n"
        "- fluency: 문장이 얼마나 자연스럽고 매끄러운가\n"
        "다음 JSON 형식으로만 답하세요(다른 텍스트 금지):\n"
        '{"meaning_preservation": n, "simplicity": n, "age_appropriateness": n, '
        '"fluency": n, "overall": n, "rationale": "한 줄 이유"}\n\n'
        "[원문]\n{{original}}\n\n"
        "[결과]\n{{simplified}}"
    ),
}

# Which prompts are "chat"/"text" and their tunable config in Langfuse.
PROMPT_CONFIG: dict[str, dict[str, Any]] = {
    "news-analyze": {"model": "gpt-4o-mini", "effort": "medium"},
    "news-simplify": {"model": "gpt-4o", "effort": "high"},
    "news-critique": {"model": "gpt-4o-mini", "effort": "medium"},
    "news-refine": {"model": "gpt-4o", "effort": "high"},
    "news-judge": {"model": "gpt-4o-mini", "effort": "medium"},
}


@dataclass
class ManagedPrompt:
    """Uniform prompt handle whether sourced from Langfuse or local defaults."""

    name: str
    template: str
    langfuse_obj: Any = None  # set only when fetched from Langfuse (for linking)

    @property
    def is_local(self) -> bool:
        return self.langfuse_obj is None

    def compile(self, **variables: Any) -> str:
        if self.langfuse_obj is not None:
            try:
                return self.langfuse_obj.compile(**variables)
            except Exception:
                pass  # fall through to local rendering
        out = self.template
        for key, value in variables.items():
            out = out.replace("{{" + key + "}}", str(value))
        return out


def get_prompt(name: str, label: str = "production") -> ManagedPrompt:
    """Fetch a prompt from Langfuse, falling back to the local default."""
    if name not in DEFAULT_PROMPTS:
        raise KeyError(f"Unknown prompt: {name!r}")

    client = tracing.get_langfuse()
    if client is not None:
        try:
            lf = client.get_prompt(name, label=label)
            template = lf.prompt if isinstance(lf.prompt, str) else DEFAULT_PROMPTS[name]
            return ManagedPrompt(name=name, template=template, langfuse_obj=lf)
        except Exception:
            pass  # not seeded yet, or offline — use local

    return ManagedPrompt(name=name, template=DEFAULT_PROMPTS[name])


def seed_prompts(label: str = "production") -> list[str]:
    """Push local default prompts into Langfuse as versioned, text prompts.

    Safe to re-run: Langfuse creates a new version each time and moves the
    given label to it. Returns the list of prompt names that were pushed.
    """
    client = tracing.get_langfuse()
    if client is None:
        raise RuntimeError(
            "Langfuse is not configured. Set LANGFUSE_PUBLIC_KEY / "
            "LANGFUSE_SECRET_KEY (and LANGFUSE_HOST) before seeding prompts."
        )

    pushed: list[str] = []
    for name, template in DEFAULT_PROMPTS.items():
        client.create_prompt(
            name=name,
            type="text",
            prompt=template,
            labels=[label],
            config=PROMPT_CONFIG.get(name, {}),
        )
        pushed.append(name)
    client.flush()
    return pushed
