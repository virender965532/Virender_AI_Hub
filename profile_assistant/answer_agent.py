"""Answer agent: answers a question strictly from the profile (``PA_ANSWER_MODEL``)."""

from __future__ import annotations

from typing import Any

from .config import ProfileAssistantSettings
from .llm_client import LLMClient, LLMResult
from .prompts import (
    ANSWER_REVISION_TEMPLATE,
    ANSWER_SYSTEM_PROMPT,
    ANSWER_USER_TEMPLATE,
    I_DONT_KNOW,
)
from .schemas import AnswerOutput

_MAX_HISTORY_MESSAGE_CHARS = 2000


def _content_to_text(content: Any) -> str:
    """Flatten chat content (str, or Gradio/OpenAI content parts) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, (list, tuple)):
        return " ".join(filter(None, (_content_to_text(part) for part in content)))
    return ""


def normalize_history(history: list | None, *, max_turns: int) -> list[dict[str, str]]:
    """Return the last ``max_turns`` user/assistant exchanges as chat messages.

    Accepts the messages format (``[{"role", "content"}, ...]``) and the legacy
    pairs format (``[[user, assistant], ...]``). Any other role (e.g. ``system``)
    is dropped so history can never inject system-level instructions.
    """
    messages: list[dict[str, str]] = []
    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            if role in ("user", "assistant"):
                text = _content_to_text(item.get("content")).strip()
                if text:
                    messages.append({"role": role, "content": text[:_MAX_HISTORY_MESSAGE_CHARS]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            for role, content in zip(("user", "assistant"), item):
                text = _content_to_text(content).strip()
                if text:
                    messages.append({"role": role, "content": text[:_MAX_HISTORY_MESSAGE_CHARS]})
    return messages[-2 * max_turns:] if max_turns > 0 else []


def is_i_dont_know(answer: str) -> bool:
    """True for any casing/punctuation variant of "I don't know"."""
    normalized = answer.strip().lower().replace("’", "'").rstrip(".! ")
    return normalized in ("i don't know", "i do not know", "i dont know")


def enforce_consistency(output: AnswerOutput) -> AnswerOutput:
    """Make the answer obey the "I don't know." contract regardless of model drift.

    Only an out-of-scope classification forces "I don't know."; an about-person
    answer whose specific detail is missing may still legitimately describe the
    relevant facts the profile does contain (the critic judges that).
    """
    if output.question_type == "out_of_scope" or is_i_dont_know(output.answer):
        return output.model_copy(update={"answer": I_DONT_KNOW, "evidence": []})
    return output.model_copy(update={"in_scope": True})


class AnswerAgent:
    """Generates (and revises) answers grounded only in the profile text."""

    def __init__(self, llm: LLMClient, settings: ProfileAssistantSettings) -> None:
        self.llm = llm
        self.settings = settings

    @property
    def model(self) -> str:
        return self.settings.answer_model

    def build_messages(
        self,
        *,
        question: str,
        profile_text: str,
        history: list | None,
        critic_feedback: str | None = None,
        previous_answer: AnswerOutput | None = None,
    ) -> list[dict[str, str]]:
        """Assemble system prompt, short-term history and the (revision) question."""
        messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT.format(profile=profile_text)}]
        messages += normalize_history(history, max_turns=self.settings.history_turns)
        if critic_feedback and previous_answer is not None:
            user = ANSWER_REVISION_TEMPLATE.format(
                previous_answer=previous_answer.answer,
                feedback=critic_feedback,
                question=question,
            )
        else:
            user = ANSWER_USER_TEMPLATE.format(question=question)
        messages.append({"role": "user", "content": user})
        return messages

    def answer(
        self,
        *,
        question: str,
        profile_text: str,
        history: list | None,
        trace_id: str,
        critic_feedback: str | None = None,
        previous_answer: AnswerOutput | None = None,
    ) -> LLMResult:
        """Produce an ``AnswerOutput``; pass ``critic_feedback`` + ``previous_answer`` to revise."""
        messages = self.build_messages(
            question=question,
            profile_text=profile_text,
            history=history,
            critic_feedback=critic_feedback,
            previous_answer=previous_answer,
        )
        result = self.llm.parse(model=self.model, messages=messages, schema=AnswerOutput, trace_id=trace_id)
        return LLMResult(
            output=enforce_consistency(result.output),  # type: ignore[arg-type]
            model=result.model,
            attempts=result.attempts,
            usage=result.usage,
        )
