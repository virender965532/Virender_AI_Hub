"""Critic agent: reviews candidate answers with a different model (``PA_CRITIC_MODEL``)."""

from __future__ import annotations

from .config import ProfileAssistantSettings
from .llm_client import LLMClient, LLMResult
from .prompts import CRITIC_SYSTEM_PROMPT, CRITIC_USER_TEMPLATE
from .schemas import AnswerOutput, CriticVerdict

PASS_THRESHOLD = 4


def apply_rubric(verdict: CriticVerdict) -> CriticVerdict:
    """Enforce the PASS rule in code so a lenient model verdict cannot slip through."""
    meets = (
        verdict.groundedness >= PASS_THRESHOLD
        and verdict.relevance >= PASS_THRESHOLD
        and verdict.scope_correct
    )
    if verdict.verdict == "PASS" and not meets:
        feedback = verdict.feedback or (
            "Scores are below the passing bar: remove anything not supported by the PROFILE "
            "and answer exactly what was asked, or answer \"I don't know.\" if the PROFILE lacks it."
        )
        return verdict.model_copy(update={"verdict": "FAIL", "feedback": feedback})
    return verdict


class CriticAgent:
    """Scores groundedness, relevance and scope of a candidate answer."""

    def __init__(self, llm: LLMClient, settings: ProfileAssistantSettings) -> None:
        self.llm = llm
        self.settings = settings

    @property
    def model(self) -> str:
        return self.settings.critic_model

    def build_messages(self, *, question: str, profile_text: str, candidate: AnswerOutput) -> list[dict[str, str]]:
        """Assemble the review prompt for one candidate answer."""
        evidence = "\n".join(f"- {snippet}" for snippet in candidate.evidence) or "(none)"
        return [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT.format(profile=profile_text)},
            {
                "role": "user",
                "content": CRITIC_USER_TEMPLATE.format(
                    question=question,
                    question_type=candidate.question_type,
                    answer=candidate.answer,
                    in_scope=candidate.in_scope,
                    found_in_profile=candidate.found_in_profile,
                    evidence=evidence,
                ),
            },
        ]

    def review(self, *, question: str, profile_text: str, candidate: AnswerOutput, trace_id: str) -> LLMResult:
        """Return a ``CriticVerdict`` for ``candidate`` (rubric enforced)."""
        messages = self.build_messages(question=question, profile_text=profile_text, candidate=candidate)
        result = self.llm.parse(model=self.model, messages=messages, schema=CriticVerdict, trace_id=trace_id)
        return LLMResult(
            output=apply_rubric(result.output),  # type: ignore[arg-type]
            model=result.model,
            attempts=result.attempts,
            usage=result.usage,
        )
