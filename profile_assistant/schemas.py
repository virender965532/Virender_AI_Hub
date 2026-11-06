"""Pydantic models for the agents' structured outputs.

These are passed as ``response_format`` to OpenAI structured outputs, so
every field is required and has no default (strict-schema requirement).
Score ranges are enforced by validators rather than JSON-schema bounds; a
violation surfaces as a parse failure that ``llm_client`` retries once.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

QuestionType = Literal["about_person", "domain", "out_of_scope"]


class AnswerOutput(BaseModel):
    """Candidate answer produced by the answer agent."""

    reasoning: str = Field(description="Brief internal reasoning. Never shown to the user.")
    question_type: QuestionType = Field(
        description="about_person: about the person; domain: a professional topic in the person's field; "
        "out_of_scope: anything else."
    )
    in_scope: bool = Field(description="True if question_type is about_person or domain.")
    found_in_profile: bool = Field(description="True if the PROFILE contains facts relevant to this question.")
    answer: str = Field(description='The final answer, or exactly "I don\'t know."')
    evidence: list[str] = Field(description="Exact PROFILE snippets that support the answer.")


class CriticVerdict(BaseModel):
    """Review of a candidate answer by the critic agent.

    ``reasoning`` is deliberately the first field so the model reasons before
    it commits to scores and a verdict (avoids post-hoc justification).
    """

    reasoning: str = Field(description="Step-by-step review of the candidate against the PROFILE.")
    groundedness: int = Field(description="1-5: every claim is supported by the PROFILE.")
    relevance: int = Field(description="1-5: the answer addresses what was asked.")
    scope_correct: bool = Field(description='True if "I don\'t know." was used exactly when required.')
    verdict: Literal["PASS", "FAIL"]
    feedback: str = Field(description="Specific, actionable fix instructions if FAIL; empty if PASS.")

    @field_validator("groundedness", "relevance")
    @classmethod
    def _score_range(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("score must be between 1 and 5")
        return value
