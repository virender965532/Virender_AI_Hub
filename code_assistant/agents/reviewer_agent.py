"""Reviewer (critic): an independent model scores the final diff with a fixed rubric.

PASS requires correctness, safety and style >= 4 and
``breaks_existing_behavior == false``; the rubric is re-checked here so a
model that says PASS with low scores is treated as FAIL.
"""

from __future__ import annotations

from ..config import CodeAssistantSettings
from ..llm_client import LLMClient
from ..observability import Trace
from ..prompts import REVIEW_USER_TEMPLATE, REVIEWER_SYSTEM
from ..schemas import ReviewVerdict

PASS_SCORE = 4
_MAX_DIFF_CHARS = 40_000


def passes(verdict: ReviewVerdict) -> bool:
    return (
        verdict.verdict == "PASS"
        and min(verdict.correctness, verdict.safety, verdict.style) >= PASS_SCORE
        and not verdict.breaks_existing_behavior
    )


class Reviewer:
    def __init__(self, llm: LLMClient, settings: CodeAssistantSettings) -> None:
        self.llm = llm
        self.settings = settings

    def review(self, task: str, diff: str, verification: str, trace: Trace) -> ReviewVerdict:
        if len(diff) > _MAX_DIFF_CHARS:
            diff = diff[:_MAX_DIFF_CHARS] + "\n...[diff truncated for review]"
        result = self.llm.parse(
            model=self.settings.reviewer_model,
            messages=[
                {"role": "system", "content": REVIEWER_SYSTEM},
                {"role": "user", "content": REVIEW_USER_TEMPLATE.format(task=task[:6000], diff=diff, verification=verification[:8000])},
            ],
            schema=ReviewVerdict, trace=trace, purpose="review",
        )
        verdict: ReviewVerdict = result.output  # type: ignore[assignment]
        if verdict.verdict == "PASS" and not passes(verdict):
            verdict = verdict.model_copy(update={
                "verdict": "FAIL",
                "feedback": (verdict.feedback + " (Rubric: every score must be >= 4 and nothing may break.)").strip(),
            })
        trace.review_verdict = verdict.verdict
        trace.emit("review", verdict.model_dump())
        return verdict
