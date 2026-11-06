"""Pydantic models for every structured output of the Code Assistant.

Models used as an OpenAI ``response_format`` (``RouteDecision``, ``Plan``,
``PlanStep``, ``ReviewVerdict``) have no defaults: strict structured outputs
require every field to be present. Field order matters: ``reasoning`` comes
first so the model reasons before it commits to a label or verdict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal[
    "explain", "answer_question", "fix_bug", "implement_feature", "refactor", "write_tests", "general_chat",
]
Difficulty = Literal["easy", "hard"]
Risk = Literal["low", "medium", "high"]
ChangeKind = Literal["edit", "create", "command"]
ChangeStatus = Literal["pending", "applied", "rejected", "rolled_back", "failed"]
ResponseStatus = Literal[
    "answered", "changes_pending_approval", "plan_pending_approval", "applied", "failed", "fallback",
]


class RouteDecision(BaseModel):
    """Router output: what the user wants and how hard it is."""

    reasoning: str = Field(description="One or two sentences on why this intent and difficulty fit.")
    intent: Intent
    difficulty: Difficulty = Field(description="'hard' = several files, a new feature, or unclear scope.")
    confidence: float = Field(description="0.0-1.0 confidence in the intent label.")


class PlanStep(BaseModel):
    """One node of the plan DAG."""

    id: str = Field(description="Short unique id such as 's1'.")
    description: str
    action: Literal["investigate", "edit"] = Field(
        description="'investigate' only reads code; 'edit' proposes changes."
    )
    files_involved: list[str]
    depends_on: list[str] = Field(description="Ids of steps that must finish first.")


class Plan(BaseModel):
    """Planner output: a goal and a dependency-ordered list of steps."""

    goal: str
    steps: list[PlanStep]


class ReviewVerdict(BaseModel):
    """Reviewer (critic) rubric. ``reasoning`` is produced before the verdict."""

    reasoning: str
    correctness: int = Field(description="1-5")
    safety: int = Field(description="1-5: no secrets, no destructive operations, no security holes.")
    style: int = Field(description="1-5: consistent with the surrounding code; no unrelated changes.")
    breaks_existing_behavior: bool
    verdict: Literal["PASS", "FAIL"]
    feedback: str = Field(description="Concrete, actionable fixes when FAIL; empty or brief when PASS.")


class ProposedChange(BaseModel):
    """A change waiting for (or past) human approval. Never written to disk until approved."""

    change_id: str
    path: str
    unified_diff: str
    explanation: str
    risk: Risk = "low"
    kind: ChangeKind = "edit"
    status: ChangeStatus = "pending"
    needs_human_review: bool = False
    review_concerns: str = ""
    command: list[str] | None = None
    # Full contents are kept server-side for applying; they are not part of the API payload.
    original_content: str | None = Field(default=None, exclude=True)
    new_content: str | None = Field(default=None, exclude=True)
    original_sha256: str | None = Field(default=None, exclude=True)


class CheckResult(BaseModel):
    """One verification command run (tests or linter)."""

    name: Literal["tests", "lint", "command"]
    command: list[str]
    exit_code: int | None
    passed: bool
    skipped: bool = False
    timed_out: bool = False
    output: str = ""
    new_failures: list[str] = Field(default_factory=list)
    preexisting_failures: list[str] = Field(default_factory=list)


class Verification(BaseModel):
    """Outcome of the verify loop."""

    attempts: int = 0
    passed: bool = True
    checks: list[CheckResult] = Field(default_factory=list)


class FinalResponse(BaseModel):
    """What the pipeline returns for one chat request."""

    summary: str
    changes: list[ProposedChange] = Field(default_factory=list)
    verification: Verification | None = None
    status: ResponseStatus
    trace_id: str
    intent: str | None = None
    plan: Plan | None = None
    review: ReviewVerdict | None = None
    needs_human_review: bool = False
    degraded: bool = False
    budget_exceeded: bool = False
    remaining: str | None = None
    metrics: dict = Field(default_factory=dict)
