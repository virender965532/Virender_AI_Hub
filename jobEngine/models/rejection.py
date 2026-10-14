"""Structured "why was I rejected?" analysis for an applied job.

The LLM fills ``RejectionAnalysis``; ``jobEngine/matching/rejection_analyzer.py``
normalizes it (every dimension in ``REJECTION_DIMENSIONS`` present exactly once,
in that order) and stores it on the applied job with its metadata, so the
Applied Jobs page shows it without calling the LLM again.

Shape, top to bottom = how the page reads it:

- final verdict (one label + a short paragraph) and an honest, candid opinion;
- fit score and how confident the analysis is;
- the ranked primary reasons for the rejection;
- one assessment per dimension (technology, experience, seniority, location,
  timezone, work authorization, remote type, job type, compensation, domain,
  education, language) — what the job asked, what the profile shows, verdict
  and how much it likely weighed;
- matched / missing skills, red flags in the posting itself;
- concrete next steps and whether re-applying to similar roles makes sense.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

REJECTION_DIMENSIONS: tuple[str, ...] = (
    "technology",
    "experience",
    "seniority",
    "location",
    "timezone",
    "work_authorization",
    "remote_type",
    "job_type",
    "compensation",
    "domain",
    "education",
    "language",
)

Dimension = Literal[
    "technology",
    "experience",
    "seniority",
    "location",
    "timezone",
    "work_authorization",
    "remote_type",
    "job_type",
    "compensation",
    "domain",
    "education",
    "language",
]
DimensionVerdict = Literal["match", "partial", "mismatch", "unknown"]
Impact = Literal["blocking", "major", "minor", "none"]
FinalVerdict = Literal[
    "strong_fit_rejected_anyway",  # profile fits; likely competition, timing or an unstated filter
    "partial_fit",                 # fits on most axes, real gaps on one or two
    "weak_fit",                    # several substantive gaps
    "not_eligible",                # a hard filter (location, authorization, timezone, ...) ruled it out
]
Level = Literal["low", "medium", "high"]


class DimensionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Dimension
    verdict: DimensionVerdict
    impact: Impact = Field(description="How much this likely weighed in the rejection.")
    job_requirement: str = Field(
        description="What the job asks for on this axis; 'Not stated in the posting.' if absent."
    )
    candidate_fact: str = Field(
        description="What the candidate profile/preferences show; 'Not stated in the profile.' if absent."
    )
    explanation: str


class RejectionReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dimension: Dimension
    reason: str
    impact: Impact


class RejectionAnalysis(BaseModel):
    """What the LLM returns. Every list is grounded in the job text and the profile.

    No length caps here (a long answer would fail validation and cost a retry);
    the analyzer clips strings and lists before storing.
    """

    model_config = ConfigDict(extra="forbid")

    final_verdict: FinalVerdict
    verdict_summary: str = Field(description="Two to four sentences: the bottom line.")
    honest_opinion: str = Field(
        description="Candid recruiter's-eye view, no sugar-coating, still grounded in facts."
    )
    fit_score: int = Field(ge=0, le=100)
    confidence: Level = Field(description="Low when the posting is vague on the deciding axes.")
    primary_reasons: list[RejectionReason] = Field(
        default_factory=list, description="Most likely reasons for the rejection, most likely first."
    )
    dimensions: list[DimensionAssessment] = Field(default_factory=list)
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(
        default_factory=list,
        description="Problems with the posting itself (country-restricted 'remote', vague role, ghost-job signs).",
    )
    improvement_actions: list[str] = Field(
        default_factory=list, description="Concrete steps for the next similar application."
    )
    reapply_similar_roles: Literal["yes", "maybe", "no"]
    reapply_advice: str
