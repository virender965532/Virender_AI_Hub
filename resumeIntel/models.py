"""Structured LLM output for resume-vs-JD gap analysis."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ResumeGapAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_fit_summary: str = Field(
        description="2-3 sentence, evidence-grounded summary of how well the resume fits this job."
    )
    matched_skills: list[str] = Field(
        default_factory=list,
        description="Skills/keywords required or preferred by the job that ARE clearly demonstrated in the resume.",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="Skills/keywords the job asks for that are NOT found anywhere in the resume.",
    )
    strong_evidence: list[str] = Field(
        default_factory=list,
        description="Matched skills backed by concrete resume evidence (a named project, role, or achievement).",
    )
    weak_evidence: list[str] = Field(
        default_factory=list,
        description="Skills mentioned in the resume only in passing, with little concrete evidence for this job.",
    )
    relevant_projects: list[str] = Field(
        default_factory=list,
        description="Resume projects/experience bullets that are most relevant to this specific job.",
    )
    resume_suggestions: list[str] = Field(
        default_factory=list,
        description=(
            "Concrete, truthful suggestions to improve this resume for this job "
            "(e.g. resurface an existing bullet, add a keyword that IS supported by existing experience). "
            "Never suggest adding a skill, tool, employer, or achievement not already present in the resume."
        ),
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Honest concerns about fit (e.g. seniority gap, location mismatch) grounded only in given evidence.",
    )
