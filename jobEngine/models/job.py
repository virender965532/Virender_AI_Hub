"""Core job schemas: raw extraction -> normalized -> matched -> final result.

Every stage boundary in the pipeline is a Pydantic model. Nothing moves
through the core engine as an untyped dict once it has left an adapter's
extraction step.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator

_MAX_TITLE_CHARS = 500
_MAX_COMPANY_CHARS = 300
_MAX_DESCRIPTION_CHARS = 20_000
_MAX_LIST_ITEMS = 100
_MAX_LIST_ITEM_CHARS = 500


def _clip(value: str, limit: int) -> str:
    value = (value or "").strip()
    return value[:limit] if len(value) > limit else value


def _clip_list(values: list[str], *, max_items: int = _MAX_LIST_ITEMS) -> list[str]:
    out: list[str] = []
    for v in values or []:
        text = _clip(str(v), _MAX_LIST_ITEM_CHARS)
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


class RawJob(BaseModel):
    """Exactly what an adapter extracted, before normalization/validation.

    Content-size guardrails are applied here too (defense in depth) so an
    oversized/malicious page can never make it further than this model.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    source_job_id: str | None = None
    title: str = ""
    company: str = ""
    location: str | None = None
    employment_type: str | None = None
    experience_required: str | None = None
    description: str = ""
    requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    salary: str | None = None
    posted_date_raw: str | None = None
    application_url: str = ""
    source_url: str = ""
    remote_type: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("title")
    @classmethod
    def _clip_title(cls, v: str) -> str:
        return _clip(v, _MAX_TITLE_CHARS)

    @field_validator("company")
    @classmethod
    def _clip_company(cls, v: str) -> str:
        return _clip(v, _MAX_COMPANY_CHARS)

    @field_validator("description")
    @classmethod
    def _clip_description(cls, v: str) -> str:
        return _clip(v, _MAX_DESCRIPTION_CHARS)

    @field_validator("requirements", "responsibilities", "skills")
    @classmethod
    def _clip_lists(cls, v: list[str]) -> list[str]:
        return _clip_list(v)


class RemoteType(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class Job(BaseModel):
    """Normalized, validated job — the only shape the matching pipeline understands."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    source: str
    source_job_id: str | None = None
    title: str
    company: str
    location: str | None = None
    employment_type: str | None = None
    experience_required: str | None = None
    description: str
    requirements: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    salary: str | None = None
    posted_date: datetime | None = None
    application_url: AnyHttpUrl
    source_url: AnyHttpUrl
    remote_type: RemoteType = RemoteType.UNKNOWN
    raw_text_hash: str
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("title", "company")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("must not be empty after normalization")
        return v


def content_hash(*, source: str, title: str, company: str, description: str) -> str:
    blob = "|".join(
        [source.strip().lower(), title.strip().lower(), company.strip().lower(), description.strip().lower()]
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class RequirementTier(str, Enum):
    MANDATORY = "mandatory"
    PREFERRED = "preferred"
    NICE_TO_HAVE = "nice_to_have"


Recommendation = Literal[
    "strong_match", "good_match", "possible_match", "poor_match", "reject"
]


class JobMatchAnalysis(BaseModel):
    """Structured LLM output for the grounded matching stage (section 20/21).

    The model must only use facts present in the candidate profile and the
    job text supplied in the prompt — enforced by prompt instructions and
    checked post-hoc by ``guardrails/output_guardrails.py`` (no claimed skill
    absent from the candidate profile).
    """

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=100)
    role_match: float = Field(ge=0, le=100)
    technical_match: float = Field(ge=0, le=100)
    ai_match: float = Field(ge=0, le=100)
    experience_match: float = Field(ge=0, le=100)

    matched_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Skills the job wants that the candidate profile explicitly demonstrates "
            "(named in the skills list AND backed by a role/project/achievement bullet, "
            "not just present in a bare skills inventory line). A given skill name must "
            "never appear in both matched_skills and missing_required_skills — decide once."
        ),
    )
    missing_required_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Skills the job requires that are NOT demonstrated in the candidate profile — "
            "either absent entirely, or only present as a bare skills-list entry with no "
            "supporting role/project/achievement evidence. Mutually exclusive with "
            "matched_skills: never list the same skill in both."
        ),
    )
    matched_requirements: list[str] = Field(default_factory=list)

    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)

    recommendation: Recommendation
    explanation: str = Field(max_length=2000)


class RuleEvaluation(BaseModel):
    """Output of the deterministic hard-rejection / mandatory-skill rule stage."""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    rejection_reasons: list[str] = Field(default_factory=list)
    mandatory_skills_matched: list[str] = Field(default_factory=list)
    mandatory_skills_missing: list[str] = Field(default_factory=list)
    rule_score: float = Field(ge=0, le=100)
    seniority_ok: bool = True
    role_related: bool = True


class MatchedJob(BaseModel):
    """Final, ranked, explainable job record — what the API returns."""

    model_config = ConfigDict(extra="forbid")

    job: Job
    match_score: float = Field(ge=0, le=100)
    recommendation: Recommendation
    rule_evaluation: RuleEvaluation
    llm_analysis: JobMatchAnalysis | None = None
    matched_skills: list[str] = Field(default_factory=list)
    missing_required_skills: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    explanation: str = ""
    rejected: bool = False
    rejection_reason: str | None = None


class GuardrailCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    passed: bool
    stage: str
    details: str | None = None


class GuardrailResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checks: list[GuardrailCheck] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[GuardrailCheck]:
        return [c for c in self.checks if not c.passed]


class ExecutionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    workflow_duration_ms: float = 0.0
    source_duration_ms: dict[str, float] = Field(default_factory=dict)
    jobs_discovered: int = 0
    jobs_validated: int = 0
    jobs_rejected_by_rules: int = 0
    jobs_sent_to_llm: int = 0
    jobs_recommended: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    llm_total_tokens: int = 0
    llm_estimated_cost_usd: float = 0.0
    llm_calls: int = 0
    llm_retries: int = 0
    browser_failures: int = 0
    source_failures: int = 0
    match_distribution: dict[str, int] = Field(default_factory=dict)


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources_processed: int
    jobs_discovered: int
    jobs_validated: int
    duplicates_removed: int
    jobs_matched: int
    jobs_recommended: int
    degraded: bool = False
    degraded_reason: str | None = None


class JobSearchResult(BaseModel):
    """The shape returned by ``POST /api/jobs/search`` once a run completes."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    status: Literal["completed", "failed", "partial"]
    summary: RunSummary
    jobs: list[MatchedJob] = Field(default_factory=list)
    source_errors: list[dict[str, Any]] = Field(default_factory=list)
    metrics: ExecutionMetrics | None = None
