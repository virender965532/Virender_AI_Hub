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


class WorldwideCategory(str, Enum):
    """Can the job be done remotely from India? (``matching/worldwide_remote.py``)"""

    WORLDWIDE_CONFIRMED = "worldwide_confirmed"  # A
    INDIA_ELIGIBLE = "india_eligible"  # B
    NEEDS_VERIFICATION = "needs_verification"  # C
    COUNTRY_RESTRICTED = "country_restricted"  # D
    INSUFFICIENT_INFO = "insufficient_info"  # E
    # Posted in India: eligible by definition, but not an international role.
    INDIA_LOCAL = "india_local"


class EmploymentArrangement(str, Enum):
    FULL_TIME_EMPLOYEE = "full_time_employee"
    FULL_TIME_CONTRACTOR = "full_time_contractor"
    PART_TIME_CONTRACTOR = "part_time_contractor"
    CONTRACTOR = "contractor"  # contract, hours not stated
    FREELANCE = "freelance"
    TEMPORARY_CONTRACT = "temporary_contract"
    PART_TIME = "part_time"  # part-time, not stated as a contract
    OTHER = "other"


class WorldwideEvidence(BaseModel):
    """One piece of text that drove the classification, quoted verbatim."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=600)
    # Where it was found: title / location / workplace_type / employment_type / description.
    found_in: str
    signal: Literal["positive", "negative", "neutral"]
    rule: str = ""


class WorldwideClassification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: WorldwideCategory
    india_eligible: bool = False
    arrangement: EmploymentArrangement = EmploymentArrangement.OTHER
    evidence: list[WorldwideEvidence] = Field(default_factory=list)
    reason: str = ""
    method: Literal["rules", "llm"] = "rules"


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
    # Display labels derived during normalization, e.g. "Global Remote".
    tags: list[str] = Field(default_factory=list)
    # Set only when JOB_WORLDWIDE_REMOTE_ENABLED is on.
    worldwide: WorldwideClassification | None = None
    # Every configured search URL that listed this job in the run.
    search_urls: list[str] = Field(default_factory=list)
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

    # The job's match score, used as-is (no sub-scores / weighted blend). Older
    # saved analyses that still carry role/technical/ai/experience sub-scores no
    # longer validate, so those jobs are re-analyzed the next time they are seen.
    score: float = Field(
        ge=0,
        le=100,
        description=(
            "Overall, evidence-grounded fit of the candidate for this job (0-100), judged the way a "
            "meticulous resume reviewer would: required skills demonstrated, seniority/experience "
            "range, role focus and location. Conservative — unsupported claims lower the score."
        ),
    )

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
    concerns: list[str] = Field(
        default_factory=list,
        description=(
            "Honest concerns about fit grounded only in the given evidence, each a full sentence stating "
            "the fact behind it — e.g. seniority gap (under- or over-qualified against the job's stated "
            "experience range), location/eligibility mismatch, required skills not demonstrated."
        ),
    )

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


ReviewTier = Literal["core", "required", "preferred", "nice_to_have"]
ReviewVerdict = Literal["met", "implied", "partial", "missing"]
ReviewDecision = Literal["keep", "downgrade", "drop_implied", "drop_covered"]
ConcernSeverity = Literal["blocking", "moderate", "minor"]
ConcernCategory = Literal["skill", "experience", "location", "other"]


class ReviewedRequirement(BaseModel):
    """One requirement line of the job, judged against the profile (``matching/concern_reviewer.py``)."""

    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(max_length=500)
    tier: ReviewTier
    verdict: ReviewVerdict
    # Profile skill that implies this requirement (verdict "implied").
    implied_by: str = ""
    # Text copied from the candidate profile (verdict "met").
    evidence: str = ""
    # Set when code overrode the LLM's verdict, e.g. "implied-skills map" / "evidence not in profile".
    adjusted: str = ""


class ReviewedItem(BaseModel):
    """The reviewer's decision on one of the matcher's concerns or missing skills,
    or a concern the reviewer added (``added_concern``, backed by a job-text quote)."""

    model_config = ConfigDict(extra="forbid")

    item: str = Field(max_length=600)
    kind: Literal["concern", "missing_skill", "added_concern"]
    decision: ReviewDecision
    severity: ConcernSeverity
    category: ConcernCategory
    reason: str = Field(default="", max_length=400)
    implied_by: str = ""
    evidence: str = ""
    adjusted: str = ""

    @property
    def kept(self) -> bool:
        return self.decision in ("keep", "downgrade")


class JobReview(BaseModel):
    """Stored result of the concern reviewer; reused while all three fingerprints match."""

    model_config = ConfigDict(extra="forbid")

    requirements: list[ReviewedRequirement] = Field(default_factory=list)
    items: list[ReviewedItem] = Field(default_factory=list)
    # Matcher (LLM) score before the review, and the score computed in code after it.
    score_before: float = Field(ge=0, le=100)
    score: float = Field(ge=0, le=100)
    score_basis: Literal["requirements", "matcher"] = "requirements"
    raw_text_hash: str
    profile_hash: str
    reviewer_version: str
    model: str = ""
    reviewed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MatchedJob(BaseModel):
    """Final, ranked, explainable job record — what the API returns."""

    model_config = ConfigDict(extra="forbid")

    job: Job
    match_score: float = Field(ge=0, le=100)
    recommendation: Recommendation
    rule_evaluation: RuleEvaluation
    llm_analysis: JobMatchAnalysis | None = None
    # Set when the concern reviewer checked this job's concerns (JOB_REVIEWER_ENABLED).
    review: JobReview | None = None
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
