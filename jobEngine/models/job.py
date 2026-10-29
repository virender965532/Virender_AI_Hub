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


class VisaStatus(str, Enum):
    """Visa-sponsorship evidence in the job text (``matching/visa_remote.py``)."""

    POSITIVE_EVIDENCE = "POSITIVE_EVIDENCE"
    NEGATIVE_EVIDENCE = "NEGATIVE_EVIDENCE"
    MENTIONED_UNCLEAR = "MENTIONED_UNCLEAR"
    NOT_MENTIONED = "NOT_MENTIONED"


class RemoteStatus(str, Enum):
    """Geographic scope of remote work stated in the job text (``matching/visa_remote.py``)."""

    GLOBAL_REMOTE_EVIDENCE = "GLOBAL_REMOTE_EVIDENCE"
    RESTRICTED_REMOTE = "RESTRICTED_REMOTE"
    REMOTE_UNCLEAR = "REMOTE_UNCLEAR"
    NOT_MENTIONED = "NOT_MENTIONED"


class RelocationSupport(str, Enum):
    CONFIRMED = "CONFIRMED"
    NOT_OFFERED = "NOT_OFFERED"
    MENTIONED_UNCLEAR = "MENTIONED_UNCLEAR"
    NOT_MENTIONED = "NOT_MENTIONED"


class EvidenceSnippet(BaseModel):
    """A sentence quoted verbatim from the posting, with the section it appeared in."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(max_length=600)
    # Heading it appeared under ("Benefits", "About the Company"), or title / location /
    # workplace_type / description when there is no heading.
    section: str
    signal: Literal["positive", "negative", "neutral", "unclear"]


class VisaRemoteClassification(BaseModel):
    """Discovery labels for visa sponsorship and remote scope. Labels only: they never
    reject a job (profile matching decides relevance)."""

    model_config = ConfigDict(extra="forbid")

    visa_status: VisaStatus = VisaStatus.NOT_MENTIONED
    remote_status: RemoteStatus = RemoteStatus.NOT_MENTIONED
    visa_keyword_found: bool = False
    remote_keyword_found: bool = False
    # Mentions visas, relocation or remote work at all.
    has_discovery_signal: bool = False
    visa_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    relocation_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    remote_evidence: list[EvidenceSnippet] = Field(default_factory=list)
    # Countries/regions a restricted remote role names; empty when none is named.
    remote_countries_allowed: list[str] = Field(default_factory=list)
    # Country of the posting location, None when it isn't a recognized country.
    job_country: str | None = None
    relocation_support: RelocationSupport = RelocationSupport.NOT_MENTIONED
    onsite_requirement: bool = False
    visa_conflict: bool = False
    remote_conflict: bool = False
    conflicting_evidence: bool = False
    # 0.9 explicit, 0.5 vague, 0.3 contradictory; None when nothing was mentioned.
    classification_confidence: float | None = Field(default=None, ge=0, le=1)
    # "llm": the matcher's eligibility read (quotes verified against the posting) replaced the rules' labels.
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
    # Visa sponsorship / remote scope labels, also only with JOB_WORLDWIDE_REMOTE_ENABLED.
    visa_remote: VisaRemoteClassification | None = None
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


class JobEligibility(BaseModel):
    """Visa / relocation / remote-scope / India read of the posting, returned by the same
    matcher call as the score (no extra LLM call). Decisive verdicts only count when
    their quotes are found verbatim in the job text (``matching/visa_remote.py``)."""

    model_config = ConfigDict(extra="forbid")

    visa_status: Literal["POSITIVE_EVIDENCE", "NEGATIVE_EVIDENCE", "MENTIONED_UNCLEAR", "NOT_MENTIONED"] = Field(
        description=(
            "POSITIVE_EVIDENCE only if the employer says it sponsors / supports work visas for this role; "
            "NEGATIVE_EVIDENCE if it says it won't sponsor or requires work authorization without sponsorship; "
            "MENTIONED_UNCLEAR if visas are discussed in relation to this role without a clear answer; "
            "NOT_MENTIONED if visas aren't discussed, or only in boilerplate unrelated to sponsorship "
            "(e.g. an anti-scam notice that the company never asks for money for visas)."
        )
    )
    visa_quotes: list[str] = Field(default_factory=list, description="Sentences copied verbatim that support visa_status.")
    relocation_support: Literal["CONFIRMED", "NOT_OFFERED", "MENTIONED_UNCLEAR", "NOT_MENTIONED"] = Field(
        description="Whether the employer helps the hire relocate. Visa sponsorship alone is not relocation support."
    )
    relocation_quotes: list[str] = Field(default_factory=list)
    remote_status: Literal["GLOBAL_REMOTE_EVIDENCE", "RESTRICTED_REMOTE", "REMOTE_UNCLEAR", "NOT_MENTIONED"] = Field(
        description=(
            "GLOBAL_REMOTE_EVIDENCE only if the text says this role can be done from any country / anywhere in "
            "the world and nothing restricts it; RESTRICTED_REMOTE if remote but limited to named countries or "
            "regions (residence, location or work-authorization requirement); REMOTE_UNCLEAR if remote with no "
            "stated geographic scope ('fully remote', 'remote-first') or contradictory statements; NOT_MENTIONED "
            "if remote work isn't mentioned. A LinkedIn location like 'United States (Remote)' alone is not global."
        )
    )
    remote_quotes: list[str] = Field(default_factory=list)
    remote_countries_allowed: list[str] = Field(
        default_factory=list, description="Countries/regions the text names as allowed for remote work; empty if none named."
    )
    india_category: Literal[
        "worldwide_confirmed", "india_eligible", "needs_verification", "country_restricted", "insufficient_info"
    ] = Field(
        description=(
            "Can someone living in India do this job remotely? worldwide_confirmed: explicitly anywhere in the world; "
            "india_eligible: India explicitly allowed; country_restricted: limited to countries excluding India; "
            "needs_verification: remote but not settled; insufficient_info: too little text."
        )
    )
    india_quotes: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="", max_length=600, description="One or two short sentences.")


FitTier = Literal["core", "required", "preferred"]
FitVerdict = Literal["met", "partial", "missing"]


class RequirementCheck(BaseModel):
    """One job requirement judged against the profile — LLM output, never a number."""

    model_config = ConfigDict(extra="forbid")

    requirement: str = Field(description="The requirement, short, copied or condensed from the job.")
    tier: FitTier = Field(
        description="core = central to the role / must-have; required = asked for; preferred = nice to have / a plus."
    )
    verdict: FitVerdict = Field(
        description="met = shown in the profile; partial = related or weaker experience; missing = not in the profile."
    )


class JobMatchAnalysis(BaseModel):
    """Structured LLM output for the grounded matching stage (section 20/21).

    The model must only use facts present in the candidate profile and the
    job text supplied in the prompt — enforced by prompt instructions and
    checked post-hoc by ``guardrails/output_guardrails.py`` (no claimed skill
    absent from the candidate profile).
    """

    model_config = ConfigDict(extra="forbid")

    # The LLM gives no score: the score is computed in code from these checks
    # (matching/fit_score.py). Older saved analyses that still carry a ``score``
    # no longer validate, so those jobs are re-analyzed the next time they are seen.
    requirements: list[RequirementCheck] = Field(
        default_factory=list,
        description=(
            "Every distinct skill / technology / responsibility requirement of the job, one entry each, "
            "judged against the candidate profile. Leave out years-of-experience, location and "
            "work-authorization lines (those are scored separately)."
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
    # Filled in the same call; None on analyses saved before this field existed.
    eligibility: JobEligibility | None = None


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


class ScoredRequirement(RequirementCheck):
    """A requirement as scored: the LLM's check after the code's corrections."""

    # Why code overrode the LLM's verdict/tier, e.g. "Python is listed in the profile".
    adjusted: str = ""


class ScoreBreakdown(BaseModel):
    """How ``match_score`` was computed from an LLM analysis (``matching/fit_score.py``)."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0, le=100)
    skills: float = Field(ge=0, le=100)  # S: weighted share of requirements met
    experience: float = Field(ge=0, le=100)  # E: years vs. the job's stated range
    keywords: float = Field(ge=0, le=100)  # K: rule-based keyword overlap
    core_missing: bool = False  # a core requirement is missing: score capped
    required_years: str = ""  # the job's range as read from its text, e.g. "3-6"
    candidate_years: float = 0.0
    requirements: list[ScoredRequirement] = Field(default_factory=list)
    formula_version: str = ""


class MatchedJob(BaseModel):
    """Final, ranked, explainable job record — what the API returns."""

    model_config = ConfigDict(extra="forbid")

    job: Job
    match_score: float = Field(ge=0, le=100)
    recommendation: Recommendation
    rule_evaluation: RuleEvaluation
    llm_analysis: JobMatchAnalysis | None = None
    # Set when the score was computed from an LLM analysis.
    score_breakdown: ScoreBreakdown | None = None
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
