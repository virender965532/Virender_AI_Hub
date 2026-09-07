"""Deterministic hard-rejection rules and mandatory/preferred/nice-to-have
requirement tiering (section 18). Runs before any embedding or LLM call so
obviously-irrelevant jobs never reach the expensive stages (section 22/40).
"""

from __future__ import annotations

import re

from IPython.display import JSON

from ..config.settings import EngineSettings
from ..models.job import Job, RequirementTier, RuleEvaluation
from ..models.profile import CandidateProfile
from .text_utils import normalize_tokens, overlap_score, skill_token_set

_JUNIOR_MARKERS = re.compile(
    r"\b(junior|jr\.?|intern(ship)?|entry[\s-]?level|fresher|trainee|graduate program)\b",
    re.IGNORECASE,
)
_SENIOR_MARKERS = re.compile(r"\b(senior|sr\.?|staff|principal|lead|architect)\b", re.IGNORECASE)
_TECH_ROLE_MARKERS = re.compile(
    r"\b(developer|engineer|architect|programmer|technologist|swe|sde)\b", re.IGNORECASE
)
_NON_TECH_ROLE_MARKERS = re.compile(
    r"\b(sales|marketing|recruiter|nurse|driver|chef|accountant|hr\b|human resources|"
    r"teacher|paralegal|receptionist|warehouse|forklift)\b",
    re.IGNORECASE,
)
_EXCLUDED_PRIMARY_LANGUAGES = {
    "java": re.compile(r"\bjava\b(?!\s*script)", re.IGNORECASE),
    "c#": re.compile(r"\bc#|c\s*sharp\b", re.IGNORECASE),
    ".net": re.compile(r"\.net\b|\bdotnet\b|\basp\.net\b", re.IGNORECASE),
    "golang": re.compile(r"\bgolang\b|\bgo\s+lang\b", re.IGNORECASE),
    "php": re.compile(r"\bphp\b", re.IGNORECASE),
    "ruby": re.compile(r"\bruby\b(?!\s*on\s*rails)?", re.IGNORECASE),
}
_REQUIRED_YEARS_RE = re.compile(r"(\d{1,2})\s*\+?\s*(?:-\s*(\d{1,2})\s*)?years?", re.IGNORECASE)
_MANDATORY_MARKERS = re.compile(r"\b(must have|required|mandatory|must be able)\b", re.IGNORECASE)
_NICE_TO_HAVE_MARKERS = re.compile(
    r"\b(nice to have|preferred|a plus|bonus|good to have)\b", re.IGNORECASE
)


def classify_requirement(text: str) -> RequirementTier:
    if _MANDATORY_MARKERS.search(text):
        return RequirementTier.MANDATORY
    if _NICE_TO_HAVE_MARKERS.search(text):
        return RequirementTier.NICE_TO_HAVE
    return RequirementTier.PREFERRED


def _job_blob(job: Job) -> str:
    return " ".join(
        [job.title, job.experience_required or "", " ".join(job.requirements), " ".join(job.skills)]
    )


def _is_role_related(job: Job) -> tuple[bool, str | None]:
    title = job.title or ""
    if _NON_TECH_ROLE_MARKERS.search(title):
        return False, f"job title {title!r} matches a non-technical role pattern"
    if not _TECH_ROLE_MARKERS.search(title) and not _TECH_ROLE_MARKERS.search(_job_blob(job)):
        return False, f"job title {title!r} does not look like a software/AI engineering role"
    return True, None


def _primary_stack_mismatch(job: Job, candidate_tokens: set[str]) -> str | None:
    blob = _job_blob(job)
    for name, pattern in _EXCLUDED_PRIMARY_LANGUAGES.items():
        if pattern.search(blob):
            job_tokens = normalize_tokens(blob)
            if not (job_tokens & candidate_tokens):
                return name
    return None


def _extract_required_years(job: Job) -> int | None:
    text = job.experience_required or ""
    m = _REQUIRED_YEARS_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _seniority_check(
    job: Job, *, settings: EngineSettings
) -> tuple[bool, str | None]:
    title = job.title or ""
    if settings.reject_junior_roles and _JUNIOR_MARKERS.search(title) and not _SENIOR_MARKERS.search(title):
        return False, f"job title {title!r} indicates a junior/entry-level role"

    required_years = _extract_required_years(job)
    if required_years is not None:
        candidate_years = settings.candidate_total_experience_years
        gap = required_years - candidate_years
        if gap > settings.max_negative_experience_gap_years:
            return False, (
                f"job requires ~{required_years}+ years, candidate has "
                f"{candidate_years} (gap {gap:.1f} > tolerance "
                f"{settings.max_negative_experience_gap_years})"
            )
    return True, None


def evaluate_rules(
    job: Job, profile: CandidateProfile, *, settings: EngineSettings
) -> RuleEvaluation:
    reasons: list[str] = []
    candidate_tokens = skill_token_set(profile.all_skills())

    role_related, role_reason = _is_role_related(job)
    print(f"evaluate_rules: role_related={role_related}, role_reason={role_reason}, candidate_tokens={candidate_tokens}, job={job}")
    if not role_related and settings.reject_unrelated_roles:
        reasons.append(role_reason or "role does not look related to candidate's field")

    mismatch = _primary_stack_mismatch(job, candidate_tokens)
    if mismatch:
        reasons.append(
            f"primary stack mismatch: job requires {mismatch} with no overlap with candidate's stack"
        )
        role_related = False

    seniority_ok, seniority_reason = _seniority_check(job, settings=settings)
    if not seniority_ok:
        reasons.append(seniority_reason or "seniority mismatch")

    job_tokens = normalize_tokens(_job_blob(job)) | skill_token_set(job.skills)
    rule_score = overlap_score(job_tokens, candidate_tokens)

    mandatory_reqs = [r for r in job.requirements if classify_requirement(r) == RequirementTier.MANDATORY]
    mandatory_matched: list[str] = []
    mandatory_missing: list[str] = []
    for req in mandatory_reqs:
        req_tokens = normalize_tokens(req)
        if req_tokens & candidate_tokens:
            mandatory_matched.append(req)
        else:
            mandatory_missing.append(req)

    return RuleEvaluation(
        passed=not reasons,
        rejection_reasons=reasons,
        mandatory_skills_matched=mandatory_matched,
        mandatory_skills_missing=mandatory_missing,
        rule_score=rule_score,
        seniority_ok=seniority_ok,
        role_related=role_related,
    )
