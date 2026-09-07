"""Guardrail set applied to every ``MatchedJob`` before it leaves the engine
(section 15). A job that fails a *hard* check here is dropped from the final
result rather than surfaced — the caller only ever sees validated records.
"""

from __future__ import annotations

import re

from ..models.job import GuardrailCheck, GuardrailResult, MatchedJob
from ..models.profile import CandidateProfile

_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)\bapi[_-]?key\s*[:=]\s*\S+"),
]


def _looks_like_secret(text: str) -> bool:
    return any(p.search(text or "") for p in _SECRET_PATTERNS)


def _contains_candidate_pii(text: str, profile: CandidateProfile) -> bool:
    text = text or ""
    phone = profile.personal_information.phone.strip()
    if phone and phone in text:
        return True
    email = profile.personal_information.email.strip()
    if email and email in text:
        return True
    return False


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 3}


def _skill_is_grounded(claimed_skill: str, profile_skill_tokens: set[str]) -> bool:
    """Fuzzy containment: tolerates naming variance (Next.js vs NextJS, AWS vs AWS Lambda)
    while still catching skills with zero token overlap with anything in the profile.
    """
    claim_tokens = _tokens(claimed_skill)
    if not claim_tokens:
        return True
    return bool(claim_tokens & profile_skill_tokens)


def evaluate_matched_job(matched: MatchedJob, *, profile: CandidateProfile) -> GuardrailResult:
    checks: list[GuardrailCheck] = []

    checks.append(
        GuardrailCheck(
            name="required_fields_present",
            stage="output",
            passed=bool(matched.job.title and matched.job.company and str(matched.job.application_url)),
        )
    )
    checks.append(
        GuardrailCheck(
            name="score_in_range",
            stage="output",
            passed=0 <= matched.match_score <= 100,
        )
    )

    explanation_blob = " ".join(
        [matched.explanation, *matched.strengths, *matched.concerns]
    )
    checks.append(
        GuardrailCheck(
            name="no_candidate_pii_leak",
            stage="output",
            passed=not _contains_candidate_pii(explanation_blob, profile),
        )
    )
    checks.append(
        GuardrailCheck(
            name="no_secrets_in_output",
            stage="output",
            passed=not _looks_like_secret(explanation_blob),
        )
    )

    if matched.llm_analysis is not None:
        profile_skill_tokens: set[str] = set()
        for skill in profile.all_skills():
            profile_skill_tokens |= _tokens(skill)
        unsupported = [
            s
            for s in matched.llm_analysis.matched_skills
            if not _skill_is_grounded(s, profile_skill_tokens)
        ]
        checks.append(
            GuardrailCheck(
                name="no_unsupported_skill_claims",
                stage="output",
                passed=not unsupported,
                details=f"unsupported: {unsupported}" if unsupported else None,
            )
        )

    return GuardrailResult(checks=checks)
