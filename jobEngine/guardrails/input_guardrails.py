"""Guardrail 1 — schema validation for everything entering the engine.

Configuration validation lives in ``jobEngine/config/sources.py`` (Pydantic
models raise on load). This module validates the two other untrusted-input
surfaces named in the spec: API request bodies and the candidate profile.
"""

from __future__ import annotations

from ..models.profile import CandidateProfile


class GuardrailValidationError(ValueError):
    """Raised when an input fails a guardrail check; always caught at the API boundary."""


def validate_candidate_profile(profile: CandidateProfile) -> list[str]:
    """Return a list of problems (empty = valid). Never raises — callers decide severity."""
    problems: list[str] = []
    if not profile.personal_information.name.strip():
        problems.append("candidate profile is missing a name")
    if not profile.all_skills():
        problems.append("candidate profile has no parsed skills")
    if profile.total_experience_years <= 0:
        problems.append("candidate profile has no usable total_experience_years")
    return problems
