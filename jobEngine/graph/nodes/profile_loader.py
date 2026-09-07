"""LoadCandidateProfile node."""

from __future__ import annotations

import logging
from typing import Any

from ...guardrails.input_guardrails import validate_candidate_profile
from ...models.errors import ErrorSeverity, ErrorType, WorkflowError
from ...services.profile_loader import load_candidate_profile

logger = logging.getLogger(__name__)


async def profile_loader_node(state: dict[str, Any]) -> dict[str, Any]:
    run_id = state.get("run_id")
    try:
        profile = load_candidate_profile()
    except FileNotFoundError as exc:
        logger.error("profile_loader_node: %s", exc)
        return {
            "errors": [
                WorkflowError(
                    error_code="profile_not_found",
                    error_type=ErrorType.CONFIGURATION,
                    message=str(exc),
                    node="profile_loader",
                    severity=ErrorSeverity.CRITICAL,
                    retryable=False,
                    run_id=run_id,
                ).model_dump(mode="json")
            ],
            "degraded": True,
            "degraded_reason": "candidate profile could not be loaded",
        }

    problems = validate_candidate_profile(profile)
    errors = [
        WorkflowError(
            error_code="profile_validation_warning",
            error_type=ErrorType.VALIDATION,
            message=problem,
            node="profile_loader",
            severity=ErrorSeverity.MEDIUM,
            run_id=run_id,
        ).model_dump(mode="json")
        for problem in problems
    ]
    logger.info(
        "profile_loader_node: loaded profile for %r with %s skills (%s validation warning(s))",
        profile.personal_information.name,
        len(profile.all_skills()),
        len(problems),
    )
    return {"candidate_profile": profile.model_dump(mode="json"), "errors": errors}
