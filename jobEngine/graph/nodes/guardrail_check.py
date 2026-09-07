"""GuardrailCheck node (section 15): final output guardrails. A job that
fails a hard check (missing required fields, candidate PII leak, secrets in
output) is dropped from the result entirely — the caller only ever sees
validated records.
"""

from __future__ import annotations

import logging
from typing import Any

from ...guardrails.output_guardrails import evaluate_matched_job
from ...models.job import MatchedJob
from ...models.profile import CandidateProfile

logger = logging.getLogger(__name__)

_HARD_CHECKS = {"required_fields_present", "no_candidate_pii_leak", "no_secrets_in_output"}


async def guardrail_check_node(state: dict[str, Any]) -> dict[str, Any]:
    profile = CandidateProfile.model_validate(state.get("candidate_profile") or {})
    matched_jobs = state.get("matched_jobs") or []

    kept: list[dict[str, Any]] = []
    guardrail_results: list[dict[str, Any]] = []
    dropped = 0

    for job_dict in matched_jobs:
        matched = MatchedJob.model_validate(job_dict)
        result = evaluate_matched_job(matched, profile=profile)
        guardrail_results.append(result.model_dump(mode="json"))

        hard_failed = [c for c in result.failed_checks if c.name in _HARD_CHECKS]
        if hard_failed:
            dropped += 1
            logger.warning(
                "guardrail_check_node: dropping job %r (%s) — failed: %s",
                matched.job.title, matched.job.company, [c.name for c in hard_failed],
            )
            continue
        kept.append(job_dict)

    if dropped:
        logger.info("guardrail_check_node: dropped %s job(s) that failed hard output guardrails", dropped)

    return {"matched_jobs": kept, "guardrail_results": guardrail_results}
