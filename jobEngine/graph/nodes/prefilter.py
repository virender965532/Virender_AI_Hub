"""PreFilterJobs node (section 18/22): cheap rule-based filter before any
paid embedding/LLM call. Hard-rejected jobs (and jobs below the configurable
``rule_prefilter_min_score`` soft floor) are dropped here and only counted,
not carried forward — this is the first and cheapest cost-optimization stage.
"""

from __future__ import annotations

import logging
from typing import Any

from ...config.settings import get_settings
from ...matching.rules import evaluate_rules
from ...models.job import Job
from ...models.profile import CandidateProfile

logger = logging.getLogger(__name__)


async def prefilter_node(state: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    profile = CandidateProfile.model_validate(state.get("candidate_profile") or {})
    jobs = state.get("validated_jobs") or []

    scored: list[dict[str, Any]] = []
    rejected_count = 0
    for job_dict in jobs:
        job = Job.model_validate(job_dict)
        rule_eval = evaluate_rules(job, profile, settings=settings)
        if not rule_eval.passed or rule_eval.rule_score < settings.rule_prefilter_min_score:
            rejected_count += 1
            continue
        scored.append({"job": job_dict, "rule_evaluation": rule_eval.model_dump(mode="json")})

    scored.sort(key=lambda r: r["rule_evaluation"]["rule_score"], reverse=True)
    capped = scored[: settings.embedding_prefilter_top_k]

    metrics = dict(state.get("metrics") or {})
    metrics["jobs_rejected_by_rules"] = metrics.get("jobs_rejected_by_rules", 0) + rejected_count
    metrics["jobs_sent_to_embedding"] = len(capped)

    logger.info(
        "prefilter_node: %s passed rules (of %s), %s rejected, top %s sent forward",
        len(scored), len(jobs), rejected_count, len(capped),
    )
    return {"prefiltered_jobs": capped, "metrics": metrics}
