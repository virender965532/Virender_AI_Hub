"""ResumeJobMatcher node (sections 16/19/20/22): every validated job gets a
deterministic rule evaluation, is ranked by rule score, and the top-K
(cost-capped) get a grounded LLM analysis; everyone else degrades to
rule-only scoring — never silently dropped, just less deeply analyzed.
Blocking LLM calls run in a worker thread so they don't block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...config.settings import get_settings
from ...matching.llm_matcher import LLMCallMetrics, LLMMatcher
from ...matching.rules import evaluate_rules
from ...matching.scorer import build_matched_job
from ...models.job import Job, RuleEvaluation
from ...models.profile import CandidateProfile

logger = logging.getLogger(__name__)


async def matching_node(state: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    request = state.get("request") or {}
    minimum_match_score = request.get("minimum_match_score")
    profile = CandidateProfile.model_validate(state.get("candidate_profile") or {})
    validated = state.get("validated_jobs") or []

    llm_matcher = LLMMatcher(settings)
    llm_metrics = LLMCallMetrics()

    scored: list[tuple[Job, RuleEvaluation]] = []
    rejected_by_rules = 0
    for job_dict in validated:
        job = Job.model_validate(job_dict)
        rule_eval = evaluate_rules(job, profile, settings=settings)
        if not rule_eval.passed:
            rejected_by_rules += 1
        scored.append((job, rule_eval))

    scored.sort(key=lambda item: item[1].rule_score, reverse=True)

    llm_available = llm_matcher.available()
    degraded = False
    degraded_reason: str | None = None
    if not llm_available and settings.degrade_on_llm_failure:
        degraded = True
        degraded_reason = (
            "LLM matching unavailable (no OPENAI_API_KEY or JOB_ENGINE_LLM_ENABLED=false); "
            "scores reflect rule-based signals only."
        )

    matched_jobs: list[dict[str, Any]] = []
    for idx, (job, rule_eval) in enumerate(scored):
        llm_analysis = None
        if llm_available and idx < settings.llm_matching_top_k:
            llm_analysis = await asyncio.to_thread(
                llm_matcher.analyze, profile=profile, job=job, metrics=llm_metrics
            )
        matched = build_matched_job(
            job=job,
            profile=profile,
            rule_eval=rule_eval,
            llm_analysis=llm_analysis,
            settings=settings,
            minimum_match_score=minimum_match_score,
        )
        matched_jobs.append(matched.model_dump(mode="json"))

    metrics = dict(state.get("metrics") or {})
    metrics["jobs_rejected_by_rules"] = metrics.get("jobs_rejected_by_rules", 0) + rejected_by_rules
    metrics["jobs_sent_to_llm"] = min(settings.llm_matching_top_k, len(scored)) if llm_available else 0
    metrics["llm_calls"] = llm_metrics.calls
    metrics["llm_retries"] = llm_metrics.retries
    metrics["llm_prompt_tokens"] = llm_metrics.prompt_tokens
    metrics["llm_completion_tokens"] = llm_metrics.completion_tokens
    metrics["llm_total_tokens"] = llm_metrics.total_tokens
    metrics["llm_estimated_cost_usd"] = round(llm_metrics.estimated_cost_usd, 4)

    logger.info(
        "matching_node: %s jobs scored (%s via LLM, %s via rule-only)",
        len(matched_jobs), metrics["llm_calls"], max(0, len(scored) - metrics["jobs_sent_to_llm"]),
    )

    result: dict[str, Any] = {"matched_jobs": matched_jobs, "metrics": metrics}
    if degraded:
        result["degraded"] = True
        result["degraded_reason"] = degraded_reason
    return result
