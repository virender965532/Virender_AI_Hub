"""ResumeJobMatcher node (sections 16/19/20/22): embedding similarity ranks
the prefiltered jobs, the top-K (cost-capped) get a grounded LLM analysis,
everyone else degrades to rule+embedding-only scoring — never silently
dropped, just less deeply analyzed. Blocking network calls (embeddings, LLM)
run in a worker thread so they don't block the event loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...config.settings import get_settings
from ...matching.embeddings import EmbeddingMatcher
from ...matching.llm_matcher import LLMCallMetrics, LLMMatcher
from ...matching.scorer import build_matched_job
from ...models.job import Job, RuleEvaluation
from ...models.profile import CandidateProfile

logger = logging.getLogger(__name__)


async def matching_node(state: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    request = state.get("request") or {}
    minimum_match_score = request.get("minimum_match_score")
    profile = CandidateProfile.model_validate(state.get("candidate_profile") or {})
    profile_text = profile.to_llm_context()
    prefiltered = state.get("prefiltered_jobs") or []

    embedding_matcher = EmbeddingMatcher(settings)
    llm_matcher = LLMMatcher(settings)
    llm_metrics = LLMCallMetrics()

    scored: list[tuple[dict[str, Any], Job, float | None]] = []
    for entry in prefiltered:
        job = Job.model_validate(entry["job"])
        similarity: float | None = None
        if embedding_matcher.available():
            job_text = f"{job.title}\n{job.description}"
            similarity = await asyncio.to_thread(embedding_matcher.similarity, profile_text, job_text)
        scored.append((entry, job, similarity))

    def _rank_key(item: tuple[dict[str, Any], Job, float | None]) -> float:
        entry, _job, similarity = item
        if similarity is not None:
            return similarity
        return float(entry["rule_evaluation"]["rule_score"]) / 100.0

    scored.sort(key=_rank_key, reverse=True)

    llm_available = llm_matcher.available()
    degraded = False
    degraded_reason: str | None = None
    if not llm_available and settings.degrade_on_llm_failure:
        degraded = True
        degraded_reason = (
            "LLM matching unavailable (no OPENAI_API_KEY or JOB_ENGINE_LLM_ENABLED=false); "
            "scores reflect rule-based and embedding signals only."
        )

    matched_jobs: list[dict[str, Any]] = []
    for idx, (entry, job, similarity) in enumerate(scored):
        rule_eval = RuleEvaluation.model_validate(entry["rule_evaluation"])
        llm_analysis = None
        if llm_available and idx < settings.llm_matching_top_k:
            llm_analysis = await asyncio.to_thread(
                llm_matcher.analyze, profile=profile, job=job, metrics=llm_metrics
            )
        matched = build_matched_job(
            job=job,
            profile=profile,
            rule_eval=rule_eval,
            embedding_similarity=similarity,
            llm_analysis=llm_analysis,
            settings=settings,
            minimum_match_score=minimum_match_score,
        )
        matched_jobs.append(matched.model_dump(mode="json"))

    metrics = dict(state.get("metrics") or {})
    metrics["jobs_sent_to_llm"] = min(settings.llm_matching_top_k, len(scored)) if llm_available else 0
    metrics["llm_calls"] = llm_metrics.calls
    metrics["llm_retries"] = llm_metrics.retries
    metrics["llm_prompt_tokens"] = llm_metrics.prompt_tokens
    metrics["llm_completion_tokens"] = llm_metrics.completion_tokens
    metrics["llm_total_tokens"] = llm_metrics.total_tokens
    metrics["llm_estimated_cost_usd"] = round(llm_metrics.estimated_cost_usd, 4)

    logger.info(
        "matching_node: %s jobs scored (%s via LLM, %s via embedding-only)",
        len(matched_jobs), metrics["llm_calls"], max(0, len(scored) - metrics["jobs_sent_to_llm"]),
    )

    result: dict[str, Any] = {"matched_jobs": matched_jobs, "metrics": metrics}
    if degraded:
        result["degraded"] = True
        result["degraded_reason"] = degraded_reason
    return result
