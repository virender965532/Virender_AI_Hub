"""ResumeJobMatcher node (sections 16/19/20/22): every validated job gets a
deterministic rule evaluation, is ranked by rule score, and the top-K
(cost-capped; ``JOB_LLM_TOP_K<=0`` means all) get a grounded LLM analysis; everyone else degrades to
rule-only scoring — never silently dropped, just less deeply analyzed.
Jobs already in the saved-jobs store with an LLM analysis reuse that stored
analysis (no rule evaluation, no LLM call); only the minimum match score of
the current request is re-applied to them.
With JOB_WORLDWIDE_REMOTE_ENABLED, shown jobs the rules left at "needs
verification" get an LLM eligibility review (capped by JOB_WORLDWIDE_LLM_MAX_JOBS;
a saved review is reused while the job's description is unchanged).
Blocking LLM calls run in worker threads, up to JOB_LLM_CONCURRENCY at a time
(every job is analyzed independently); each call counts into its own metrics,
merged afterwards.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...config.settings import get_settings
from ...matching.llm_matcher import LLMCallMetrics, LLMMatcher
from ...matching.rules import evaluate_rules
from ...matching.scorer import build_matched_job, is_below_minimum
from ...matching.worldwide_llm import WorldwideReviewer
from ...models.job import Job, MatchedJob, RuleEvaluation, WorldwideCategory, WorldwideClassification
from ...models.profile import CandidateProfile
from ...persistence.saved_jobs import list_saved_jobs, saved_job_key

logger = logging.getLogger(__name__)

_MATCH_FIELDS = tuple(name for name in MatchedJob.model_fields if name != "job")


def _saved_jobs() -> list[dict[str, Any]]:
    try:
        return list_saved_jobs()
    except Exception:  # noqa: BLE001
        logger.exception("matching_node: could not read saved jobs; matching every job")
        return []


def _saved_analyses(saved: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Saved jobs that were analyzed by the LLM, keyed by ``saved_job_key``."""
    return {s["saved_key"]: s for s in saved if s.get("saved_key") and s.get("llm_analysis")}


def _saved_worldwide_reviews(saved: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Saved jobs whose worldwide classification came from an LLM review, by key."""
    return {
        s["saved_key"]: s["job"]
        for s in saved
        if s.get("saved_key") and ((s.get("job") or {}).get("worldwide") or {}).get("method") == "llm"
    }


async def _review_worldwide(matched_jobs: list[dict[str, Any]], saved: list[dict[str, Any]], *, settings, metrics) -> int:
    """LLM review of shown "needs verification" jobs; returns how many were sent to the LLM."""
    reviewer = WorldwideReviewer(settings)
    if not reviewer.available():
        return 0
    previous = _saved_worldwide_reviews(saved)
    pending: list[dict[str, Any]] = []
    for matched in matched_jobs:
        job_dict = matched.get("job") or {}
        current = job_dict.get("worldwide")
        if matched.get("rejected") or not current or current.get("category") != WorldwideCategory.NEEDS_VERIFICATION.value:
            continue
        prior = previous.get(saved_job_key(matched) or "")
        if prior and prior.get("raw_text_hash") == job_dict.get("raw_text_hash"):
            job_dict["worldwide"] = prior["worldwide"]
            continue
        if len(pending) < settings.worldwide_llm_max_jobs:
            pending.append(job_dict)

    if pending:
        reviewer._structured_llm()  # build the client once; the worker threads share it
    semaphore = asyncio.Semaphore(settings.llm_concurrency)

    async def _review(job_dict: dict[str, Any]) -> None:
        async with semaphore:
            call_metrics = LLMCallMetrics()
            reviewed = await asyncio.to_thread(
                reviewer.review,
                Job.model_validate(job_dict),
                WorldwideClassification.model_validate(job_dict["worldwide"]),
                call_metrics,
            )
        job_dict["worldwide"] = reviewed.model_dump(mode="json")
        metrics.merge(call_metrics)

    await asyncio.gather(*(_review(j) for j in pending))
    return len(pending)


def _reuse_saved(job_dict: dict[str, Any], saved: dict[str, Any], *, settings, minimum_match_score) -> dict[str, Any] | None:
    """The saved analysis applied to the freshly scraped job, or None if it no longer validates."""
    # Saved jobs aren't opened again, so the scrape may lack what only the detail pane
    # shows (posted age on a "Viewed" card, employment type); take it from the saved copy.
    saved_job = saved.get("job") or {}
    restored: dict[str, Any] = {}
    if not job_dict.get("posted_date") and saved_job.get("posted_date"):
        restored["posted_date"] = saved_job["posted_date"]
    if not job_dict.get("employment_type") and saved_job.get("employment_type"):
        restored["employment_type"] = saved_job["employment_type"]
        if saved_job.get("worldwide") and saved_job.get("raw_text_hash") == job_dict.get("raw_text_hash"):
            restored["worldwide"] = saved_job["worldwide"]
    if restored:
        job_dict = {**job_dict, **restored}
    try:
        matched = MatchedJob.model_validate({"job": job_dict, **{k: saved[k] for k in _MATCH_FIELDS if k in saved}})
    except Exception:  # noqa: BLE001
        return None
    rejected = not matched.rule_evaluation.passed or is_below_minimum(
        matched.match_score, matched.recommendation, settings=settings, minimum_match_score=minimum_match_score
    )
    matched = matched.model_copy(
        update={
            "rejected": rejected,
            "rejection_reason": (matched.rejection_reason or matched.explanation or "below minimum match score")
            if rejected
            else None,
        }
    )
    return matched.model_dump(mode="json")


async def matching_node(state: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()
    request = state.get("request") or {}
    minimum_match_score = request.get("minimum_match_score")
    profile = CandidateProfile.model_validate(state.get("candidate_profile") or {})
    validated = state.get("validated_jobs") or []

    llm_matcher = LLMMatcher(settings)
    llm_metrics = LLMCallMetrics()

    saved_jobs = _saved_jobs()
    saved_analyses = _saved_analyses(saved_jobs)
    reused: list[dict[str, Any]] = []
    scored: list[tuple[Job, RuleEvaluation]] = []
    rejected_by_rules = 0
    for job_dict in validated:
        saved = saved_analyses.get(saved_job_key({"job": job_dict}) or "")
        if saved is not None:
            matched = _reuse_saved(job_dict, saved, settings=settings, minimum_match_score=minimum_match_score)
            if matched is not None:
                reused.append(matched)
                continue
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

    top_k = settings.llm_matching_top_k
    llm_limit = len(scored) if top_k <= 0 else min(top_k, len(scored))

    semaphore = asyncio.Semaphore(settings.llm_concurrency)

    async def _analyze(job: Job):
        async with semaphore:
            call_metrics = LLMCallMetrics()
            analysis = await asyncio.to_thread(llm_matcher.analyze, profile=profile, job=job, metrics=call_metrics)
        llm_metrics.merge(call_metrics)
        return analysis

    to_analyze = [job for job, _ in scored[:llm_limit]] if llm_available else []
    if to_analyze:
        llm_matcher._get_structured_llm()  # build the client once; the worker threads share it
    analyses = await asyncio.gather(*(_analyze(job) for job in to_analyze))

    matched_jobs: list[dict[str, Any]] = list(reused)
    for idx, (job, rule_eval) in enumerate(scored):
        llm_analysis = analyses[idx] if idx < len(analyses) else None
        matched = build_matched_job(
            job=job,
            profile=profile,
            rule_eval=rule_eval,
            llm_analysis=llm_analysis,
            settings=settings,
            minimum_match_score=minimum_match_score,
        )
        matched_jobs.append(matched.model_dump(mode="json"))

    worldwide_reviews = 0
    if settings.worldwide_remote_enabled:
        worldwide_reviews = await _review_worldwide(matched_jobs, saved_jobs, settings=settings, metrics=llm_metrics)

    metrics = dict(state.get("metrics") or {})
    metrics["worldwide_llm_reviews"] = worldwide_reviews
    metrics["jobs_rejected_by_rules"] = metrics.get("jobs_rejected_by_rules", 0) + rejected_by_rules
    metrics["jobs_reused_from_saved"] = len(reused)
    metrics["jobs_sent_to_llm"] = llm_limit if llm_available else 0
    metrics["llm_calls"] = llm_metrics.calls
    metrics["llm_retries"] = llm_metrics.retries
    metrics["llm_prompt_tokens"] = llm_metrics.prompt_tokens
    metrics["llm_completion_tokens"] = llm_metrics.completion_tokens
    metrics["llm_total_tokens"] = llm_metrics.total_tokens
    metrics["llm_estimated_cost_usd"] = round(llm_metrics.estimated_cost_usd, 4)

    logger.info(
        "matching_node: %s jobs matched (%s reused from saved jobs, %s via LLM, %s via rule-only)",
        len(matched_jobs), len(reused), metrics["llm_calls"], max(0, len(scored) - metrics["jobs_sent_to_llm"]),
    )

    result: dict[str, Any] = {"matched_jobs": matched_jobs, "metrics": metrics}
    if degraded:
        result["degraded"] = True
        result["degraded_reason"] = degraded_reason
    return result
