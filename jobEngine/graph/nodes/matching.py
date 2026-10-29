"""ResumeJobMatcher node (sections 16/19/20/22): every validated job gets a
deterministic rule evaluation, is ranked by rule score, and the top-K
(cost-capped; ``JOB_LLM_TOP_K<=0`` means all) get a grounded LLM analysis; everyone else degrades to
rule-only scoring — never silently dropped, just less deeply analyzed.
The score is computed in code from the analysis (matching/fit_score.py), so the
same analysis always gives the same score.
Jobs already in the saved-jobs store with an LLM analysis reuse that stored
analysis (no rule evaluation, no LLM call); their score is recomputed from it and
the minimum match score of the current request is re-applied. The same goes for
jobs an earlier run rejected (persistence/rejected_jobs.py) while their text and
the candidate profile are unchanged, so a job turned down once is never paid for
again. Any other job whose text, profile and matcher version were analyzed before
reuses that analysis from persistence/analysis_cache.py instead of a new call.
Jobs the rules reject, and (with JOB_WORLDWIDE_REMOTE_ENABLED and
JOB_REJECT_COUNTRY_RESTRICTED) jobs classified country-restricted — not doable from
India, and not offering visa sponsorship — are rejected without an LLM call. By
default they are kept and only labelled (job.worldwide / job.visa_remote).
With JOB_WORLDWIDE_REMOTE_ENABLED, the matcher's single call per job also returns
the job's eligibility (visa sponsorship, relocation, remote scope, India), merged
into job.visa_remote / job.worldwide (grounded quotes only); a saved analysis
re-applies its stored read with no new call. Only shown jobs without that read
(analyses saved before it existed) and left at "needs verification" get the separate
worldwide LLM review (capped by JOB_WORLDWIDE_LLM_MAX_JOBS; a saved review is reused
while the job's description is unchanged).
Blocking LLM calls run in worker threads, up to JOB_LLM_CONCURRENCY at a time
(every job is analyzed independently); each call counts into its own metrics,
merged afterwards.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...config.settings import get_settings
from ...matching.fit_score import compute_fit_score, profile_fingerprint
from ...matching.llm_matcher import LLMCallMetrics, LLMMatcher
from ...matching.rules import evaluate_rules
from ...matching.scorer import build_matched_job, is_below_minimum, recommendation_for_score
from ...matching.visa_remote import apply_llm_eligibility, india_quotes_support, with_global_remote_tag
from ...matching.worldwide_llm import WorldwideReview, WorldwideReviewer
from ...matching.worldwide_llm import apply_review as apply_worldwide_review
from ...models.job import (
    Job,
    JobEligibility,
    JobMatchAnalysis,
    MatchedJob,
    RuleEvaluation,
    VisaRemoteClassification,
    VisaStatus,
    WorldwideCategory,
    WorldwideClassification,
)
from ...models.profile import CandidateProfile
from ...persistence.analysis_cache import analysis_key, cached_analyses, store_analyses
from ...persistence.rejected_jobs import rejected_jobs_by_key
from ...persistence.saved_jobs import list_saved_jobs, saved_job_key

logger = logging.getLogger(__name__)

_MATCH_FIELDS = tuple(name for name in MatchedJob.model_fields if name != "job")


def _saved_jobs() -> list[dict[str, Any]]:
    try:
        return list_saved_jobs()
    except Exception:  # noqa: BLE001
        logger.exception("matching_node: could not read saved jobs; matching every job")
        return []


def _rejected_jobs(keys: list[str]) -> dict[str, dict[str, Any]]:
    try:
        return rejected_jobs_by_key(keys)
    except Exception:  # noqa: BLE001
        logger.exception("matching_node: could not read rejected jobs; matching them again")
        return {}


def _cached_analyses(keys: list[str]) -> dict[str, JobMatchAnalysis]:
    try:
        stored = cached_analyses(keys)
    except Exception:  # noqa: BLE001
        logger.exception("matching_node: could not read the analysis cache; analyzing every job")
        return {}
    out: dict[str, JobMatchAnalysis] = {}
    for key, analysis in stored.items():
        try:
            out[key] = JobMatchAnalysis.model_validate(analysis)
        except Exception:  # noqa: BLE001
            continue  # no longer validates: analyzed again
    return out


def _store_analyses(analyses: dict[str, dict[str, Any]]) -> None:
    try:
        store_analyses(analyses)
    except Exception:  # noqa: BLE001
        logger.exception("matching_node: could not write the analysis cache")


def _not_eligible_reason(job_dict: dict[str, Any], settings) -> str | None:
    """Why a job is rejected as not doable from India, or None. Only with both
    JOB_WORLDWIDE_REMOTE_ENABLED and JOB_REJECT_COUNTRY_RESTRICTED; a job abroad
    that offers visa sponsorship is never rejected for its location."""
    if not settings.worldwide_remote_enabled or not settings.reject_country_restricted:
        return None
    worldwide = job_dict.get("worldwide") or {}
    if worldwide.get("category") != WorldwideCategory.COUNTRY_RESTRICTED.value:
        return None
    if (job_dict.get("visa_remote") or {}).get("visa_status") == VisaStatus.POSITIVE_EVIDENCE.value:
        return None
    return f"Not eligible from India: {worldwide.get('reason') or 'remote only within other countries'}"


def _rejection_is_current(prior: dict[str, Any], job_dict: dict[str, Any], *, profile_hash: str, settings) -> bool:
    """A stored rejection still holds: same job text, same profile, and it cost an LLM call
    (rule rejections are cheap to redo, and redoing them picks up changed rule settings)."""
    prior_job = prior.get("job") or {}
    if prior_job.get("raw_text_hash") != job_dict.get("raw_text_hash") or prior.get("profile_hash") != profile_hash:
        return False
    return bool(prior.get("llm_analysis")) or _not_eligible_reason(prior_job, settings) is not None


def _apply_eligibility(job_dict: dict[str, Any], analysis: dict[str, Any] | None) -> dict[str, Any]:
    """The job with the matcher's eligibility read merged into its rule labels (no LLM call)."""
    eligibility = (analysis or {}).get("eligibility")
    if not eligibility or not job_dict.get("visa_remote"):
        return job_dict
    read = JobEligibility.model_validate(eligibility)
    text = job_dict.get("description") or ""
    visa_remote = apply_llm_eligibility(VisaRemoteClassification.model_validate(job_dict["visa_remote"]), read, text)
    update: dict[str, Any] = {
        "visa_remote": visa_remote.model_dump(mode="json"),
        "tags": with_global_remote_tag(job_dict.get("tags") or [], visa_remote),
    }
    worldwide = job_dict.get("worldwide")
    if worldwide and worldwide.get("category") == WorldwideCategory.NEEDS_VERIFICATION.value:
        review = WorldwideReview(
            category=read.india_category, evidence_quotes=india_quotes_support(read.india_category, read.india_quotes), reasoning=read.reasoning or "matcher read"
        )
        update["worldwide"] = apply_worldwide_review(WorldwideClassification.model_validate(worldwide), review, text).model_dump(mode="json")
    return {**job_dict, **update}


def _rejected_without_llm(job: Job, rule_eval: RuleEvaluation, reason: str) -> dict[str, Any]:
    return MatchedJob(
        job=job,
        match_score=0.0,
        recommendation="reject",
        rule_evaluation=rule_eval,
        explanation=reason,
        rejected=True,
        rejection_reason=reason,
    ).model_dump(mode="json")


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
        if (matched.get("llm_analysis") or {}).get("eligibility"):
            continue  # the matcher call already read eligibility; never a second call for this job
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


def _reuse_saved(
    job_dict: dict[str, Any], saved: dict[str, Any], *, profile: CandidateProfile, settings, minimum_match_score
) -> dict[str, Any] | None:
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
    job_dict = _apply_eligibility(job_dict, saved.get("llm_analysis"))
    try:
        matched = MatchedJob.model_validate({"job": job_dict, **{k: saved[k] for k in _MATCH_FIELDS if k in saved}})
    except Exception:  # noqa: BLE001
        return None
    if matched.llm_analysis is not None and matched.rule_evaluation.passed:
        # Recomputed, not taken from the stored copy, so every job is on the current formula.
        breakdown = compute_fit_score(
            analysis=matched.llm_analysis,
            job=matched.job,
            profile=profile,
            rule_eval=matched.rule_evaluation,
            settings=settings,
        )
        matched = matched.model_copy(
            update={
                "match_score": breakdown.score,
                "recommendation": recommendation_for_score(breakdown.score),
                "score_breakdown": breakdown,
            }
        )
    not_eligible = _not_eligible_reason(job_dict, settings)
    rejected = (
        not matched.rule_evaluation.passed
        or not_eligible is not None
        or is_below_minimum(
            matched.match_score, matched.recommendation, settings=settings, minimum_match_score=minimum_match_score
        )
    )
    matched = matched.model_copy(
        update={
            "rejected": rejected,
            "rejection_reason": (
                not_eligible or matched.rejection_reason or matched.explanation or "below minimum match score"
            )
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

    # Both stores are blocking MongoDB reads; run them side by side in worker threads.
    saved_jobs, rejected_before = await asyncio.gather(
        asyncio.to_thread(_saved_jobs),
        asyncio.to_thread(_rejected_jobs, [saved_job_key({"job": j}) or "" for j in validated]),
    )
    saved_analyses = _saved_analyses(saved_jobs)
    profile_hash = profile_fingerprint(profile)
    reused: list[dict[str, Any]] = []
    reused_rejected = 0
    # Rejected here without an LLM call: its analysis would be thrown away.
    rejected_up_front: list[dict[str, Any]] = []
    scored: list[tuple[Job, RuleEvaluation]] = []
    rejected_by_rules = 0
    not_eligible = 0
    for job_dict in validated:
        key = saved_job_key({"job": job_dict}) or ""
        saved = saved_analyses.get(key)
        if saved is not None:
            matched = _reuse_saved(
                job_dict, saved, profile=profile, settings=settings, minimum_match_score=minimum_match_score
            )
            if matched is not None:
                reused.append(matched)
                continue
        prior = rejected_before.get(key)
        if prior is not None and _rejection_is_current(prior, job_dict, profile_hash=profile_hash, settings=settings):
            # The stored worldwide verdict may come from an LLM review the rules can't repeat.
            stored_worldwide = (prior.get("job") or {}).get("worldwide")
            restored = {**job_dict, "worldwide": stored_worldwide} if stored_worldwide else job_dict
            matched = _reuse_saved(
                restored, prior, profile=profile, settings=settings, minimum_match_score=minimum_match_score
            )
            if matched is not None:
                reused.append(matched)
                reused_rejected += 1
                continue
        job = Job.model_validate(job_dict)
        rule_eval = evaluate_rules(job, profile, settings=settings)
        if not rule_eval.passed:
            rejected_by_rules += 1
            rejected_up_front.append(
                build_matched_job(
                    job=job,
                    profile=profile,
                    rule_eval=rule_eval,
                    llm_analysis=None,
                    settings=settings,
                    minimum_match_score=minimum_match_score,
                ).model_dump(mode="json")
            )
            continue
        reason = _not_eligible_reason(job_dict, settings)
        if reason:
            not_eligible += 1
            rejected_up_front.append(_rejected_without_llm(job, rule_eval, reason))
            continue
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
    # Same job text, profile and matcher version as an earlier analysis: reuse it,
    # so the score doesn't move between runs.
    keys = [
        analysis_key(job.raw_text_hash, profile_hash=profile_hash, matcher_version=llm_matcher.version)
        for job in to_analyze
    ]
    cached = await asyncio.to_thread(_cached_analyses, keys) if keys else {}
    analyses: list[JobMatchAnalysis | None] = [cached.get(key) for key in keys]
    pending = [idx for idx, analysis in enumerate(analyses) if analysis is None]
    if pending:
        llm_matcher._get_structured_llm()  # build the client once; the worker threads share it
    fresh = await asyncio.gather(*(_analyze(to_analyze[idx]) for idx in pending))
    for idx, analysis in zip(pending, fresh):
        analyses[idx] = analysis
    new_analyses = {keys[idx]: a.model_dump(mode="json") for idx, a in zip(pending, fresh) if a is not None}
    if new_analyses:
        await asyncio.to_thread(_store_analyses, new_analyses)

    matched_jobs: list[dict[str, Any]] = list(reused) + rejected_up_front
    for idx, (job, rule_eval) in enumerate(scored):
        llm_analysis = analyses[idx] if idx < len(analyses) else None
        if llm_analysis is not None and llm_analysis.eligibility is not None:
            job = Job.model_validate(
                _apply_eligibility(job.model_dump(mode="json"), llm_analysis.model_dump(mode="json"))
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

    worldwide_reviews = 0
    if settings.worldwide_remote_enabled:
        worldwide_reviews = await _review_worldwide(matched_jobs, saved_jobs, settings=settings, metrics=llm_metrics)
        # The review settles "needs verification" jobs; drop the ones it found restricted.
        for matched in matched_jobs:
            reason = None if matched.get("rejected") else _not_eligible_reason(matched.get("job") or {}, settings)
            if reason:
                matched["rejected"] = True
                matched["rejection_reason"] = reason
                not_eligible += 1

    metrics = dict(state.get("metrics") or {})
    metrics["jobs_reused_from_analysis_cache"] = len(to_analyze) - len(pending)
    metrics["worldwide_llm_reviews"] = worldwide_reviews
    metrics["jobs_rejected_by_rules"] = metrics.get("jobs_rejected_by_rules", 0) + rejected_by_rules
    metrics["jobs_reused_from_saved"] = len(reused) - reused_rejected
    metrics["jobs_reused_from_rejected"] = reused_rejected
    metrics["jobs_rejected_not_eligible"] = not_eligible
    metrics["jobs_sent_to_llm"] = len(pending)
    metrics["llm_calls"] = llm_metrics.calls
    metrics["llm_retries"] = llm_metrics.retries
    metrics["llm_prompt_tokens"] = llm_metrics.prompt_tokens
    metrics["llm_completion_tokens"] = llm_metrics.completion_tokens
    metrics["llm_total_tokens"] = llm_metrics.total_tokens
    metrics["llm_estimated_cost_usd"] = round(llm_metrics.estimated_cost_usd, 4)

    logger.info(
        "matching_node: %s jobs matched (%s reused from saved/rejected jobs, %s rejected without an LLM call, "
        "%s via LLM, %s via rule-only)",
        len(matched_jobs), len(reused), len(rejected_up_front), metrics["llm_calls"],
        max(0, len(scored) - metrics["jobs_sent_to_llm"]),
    )

    result: dict[str, Any] = {"matched_jobs": matched_jobs, "metrics": metrics}
    if degraded:
        result["degraded"] = True
        result["degraded_reason"] = degraded_reason
    return result
