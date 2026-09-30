"""Combines rule evaluation + LLM analysis into one explainable,
configurable-weight final score (section 17) and the resulting
``MatchedJob`` record. The final recommendation is always derived from the
computed weighted score, never taken verbatim from the LLM (section 71 —
"no blind LLM decisions"); the LLM's own analysis is kept for explainability.
"""

from __future__ import annotations

from ..config.settings import EngineSettings
from ..models.job import Job, JobMatchAnalysis, MatchedJob, Recommendation, RuleEvaluation
from ..models.profile import CandidateProfile
from .text_utils import normalize_tokens, overlap_score, skill_token_set

_AI_KEYWORDS = {
    "llm", "rag", "langgraph", "langchain", "agentic", "genai", "generative",
    "openai", "vector", "embedding", "ai", "ml", "machinelearning", "gpt",
}
_ARCHITECTURE_KEYWORDS = {
    "microservice", "microservices", "cloud", "distributed", "api", "apis",
    "cicd", "ci", "cd", "production", "scalable", "scalability", "docker",
    "kubernetes", "serverless", "architecture",
}

# Shared neutral midpoint for low-weight dimensions with sparse/no signal:
# domain-vertical experience (section 3 "domain_experience") isn't reliably
# derivable from Profile/Summary.txt without inventing categories, so it's
# left unpopulated by the profile loader; and a job posting that simply
# doesn't mention architecture/cloud terms isn't evidence against fit either.
# A neutral midpoint avoids letting an evidence-free, low-weight dimension
# systematically drag every job's score down — see
# docs/job_engine/ARCHITECTURE.md "Matching pipeline".
_NEUTRAL_SPARSE_DIMENSION_SCORE = 50.0


def _job_tokens(job: Job) -> set[str]:
    blob = " ".join(
        [job.title, job.description, " ".join(job.requirements), " ".join(job.responsibilities), " ".join(job.skills)]
    )
    return normalize_tokens(blob)


def compute_dimension_scores(
    *,
    job: Job,
    profile: CandidateProfile,
    rule_eval: RuleEvaluation,
    llm_analysis: JobMatchAnalysis | None,
) -> dict[str, float]:
    job_tokens = _job_tokens(job)
    candidate_tokens = skill_token_set(profile.all_skills())

    if llm_analysis is not None:
        role = llm_analysis.role_match
        technical = llm_analysis.technical_match
        ai = llm_analysis.ai_match
        experience = llm_analysis.experience_match
    else:
        role = 100.0 if rule_eval.role_related else 0.0
        technical = rule_eval.rule_score
        ai = overlap_score(job_tokens & _AI_KEYWORDS, candidate_tokens) if (job_tokens & _AI_KEYWORDS) else 0.0
        experience = 100.0 if rule_eval.seniority_ok else 30.0

    # A job posting that simply doesn't discuss architecture/cloud concerns is
    # not evidence the role is architecturally irrelevant to the candidate —
    # only jobs that *do* mention it are scored against the candidate's cloud
    # tokens; otherwise this sparse, low-weight (10%) dimension defaults to
    # the same neutral midpoint as "domain" instead of unfairly zeroing out.
    arch_terms_in_job = job_tokens & _ARCHITECTURE_KEYWORDS
    architecture_cloud = (
        overlap_score(arch_terms_in_job, candidate_tokens) if arch_terms_in_job else _NEUTRAL_SPARSE_DIMENSION_SCORE
    )

    return {
        "technical_skills": round(technical, 1),
        "role_alignment": round(role, 1),
        "experience": round(experience, 1),
        "ai_alignment": round(ai, 1),
        "architecture_cloud": round(architecture_cloud, 1),
        "domain": _NEUTRAL_SPARSE_DIMENSION_SCORE,
    }


def compute_final_score(dimension_scores: dict[str, float], weights: dict[str, float]) -> float:
    total = sum(dimension_scores[key] * weight for key, weight in weights.items())
    return round(min(100.0, max(0.0, total)), 1)


def _reconcile_skill_lists(
    profile: CandidateProfile, matched_skills: list[str], missing: list[str]
) -> tuple[list[str], list[str]]:
    """The LLM scans one flat comma-separated list of ~40 candidate skills and
    can miss one that's genuinely present (observed: it called "Python"
    missing even though it's verbatim in the candidate's skills list) — this
    catches that class of oversight deterministically rather than trusting
    the LLM's list verbatim.
    """
    candidate_tokens = skill_token_set(profile.all_skills())
    reconciled_matched = list(matched_skills)
    reconciled_missing: list[str] = []
    seen_matched = {s.strip().lower() for s in reconciled_matched}
    for skill in missing:
        tokens = skill_token_set([skill])
        if tokens and tokens <= candidate_tokens:
            key = skill.strip().lower()
            if key not in seen_matched:
                reconciled_matched.append(skill)
                seen_matched.add(key)
        else:
            reconciled_missing.append(skill)
    return reconciled_matched, reconciled_missing


def recommendation_for_score(score: float) -> Recommendation:
    if score >= 85:
        return "strong_match"
    if score >= 70:
        return "good_match"
    if score >= 50:
        return "possible_match"
    if score >= 30:
        return "poor_match"
    return "reject"


def build_matched_job(
    *,
    job: Job,
    profile: CandidateProfile,
    rule_eval: RuleEvaluation,
    llm_analysis: JobMatchAnalysis | None,
    settings: EngineSettings,
    minimum_match_score: float | None = None,
) -> MatchedJob:
    """``minimum_match_score`` overrides ``settings.minimum_match_score`` when
    the caller (a specific search request) asked for a different bar — see
    ``JobSearchRequest.minimum_match_score`` (section 36).
    """
    effective_minimum = settings.minimum_match_score if minimum_match_score is None else minimum_match_score
    if not rule_eval.passed:
        return MatchedJob(
            job=job,
            match_score=0.0,
            recommendation="reject",
            rule_evaluation=rule_eval,
            llm_analysis=llm_analysis,
            rejected=True,
            rejection_reason="; ".join(rule_eval.rejection_reasons) or "failed hard-rejection rules",
        )

    dims = compute_dimension_scores(
        job=job,
        profile=profile,
        rule_eval=rule_eval,
        llm_analysis=llm_analysis,
    )
    score = compute_final_score(dims, settings.weights.as_dict())
    recommendation = recommendation_for_score(score)

    if llm_analysis is not None:
        matched_skills, missing = _reconcile_skill_lists(
            profile, llm_analysis.matched_skills, llm_analysis.missing_required_skills
        )
        strengths = llm_analysis.strengths
        concerns = llm_analysis.concerns
        explanation = llm_analysis.explanation
    else:
        matched_skills = rule_eval.mandatory_skills_matched
        missing = rule_eval.mandatory_skills_missing
        strengths = []
        concerns = ["LLM analysis unavailable for this job; score reflects rule-based signals only."]
        explanation = (
            f"Rule-based match only (LLM unavailable): {rule_eval.rule_score}% keyword/stack overlap "
            f"with the candidate profile."
        )

    # The "reject" band (<30) only applies when the caller gave no explicit
    # minimum; otherwise a request-level bar below 30 would be silently ignored.
    rejected = score < effective_minimum or (minimum_match_score is None and recommendation == "reject")
    rejection_reason = None
    if rejected:
        rejection_reason = explanation if explanation else f"score {score} below minimum {effective_minimum}"

    return MatchedJob(
        job=job,
        match_score=score,
        recommendation=recommendation,
        rule_evaluation=rule_eval,
        llm_analysis=llm_analysis,
        matched_skills=matched_skills,
        missing_required_skills=missing,
        strengths=strengths,
        concerns=concerns,
        explanation=explanation,
        rejected=rejected,
        rejection_reason=rejection_reason,
    )
