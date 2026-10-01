from __future__ import annotations

from jobEngine.matching.rules import evaluate_rules
from jobEngine.matching.scorer import build_matched_job, compute_final_score, recommendation_for_score
from jobEngine.models.job import Job, JobMatchAnalysis


def _job(**overrides) -> Job:
    defaults = dict(
        job_id="j1",
        source="test",
        title="Senior Full Stack AI Engineer",
        company="Acme",
        description="Build agentic systems with LangGraph and RAG using Python and Next.js.",
        skills=["Python", "LangGraph", "RAG", "Next.js"],
        application_url="https://example.com/apply",
        source_url="https://example.com/job/1",
        raw_text_hash="hash1",
    )
    defaults.update(overrides)
    return Job(**defaults)


def test_compute_final_score_respects_weights():
    dims = {
        "technical_skills": 100.0, "role_alignment": 0.0, "experience": 0.0,
        "ai_alignment": 0.0, "architecture_cloud": 0.0, "domain": 0.0,
    }
    weights = {
        "technical_skills": 0.3, "role_alignment": 0.2, "experience": 0.15,
        "ai_alignment": 0.2, "architecture_cloud": 0.1, "domain": 0.05,
    }
    assert compute_final_score(dims, weights) == 30.0


def test_recommendation_thresholds():
    assert recommendation_for_score(94) == "strong_match"
    assert recommendation_for_score(75) == "good_match"
    assert recommendation_for_score(55) == "possible_match"
    assert recommendation_for_score(35) == "poor_match"
    assert recommendation_for_score(10) == "reject"


def test_hard_rejected_job_never_scores_above_zero(candidate_profile, settings):
    job = _job(title="Junior Java Developer", skills=["Java"], description="Entry level Java role.")
    rule_eval = evaluate_rules(job, candidate_profile, settings=settings)
    matched = build_matched_job(
        job=job, profile=candidate_profile, rule_eval=rule_eval,
        llm_analysis=None, settings=settings,
    )
    assert matched.rejected
    assert matched.match_score == 0.0
    assert matched.recommendation == "reject"


def test_llm_analysis_drives_score_when_available(candidate_profile, settings):
    job = _job()
    rule_eval = evaluate_rules(job, candidate_profile, settings=settings)
    llm_analysis = JobMatchAnalysis(
        score=95, role_match=95, technical_match=95, ai_match=98, experience_match=95,
        matched_skills=["Python", "LangGraph", "RAG"],
        recommendation="strong_match",
        explanation="Excellent alignment with the candidate's AI/full-stack background.",
    )
    matched = build_matched_job(
        job=job, profile=candidate_profile, rule_eval=rule_eval,
        llm_analysis=llm_analysis, settings=settings,
    )
    assert not matched.rejected
    assert matched.match_score >= 85
    assert matched.recommendation == "strong_match"


def test_request_level_minimum_score_overrides_settings_default(candidate_profile, settings):
    job = _job()
    rule_eval = evaluate_rules(job, candidate_profile, settings=settings)
    llm_analysis = JobMatchAnalysis(
        score=60, role_match=60, technical_match=60, ai_match=60, experience_match=60,
        recommendation="possible_match", explanation="Partial match.",
    )
    # With the engine default (70), this scores just under it and would be rejected...
    default_result = build_matched_job(
        job=job, profile=candidate_profile, rule_eval=rule_eval,
        llm_analysis=llm_analysis, settings=settings,
    )
    assert default_result.rejected

    # ...but a caller-supplied lower threshold must actually be honored.
    override_result = build_matched_job(
        job=job, profile=candidate_profile, rule_eval=rule_eval,
        llm_analysis=llm_analysis, settings=settings,
        minimum_match_score=10.0,
    )
    assert not override_result.rejected
