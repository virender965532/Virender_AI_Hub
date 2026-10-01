from __future__ import annotations

from jobEngine.matching.rules import classify_requirement, evaluate_rules
from jobEngine.models.job import Job, RequirementTier


def _job(**overrides) -> Job:
    defaults = dict(
        job_id="j1",
        source="test",
        title="Senior Full Stack AI Engineer",
        company="Acme",
        description="Build agentic systems with LangGraph and RAG using Python and Next.js.",
        skills=["Python", "LangGraph", "RAG", "Next.js"],
        requirements=[],
        application_url="https://example.com/apply",
        source_url="https://example.com/job/1",
        raw_text_hash="hash1",
    )
    defaults.update(overrides)
    return Job(**defaults)


def test_classify_requirement_tiers():
    assert classify_requirement("Must have 5+ years of Python") == RequirementTier.MANDATORY
    assert classify_requirement("Kubernetes experience is a plus") == RequirementTier.NICE_TO_HAVE
    assert classify_requirement("Experience with cloud platforms") == RequirementTier.PREFERRED


def test_good_job_passes_rules(candidate_profile, settings):
    job = _job()
    result = evaluate_rules(job, candidate_profile, settings=settings)
    assert result.passed
    assert result.role_related
    assert result.seniority_ok
    assert result.rule_score > 0


def test_title_never_rejects_a_job(candidate_profile, settings):
    for title in ("Senior FDE, Full Stack App Development", "Technical Lead", "Junior Python Developer"):
        result = evaluate_rules(_job(title=title), candidate_profile, settings=settings)
        assert result.passed, title
        assert result.role_related, title


def test_primary_stack_mismatch_is_hard_rejected(candidate_profile, settings):
    job = _job(
        title="Senior Java Backend Developer",
        description="Build backend services in Java and Spring Boot.",
        skills=["Java", "Spring Boot"],
    )
    result = evaluate_rules(job, candidate_profile, settings=settings)
    assert not result.passed
    assert any("primary stack mismatch" in r for r in result.rejection_reasons)


def test_java_mentioned_alongside_candidate_stack_is_not_rejected(candidate_profile, settings):
    # "javascript" must not trigger the "java" exclusion pattern, and jobs that
    # merely interoperate with a Java backend (without requiring Java from the
    # candidate) should not be penalized just because the word appears.
    job = _job(
        title="Senior Full Stack AI Engineer",
        description="Frontend in Next.js/React/JavaScript talking to an existing Java backend via REST APIs.",
        skills=["JavaScript", "Next.js", "React", "Python"],
    )
    result = evaluate_rules(job, candidate_profile, settings=settings)
    assert result.passed


def test_excessive_experience_gap_is_hard_rejected(candidate_profile, settings):
    job = _job(experience_required="12+ years required")
    result = evaluate_rules(job, candidate_profile, settings=settings)
    assert not result.passed
    assert not result.seniority_ok


def test_mandatory_requirements_tracked(candidate_profile, settings):
    job = _job(requirements=["Must have Python and LangGraph", "Nice to have Kubernetes", "Must have Rust"])
    result = evaluate_rules(job, candidate_profile, settings=settings)
    assert "Must have Python and LangGraph" in result.mandatory_skills_matched
    assert "Must have Rust" in result.mandatory_skills_missing
