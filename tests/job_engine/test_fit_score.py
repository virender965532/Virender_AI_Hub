"""The match score formula (matching/fit_score.py) and the analysis cache that
keeps a job's score the same across runs (persistence/analysis_cache.py)."""

from __future__ import annotations

import asyncio

import pytest

from jobEngine.config.settings import get_settings
from jobEngine.graph.nodes import matching
from jobEngine.matching import fit_score as fs
from jobEngine.models.job import Job, JobMatchAnalysis, RequirementCheck, RuleEvaluation


def _job(**overrides) -> Job:
    defaults = dict(
        job_id="j1", source="linkedin", source_job_id="1", title="Full-Stack Developer", company="Acme",
        description="Build supply chain analytics dashboards.",
        application_url="https://www.linkedin.com/jobs/view/1/", source_url="https://www.linkedin.com/jobs/view/1/",
        raw_text_hash="hash-1",
    )
    defaults.update(overrides)
    return Job(**defaults)


def _req(text: str, tier: str = "required", verdict: str = "missing") -> RequirementCheck:
    return RequirementCheck(requirement=text, tier=tier, verdict=verdict)


def _score(reqs, profile, settings, *, job=None, rule_score=50.0):
    return fs.compute_fit_score(
        analysis=JobMatchAnalysis(requirements=reqs, recommendation="good_match", explanation="x"),
        job=job or _job(),
        profile=profile,
        rule_eval=RuleEvaluation(passed=True, rule_score=rule_score),
        settings=settings,
        implied=fs.ImpliedSkills.from_mapping({"next.js": {"aliases": ["next.js"], "implies": ["ssr"]}}),
    )


# --- formula ----------------------------------------------------------------------


def test_weighted_requirements_experience_and_keywords(candidate_profile, settings):
    reqs = [
        _req("Mainframe COBOL", "core", "met"),
        _req("Fortran numerics", "required", "partial"),
        _req("SAP ABAP", "preferred", "missing"),
    ]
    b = _score(reqs, candidate_profile, settings, rule_score=60.0)
    # S = 100 × (3·1 + 2·0.5 + 1·0) / 6 = 66.7; E = 100 (no range); K = 60
    assert b.skills == 66.7
    assert b.experience == 100.0
    assert b.score == round(0.90 * (400 / 6) + 0.05 * 100 + 0.05 * 60)


def test_missing_core_requirement_caps_the_score(candidate_profile, settings):
    reqs = [_req("Mainframe COBOL", "core", "missing")] + [_req(f"Skill {c}", "required", "met") for c in "ABCDEFGHIJ"]
    b = _score(reqs, candidate_profile, settings, rule_score=100.0)
    assert b.core_missing
    assert b.score == 60.0


def test_no_requirements_falls_back_to_keywords(candidate_profile, settings):
    assert _score([], candidate_profile, settings, rule_score=40.0).skills == 40.0


@pytest.mark.parametrize(
    "years, expected",
    [
        (None, 100.0),
        ((5, 8), 100.0),
        ((3, 6), 90.0),  # 1 year over
        ((5, None), 100.0),
        ((9, None), 50.0),  # 2 years short
        ((12, None), 0.0),
        ((1, 2), 70.0),  # 5 years over → floor 70
    ],
)
def test_experience_score(years, expected):
    assert fs.experience_score(7.0, years) == expected


def test_years_range_is_read_from_the_job_text():
    assert fs.required_years_range(_job(experience_required="3-6 years")) == (3.0, 6.0)
    assert fs.required_years_range(_job(description="You have 5+ years of backend work.")) == (5.0, None)
    assert fs.required_years_range(_job(description="We have 2 to 4 yrs of runway.")) == (2.0, 4.0)
    assert fs.required_years_range(_job()) is None


# --- corrections that override the LLM -------------------------------------------


def test_requirement_naming_a_profile_skill_is_met(candidate_profile, settings):
    b = _score([_req("Strong PostgreSQL skills", "core", "missing")], candidate_profile, settings)
    (r,) = b.requirements
    assert r.verdict == "met" and "PostgreSQL" in r.adjusted
    assert not b.core_missing


def test_implied_sub_topic_is_met(candidate_profile, settings):
    profile = candidate_profile.model_copy(update={"frontend_skills": ["Next.js"]})
    (r,) = _score([_req("Experience with SSR", "required", "missing")], profile, settings).requirements
    assert r.verdict == "met" and r.adjusted == "implied by next.js"


def test_optional_marker_makes_it_preferred(candidate_profile, settings):
    (r,) = _score([_req("Kafka is a plus", "core", "missing")], candidate_profile, settings).requirements
    assert r.tier == "preferred"


def test_years_line_is_left_to_the_experience_score(candidate_profile, settings):
    b = _score([_req("5+ years of software engineering", "core", "missing")], candidate_profile, settings)
    assert b.requirements == [] and not b.core_missing


def test_same_analysis_always_gives_the_same_score(candidate_profile, settings):
    reqs = [_req("Mainframe COBOL", "core", "partial"), _req("Docker", "required", "missing")]
    scores = {_score(reqs, candidate_profile, settings).score for _ in range(5)}
    assert len(scores) == 1


# --- the analysis cache -------------------------------------------------------------


@pytest.fixture()
def llm(monkeypatch):
    monkeypatch.setenv("JOB_LLM_TOP_K", "-1")
    get_settings.cache_clear()
    calls: list[str] = []
    verdicts = iter(["met", "missing"])  # a second call would label it differently

    def analyze(self, *, profile, job, metrics):
        calls.append(job.source_job_id)
        metrics.calls += 1
        return JobMatchAnalysis(
            requirements=[_req("Mainframe COBOL", "core", next(verdicts))],
            recommendation="good_match", explanation="fit",
        )

    monkeypatch.setattr(matching.LLMMatcher, "available", lambda self: True)
    monkeypatch.setattr(matching.LLMMatcher, "_get_structured_llm", lambda self: None)
    monkeypatch.setattr(matching.LLMMatcher, "analyze", analyze)
    yield calls
    get_settings.cache_clear()


def _run(job, profile):
    state = {"validated_jobs": [job.model_dump(mode="json")], "candidate_profile": profile.model_dump(mode="json"),
             "request": {"minimum_match_score": 0}}
    return asyncio.run(matching.matching_node(state))


def test_job_seen_again_reuses_its_analysis_and_score(llm, candidate_profile):
    """Not saved, not rejected (what clearing the saved jobs leaves): still the same score."""
    first = _run(_job(), candidate_profile)
    second = _run(_job(), candidate_profile)
    assert llm == ["1"]
    assert first["matched_jobs"][0]["match_score"] == second["matched_jobs"][0]["match_score"]
    assert second["metrics"]["jobs_reused_from_analysis_cache"] == 1
    assert second["metrics"]["jobs_sent_to_llm"] == 0


def test_changed_job_text_is_analyzed_again(llm, candidate_profile):
    _run(_job(), candidate_profile)
    _run(_job(raw_text_hash="edited"), candidate_profile)
    assert llm == ["1", "1"]
