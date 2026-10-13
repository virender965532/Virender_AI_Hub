"""Concern reviewer: grounding checks, the implied-skills map, the score formula,
the gate, and reuse of stored reviews in the matching node."""

from __future__ import annotations

import asyncio

import pytest

from jobEngine.config.settings import EngineSettings, get_settings
from jobEngine.graph.nodes import matching
from jobEngine.matching import concern_reviewer as cr
from jobEngine.models.job import Job, JobMatchAnalysis, ReviewedItem, ReviewedRequirement
from jobEngine.models.profile import CandidateProfile, PersonalInfo, RoleInfo

# The Next.js job from the Saved Jobs page that the matcher scored 85.
NEXTJS_JD = """Job Summary
We are looking for a skilled Next.js Developer to join our team.

Key Responsibilities
Develop and maintain web applications using Next.js
Build reusable and scalable components using React.js
Implement server-side rendering (SSR) and static site generation (SSG)

Required Skills & Qualifications
Strong experience with Next.js and React.js
Proficiency in JavaScript (ES6+) / TypeScript
Understanding of SSR, SSG, and CSR concepts
Experience with RESTful APIs and/or GraphQL
Familiarity with version control tools like Git
Knowledge of HTML5, CSS3, and responsive design principles
Experience with state management libraries (Redux, Context API, etc.)

Preferred Qualifications
Experience with backend technologies (Node.js, Express)
Familiarity with cloud platforms (AWS, Vercel, etc.)
Knowledge of SEO best practices
Experience with CI/CD pipelines
Understanding of web performance optimization techniques
"""

CONCERNS = [
    "The candidate lacks explicit experience with server-side rendering (SSR), which is a key responsibility of the job.",
    "The candidate does not demonstrate experience with state management libraries like Redux or Context API, which are required for the role.",
    "The candidate's profile does not mention familiarity with SEO best practices, which is a preferred qualification for the job.",
]
MISSING = ["Server-side rendering (SSR)", "State management libraries (Redux, Context API)", "SEO best practices"]


@pytest.fixture()
def profile() -> CandidateProfile:
    return CandidateProfile(
        personal_information=PersonalInfo(name="Test Candidate", location="Noida, India"),
        professional_summary="Full Stack Developer with 7 years of experience.",
        total_experience_years=7.0,
        current_role=RoleInfo(
            company="Acme", title="Senior Developer", date_range_raw="2024 - Present", is_current=True,
            achievements=["Reduced API calls by 80% with Next.js Static Site Generation (SSG), caching, and performance optimization strategies."],
        ),
        frontend_skills=["Next.js", "React.js", "JavaScript", "TypeScript", "HTML", "CSS"],
        backend_skills=["Node.js", "GraphQL", "REST APIs", "Express"],
        cloud_skills=["Vercel", "AWS Lambda", "CI/CD Pipelines", "Docker", "Git"],
        database_skills=["MongoDB", "PostgreSQL"],
    )


@pytest.fixture()
def implied() -> cr.ImpliedSkills:
    return cr.load_implied_skills()


def _job(**overrides) -> Job:
    values = dict(
        job_id="j1", source="linkedin", source_job_id="42", title="Back End Developer - Next JS", company="Talentgigs",
        location="Greater Bengaluru Area (On-site)", description=NEXTJS_JD,
        application_url="https://example.com/apply", source_url="https://example.com/job/42", raw_text_hash="hash-v1",
    )
    values.update(overrides)
    return Job(**values)


def _req(text, tier, verdict, **kw):
    return cr.ReviewOutputRequirement(requirement=text, tier=tier, verdict=verdict, **kw)


def _item(text, kind, decision, severity="moderate", category="skill", **kw):
    return cr.ReviewOutputItem(item=text, kind=kind, decision=decision, severity=severity, category=category, **kw)


def _nextjs_output() -> cr.ReviewOutput:
    """What a reviewer could plausibly return — including one mistake (keeps the SSR missing skill)."""
    return cr.ReviewOutput(
        requirements=[
            _req("Strong experience with Next.js and React.js", "core", "met", evidence="Next.js"),
            _req("Proficiency in JavaScript (ES6+) / TypeScript", "required", "met", evidence="JavaScript, TypeScript"),
            _req("Understanding of SSR, SSG, and CSR concepts", "required", "missing"),
            _req("Experience with RESTful APIs and/or GraphQL", "required", "met", evidence="REST APIs"),
            _req("Familiarity with version control tools like Git", "required", "met", evidence="Git"),
            _req("Knowledge of HTML5, CSS3, and responsive design principles", "required", "met", evidence="HTML, CSS"),
            _req("Experience with state management libraries (Redux, Context API, etc.)", "required", "implied", implied_by="React"),
            _req("Experience with backend technologies (Node.js, Express)", "preferred", "met", evidence="Node.js"),
            _req("Familiarity with cloud platforms (AWS, Vercel, etc.)", "preferred", "met", evidence="Vercel"),
            _req("Knowledge of SEO best practices", "preferred", "partial"),
            _req("Experience with CI/CD pipelines", "preferred", "met", evidence="CI/CD"),
            _req("Understanding of web performance optimization techniques", "preferred", "met", evidence="performance optimization"),
        ],
        items=[
            _item(CONCERNS[0], "concern", "drop_implied", implied_by="Next.js"),
            _item(CONCERNS[1], "concern", "drop_implied", implied_by="React"),
            _item(CONCERNS[2], "concern", "downgrade"),
            _item(MISSING[0], "missing_skill", "keep"),
            _item(MISSING[1], "missing_skill", "drop_implied", implied_by="React"),
            _item(MISSING[2], "missing_skill", "downgrade"),
        ],
        added_concerns=[
            cr.ReviewOutputConcern(
                text="The role is on-site in Bengaluru.", severity="moderate", category="location",
                evidence="Greater Bengaluru Area (On-site)",
            )
        ],
    )


def _build(output, profile, implied, *, concerns=CONCERNS, missing=MISSING, score_before=85.0, job=None):
    job = job or _job()
    return cr.build_review(
        output, concerns=list(concerns), missing=list(missing), score_before=score_before,
        job_text=cr.job_review_text(job), profile_text=profile.to_llm_context(),
        profile_skills=profile.all_skills(), implied=implied,
        raw_text_hash=job.raw_text_hash, profile_hash=cr.profile_fingerprint(profile), version="v-test",
    )


# --- the Next.js example -------------------------------------------------------------


def test_nextjs_job_scores_high_once_subtopic_concerns_are_dropped(profile, implied):
    review = _build(_nextjs_output(), profile, implied)

    # 3 (core) + 6 x 2 (required) + 4.5 of 5 (preferred, SEO partial) = 19.5 / 20
    assert review.score == 97.5
    assert review.score_before == 85.0
    assert review.score_basis == "requirements"

    by_item = {i.item: i for i in review.items}
    assert by_item[CONCERNS[0]].decision == "drop_implied" and by_item[CONCERNS[0]].implied_by == "next.js"
    assert by_item[CONCERNS[1]].decision == "drop_implied" and by_item[CONCERNS[1]].implied_by == "react"
    assert by_item[CONCERNS[2]].decision == "downgrade" and by_item[CONCERNS[2]].severity == "minor"
    # The LLM wrongly kept this one; the implied-skills map overrides it.
    assert by_item[MISSING[0]].decision == "drop_implied"
    assert by_item[MISSING[0]].adjusted == "implied-skills map"

    ssr = next(r for r in review.requirements if r.requirement.startswith("Understanding of SSR"))
    assert ssr.verdict == "implied" and ssr.implied_by == "next.js"

    # The on-site location is kept as a concern but does not lower the score.
    added = [i for i in review.items if i.kind == "added_concern"]
    assert [a.category for a in added] == ["location"]


def test_apply_review_updates_score_concerns_and_minimum(profile, implied, settings):
    review = _build(_nextjs_output(), profile, implied)
    matched = {
        "job": _job().model_dump(mode="json"), "match_score": 85.0, "recommendation": "strong_match",
        "rule_evaluation": {"passed": True, "rule_score": 70}, "concerns": CONCERNS, "missing_required_skills": MISSING,
        "explanation": "x", "rejected": False,
    }
    out = cr.apply_review(matched, review, settings=settings, minimum_match_score=90)

    assert out["match_score"] == 97.5
    assert out["recommendation"] == "strong_match"
    assert not out["rejected"]  # 97.5 clears the 90 bar the matcher's 85 missed
    assert out["concerns"] == [CONCERNS[2], "The role is on-site in Bengaluru."]
    assert out["missing_required_skills"] == ["SEO best practices"]
    assert out["review"]["score"] == 97.5


# --- grounding checks ---------------------------------------------------------------


def test_implied_needs_the_map_not_just_the_llms_word(profile, implied):
    output = cr.ReviewOutput(
        requirements=[_req("Hands-on Redux Toolkit experience", "required", "implied", implied_by="React")],
        items=[_item("Redux Toolkit", "missing_skill", "drop_implied", implied_by="React")],
    )
    review = _build(output, profile, implied, concerns=[], missing=["Redux Toolkit"])
    assert review.requirements[0].verdict == "partial"
    assert review.requirements[0].adjusted == "not in implied-skills map"
    assert review.items[0].decision == "keep"


def test_met_without_profile_evidence_is_downgraded(profile, implied):
    output = cr.ReviewOutput(
        requirements=[
            _req("Kubernetes in production", "required", "met", evidence="Kubernetes"),
            # Real profile text, but unrelated to the requirement — can't vouch for it.
            _req("Kafka streaming", "required", "met", evidence="Next.js"),
        ]
    )
    review = _build(output, profile, implied, concerns=[], missing=[])
    assert [r.verdict for r in review.requirements] == ["partial", "partial"]


def test_requirement_naming_a_profile_skill_is_met_without_quote(profile, implied):
    output = cr.ReviewOutput(
        requirements=[
            _req("Experience with backend technologies (Node.js, Express)", "preferred", "met",
                 evidence="Built backend services in Node"),  # paraphrased, not in the profile
        ]
    )
    review = _build(output, profile, implied, concerns=[], missing=[])
    assert review.requirements[0].verdict == "met"


def test_experience_concern_needs_a_stated_range(profile, implied):
    concern = "The candidate has 7 years of experience, which may indicate overqualification."
    output = cr.ReviewOutput(items=[_item(concern, "concern", "keep", category="experience")])
    reqs = [_req("Strong experience with Next.js and React.js", "core", "met", evidence="Next.js")]
    output.requirements = reqs

    # The Next.js JD states no years: the concern stays visible but minor, and costs nothing.
    review = _build(output, profile, implied, concerns=[concern], missing=[])
    assert (review.items[0].decision, review.items[0].severity) == ("downgrade", "minor")
    assert review.score == 100.0

    # With a stated range it is a real (moderate) concern: -5.
    job = _job(description=NEXTJS_JD + "\nExperience: 2 to 6 years\n", raw_text_hash="hash-range")
    review = _build(output, profile, implied, concerns=[concern], missing=[], job=job)
    assert (review.items[0].decision, review.items[0].severity) == ("keep", "moderate")
    assert review.score == 95.0


def test_kept_skill_concern_naming_an_implied_subtopic_is_downgraded(profile, implied):
    concern = "The candidate does not demonstrate server-side rendering (SSR) or Kubernetes experience."
    output = cr.ReviewOutput(items=[_item(concern, "concern", "keep", severity="blocking")])
    review = _build(output, profile, implied, concerns=[concern], missing=[])
    item = review.items[0]
    assert (item.decision, item.severity) == ("downgrade", "minor")
    assert "next.js" in item.adjusted


def test_kept_skill_concern_naming_a_listed_skill_is_downgraded(profile, implied):
    concern = "The candidate's profile does not indicate experience with GraphQL."
    output = cr.ReviewOutput(items=[_item(concern, "concern", "keep")])
    item = _build(output, profile, implied, concerns=[concern], missing=[]).items[0]
    assert (item.decision, item.severity, item.adjusted) == ("downgrade", "minor", "GraphQL is listed in the profile")

    other = "No Kubernetes experience."
    output = cr.ReviewOutput(items=[_item(other, "concern", "keep")])
    assert _build(output, profile, implied, concerns=[other], missing=[]).items[0].decision == "keep"


def test_drop_covered_needs_profile_evidence(profile, implied):
    output = cr.ReviewOutput(
        items=[
            _item("No GraphQL experience shown.", "concern", "drop_covered", evidence="GraphQL"),
            _item("No Kubernetes experience shown.", "concern", "drop_covered", evidence="Kubernetes"),
        ]
    )
    review = _build(
        output, profile, implied, concerns=["No GraphQL experience shown.", "No Kubernetes experience shown."], missing=[]
    )
    assert [i.decision for i in review.items] == ["drop_covered", "keep"]


def test_added_concern_needs_a_quote_from_the_job(profile, implied):
    output = cr.ReviewOutput(
        added_concerns=[
            cr.ReviewOutputConcern(text="Requires 10+ years.", severity="blocking", category="experience",
                                   evidence="10+ years of experience required"),  # not in the JD
        ]
    )
    review = _build(output, profile, implied, concerns=[], missing=[])
    assert review.items == []


def test_items_the_reviewer_skipped_are_kept(profile, implied):
    review = _build(cr.ReviewOutput(), profile, implied, concerns=["Needs a security clearance."], missing=["Rust"])
    assert [(i.item, i.decision) for i in review.items] == [("Needs a security clearance.", "keep"), ("Rust", "keep")]
    assert all(i.adjusted == "not returned by reviewer" for i in review.items)
    assert review.score_basis == "matcher" and review.score == 85.0

    # A skipped concern never costs points.
    reqs = [_req("Strong experience with Next.js and React.js", "core", "met", evidence="Next.js")]
    review = _build(cr.ReviewOutput(requirements=reqs), profile, implied, concerns=["Needs a security clearance."], missing=[])
    assert review.score == 100.0


# --- score formula --------------------------------------------------------------------


def _r(tier, verdict):
    return ReviewedRequirement(requirement="r", tier=tier, verdict=verdict)


def _i(severity, category, decision="keep"):
    return ReviewedItem(item="c", kind="concern", decision=decision, severity=severity, category=category)


def test_score_formula_penalties_and_caps():
    reqs = [_r("core", "met"), _r("required", "met")]
    assert cr.compute_review_score(reqs, [], 50)[0] == 100.0
    # Location never moves the score; a moderate experience concern costs 5.
    assert cr.compute_review_score(reqs, [_i("moderate", "location")], 50)[0] == 100.0
    assert cr.compute_review_score(reqs, [_i("moderate", "experience")], 50)[0] == 95.0
    # Skill gaps count through the requirements, not as a second penalty.
    assert cr.compute_review_score(reqs, [_i("moderate", "skill")], 50)[0] == 100.0
    assert cr.compute_review_score(reqs, [_i("blocking", "experience")], 50)[0] == 75.0
    assert cr.compute_review_score(reqs, [_i("blocking", "location")], 50)[0] == 100.0
    assert cr.compute_review_score(reqs, [_i("blocking", "skill")], 50)[0] == 100.0
    # Dropped concerns don't count.
    assert cr.compute_review_score(reqs, [_i("blocking", "experience", "drop_implied")], 50)[0] == 100.0
    # A missing core requirement caps the score.
    assert cr.compute_review_score([_r("core", "missing"), _r("required", "met")] * 5, [], 50)[0] == 40.0
    assert cr.compute_review_score([_r("core", "missing")] + [_r("required", "met")] * 20, [], 50)[0] == 60.0


# --- gate ---------------------------------------------------------------------------------


def _matched(score=85.0, concerns=("c",), missing=(), passed=True):
    return {
        "rule_evaluation": {"passed": passed},
        "llm_analysis": {"score": score, "concerns": list(concerns), "missing_required_skills": list(missing)},
    }


def test_gate():
    s = EngineSettings(reviewer_min_score=50)
    assert cr.needs_review(_matched(), s)
    assert cr.needs_review(_matched(concerns=(), missing=("SSR",)), s)
    assert not cr.needs_review(_matched(score=49.9), s)
    assert not cr.needs_review(_matched(concerns=()), s)
    assert not cr.needs_review(_matched(passed=False), s)
    assert not cr.needs_review({"rule_evaluation": {"passed": True}, "llm_analysis": None}, s)


# --- matching node: review, reuse, failure ---------------------------------------------------


@pytest.fixture()
def node_env(monkeypatch, profile):
    """Matcher returns the 85 analysis; the reviewer LLM returns the Next.js output (counted)."""
    monkeypatch.setenv("JOB_LLM_TOP_K", "-1")
    get_settings.cache_clear()
    calls = {"reviewer": 0}

    def fake_analyze(self, *, profile, job, metrics):
        return JobMatchAnalysis(
            score=85, matched_skills=["Next.js"], missing_required_skills=MISSING, concerns=CONCERNS,
            recommendation="strong_match", explanation="Strong Next.js match.",
        )

    class FakeLLM:
        def __init__(self, result):
            self.result = result

        def invoke(self, messages):
            calls["reviewer"] += 1
            if isinstance(self.result, Exception):
                raise self.result
            return {"raw": None, "parsed": self.result}

    monkeypatch.setattr(matching.LLMMatcher, "available", lambda self: True)
    monkeypatch.setattr(matching.LLMMatcher, "_get_structured_llm", lambda self: None)
    monkeypatch.setattr(matching.LLMMatcher, "analyze", fake_analyze)
    monkeypatch.setattr(cr.ConcernReviewer, "available", lambda self: True)
    reviewer_result = {"value": _nextjs_output()}
    monkeypatch.setattr(cr.ConcernReviewer, "_structured_llm", lambda self: FakeLLM(reviewer_result["value"]))

    def run(job: Job, saved=()):
        monkeypatch.setattr(matching, "list_saved_jobs", lambda: list(saved))
        state = {"validated_jobs": [job.model_dump(mode="json")], "candidate_profile": profile.model_dump(mode="json"),
                 "request": {"minimum_match_score": 55}}
        return asyncio.run(matching.matching_node(state))

    yield run, calls, reviewer_result
    get_settings.cache_clear()


def test_node_reviews_then_reuses_the_stored_review(node_env):
    run, calls, _ = node_env
    first = run(_job())
    matched = first["matched_jobs"][0]
    assert matched["match_score"] == 97.5
    assert matched["review"]["score_before"] == 85.0
    assert first["metrics"]["concern_reviews"] == 1
    assert calls["reviewer"] == 1

    saved = {**matched, "saved_key": "linkedin:42"}
    second = run(_job(), saved=[saved])
    assert calls["reviewer"] == 1  # no new call
    assert second["metrics"]["concern_reviews_reused"] == 1
    assert second["matched_jobs"][0]["match_score"] == 97.5

    # The job description changed: review again.
    third = run(_job(raw_text_hash="hash-v2"), saved=[saved])
    assert calls["reviewer"] == 2
    assert third["metrics"]["concern_reviews"] == 1


def test_node_keeps_matcher_result_when_the_review_fails(node_env):
    run, calls, reviewer_result = node_env
    reviewer_result["value"] = RuntimeError("OpenAI down")
    out = run(_job())
    matched = out["matched_jobs"][0]
    assert calls["reviewer"] == 1
    assert matched["match_score"] == 85.0
    assert matched["review"] is None
    assert matched["concerns"] == CONCERNS
