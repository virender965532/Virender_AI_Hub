"""Rejected jobs are remembered (persistence/rejected_jobs.py) and never paid for
again; rule-rejected and country-restricted jobs never reach the LLM."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from jobEngine.config.settings import get_settings
from jobEngine.graph.nodes import matching, source_discovery
from jobEngine.matching.fit_score import profile_fingerprint
from jobEngine.models.job import Job, JobMatchAnalysis, RequirementCheck
from jobEngine.persistence import rejected_jobs, saved_jobs

_DESCRIPTION = "Build Node.js and React services on AWS with TypeScript."


@pytest.fixture()
def llm(monkeypatch):
    """Fake matcher: every job meets (``llm.fits``) or misses its one core requirement, so it
    scores well or below 50; ``llm.calls`` lists the jobs sent to it."""
    monkeypatch.setenv("JOB_LLM_TOP_K", "-1")
    monkeypatch.setenv("JOB_WORLDWIDE_REMOTE_ENABLED", "true")
    monkeypatch.setenv("JOB_WORLDWIDE_LLM_REVIEW", "false")
    get_settings.cache_clear()

    class Fake:
        fits = False
        calls: list[str] = []

    def analyze(self, *, profile, job, metrics):
        Fake.calls.append(job.source_job_id)
        metrics.calls += 1
        verdict = "met" if Fake.fits else "missing"
        return JobMatchAnalysis(
            requirements=[RequirementCheck(requirement="Mainframe COBOL batch jobs", tier="core", verdict=verdict)],
            recommendation="poor_match", explanation="weak fit",
        )

    monkeypatch.setattr(matching.LLMMatcher, "available", lambda self: True)
    monkeypatch.setattr(matching.LLMMatcher, "_get_structured_llm", lambda self: None)
    monkeypatch.setattr(matching.LLMMatcher, "analyze", analyze)
    yield Fake
    get_settings.cache_clear()


def _job(job_id: str, *, title="Senior Node.js Engineer", description=_DESCRIPTION, worldwide=None) -> dict:
    return Job(
        job_id=f"hash-{job_id}", source="linkedin", source_job_id=job_id, title=title, company="Acme",
        description=description, application_url=f"https://www.linkedin.com/jobs/view/{job_id}/",
        source_url=f"https://www.linkedin.com/jobs/view/{job_id}/", raw_text_hash=f"hash-{job_id}",
        worldwide=worldwide,
    ).model_dump(mode="json")


def _restricted() -> dict:
    return {"category": "country_restricted", "reason": "US residents only", "method": "rules"}


def _run(jobs, profile, *, minimum=50):
    state = {"validated_jobs": jobs, "candidate_profile": profile.model_dump(mode="json"),
             "request": {"minimum_match_score": minimum}}
    return asyncio.run(matching.matching_node(state))


def _store(out, profile):
    rejected_jobs.save_rejected_jobs(out["matched_jobs"], run_id="r1", profile_hash=profile_fingerprint(profile))


# --- store -----------------------------------------------------------------


def test_store_keeps_only_rejected_jobs_and_hides_expired():
    now = datetime.now(timezone.utc)
    jobs = [{"job": _job("1"), "rejected": True}, {"job": _job("2"), "rejected": False}]
    assert rejected_jobs.save_rejected_jobs(jobs, run_id="r1", profile_hash="p") == 1
    assert set(rejected_jobs.rejected_jobs_by_key(["linkedin:1", "linkedin:2"])) == {"linkedin:1"}
    assert rejected_jobs.rejected_descriptions("linkedin") == {"1": _DESCRIPTION}
    assert rejected_jobs.rejected_descriptions("other") == {}

    rejected_jobs.save_rejected_jobs(
        [{"job": _job("3"), "rejected": True}], run_id="r0", profile_hash="p", now=now - timedelta(days=30)
    )
    assert rejected_jobs.rejected_jobs_by_key(["linkedin:3"]) == {}


def test_source_nodes_treat_rejected_jobs_as_known(monkeypatch):
    rejected_jobs.save_rejected_jobs([{"job": _job("9"), "rejected": True}], run_id="r1", profile_hash="p")
    assert source_discovery._known_descriptions("linkedin") == {"9": _DESCRIPTION}


# --- matching --------------------------------------------------------------


def test_rule_rejected_job_is_not_sent_to_llm(llm, candidate_profile):
    too_senior = {**_job("1"), "experience_required": "15+ years"}
    out = _run([too_senior, _job("2")], candidate_profile)
    by_id = {m["job"]["source_job_id"]: m for m in out["matched_jobs"]}
    assert by_id["1"]["rule_evaluation"]["passed"] is False
    assert llm.calls == ["2"]
    assert by_id["1"]["rejected"] is True and by_id["1"]["llm_analysis"] is None


@pytest.fixture()
def reject_restricted(llm, monkeypatch):
    monkeypatch.setenv("JOB_REJECT_COUNTRY_RESTRICTED", "true")
    get_settings.cache_clear()


def test_country_restricted_job_is_kept_and_matched_by_default(llm, candidate_profile, monkeypatch):
    monkeypatch.setenv("JOB_REJECT_COUNTRY_RESTRICTED", "false")
    get_settings.cache_clear()
    llm.fits = True
    out = _run([_job("1", worldwide=_restricted()), _job("2")], candidate_profile)
    by_id = {m["job"]["source_job_id"]: m for m in out["matched_jobs"]}
    assert sorted(llm.calls) == ["1", "2"]
    assert by_id["1"]["rejected"] is False
    assert by_id["1"]["job"]["worldwide"]["category"] == "country_restricted"
    assert out["metrics"]["jobs_rejected_not_eligible"] == 0


def test_country_restricted_job_is_rejected_without_llm(reject_restricted, llm, candidate_profile):
    out = _run([_job("1", worldwide=_restricted()), _job("2")], candidate_profile)
    by_id = {m["job"]["source_job_id"]: m for m in out["matched_jobs"]}
    assert llm.calls == ["2"]
    assert by_id["1"]["rejected"] is True
    assert by_id["1"]["rejection_reason"].startswith("Not eligible from India")
    assert out["metrics"]["jobs_rejected_not_eligible"] == 1


def test_llm_rejected_job_is_not_matched_again(llm, candidate_profile):
    _store(_run([_job("1")], candidate_profile), candidate_profile)
    assert llm.calls == ["1"]

    out = _run([_job("1")], candidate_profile)
    assert llm.calls == ["1"]  # no second call
    (matched,) = out["matched_jobs"]
    assert matched["rejected"] is True and matched["match_score"] < 50
    assert out["metrics"]["jobs_reused_from_rejected"] == 1

    # A lower bar on a later (manual) search shows it, still without an LLM call.
    (matched,) = _run([_job("1")], candidate_profile, minimum=1)["matched_jobs"]
    assert matched["rejected"] is False
    assert llm.calls == ["1"]


def test_changed_job_text_or_profile_is_matched_again(llm, candidate_profile):
    _store(_run([_job("1")], candidate_profile), candidate_profile)

    edited = {**_job("1"), "raw_text_hash": "edited"}
    _run([edited], candidate_profile)
    assert llm.calls == ["1", "1"]

    other_profile = candidate_profile.model_copy(update={"professional_summary": "Changed summary."})
    _run([_job("1")], other_profile)
    assert llm.calls == ["1", "1", "1"]


def test_stored_restriction_wins_over_fresh_rules_verdict(reject_restricted, llm, candidate_profile):
    """An LLM worldwide review found it restricted; the rules alone only say "needs verification"."""
    llm.fits = True
    stored = {"job": _job("1", worldwide={**_restricted(), "method": "llm"}), "rejected": True,
              "match_score": 0.0, "recommendation": "reject",
              "rule_evaluation": {"passed": True, "rule_score": 50.0}}
    rejected_jobs.save_rejected_jobs([stored], run_id="r1", profile_hash=profile_fingerprint(candidate_profile))

    fresh = _job("1", worldwide={"category": "needs_verification", "reason": "Remote", "method": "rules"})
    (matched,) = _run([fresh], candidate_profile)["matched_jobs"]
    assert llm.calls == []
    assert matched["rejected"] is True


# --- saved jobs page -------------------------------------------------------


def test_saved_jobs_page_can_leave_out_restricted_jobs():
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [
        {"job": _job("1", worldwide=_restricted()), "match_score": 80},
        {"job": _job("2"), "match_score": 80},
    ]})
    assert len(saved_jobs.list_saved_jobs()) == 2
    assert [j["saved_key"] for j in saved_jobs.list_saved_jobs(include_not_eligible=False)] == ["linkedin:2"]
