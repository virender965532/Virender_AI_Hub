import pytest
from flask import Flask

from jobEngine.matching import rejection_analyzer
from jobEngine.matching.rejection_analyzer import (
    RejectionAnalysisError,
    RejectionAnalyzer,
    job_text,
    normalize,
    preferences_text,
)
from jobEngine.models.profile import CandidateProfile, PersonalInfo
from jobEngine.models.rejection import REJECTION_DIMENSIONS, RejectionAnalysis
from jobEngine.persistence import applied_jobs, saved_jobs


def _matched(source_job_id):
    return {
        "job": {
            "job_id": f"hash-{source_job_id}",
            "source": "linkedin",
            "source_job_id": source_job_id,
            "title": "Senior Backend Engineer",
            "company": "Acme",
            "location": "Remote (US only)",
            "salary": "$150k",
            "description": "Build APIs in Go. Must overlap 9-5 EST.",
        },
        "match_score": 60,
        "concerns": ["Go is not in the profile."],
    }


def _analysis(**overrides):
    data = {
        "final_verdict": "not_eligible",
        "verdict_summary": "US-only remote role.",
        "honest_opinion": "This was never going to work from India.",
        "fit_score": 55,
        "confidence": "high",
        "primary_reasons": [{"dimension": "location", "reason": "Hires in the US only.", "impact": "blocking"}],
        "dimensions": [
            {
                "dimension": "location",
                "verdict": "mismatch",
                "impact": "blocking",
                "job_requirement": "US only",
                "candidate_fact": "Noida, India",
                "explanation": "Outside the hiring country.",
            },
            {
                "dimension": "location",
                "verdict": "match",
                "impact": "none",
                "job_requirement": "duplicate",
                "candidate_fact": "duplicate",
                "explanation": "duplicate",
            },
            {
                "dimension": "technology",
                "verdict": "partial",
                "impact": "major",
                "job_requirement": "Go",
                "candidate_fact": "Node.js",
                "explanation": "No Go.",
            },
        ],
        "matched_skills": ["REST APIs", "Go"],
        "missing_skills": ["Go"],
        "red_flags": ["'Remote' but US only"],
        "improvement_actions": ["Filter out US-only roles."],
        "reapply_similar_roles": "no",
        "reapply_advice": "Skip US-only roles.",
    }
    data.update(overrides)
    return RejectionAnalysis(**data)


def test_normalize_covers_every_dimension_once_in_order():
    data = normalize(_analysis())
    assert [d["dimension"] for d in data["dimensions"]] == list(REJECTION_DIMENSIONS)
    by_dim = {d["dimension"]: d for d in data["dimensions"]}
    assert by_dim["location"]["verdict"] == "mismatch"  # the first entry wins
    assert by_dim["timezone"]["verdict"] == "unknown" and by_dim["timezone"]["impact"] == "none"
    # A skill is never both matched and missing.
    assert data["matched_skills"] == ["REST APIs"] and data["missing_skills"] == ["Go"]


def test_prompt_inputs_state_unknowns_explicitly():
    header, body = job_text({"job": {"title": "Dev", "remote_type": "unknown"}, "concerns": ["No Go"]})
    assert "Remote type: Not stated in the posting." in header
    assert "Salary: Not stated in the posting." in header
    assert "No Go" in header and body == ""
    text = preferences_text({"timezone": "Asia/Kolkata", "job_types": []})
    assert "Timezone: Asia/Kolkata" in text
    assert "Preferred job types: Not stated in the profile." in text


def test_analyzer_refuses_without_llm(settings):
    with pytest.raises(RejectionAnalysisError):
        RejectionAnalyzer(settings).analyze({"job": {}}, CandidateProfile())


def test_analyzer_messages_wrap_untrusted_job_text(settings):
    profile = CandidateProfile(personal_information=PersonalInfo(name="V", location="Noida, India", phone="+91 1"))
    messages = RejectionAnalyzer(settings).build_messages(_matched("1"), profile, {"timezone": "Asia/Kolkata"})
    user = messages[1]["content"]
    assert "UNTRUSTED_JOB_CONTENT_START" in user and "Must overlap 9-5 EST" in user
    assert "Timezone: Asia/Kolkata" in user and "+91" not in user


def _applied(key="1"):
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched(key)]})
    saved_jobs.update_saved_job_status(f"linkedin:{key}", "applied")
    return f"linkedin:{key}"


def test_rejection_and_analysis_survive_a_fresh_snapshot():
    key = _applied()
    applied_jobs.set_rejected(key, True)
    applied_jobs.save_rejection_analysis(key, {"final_verdict": "weak_fit"})
    # Re-applying from Saved Jobs re-snapshots the job.
    saved_jobs.update_saved_job_status(key, "applied")
    job = applied_jobs.get_applied_job(key)
    assert job["outcome"] == "rejected" and job["rejected_at"]
    assert job["rejection_analysis"] == {"final_verdict": "weak_fit"}

    job = applied_jobs.set_rejected(key, False)
    assert "outcome" not in job and "rejected_at" not in job
    assert job["rejection_analysis"] == {"final_verdict": "weak_fit"}


@pytest.fixture()
def client(monkeypatch):
    from routes import remote_job_finder_routes as routes

    calls = []

    class FakeAnalyzer:
        def analyze(self, job, profile):
            calls.append(job["saved_key"])
            return {**normalize(_analysis()), "evaluated_at": "2026-10-13T00:00:00+00:00", "model": "fake"}

    monkeypatch.setattr(routes, "_rejection_analyzer", FakeAnalyzer())
    monkeypatch.setattr(routes, "load_candidate_profile", lambda: CandidateProfile())
    app = Flask(__name__)
    app.register_blueprint(routes.remote_job_finder_bp)
    c = app.test_client()
    c.calls = calls
    return c


def test_marking_rejected_evaluates_once_and_reevaluate_reruns(client):
    key = _applied()
    res = client.post(f"/api/applied-jobs/{key}/rejected", json={"rejected": True}).get_json()
    assert res["ok"] and res["analysis_error"] is None
    assert res["job"]["outcome"] == "rejected"
    assert res["job"]["rejection_analysis"]["final_verdict"] == "not_eligible"
    assert client.calls == [key]

    # Unmark + mark again: the stored analysis is reused, no new call.
    client.post(f"/api/applied-jobs/{key}/rejected", json={"rejected": False})
    client.post(f"/api/applied-jobs/{key}/rejected", json={"rejected": True})
    assert client.calls == [key]

    assert client.post(f"/api/applied-jobs/{key}/evaluate").get_json()["ok"]
    assert client.calls == [key, key]

    stored = client.get("/api/applied-jobs").get_json()["jobs"][0]
    assert stored["rejection_analysis"]["model"] == "fake"


def test_rejection_is_kept_when_analysis_fails(client, monkeypatch):
    from routes import remote_job_finder_routes as routes

    class Broken:
        def analyze(self, job, profile):
            raise RejectionAnalysisError("LLM is not configured")

    monkeypatch.setattr(routes, "_rejection_analyzer", Broken())
    key = _applied()
    res = client.post(f"/api/applied-jobs/{key}/rejected", json={"rejected": True}).get_json()
    assert res["ok"] and res["analysis_error"] == "LLM is not configured"
    assert res["job"]["outcome"] == "rejected" and "rejection_analysis" not in res["job"]
    assert client.post(f"/api/applied-jobs/{key}/evaluate").status_code == 502
    assert client.post("/api/applied-jobs/linkedin:nope/rejected", json={}).status_code == 404


def test_preferences_file_loads():
    prefs = rejection_analyzer.load_candidate_preferences()
    assert prefs.get("timezone") == "Asia/Kolkata"
