from datetime import datetime, timedelta, timezone

import pytest
from flask import Flask

from jobEngine.persistence import applied_jobs, saved_jobs
from jobEngine.persistence.mongo import SAVED_JOBS_COLLECTION, get_collection


def _matched(source_job_id, score=60):
    return {
        "job": {
            "job_id": f"hash-{source_job_id}",
            "source": "linkedin",
            "source_job_id": source_job_id,
            "title": "Backend Engineer",
            "company": "Acme",
            "description": "Build APIs",
        },
        "match_score": score,
        "strengths": ["Node.js"],
        "concerns": ["No Redis"],
    }


def test_marking_applied_snapshots_job_and_unmarking_removes_it():
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1"), _matched("2")]})
    stamped = saved_jobs.update_saved_job_status("linkedin:1", "applied")["applied_at"]

    jobs = applied_jobs.list_applied_jobs()
    assert [j["saved_key"] for j in jobs] == ["linkedin:1"]
    job = jobs[0]
    assert job["status"] == "applied" and job["applied_at"] == stamped
    assert job["job"]["description"] == "Build APIs" and job["strengths"] == ["Node.js"]
    assert "_id" not in job and "age_ref" not in job

    saved_jobs.update_saved_job_status("linkedin:1", "interested")
    assert applied_jobs.list_applied_jobs() == []


def test_applied_jobs_survive_the_saved_jobs_purge(monkeypatch):
    monkeypatch.delenv("SAVED_JOBS_MAX_AGE_HOURS", raising=False)
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1")]})
    saved_jobs.update_saved_job_status("linkedin:1", "applied")
    get_collection(SAVED_JOBS_COLLECTION).update_one(
        {"_id": "linkedin:1"}, {"$set": {"age_ref": datetime.now(timezone.utc) - timedelta(hours=48)}}
    )
    assert saved_jobs.list_saved_jobs() == []
    assert [j["saved_key"] for j in applied_jobs.list_applied_jobs()] == ["linkedin:1"]


def test_backfill_runs_once_so_removed_jobs_stay_removed():
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1"), _matched("2")]})
    # Applied before the applied_jobs collection existed.
    get_collection(SAVED_JOBS_COLLECTION).update_many({}, {"$set": {"status": "applied", "applied_at": "2026-10-01"}})

    assert {j["saved_key"] for j in applied_jobs.list_applied_jobs()} == {"linkedin:1", "linkedin:2"}
    assert applied_jobs.delete_applied_job("linkedin:1") is True
    assert applied_jobs.delete_applied_job("linkedin:1") is False
    assert [j["saved_key"] for j in applied_jobs.list_applied_jobs()] == ["linkedin:2"]
    # Removing from the applied list leaves the saved job alone.
    assert saved_jobs.get_saved_job("linkedin:1")["status"] == "applied"


def test_list_is_newest_applied_first_and_clear():
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1"), _matched("2")]})
    saved_jobs.update_saved_job_status("linkedin:1", "applied")
    saved_jobs.update_saved_job_status("linkedin:2", "applied")
    get_collection("applied_jobs").update_one({"_id": "linkedin:1"}, {"$set": {"applied_at": "2026-01-01T00:00:00+00:00"}})
    assert [j["saved_key"] for j in applied_jobs.list_applied_jobs()] == ["linkedin:2", "linkedin:1"]
    assert applied_jobs.clear_applied_jobs() == 2
    assert applied_jobs.list_applied_jobs() == []


@pytest.fixture()
def client():
    from routes.remote_job_finder_routes import remote_job_finder_bp

    app = Flask(__name__)
    app.register_blueprint(remote_job_finder_bp)
    return app.test_client()


def test_applied_jobs_api(client):
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1"), _matched("2")]})
    client.post("/api/saved-jobs/linkedin:1/status", json={"status": "applied"})
    client.post("/api/saved-jobs/linkedin:2/status", json={"status": "applied"})

    data = client.get("/api/applied-jobs").get_json()
    assert data["ok"] and data["count"] == 2

    assert client.delete("/api/applied-jobs/linkedin:1").status_code == 200
    assert client.delete("/api/applied-jobs/linkedin:1").status_code == 404
    assert client.get("/api/applied-jobs").get_json()["count"] == 1

    assert client.delete("/api/applied-jobs").get_json() == {"ok": True, "removed": 1}
    assert client.get("/api/applied-jobs").get_json()["count"] == 0
