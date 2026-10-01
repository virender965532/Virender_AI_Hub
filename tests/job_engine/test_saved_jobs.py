from datetime import datetime, timedelta, timezone

import pytest

from jobEngine.persistence import saved_jobs
from jobEngine.persistence.mongo import SAVED_JOBS_COLLECTION, get_collection


def _matched(source_job_id, score, *, rejected=False, title="Backend Engineer"):
    return {
        "job": {
            "job_id": f"hash-{source_job_id}",
            "source": "linkedin",
            "source_job_id": source_job_id,
            "title": title,
            "company": "Acme",
            "description": "Build APIs",
        },
        "match_score": score,
        "strengths": ["Node.js"],
        "concerns": ["No Redis"],
        "rejected": rejected,
    }


def test_upsert_dedupes_by_source_job_id_and_keeps_status():
    assert saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1", 60), _matched("2", 50)]}) == 2
    saved_jobs.update_saved_job_status("linkedin:1", "applied")

    added = saved_jobs.save_jobs_from_run({"run_id": "r2", "jobs": [_matched("1", 72, title="Backend Engineer II")]})
    assert added == 0

    by_key = {j["saved_key"]: j for j in saved_jobs.list_saved_jobs()}
    assert len(by_key) == 2
    job = by_key["linkedin:1"]
    assert job["match_score"] == 72
    assert job["job"]["title"] == "Backend Engineer II"
    assert job["status"] == "applied"
    assert job["run_ids"] == ["r1", "r2"]
    assert job["first_seen_at"] <= job["last_seen_at"]
    assert job["strengths"] == ["Node.js"] and job["concerns"] == ["No Redis"]
    assert "_id" not in job and "age_ref" not in job


def test_rejected_jobs_are_not_saved():
    assert saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1", 10, rejected=True)]}) == 0
    assert saved_jobs.list_saved_jobs() == []


def test_status_validation_and_delete():
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1", 60)]})
    with pytest.raises(ValueError):
        saved_jobs.update_saved_job_status("linkedin:1", "bogus")
    assert saved_jobs.update_saved_job_status("linkedin:missing", "applied") is None
    assert saved_jobs.get_saved_job("linkedin:1")["saved_key"] == "linkedin:1"
    assert saved_jobs.delete_saved_job("linkedin:1") is True
    assert saved_jobs.delete_saved_job("linkedin:1") is False
    assert saved_jobs.get_saved_job("linkedin:1") is None


def test_applied_at_is_stamped_kept_across_runs_and_cleared():
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1", 60)]})
    stamped = saved_jobs.update_saved_job_status("linkedin:1", "applied")["applied_at"]
    assert stamped
    assert saved_jobs.update_saved_job_status("linkedin:1", "applied")["applied_at"] == stamped

    saved_jobs.save_jobs_from_run({"run_id": "r2", "jobs": [_matched("1", 70)]})
    job = saved_jobs.list_saved_jobs()[0]
    assert (job["status"], job["applied_at"]) == ("applied", stamped)

    assert "applied_at" not in saved_jobs.update_saved_job_status("linkedin:1", "interested")


def _iso(hours_ago):
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def test_jobs_older_than_24h_are_purged_on_save_and_list(monkeypatch):
    monkeypatch.delenv("SAVED_JOBS_MAX_AGE_HOURS", raising=False)
    old = _matched("old", 60)
    old["job"]["posted_date"] = _iso(30)
    fresh = _matched("fresh", 60)
    fresh["job"]["posted_date"] = _iso(2)
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [old, fresh]})
    assert [j["saved_key"] for j in saved_jobs.list_saved_jobs()] == ["linkedin:fresh"]

    # Simulate the job having been first seen 25h ago: the purge also runs on list.
    get_collection(SAVED_JOBS_COLLECTION).update_one(
        {"_id": "linkedin:fresh"}, {"$set": {"age_ref": datetime.now(timezone.utc) - timedelta(hours=25)}}
    )
    assert saved_jobs.list_saved_jobs() == []


def test_job_without_posted_date_ages_from_first_seen():
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1", 60)]})
    doc = get_collection(SAVED_JOBS_COLLECTION).find_one({"_id": "linkedin:1"})
    # BSON dates keep milliseconds only.
    assert abs(doc["age_ref"] - saved_jobs.job_age_reference(doc)) < timedelta(milliseconds=1)


def test_purge_can_be_disabled(monkeypatch):
    monkeypatch.setenv("SAVED_JOBS_MAX_AGE_HOURS", "0")
    old = _matched("old", 60)
    old["job"]["posted_date"] = _iso(100)
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [old]})
    assert len(saved_jobs.list_saved_jobs()) == 1


def test_saved_job_keys():
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1", 60), _matched("2", 50)]})
    assert saved_jobs.saved_job_keys() == {"linkedin:1", "linkedin:2"}


def test_clear_saved_jobs():
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1", 60), _matched("2", 50)]})
    assert saved_jobs.clear_saved_jobs() == 2
    assert saved_jobs.list_saved_jobs() == []
