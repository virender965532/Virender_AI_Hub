import json

import pytest

from jobEngine.persistence import saved_jobs


@pytest.fixture
def store_paths(tmp_path, monkeypatch):
    history = tmp_path / "history"
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(saved_jobs, "JOB_ENGINE_HISTORY_DIR", history)
    monkeypatch.setattr(saved_jobs, "JOB_ENGINE_OUTPUT_DIR", output)
    monkeypatch.setattr(saved_jobs, "_SAVED_JOBS_FILE", history / "saved_jobs.json")
    return output


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


def test_upsert_dedupes_by_source_job_id_and_keeps_status(store_paths):
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


def test_rejected_jobs_are_not_saved(store_paths):
    assert saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1", 10, rejected=True)]}) == 0
    assert saved_jobs.list_saved_jobs() == []


def test_backfills_from_existing_run_files(store_paths):
    (store_paths / "job-search-1.json").write_text(
        json.dumps({"run_id": "job-search-1", "jobs": [_matched("9", 55)]}), encoding="utf-8"
    )
    jobs = saved_jobs.list_saved_jobs()
    assert [j["saved_key"] for j in jobs] == ["linkedin:9"]
    assert jobs[0]["run_ids"] == ["job-search-1"]


def test_status_validation_and_delete(store_paths):
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1", 60)]})
    with pytest.raises(ValueError):
        saved_jobs.update_saved_job_status("linkedin:1", "bogus")
    assert saved_jobs.update_saved_job_status("linkedin:missing", "applied") is None
    assert saved_jobs.delete_saved_job("linkedin:1") is True
    assert saved_jobs.delete_saved_job("linkedin:1") is False


def _iso(hours_ago):
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def test_jobs_older_than_24h_are_purged_on_save_and_list(store_paths, monkeypatch):
    monkeypatch.delenv("SAVED_JOBS_MAX_AGE_HOURS", raising=False)
    old = _matched("old", 60)
    old["job"]["posted_date"] = _iso(30)
    fresh = _matched("fresh", 60)
    fresh["job"]["posted_date"] = _iso(2)
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [old, fresh]})
    assert [j["saved_key"] for j in saved_jobs.list_saved_jobs()] == ["linkedin:fresh"]

    # No posted date: falls back to first_seen_at, and the purge also runs on list.
    store = json.loads(saved_jobs._SAVED_JOBS_FILE.read_text(encoding="utf-8"))
    store["linkedin:fresh"]["job"]["posted_date"] = None
    store["linkedin:fresh"]["first_seen_at"] = _iso(25)
    saved_jobs._SAVED_JOBS_FILE.write_text(json.dumps(store), encoding="utf-8")
    assert saved_jobs.list_saved_jobs() == []


def test_purge_can_be_disabled(store_paths, monkeypatch):
    monkeypatch.setenv("SAVED_JOBS_MAX_AGE_HOURS", "0")
    old = _matched("old", 60)
    old["job"]["posted_date"] = _iso(100)
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [old]})
    assert len(saved_jobs.list_saved_jobs()) == 1


def test_clear_saved_jobs(store_paths):
    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [_matched("1", 60), _matched("2", 50)]})
    assert saved_jobs.clear_saved_jobs() == 2
    assert saved_jobs.list_saved_jobs() == []
