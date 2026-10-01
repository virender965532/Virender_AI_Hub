from datetime import datetime, timedelta, timezone

from jobEngine.persistence import repository
from jobEngine.persistence.mongo import RUNS_COLLECTION, get_collection


def _age(run_id, hours):
    get_collection(RUNS_COLLECTION).update_one(
        {"_id": run_id}, {"$set": {"saved_at": datetime.now(timezone.utc) - timedelta(hours=hours)}}
    )


def test_save_get_and_list_runs():
    repository.save_run({"run_id": "job-search-1", "status": "completed", "summary": {"jobs_recommended": 3}, "jobs": []})
    repository.save_run({"run_id": "job-search-2", "status": "partial", "summary": {"sources_processed": 2}, "jobs": []})
    _age("job-search-1", 1)

    assert repository.get_run("job-search-1") == {
        "run_id": "job-search-1", "status": "completed", "summary": {"jobs_recommended": 3}, "jobs": [],
    }
    assert repository.get_run("missing") is None
    runs = repository.list_runs()
    assert [r["run_id"] for r in runs] == ["job-search-2", "job-search-1"]
    assert runs[1]["jobs_recommended"] == 3 and runs[0]["sources_processed"] == 2
    assert isinstance(runs[0]["saved_at"], str)
    assert len(repository.list_runs(limit=1)) == 1


def test_save_run_upserts_by_run_id():
    repository.save_run({"run_id": "job-search-1", "status": "running"})
    repository.save_run({"run_id": "job-search-1", "status": "completed"})
    assert repository.get_run("job-search-1")["status"] == "completed"
    assert len(repository.list_runs()) == 1


def test_save_run_purges_runs_older_than_25_hours(monkeypatch):
    monkeypatch.delenv("RUN_FILES_MAX_AGE_HOURS", raising=False)
    repository.save_run({"run_id": "job-search-old"})
    repository.save_run({"run_id": "job-search-recent"})
    _age("job-search-old", 26)
    _age("job-search-recent", 24)

    repository.save_run({"run_id": "job-search-new"})

    assert sorted(r["run_id"] for r in repository.list_runs()) == ["job-search-new", "job-search-recent"]


def test_purge_disabled_with_zero_max_age(monkeypatch):
    monkeypatch.setenv("RUN_FILES_MAX_AGE_HOURS", "0")
    repository.save_run({"run_id": "job-search-old"})
    _age("job-search-old", 1000)

    repository.save_run({"run_id": "job-search-new"})

    assert repository.get_run("job-search-old") is not None


def test_find_job_in_runs_returns_newest_match():
    job = lambda score: {"job": {"job_id": "abc"}, "match_score": score}  # noqa: E731
    repository.save_run({"run_id": "job-search-1", "jobs": [job(50)]})
    repository.save_run({"run_id": "job-search-2", "jobs": [{"job": {"job_id": "other"}}, job(80)]})
    _age("job-search-1", 2)

    matched, run_id = repository.find_job_in_runs("abc")
    assert (run_id, matched["match_score"]) == ("job-search-2", 80)
    assert repository.find_job_in_runs("nope") is None
