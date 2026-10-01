import json
import os
import time

from jobEngine.persistence import repository


def _point_at(tmp_path, monkeypatch):
    out, hist = tmp_path / "output", tmp_path / "history"
    out.mkdir()
    hist.mkdir()
    monkeypatch.setattr(repository, "JOB_ENGINE_OUTPUT_DIR", out)
    monkeypatch.setattr(repository, "JOB_ENGINE_HISTORY_DIR", hist)
    monkeypatch.setattr(repository, "_RUNS_INDEX_FILE", hist / "runs_index.json")
    return out, hist


def _age(path, hours):
    stamp = time.time() - hours * 3600
    os.utime(path, (stamp, stamp))


def test_save_run_purges_run_files_older_than_25_hours(tmp_path, monkeypatch):
    monkeypatch.delenv("RUN_FILES_MAX_AGE_HOURS", raising=False)
    out, hist = _point_at(tmp_path, monkeypatch)
    repository.save_run({"run_id": "job-search-old"})
    repository.save_run({"run_id": "job-search-recent"})
    _age(out / "job-search-old.json", 26)
    _age(out / "job-search-recent.json", 24)
    (out / "notes.json").write_text("{}")
    _age(out / "notes.json", 100)

    repository.save_run({"run_id": "job-search-new"})

    assert sorted(p.name for p in out.iterdir()) == [
        "job-search-new.json", "job-search-recent.json", "notes.json",
    ]
    index_ids = [e["run_id"] for e in json.loads((hist / "runs_index.json").read_text())]
    assert index_ids == ["job-search-recent", "job-search-new"]


def test_purge_disabled_with_zero_max_age(tmp_path, monkeypatch):
    monkeypatch.setenv("RUN_FILES_MAX_AGE_HOURS", "0")
    out, _ = _point_at(tmp_path, monkeypatch)
    repository.save_run({"run_id": "job-search-old"})
    _age(out / "job-search-old.json", 1000)

    repository.save_run({"run_id": "job-search-new"})

    assert (out / "job-search-old.json").exists()
