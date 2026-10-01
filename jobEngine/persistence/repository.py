"""Flat-file persistence for job search runs (section 24/60).

Follows the existing ``jobApplyPro/history.py`` convention (JSON files under
a project-level directory) rather than introducing a database dependency
this project doesn't otherwise have — see docs/job_engine/PROJECT_ANALYSIS.md
"Recommended integration strategy". Every function here is written behind a
narrow interface so a real database could be swapped in later (e.g. behind
the same ``save_run``/``get_run``/``list_runs`` signatures) without touching
callers.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..config.settings import JOB_ENGINE_HISTORY_DIR, JOB_ENGINE_OUTPUT_DIR

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_RUNS_INDEX_FILE = JOB_ENGINE_HISTORY_DIR / "runs_index.json"


def _ensure_dirs() -> None:
    JOB_ENGINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JOB_ENGINE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _run_file(run_id: str) -> Path:
    return JOB_ENGINE_OUTPUT_DIR / f"{run_id}.json"


def _run_files_max_age() -> timedelta | None:
    """``RUN_FILES_MAX_AGE_HOURS`` (default 25); 0 or negative keeps every run file."""
    try:
        hours = float((os.getenv("RUN_FILES_MAX_AGE_HOURS") or "25").strip())
    except ValueError:
        hours = 25.0
    return timedelta(hours=hours) if hours > 0 else None


def _purge_old_run_files(*, keep: str) -> int:
    """Delete ``job-search-*.json`` run files last written before the cutoff.

    Safe because saved jobs live in ``history/job_engine/saved_jobs.json``; run
    files are only read back to seed that store when it doesn't exist yet."""
    max_age = _run_files_max_age()
    if max_age is None:
        return 0
    cutoff = (datetime.now(timezone.utc) - max_age).timestamp()
    removed = 0
    for path in JOB_ENGINE_OUTPUT_DIR.glob("job-search-*.json"):
        if path.stem == keep:
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            logger.warning("Could not remove old run file %s", path, exc_info=True)
    if removed:
        logger.info("Removed %s job-engine run files older than %s", removed, max_age)
    return removed


def _load_index() -> list[dict[str, Any]]:
    if not _RUNS_INDEX_FILE.exists():
        return []
    try:
        with _RUNS_INDEX_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        logger.warning("Could not read job-engine runs index; treating as empty", exc_info=True)
        return []


def save_run(result: dict[str, Any]) -> None:
    """Persist a full ``JobSearchResult`` dict and append a lightweight index entry."""
    _ensure_dirs()
    run_id = result.get("run_id", "")
    if not run_id:
        raise ValueError("result must include run_id")

    with _lock:
        run_path = _run_file(run_id)
        with run_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False, default=str)

        _purge_old_run_files(keep=run_id)
        # Index entries of purged runs would only 404 when opened.
        index = [e for e in _load_index() if _run_file(e.get("run_id", "")).exists()]
        summary = result.get("summary") or {}
        index.append(
            {
                "run_id": run_id,
                "status": result.get("status"),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "jobs_recommended": summary.get("jobs_recommended", 0),
                "sources_processed": summary.get("sources_processed", 0),
            }
        )
        with _RUNS_INDEX_FILE.open("w", encoding="utf-8") as fh:
            json.dump(index[-500:], fh, indent=2, ensure_ascii=False)

    logger.info("Persisted job-engine run %s (status=%s)", run_id, result.get("status"))


def get_run(run_id: str) -> dict[str, Any] | None:
    path = _run_file(run_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        logger.warning("Could not read persisted run %s", run_id, exc_info=True)
        return None


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    index = _load_index()
    return list(reversed(index))[: max(1, limit)]
