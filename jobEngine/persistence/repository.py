"""MongoDB persistence for job search runs (section 24/60).

One document per run in the ``job_engine_runs`` collection: the full
``JobSearchResult`` dict, ``_id`` = run_id, plus a ``saved_at`` timestamp.
The ``save_run``/``get_run``/``list_runs`` signatures are unchanged from the
earlier flat-file store, so callers did not change.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import DESCENDING

from .mongo import RUNS_COLLECTION, get_collection, to_bson_safe

logger = logging.getLogger(__name__)

_HIDDEN_FIELDS = {"_id": 0, "saved_at": 0}


def _runs():
    return get_collection(RUNS_COLLECTION)


def _runs_max_age() -> timedelta | None:
    """``RUN_FILES_MAX_AGE_HOURS`` (default 25); 0 or negative keeps every run."""
    try:
        hours = float((os.getenv("RUN_FILES_MAX_AGE_HOURS") or "25").strip())
    except ValueError:
        hours = 25.0
    return timedelta(hours=hours) if hours > 0 else None


def _purge_old_runs(*, keep: str) -> int:
    """Delete runs saved before the cutoff. Saved jobs live in their own collection."""
    max_age = _runs_max_age()
    if max_age is None:
        return 0
    cutoff = datetime.now(timezone.utc) - max_age
    removed = _runs().delete_many({"_id": {"$ne": keep}, "saved_at": {"$lt": cutoff}}).deleted_count
    if removed:
        logger.info("Removed %s job-engine runs older than %s", removed, max_age)
    return removed


def save_run(result: dict[str, Any]) -> None:
    """Persist a full ``JobSearchResult`` dict (upsert by run_id)."""
    run_id = result.get("run_id", "")
    if not run_id:
        raise ValueError("result must include run_id")

    doc = {**to_bson_safe(result), "_id": run_id, "saved_at": datetime.now(timezone.utc)}
    _runs().replace_one({"_id": run_id}, doc, upsert=True)
    _purge_old_runs(keep=run_id)
    logger.info("Persisted job-engine run %s (status=%s)", run_id, result.get("status"))


def get_run(run_id: str) -> dict[str, Any] | None:
    return _runs().find_one({"_id": run_id}, _HIDDEN_FIELDS)


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    """Lightweight summaries of the most recent runs, newest first."""
    cursor = (
        _runs()
        .find({}, {"run_id": 1, "status": 1, "saved_at": 1, "summary": 1})
        .sort("saved_at", DESCENDING)
        .limit(max(1, limit))
    )
    runs = []
    for doc in cursor:
        summary = doc.get("summary") or {}
        saved_at = doc.get("saved_at")
        runs.append(
            {
                "run_id": doc.get("run_id") or doc["_id"],
                "status": doc.get("status"),
                "saved_at": saved_at.isoformat() if isinstance(saved_at, datetime) else saved_at,
                "jobs_recommended": summary.get("jobs_recommended", 0),
                "sources_processed": summary.get("sources_processed", 0),
            }
        )
    return runs


def find_job_in_runs(job_id: str) -> tuple[dict[str, Any], str] | None:
    """The matched job with this ``job_id`` from the newest run containing it, and that run's id."""
    run = _runs().find_one(
        {"jobs.job.job_id": job_id}, {"run_id": 1, "jobs": 1}, sort=[("saved_at", DESCENDING)]
    )
    if not run:
        return None
    for matched in run.get("jobs") or []:
        if (matched.get("job") or {}).get("job_id") == job_id:
            return matched, run.get("run_id") or run["_id"]
    return None
