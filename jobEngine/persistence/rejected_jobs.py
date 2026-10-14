"""Jobs a search run rejected (MongoDB ``rejected_jobs``), so later runs skip them.

The saved-jobs store keeps only recommended jobs, so without this a job the
rules, the LLM score or the worldwide-remote check turned down came back on the
next run and was opened, matched and paid for again. Every rejected job of a
run is stored here with its full analysis, keyed by the same ``saved_key``:

  * the source nodes treat it as known — LinkedIn doesn't open its detail pane;
  * the matching node reuses the stored analysis instead of calling the LLM,
    as long as the job text (``raw_text_hash``) and the candidate profile
    (``profile_hash``) are unchanged; the request's minimum score is re-applied.

Documents expire ``REJECTED_JOBS_MAX_AGE_HOURS`` (default 168 = 7 days) after
the job was last rejected, via a TTL index on ``expires_at``. Nothing here is
shown on the job pages.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from pymongo import ReplaceOne

from .mongo import REJECTED_JOBS_COLLECTION, get_collection, to_bson_safe
from .saved_jobs import saved_job_key

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_HOURS = 168.0


def _rejected_jobs():
    return get_collection(REJECTED_JOBS_COLLECTION)


def _max_age() -> timedelta:
    try:
        hours = float((os.getenv("REJECTED_JOBS_MAX_AGE_HOURS") or "").strip() or _DEFAULT_MAX_AGE_HOURS)
    except ValueError:
        hours = _DEFAULT_MAX_AGE_HOURS
    return timedelta(hours=max(1.0, hours))


def save_rejected_jobs(
    matched_jobs: Iterable[dict[str, Any]], *, run_id: str, profile_hash: str, now: datetime | None = None
) -> int:
    """Store every rejected job of a run; returns how many were written."""
    now = now or datetime.now(timezone.utc)
    expires_at = now + _max_age()
    ops = []
    for matched in matched_jobs:
        if not isinstance(matched, dict) or not matched.get("rejected"):
            continue
        key = saved_job_key(matched)
        if not key:
            continue
        doc = {
            **to_bson_safe(matched),
            "_id": key,
            "saved_key": key,
            "profile_hash": profile_hash,
            "run_id": run_id,
            "rejected_at": now.isoformat(),
            "expires_at": expires_at,
        }
        ops.append(ReplaceOne({"_id": key}, doc, upsert=True))
    if ops:
        _rejected_jobs().bulk_write(ops, ordered=False)
        logger.info("Rejected jobs store: %s jobs recorded from run %s", len(ops), run_id)
    return len(ops)


def _not_expired(now: datetime | None = None) -> dict[str, Any]:
    # The TTL monitor runs about once a minute; don't hand out what it is about to delete.
    return {"expires_at": {"$gt": now or datetime.now(timezone.utc)}}


def rejected_jobs_by_key(keys: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Stored rejected jobs for these keys."""
    wanted = [k for k in keys if k]
    if not wanted:
        return {}
    query = {"_id": {"$in": wanted}, **_not_expired()}
    return {doc["_id"]: doc for doc in _rejected_jobs().find(query, {"expires_at": 0})}


def rejected_descriptions(source_id: str) -> dict[str, str]:
    """Descriptions of this source's rejected jobs by ``source_job_id`` (for the source nodes)."""
    query = {"job.source": source_id, "job.source_job_id": {"$nin": [None, ""]}, **_not_expired()}
    projection = {"job.source_job_id": 1, "job.description": 1}
    known: dict[str, str] = {}
    for doc in _rejected_jobs().find(query, projection):
        job = doc.get("job") or {}
        if job.get("description"):
            known[str(job["source_job_id"])] = job["description"]
    return known


def clear_rejected_jobs() -> int:
    count = _rejected_jobs().delete_many({}).deleted_count
    logger.info("Rejected jobs store cleared (%s jobs removed)", count)
    return count
