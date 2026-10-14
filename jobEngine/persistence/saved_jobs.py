"""Cumulative store of every recommended job across search runs (MongoDB).

``repository.py`` keeps one document per run; this module keeps a single
de-duplicated library of matched jobs (LLM score, strengths, concerns, full
description, skills, ...) for the Saved Jobs page, in the ``saved_jobs``
collection — one document per job, ``_id`` = ``saved_key``.

A job is keyed by ``<source>:<source_job_id>`` when the source exposes a stable
id (LinkedIn does), otherwise by the content-hash ``job_id``. Seeing a job again
in a later run refreshes its scores and ``last_seen_at``; ``first_seen_at`` and
the user's ``status`` are kept.

Jobs older than ``SAVED_JOBS_MAX_AGE_HOURS`` (default 24) are purged whenever the
store is written by a search run or read by the Saved Jobs page. Age is taken
from the job's posted date, falling back to when it was first saved; it is
stored as ``age_ref`` (a BSON date) so the purge is a single ``delete_many``.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo import DESCENDING, ReplaceOne, ReturnDocument

from .applied_jobs import forget_applied_job, record_applied_job
from .mongo import SAVED_JOBS_COLLECTION, get_collection, to_bson_safe

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_MAX_RUN_IDS = 20
# Internal fields never returned to callers (saved_key already mirrors _id).
_HIDDEN_FIELDS = {"_id": 0, "age_ref": 0}

SAVED_JOB_STATUSES = ("new", "interested", "applied", "not_interested")


def _saved_jobs():
    return get_collection(SAVED_JOBS_COLLECTION)


def _max_age() -> timedelta | None:
    """``SAVED_JOBS_MAX_AGE_HOURS``; 0 or negative disables the purge."""
    try:
        hours = float((os.getenv("SAVED_JOBS_MAX_AGE_HOURS") or "24").strip())
    except ValueError:
        hours = 24.0
    return timedelta(hours=hours) if hours > 0 else None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def job_age_reference(saved: dict[str, Any]) -> datetime | None:
    """Posted date when the source gave one, else when the job was first saved."""
    return _parse_ts((saved.get("job") or {}).get("posted_date")) or _parse_ts(saved.get("first_seen_at"))


def _purge_expired(*, now: datetime | None = None) -> int:
    max_age = _max_age()
    if max_age is None:
        return 0
    cutoff = (now or datetime.now(timezone.utc)) - max_age
    # Jobs with no usable date have age_ref null, which $lt never matches: kept.
    removed = _saved_jobs().delete_many({"age_ref": {"$lt": cutoff}}).deleted_count
    if removed:
        logger.info("Saved jobs store: purged %s jobs older than %s", removed, max_age)
    return removed


def saved_job_key(matched: dict[str, Any]) -> str | None:
    job = matched.get("job") or {}
    source = job.get("source") or "unknown"
    if job.get("source_job_id"):
        return f"{source}:{job['source_job_id']}"
    if job.get("job_id"):
        return f"{source}:{job['job_id']}"
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merged_doc(matched: dict[str, Any], previous: dict[str, Any] | None, *, key: str, run_id: str, seen_at: str) -> dict[str, Any]:
    run_ids = list((previous or {}).get("run_ids") or [])
    if run_id and run_id not in run_ids:
        run_ids.append(run_id)
    # A re-run may miss the posted time (e.g. detail pane skipped); keep the one we had.
    prev_job = (previous or {}).get("job") or {}
    prev_posted = prev_job.get("posted_date")
    if prev_posted and not (matched.get("job") or {}).get("posted_date"):
        matched = {**matched, "job": {**matched["job"], "posted_date": prev_posted}}
    # Same for the employment type (read from the detail pane, skipped for saved jobs),
    # and keep every search URL that has ever listed the job.
    if prev_job.get("employment_type") and not matched["job"].get("employment_type"):
        restored = {"employment_type": prev_job["employment_type"]}
        # The previous classification saw the employment type; this one didn't.
        if prev_job.get("worldwide") and prev_job.get("raw_text_hash") == matched["job"].get("raw_text_hash"):
            restored["worldwide"] = prev_job["worldwide"]
        matched = {**matched, "job": {**matched["job"], **restored}}
    if prev_job.get("search_urls"):
        urls = list(matched["job"].get("search_urls") or [])
        urls += [u for u in prev_job["search_urls"] if u not in urls]
        matched = {**matched, "job": {**matched["job"], "search_urls": urls}}
    doc = {
        **to_bson_safe(matched),
        "_id": key,
        "saved_key": key,
        "status": (previous or {}).get("status", "new"),
        "applied_at": (previous or {}).get("applied_at"),
        "first_seen_at": (previous or {}).get("first_seen_at", seen_at),
        "last_seen_at": seen_at,
        "run_ids": run_ids[-_MAX_RUN_IDS:],
    }
    doc["age_ref"] = job_age_reference(doc)
    return doc


def upsert_jobs(jobs: list[dict[str, Any]], *, run_id: str, seen_at: str) -> int:
    """Merge matched jobs into the store; returns how many were new."""
    by_key: dict[str, dict[str, Any]] = {}
    for matched in jobs:
        if not isinstance(matched, dict) or matched.get("rejected"):
            continue
        key = saved_job_key(matched)
        if key:
            by_key[key] = matched  # a later duplicate in the same run wins
    if not by_key:
        return 0

    coll = _saved_jobs()
    previous = {doc["_id"]: doc for doc in coll.find({"_id": {"$in": list(by_key)}})}
    ops = [
        ReplaceOne(
            {"_id": key},
            _merged_doc(matched, previous.get(key), key=key, run_id=run_id, seen_at=seen_at),
            upsert=True,
        )
        for key, matched in by_key.items()
    ]
    coll.bulk_write(ops, ordered=False)
    return sum(1 for key in by_key if key not in previous)


def save_jobs_from_run(result: dict[str, Any]) -> int:
    """Upsert every non-rejected job of a finished run; returns how many were new."""
    jobs = result.get("jobs") or []
    if not jobs:
        return 0
    with _lock:
        added = upsert_jobs(jobs, run_id=result.get("run_id", ""), seen_at=_now())
        _purge_expired()
    logger.info("Saved jobs store: %s new / %s total from run %s", added, len(jobs), result.get("run_id"))
    return added


def list_saved_jobs(*, include_not_eligible: bool = True) -> list[dict[str, Any]]:
    """All saved jobs (expired ones purged first), most recently seen first.

    ``include_not_eligible=False`` leaves out jobs classified country-restricted
    (not doable from India); saved before that check rejected them, they stay
    in the store until purged so the search can still reuse their analysis.
    """
    with _lock:
        _purge_expired()
    query: dict[str, Any] = {} if include_not_eligible else {"job.worldwide.category": {"$ne": "country_restricted"}}
    return list(_saved_jobs().find(query, _HIDDEN_FIELDS).sort("last_seen_at", DESCENDING))


def saved_job_keys() -> set[str]:
    """Keys of every saved job — a cheap existence check (no purge, ids only)."""
    return {doc["_id"] for doc in _saved_jobs().find({}, {"_id": 1})}


def get_saved_job(key: str) -> dict[str, Any] | None:
    return _saved_jobs().find_one({"_id": key}, _HIDDEN_FIELDS)


def update_saved_job_status(key: str, status: str) -> dict[str, Any] | None:
    if status not in SAVED_JOB_STATUSES:
        raise ValueError(f"status must be one of {SAVED_JOB_STATUSES}")
    coll = _saved_jobs()
    if status == "applied":
        # Stamp applied_at only the first time (matches a null or missing field).
        coll.update_one({"_id": key, "applied_at": None}, {"$set": {"applied_at": _now()}})
        update: dict[str, Any] = {"$set": {"status": status}}
    else:
        update = {"$set": {"status": status}, "$unset": {"applied_at": ""}}
    job = coll.find_one_and_update(
        {"_id": key}, update, projection=_HIDDEN_FIELDS, return_document=ReturnDocument.AFTER
    )
    # Mirror into the applied-jobs collection, which outlives the saved-jobs purge.
    if job is not None:
        if status == "applied":
            record_applied_job(job)
        else:
            forget_applied_job(key)
    return job


def delete_saved_job(key: str) -> bool:
    return _saved_jobs().delete_one({"_id": key}).deleted_count == 1


def clear_saved_jobs() -> int:
    """Remove every saved job; returns how many were removed."""
    count = _saved_jobs().delete_many({}).deleted_count
    logger.info("Saved jobs store cleared (%s jobs removed)", count)
    return count
