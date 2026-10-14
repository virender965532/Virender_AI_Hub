"""Permanent record of every job the user applied to (MongoDB ``applied_jobs``).

Saved jobs are purged after ``SAVED_JOBS_MAX_AGE_HOURS``; applied jobs must not
be. When a saved job's status becomes ``applied`` (see
``saved_jobs.update_saved_job_status``) a snapshot of it — score, analysis, full
description, ``applied_at`` — is copied here, keyed by the same ``saved_key``.
Moving the status away from ``applied`` removes the snapshot. Removing a job
from this collection never touches the saved-jobs store.

The user can mark an applied job rejected (``outcome``/``rejected_at``) and keep
an LLM rejection analysis on it (``rejection_analysis``, see
``jobEngine/matching/rejection_analyzer.py``). Both survive a fresh snapshot.

The first time the collection is used it is backfilled with the saved jobs
already marked applied, so applications made before it existed are kept.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from pymongo import DESCENDING, ReturnDocument

from .mongo import APPLIED_JOBS_COLLECTION, SAVED_JOBS_COLLECTION, get_collection, get_db

logger = logging.getLogger(__name__)

_HIDDEN_FIELDS = {"_id": 0}
# Set on the applied job itself, never by a snapshot of the saved job.
_OWN_FIELDS = ("outcome", "rejected_at", "rejection_analysis")


def _applied_jobs():
    is_new = APPLIED_JOBS_COLLECTION not in get_db().list_collection_names()
    coll = get_collection(APPLIED_JOBS_COLLECTION)
    if is_new:
        _backfill_from_saved_jobs(coll)
    return coll


def _snapshot(saved: dict[str, Any]) -> dict[str, Any]:
    doc = {k: v for k, v in saved.items() if k not in ("_id", "age_ref")}
    doc["_id"] = saved["saved_key"]
    doc["status"] = "applied"
    return doc


def record_applied_job(saved: dict[str, Any]) -> None:
    """Store (or refresh) the snapshot of a saved job the user applied to."""
    if not saved or not saved.get("saved_key"):
        return
    coll = _applied_jobs()
    doc = _snapshot(saved)
    previous = coll.find_one({"_id": doc["_id"]}, {f: 1 for f in _OWN_FIELDS}) or {}
    doc.update({f: previous[f] for f in _OWN_FIELDS if f in previous})
    coll.replace_one({"_id": doc["_id"]}, doc, upsert=True)


def forget_applied_job(key: str) -> bool:
    return _applied_jobs().delete_one({"_id": key}).deleted_count == 1


def _backfill_from_saved_jobs(coll) -> int:
    """Copy saved jobs marked applied before this collection existed; returns how many."""
    added = 0
    for saved in get_collection(SAVED_JOBS_COLLECTION).find({"status": "applied"}):
        coll.replace_one({"_id": saved["_id"]}, _snapshot(saved), upsert=True)
        added += 1
    if added:
        logger.info("Applied jobs store: backfilled %s jobs from saved jobs", added)
    return added


def list_applied_jobs() -> list[dict[str, Any]]:
    """Every applied job, most recently applied first."""
    return list(_applied_jobs().find({}, _HIDDEN_FIELDS).sort("applied_at", DESCENDING))


def get_applied_job(key: str) -> dict[str, Any] | None:
    return _applied_jobs().find_one({"_id": key}, _HIDDEN_FIELDS)


def delete_applied_job(key: str) -> bool:
    return forget_applied_job(key)


def clear_applied_jobs() -> int:
    """Remove every applied job; returns how many were removed."""
    count = _applied_jobs().delete_many({}).deleted_count
    logger.info("Applied jobs store cleared (%s jobs removed)", count)
    return count


def set_rejected(key: str, rejected: bool) -> dict[str, Any] | None:
    """Mark (or unmark) an applied job as rejected; a stored analysis is kept either way."""
    if rejected:
        update = {"$set": {"outcome": "rejected", "rejected_at": datetime.now(timezone.utc).isoformat()}}
    else:
        update = {"$unset": {"outcome": "", "rejected_at": ""}}
    return _applied_jobs().find_one_and_update(
        {"_id": key}, update, projection=_HIDDEN_FIELDS, return_document=ReturnDocument.AFTER
    )


def save_rejection_analysis(key: str, analysis: dict[str, Any]) -> dict[str, Any] | None:
    return _applied_jobs().find_one_and_update(
        {"_id": key},
        {"$set": {"rejection_analysis": analysis}},
        projection=_HIDDEN_FIELDS,
        return_document=ReturnDocument.AFTER,
    )
