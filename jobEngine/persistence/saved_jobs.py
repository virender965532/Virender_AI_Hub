"""Cumulative store of every recommended job across search runs.

``repository.py`` keeps one JSON file per run; this module keeps a single
de-duplicated library of matched jobs (LLM score, strengths, concerns, full
description, skills, ...) for the Saved Jobs page. Same flat-file convention —
one JSON document, guarded by a process-wide lock.

A job is keyed by ``<source>:<source_job_id>`` when the source exposes a stable
id (LinkedIn does), otherwise by the content-hash ``job_id``. Seeing a job again
in a later run refreshes its scores and ``last_seen_at``; ``first_seen_at`` and
the user's ``status`` are kept.

Jobs older than ``SAVED_JOBS_MAX_AGE_HOURS`` (default 24) are purged whenever the
store is written by a search run or read by the Saved Jobs page. Age is taken
from the job's posted date, falling back to when it was first saved.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config.settings import JOB_ENGINE_HISTORY_DIR, JOB_ENGINE_OUTPUT_DIR

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_SAVED_JOBS_FILE = JOB_ENGINE_HISTORY_DIR / "saved_jobs.json"
_MAX_RUN_IDS = 20

SAVED_JOB_STATUSES = ("new", "interested", "applied", "not_interested")


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


def _purge_expired(store: dict[str, dict[str, Any]], *, now: datetime | None = None) -> int:
    max_age = _max_age()
    if max_age is None:
        return 0
    cutoff = (now or datetime.now(timezone.utc)) - max_age
    expired = [key for key, saved in store.items() if (job_age_reference(saved) or cutoff) < cutoff]
    for key in expired:
        del store[key]
    if expired:
        logger.info("Saved jobs store: purged %s jobs older than %s", len(expired), max_age)
    return len(expired)


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


def _load() -> dict[str, dict[str, Any]]:
    if not _SAVED_JOBS_FILE.exists():
        return {}
    try:
        with _SAVED_JOBS_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:  # noqa: BLE001
        logger.warning("Could not read saved jobs store; treating as empty", exc_info=True)
        return {}


def _write(store: dict[str, dict[str, Any]]) -> None:
    JOB_ENGINE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _SAVED_JOBS_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2, ensure_ascii=False, default=str)
    tmp.replace(_SAVED_JOBS_FILE)


def _merge(store: dict[str, dict[str, Any]], jobs: list[dict[str, Any]], *, run_id: str, seen_at: str) -> int:
    added = 0
    for matched in jobs:
        if not isinstance(matched, dict) or matched.get("rejected"):
            continue
        key = saved_job_key(matched)
        if not key:
            continue
        previous = store.get(key)
        run_ids = list((previous or {}).get("run_ids") or [])
        if run_id and run_id not in run_ids:
            run_ids.append(run_id)
        store[key] = {
            **matched,
            "saved_key": key,
            "status": (previous or {}).get("status", "new"),
            "applied_at": (previous or {}).get("applied_at"),
            "first_seen_at": (previous or {}).get("first_seen_at", seen_at),
            "last_seen_at": seen_at,
            "run_ids": run_ids[-_MAX_RUN_IDS:],
        }
        if previous is None:
            added += 1
    return added


def save_jobs_from_run(result: dict[str, Any]) -> int:
    """Upsert every non-rejected job of a finished run; returns how many were new."""
    jobs = result.get("jobs") or []
    if not jobs:
        return 0
    with _lock:
        store = _load_or_backfill()
        added = _merge(store, jobs, run_id=result.get("run_id", ""), seen_at=_now())
        _purge_expired(store)
        _write(store)
    logger.info("Saved jobs store: %s new / %s total from run %s", added, len(jobs), result.get("run_id"))
    return added


def _load_or_backfill() -> dict[str, dict[str, Any]]:
    """First use: seed the store from runs already persisted in ``output/job_engine``."""
    if _SAVED_JOBS_FILE.exists():
        return _load()
    store: dict[str, dict[str, Any]] = {}
    run_files = sorted(JOB_ENGINE_OUTPUT_DIR.glob("*.json")) if JOB_ENGINE_OUTPUT_DIR.exists() else []
    for path in run_files:  # oldest first, so later runs refresh scores
        try:
            with path.open("r", encoding="utf-8") as fh:
                run = json.load(fh)
        except Exception:  # noqa: BLE001
            logger.warning("Skipping unreadable run file %s during backfill", path)
            continue
        seen_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        _merge(store, run.get("jobs") or [], run_id=run.get("run_id", path.stem), seen_at=seen_at)
    if store:
        logger.info("Backfilled saved jobs store with %s jobs from %s runs", len(store), len(run_files))
    _write(store)
    return store


def list_saved_jobs() -> list[dict[str, Any]]:
    """All saved jobs (expired ones purged first), most recently seen first."""
    with _lock:
        store = _load_or_backfill()
        if _purge_expired(store):
            _write(store)
    return sorted(store.values(), key=lambda j: j.get("last_seen_at") or "", reverse=True)


def update_saved_job_status(key: str, status: str) -> dict[str, Any] | None:
    if status not in SAVED_JOB_STATUSES:
        raise ValueError(f"status must be one of {SAVED_JOB_STATUSES}")
    with _lock:
        store = _load_or_backfill()
        job = store.get(key)
        if job is None:
            return None
        job["status"] = status
        if status == "applied":
            job["applied_at"] = job.get("applied_at") or _now()
        else:
            job.pop("applied_at", None)
        _write(store)
    return job


def delete_saved_job(key: str) -> bool:
    with _lock:
        store = _load_or_backfill()
        if store.pop(key, None) is None:
            return False
        _write(store)
    return True


def clear_saved_jobs() -> int:
    """Remove every saved job; returns how many were removed."""
    with _lock:
        count = len(_load_or_backfill())
        _write({})
    logger.info("Saved jobs store cleared (%s jobs removed)", count)
    return count
