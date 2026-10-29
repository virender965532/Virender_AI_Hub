"""MongoDB connection shared by every job-engine store.

``MONGODB_URI`` / ``MONGODB_DB_NAME`` (default ``vai_hub``) select the database.
One ``MongoClient`` per process (it is thread-safe and pools connections); the
Flask app, its background search threads and ``jobEngine.cron`` each get their
own. Indexes are created once per collection on first use.

Collections:
    job_engine_runs   one document per search run (``_id`` = run_id)
    saved_jobs        de-duplicated library of matched jobs (``_id`` = saved_key)
    applied_jobs      snapshot of every job marked applied, never purged (``_id`` = saved_key)
    rejected_jobs     jobs a run rejected, so later runs skip them; TTL-expired (``_id`` = saved_key)
    browser_sessions  Playwright storage_state (cookies) per source (``_id`` = source id)
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

RUNS_COLLECTION = "job_engine_runs"
SAVED_JOBS_COLLECTION = "saved_jobs"
APPLIED_JOBS_COLLECTION = "applied_jobs"
REJECTED_JOBS_COLLECTION = "rejected_jobs"
ANALYSIS_CACHE_COLLECTION = "job_analysis_cache"
SESSIONS_COLLECTION = "browser_sessions"

_INDEXES: dict[str, list[tuple[list[tuple[str, int]], dict[str, Any]]]] = {
    RUNS_COLLECTION: [
        ([("saved_at", DESCENDING)], {}),
        ([("jobs.job.job_id", ASCENDING)], {}),
    ],
    SAVED_JOBS_COLLECTION: [
        ([("last_seen_at", DESCENDING)], {}),
        ([("age_ref", ASCENDING)], {}),
    ],
    APPLIED_JOBS_COLLECTION: [
        ([("applied_at", DESCENDING)], {}),
    ],
    REJECTED_JOBS_COLLECTION: [
        # TTL: MongoDB deletes each document once its own expires_at has passed.
        ([("expires_at", ASCENDING)], {"expireAfterSeconds": 0}),
        ([("job.source", ASCENDING)], {}),
    ],
    ANALYSIS_CACHE_COLLECTION: [
        ([("expires_at", ASCENDING)], {"expireAfterSeconds": 0}),
    ],
    SESSIONS_COLLECTION: [],
}

_lock = threading.Lock()
_client: MongoClient | None = None
_indexed: set[str] = set()


class MongoNotConfigured(RuntimeError):
    pass


def get_db() -> Database:
    global _client
    with _lock:
        if _client is None:
            uri = (os.getenv("MONGODB_URI") or "").strip()
            if not uri:
                raise MongoNotConfigured("MONGODB_URI is not set")
            _client = MongoClient(uri, appname="viru-ai-hub", tz_aware=True, serverSelectionTimeoutMS=10_000)
        client = _client
    return client[(os.getenv("MONGODB_DB_NAME") or "").strip() or "vai_hub"]


def get_collection(name: str) -> Collection:
    coll = get_db()[name]
    if name not in _indexed:
        for keys, options in _INDEXES.get(name, []):
            coll.create_index(keys, **options)
        _indexed.add(name)
    return coll


def ping() -> None:
    get_db().command("ping")


def to_bson_safe(value: Any) -> Any:
    """JSON round-trip: what the flat files stored (``default=str``), now as plain BSON types."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def reset_for_tests(client: Any | None = None) -> None:
    """Swap in a client (e.g. ``mongomock.MongoClient()``) or drop the cached one."""
    global _client
    with _lock:
        _client = client
        _indexed.clear()
