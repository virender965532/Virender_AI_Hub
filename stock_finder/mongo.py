"""MongoDB connection for Stock Finder (stock research database).

Mirrors ``jobEngine/persistence/mongo.py`` but is a separate connection with its
own settings, so Stock Finder never depends on the job engine:

``SF_MONGODB_URI`` (falls back to ``MONGODB_URI``) and ``SF_MONGODB_DB_NAME``
(default ``stock_research``) select the database. One ``MongoClient`` per
process (it is thread-safe and pools connections). Indexes are created once
per collection on first use.

Collections:
    sf_holdings   one document per holding (``_id`` = "<portfolio_id>:<symbol>")
    sf_reports    every generated report with prompt version, models and data snapshot (``_id`` = report_id)
"""

from __future__ import annotations

import json
import threading
from typing import Any

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from .config import get_settings

HOLDINGS_COLLECTION = "sf_holdings"
REPORTS_COLLECTION = "sf_reports"

_INDEXES: dict[str, list[tuple[list[tuple[str, int]], dict[str, Any]]]] = {
    HOLDINGS_COLLECTION: [
        ([("portfolio_id", ASCENDING), ("symbol", ASCENDING)], {"unique": True}),
    ],
    REPORTS_COLLECTION: [
        ([("created_at", DESCENDING)], {}),
        ([("kind", ASCENDING), ("created_at", DESCENDING)], {}),
    ],
}

_lock = threading.Lock()
_client: MongoClient | None = None
_indexed: set[str] = set()


class MongoNotConfigured(RuntimeError):
    pass


def get_db() -> Database:
    global _client
    settings = get_settings()
    with _lock:
        if _client is None:
            if not settings.mongodb_uri:
                raise MongoNotConfigured("SF_MONGODB_URI (or MONGODB_URI) is not set")
            _client = MongoClient(settings.mongodb_uri, appname="viru-ai-hub-stock-finder", tz_aware=True,
                                  serverSelectionTimeoutMS=10_000)
        client = _client
    return client[settings.mongodb_db_name]


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
    """JSON round-trip into plain BSON types (dates become ISO strings, NaN-free numbers)."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def reset_for_tests(client: Any | None = None) -> None:
    """Swap in a client (e.g. ``mongomock.MongoClient()``) or drop the cached one."""
    global _client
    with _lock:
        _client = client
        _indexed.clear()
