"""Browser session (cookie) store in the ``browser_sessions`` collection.

One document per auth-gated source (``_id`` = source id, e.g. ``linkedin``)
holding the Playwright ``storage_state`` dict (``cookies`` + ``origins``) that
``BrowserContext.storage_state()`` returns and ``Browser.new_context`` accepts.

Sessions saved before the move to MongoDB lived in a JSON file; pass that path
as ``legacy_path`` and, if MongoDB has no session yet, the file is imported
once so the next run does not have to sign in again.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .mongo import SESSIONS_COLLECTION, get_collection

logger = logging.getLogger(__name__)


def _sessions():
    return get_collection(SESSIONS_COLLECTION)


def load_storage_state(source_id: str, *, legacy_path: Path | None = None) -> dict[str, Any] | None:
    doc = _sessions().find_one({"_id": source_id}, {"storage_state": 1})
    if doc and doc.get("storage_state"):
        return doc["storage_state"]
    if legacy_path is not None and legacy_path.exists():
        try:
            state = json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logger.warning("Could not read legacy %s session file %s", source_id, legacy_path, exc_info=True)
            return None
        if isinstance(state, dict) and state.get("cookies"):
            save_storage_state(source_id, state)
            logger.info("Imported %s session from %s into MongoDB", source_id, legacy_path)
            return state
    return None


def save_storage_state(source_id: str, state: dict[str, Any]) -> None:
    _sessions().replace_one(
        {"_id": source_id},
        {
            "_id": source_id,
            "storage_state": state,
            "cookie_count": len(state.get("cookies") or []),
            "updated_at": datetime.now(timezone.utc),
        },
        upsert=True,
    )


def has_storage_state(source_id: str) -> bool:
    return _sessions().count_documents({"_id": source_id, "storage_state.cookies.0": {"$exists": True}}, limit=1) > 0


def delete_storage_state(source_id: str) -> bool:
    return _sessions().delete_one({"_id": source_id}).deleted_count == 1
