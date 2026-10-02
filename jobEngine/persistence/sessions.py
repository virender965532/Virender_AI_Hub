"""Browser session (cookie) store in the ``browser_sessions`` collection.

One document per auth-gated source (``_id`` = source id, e.g. ``linkedin``)
holding the Playwright ``storage_state`` dict (``cookies`` + ``origins``) that
``BrowserContext.storage_state()`` returns and ``Browser.new_context`` accepts.

Sessions saved before the move to MongoDB lived in a JSON file; pass that path
as ``legacy_path`` and, if MongoDB has no session yet, the file is imported
once so the next run does not have to sign in again.

Only a signed-in state (an unexpired ``li_at`` auth cookie) is ever stored: a
run that LinkedIn signed out must not overwrite the good session with guest
cookies. Such a run calls ``mark_signed_out`` instead, so the stored session
reads as disconnected until the next sign-in saves a fresh one.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .mongo import SESSIONS_COLLECTION, get_collection

logger = logging.getLogger(__name__)


def _sessions():
    return get_collection(SESSIONS_COLLECTION)


def load_storage_state(source_id: str, *, legacy_path: Path | None = None) -> dict[str, Any] | None:
    doc = _sessions().find_one({"_id": source_id}, {"storage_state": 1, "signed_out_at": 1})
    if doc and doc.get("signed_out_at"):
        return None  # rejected by the site: replaying it only gets the run signed out again
    if doc and auth_cookie_expiry(doc.get("storage_state")) is not None:
        return doc["storage_state"]
    if legacy_path is not None and legacy_path.exists():
        try:
            state = json.loads(legacy_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            logger.warning("Could not read legacy %s session file %s", source_id, legacy_path, exc_info=True)
            return None
        if isinstance(state, dict) and save_storage_state(source_id, state):
            logger.info("Imported %s session from %s into MongoDB", source_id, legacy_path)
            return state
    return None


AUTH_COOKIE = "li_at"


def auth_cookie_expiry(state: dict[str, Any] | None) -> float | None:
    """Expiry (epoch seconds, ``-1`` = browser-session cookie) of a live auth cookie, else None."""
    for cookie in (state or {}).get("cookies") or []:
        if cookie.get("name") != AUTH_COOKIE or not cookie.get("value"):
            continue
        expires = cookie.get("expires", -1)
        if expires is None or expires < 0 or expires > time.time():
            return -1 if expires is None else expires
    return None


def save_storage_state(source_id: str, state: dict[str, Any]) -> bool:
    """Store a signed-in state; a state without a live auth cookie is refused (returns False)."""
    expiry = auth_cookie_expiry(state)
    if expiry is None:
        logger.warning("Not saving %s session: no live %s cookie (signed out?)", source_id, AUTH_COOKIE)
        return False
    _sessions().replace_one(
        {"_id": source_id},
        {
            "_id": source_id,
            "storage_state": state,
            "cookie_count": len(state.get("cookies") or []),
            "auth_expires_at": datetime.fromtimestamp(expiry, timezone.utc) if expiry > 0 else None,
            "updated_at": datetime.now(timezone.utc),
            "signed_out_at": None,
        },
        upsert=True,
    )
    return True


def mark_signed_out(source_id: str, reason: str) -> None:
    """Flag the stored session as rejected by the site; its cookies are kept for inspection."""
    _sessions().update_one(
        {"_id": source_id},
        {"$set": {"signed_out_at": datetime.now(timezone.utc), "signed_out_reason": reason}},
    )
    logger.warning("%s session marked signed out: %s", source_id, reason)


def has_storage_state(source_id: str) -> bool:
    return storage_state_info(source_id) is not None


def storage_state_info(source_id: str) -> dict[str, Any] | None:
    """Save time and auth-cookie expiry of a usable stored session, without the cookies.

    None when there is no session, it has no live auth cookie, or a run found
    it signed out.
    """
    doc = _sessions().find_one({"_id": source_id})
    if not doc or doc.get("signed_out_at") or auth_cookie_expiry(doc.get("storage_state")) is None:
        return None
    updated, expires = doc.get("updated_at"), doc.get("auth_expires_at")
    return {
        "cookie_count": doc.get("cookie_count", 0),
        "updated_at": updated.isoformat() if updated else None,
        "expires_at": expires.isoformat() if expires else None,
    }


def delete_storage_state(source_id: str) -> bool:
    return _sessions().delete_one({"_id": source_id}).deleted_count == 1
