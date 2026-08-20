from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from .config import HISTORY_DIR, HISTORY_FILE

logger = logging.getLogger(__name__)


def _ensure_history_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def load_application_history() -> list[dict[str, Any]]:
    _ensure_history_dir()
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning("Could not read application history: %s", e)
        return []


def save_application_history(entry: dict[str, Any]) -> dict[str, Any]:
    _ensure_history_dir()
    records = load_application_history()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    records.append(record)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return record
