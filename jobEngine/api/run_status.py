"""In-memory run-status registry so ``GET /api/jobs/search/<run_id>`` can
distinguish "still running" from "not found", mirroring the existing
``jobSearch/utils/scrape_progress.py`` in-memory-dict convention. Process
-local by design (documented limitation for multi-process deployments).
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_STATUS: dict[str, str] = {}


def set_status(run_id: str, status: str) -> None:
    with _lock:
        _STATUS[run_id] = status


def get_status(run_id: str) -> str | None:
    with _lock:
        return _STATUS.get(run_id)
