"""In-memory scrape progress for the job-search UI (poll while scrape runs)."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any


_lock = threading.Lock()
_PROGRESS: dict[str, dict[str, Any]] = {}
_TTL_SECONDS = 60 * 30


def create_progress(
    *, target: int, keyword: str = "", run_id: str | None = None
) -> str:
    """Register a new scrape run and return its id."""
    rid = (run_id or "").strip() or uuid.uuid4().hex
    now = time.time()
    with _lock:
        _PROGRESS[rid] = {
            "id": rid,
            "status": "starting",
            "phase": "starting",
            "message": "Starting job search…",
            "found": 0,
            "target": max(1, int(target)),
            "scanned": 0,
            "page": 0,
            "keyword": keyword or "",
            "percent": 0.0,
            "done": False,
            "error": None,
            "jobs": None,
            "errors": [],
            "relevance_min_pct": None,
            "display_complete": None,
            "updated_at": now,
            "created_at": now,
        }
        _prune_locked(now)
    return rid


def update_progress(run_id: str | None, **fields: Any) -> None:
    if not run_id:
        return
    now = time.time()
    with _lock:
        row = _PROGRESS.get(run_id)
        if not row:
            return
        row.update(fields)
        target = max(1, int(row.get("target") or 1))
        found = max(0, int(row.get("found") or 0))
        row["found"] = found
        row["target"] = target
        row["percent"] = round(min(100.0, (found / target) * 100.0), 1)
        if row.get("done"):
            if found >= target or row.get("status") == "done":
                row["percent"] = max(float(row["percent"]), 100.0) if not row.get("error") else float(row["percent"])
            if not row.get("error"):
                row["status"] = "done"
                row["percent"] = 100.0
            else:
                row["status"] = "error"
        row["updated_at"] = now


def get_progress(run_id: str) -> dict[str, Any] | None:
    with _lock:
        row = _PROGRESS.get(run_id)
        if not row:
            return None
        return dict(row)


def finish_progress(
    run_id: str | None,
    *,
    found: int | None = None,
    error: str | None = None,
    jobs: list[Any] | None = None,
    errors: list[str] | None = None,
    relevance_min_pct: float | None = None,
    display_complete: bool | None = None,
) -> None:
    if not run_id:
        return
    fields: dict[str, Any] = {
        "done": True,
        "status": "error" if error else "done",
        "phase": "done",
        "message": error or "Scraping complete",
        "error": error,
    }
    if found is not None:
        fields["found"] = found
    if jobs is not None:
        fields["jobs"] = jobs
        fields["found"] = found if found is not None else len(jobs)
    if errors is not None:
        fields["errors"] = errors
    if relevance_min_pct is not None:
        fields["relevance_min_pct"] = relevance_min_pct
    if display_complete is not None:
        fields["display_complete"] = display_complete
    update_progress(run_id, **fields)


def _prune_locked(now: float) -> None:
    stale = [
        key
        for key, row in _PROGRESS.items()
        if now - float(row.get("updated_at") or 0) > _TTL_SECONDS
    ]
    for key in stale:
        _PROGRESS.pop(key, None)
