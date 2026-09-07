"""Flat-file persistence for job search runs (section 24/60).

Follows the existing ``jobApplyPro/history.py`` convention (JSON files under
a project-level directory) rather than introducing a database dependency
this project doesn't otherwise have — see docs/job_engine/PROJECT_ANALYSIS.md
"Recommended integration strategy". Every function here is written behind a
narrow interface so a real database could be swapped in later (e.g. behind
the same ``save_run``/``get_run``/``list_runs`` signatures) without touching
callers.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.settings import JOB_ENGINE_HISTORY_DIR, JOB_ENGINE_OUTPUT_DIR

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_RUNS_INDEX_FILE = JOB_ENGINE_HISTORY_DIR / "runs_index.json"


def _ensure_dirs() -> None:
    JOB_ENGINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    JOB_ENGINE_HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _run_file(run_id: str) -> Path:
    return JOB_ENGINE_OUTPUT_DIR / f"{run_id}.json"


def _load_index() -> list[dict[str, Any]]:
    if not _RUNS_INDEX_FILE.exists():
        return []
    try:
        with _RUNS_INDEX_FILE.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        logger.warning("Could not read job-engine runs index; treating as empty", exc_info=True)
        return []


def save_run(result: dict[str, Any]) -> None:
    """Persist a full ``JobSearchResult`` dict and append a lightweight index entry."""
    _ensure_dirs()
    run_id = result.get("run_id", "")
    if not run_id:
        raise ValueError("result must include run_id")

    with _lock:
        run_path = _run_file(run_id)
        with run_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False, default=str)

        index = _load_index()
        summary = result.get("summary") or {}
        index.append(
            {
                "run_id": run_id,
                "status": result.get("status"),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "jobs_recommended": summary.get("jobs_recommended", 0),
                "sources_processed": summary.get("sources_processed", 0),
            }
        )
        with _RUNS_INDEX_FILE.open("w", encoding="utf-8") as fh:
            json.dump(index[-500:], fh, indent=2, ensure_ascii=False)

    logger.info("Persisted job-engine run %s (status=%s)", run_id, result.get("status"))


def get_run(run_id: str) -> dict[str, Any] | None:
    path = _run_file(run_id)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001
        logger.warning("Could not read persisted run %s", run_id, exc_info=True)
        return None


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    index = _load_index()
    return list(reversed(index))[: max(1, limit)]
