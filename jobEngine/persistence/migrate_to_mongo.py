"""One-time import of the pre-MongoDB flat files into MongoDB.

    .venv\\Scripts\\python.exe -m jobEngine.persistence.migrate_to_mongo

Imports, if present:
    output/job_engine/job-search-*.json        -> job_engine_runs
    history/job_engine/saved_jobs.json         -> saved_jobs (status/applied_at kept)
    history/job_engine/linkedin_storage_state.json -> browser_sessions ("linkedin")

Safe to re-run: runs and sessions are upserted by id, and a saved job already in
MongoDB is left as is. The files are not deleted.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from ..config.settings import JOB_ENGINE_HISTORY_DIR, JOB_ENGINE_OUTPUT_DIR  # noqa: E402
from .mongo import RUNS_COLLECTION, SAVED_JOBS_COLLECTION, get_collection, to_bson_safe  # noqa: E402
from .saved_jobs import job_age_reference  # noqa: E402
from .sessions import load_storage_state  # noqa: E402

logger = logging.getLogger("jobEngine.migrate_to_mongo")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        logger.warning("Skipping unreadable file %s", path)
        return None


def migrate_runs(output_dir: Path = JOB_ENGINE_OUTPUT_DIR) -> int:
    runs = get_collection(RUNS_COLLECTION)
    count = 0
    for path in sorted(output_dir.glob("job-search-*.json")) if output_dir.exists() else []:
        run = _read_json(path)
        if not isinstance(run, dict):
            continue
        run_id = run.get("run_id") or path.stem
        saved_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        runs.replace_one(
            {"_id": run_id}, {**to_bson_safe(run), "run_id": run_id, "_id": run_id, "saved_at": saved_at}, upsert=True
        )
        count += 1
    return count


def migrate_saved_jobs(path: Path = JOB_ENGINE_HISTORY_DIR / "saved_jobs.json") -> int:
    store = _read_json(path) if path.exists() else None
    if not isinstance(store, dict):
        return 0
    coll = get_collection(SAVED_JOBS_COLLECTION)
    count = 0
    for key, saved in store.items():
        if not isinstance(saved, dict):
            continue
        doc = {**to_bson_safe(saved), "_id": key, "saved_key": key}
        doc["age_ref"] = job_age_reference(doc)
        if coll.update_one({"_id": key}, {"$setOnInsert": doc}, upsert=True).upserted_id is not None:
            count += 1
    return count


def migrate_sessions(path: Path = JOB_ENGINE_HISTORY_DIR / "linkedin_storage_state.json") -> bool:
    raw = (os.getenv("LINKEDIN_STORAGE_STATE_PATH") or "").strip()
    return load_storage_state("linkedin", legacy_path=Path(raw) if raw else path) is not None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger.info("Runs imported: %s", migrate_runs())
    logger.info("Saved jobs imported (new): %s", migrate_saved_jobs())
    logger.info("LinkedIn session in MongoDB: %s", migrate_sessions())
    return 0


if __name__ == "__main__":
    sys.exit(main())
