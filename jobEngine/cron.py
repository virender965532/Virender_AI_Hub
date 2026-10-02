"""Scheduled job search + Pushover notification.

Run once per schedule tick (Windows Task Scheduler / cron, hourly):

    .venv\\Scripts\\python.exe -m jobEngine.cron

Runs the same search as the Remote Job Finder page with its default sources
and the keywords already in ``config/job_sources.yaml``. Results are saved to
the MongoDB saved-jobs collection like a manual search. A push notification lists the jobs
this run added (jobs already saved by an earlier run are not re-announced);
a failed run is pushed too, so a broken schedule doesn't go unnoticed.

Environment overrides (defaults in brackets):
    JOB_CRON_MIN_MATCH_SCORE [30]   minimum match score
    JOB_CRON_MAX_RESULTS     [-1]   jobs per search URL, -1 = all
    JOB_CRON_REMOTE_ONLY     [true] keep only remote/hybrid roles
    JOB_CRON_SOURCES         [page defaults] comma-separated source ids
    JOB_CRON_SAVED_JOBS_URL  [https://ai.ritishka.in/saved-jobs] link in the push
    JOB_CRON_ENABLED         [true] false = exit without searching
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

from .api.schemas import JobSearchRequest  # noqa: E402  (after load_dotenv: settings read env)
from .config.sources import default_search_source_ids, load_job_sources  # noqa: E402
from .notifications import send_pushover  # noqa: E402
from .persistence.saved_jobs import saved_job_key, saved_job_keys  # noqa: E402
from .services.job_engine_service import run_job_search  # noqa: E402

logger = logging.getLogger("jobEngine.cron")

def _build_request() -> dict[str, Any]:
    sources_env = os.getenv("JOB_CRON_SOURCES", "").strip()
    sources = [s.strip() for s in sources_env.split(",") if s.strip()] or default_search_source_ids(load_job_sources())
    parsed = JobSearchRequest(
        sources=sources,
        minimum_match_score=float(os.getenv("JOB_CRON_MIN_MATCH_SCORE", "30")),
        max_jobs=int(os.getenv("JOB_CRON_MAX_RESULTS", "-1")),
        max_results=500,
        remote_only=os.getenv("JOB_CRON_REMOTE_ONLY", "true").strip().lower() in ("1", "true", "yes", "on"),
    )
    return parsed.model_dump(mode="json")


_DEFAULT_SAVED_JOBS_URL = "https://ai.ritishka.in/saved-jobs"


def _saved_jobs_url() -> str:
    """Public link to the saved-jobs page, opened from the push notification."""
    return os.getenv("JOB_CRON_SAVED_JOBS_URL", "").strip() or _DEFAULT_SAVED_JOBS_URL


def _format_summary(summary: dict[str, Any]) -> str:
    return f"<i>{summary.get('jobs_discovered', 0)} scanned · {summary.get('jobs_recommended', 0)} matched</i>"


async def run_once() -> int:
    request = _build_request()
    try:
        known_keys = saved_job_keys()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not read saved jobs from MongoDB")
        send_pushover(f"Scheduled job search could not reach MongoDB: {exc}", title="Job search failed", priority=0)
        return 1

    try:
        result = await run_job_search(request)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scheduled job search crashed")
        send_pushover(f"Scheduled job search crashed: {exc}", title="Job search failed", priority=0)
        return 1

    summary = result.get("summary") or {}
    if result.get("status") == "failed":
        errors = "; ".join(e.get("message", "") for e in (result.get("source_errors") or [])[:3])
        send_pushover(errors or "The run failed; check the server logs.", title="Job search failed")
        return 1

    new_jobs = [j for j in result.get("jobs") or [] if saved_job_key(j) not in known_keys]  # None key = unsaved = new
    logger.info("Scheduled job search %s: %s new job(s)", result.get("run_id"), len(new_jobs))
    if not new_jobs:
        return 0

    send_pushover(
        _format_summary(summary),
        title=f"{len(new_jobs)} new remote job{'s' if len(new_jobs) != 1 else ''}",
        url=_saved_jobs_url(),
        url_title="Open saved jobs",
        html=True,
    )
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Same switch as the in-app schedule, so JOB_CRON_ENABLED=false in .env also
    # silences an OS-level task (Windows Task Scheduler / crontab) on that machine.
    if os.getenv("JOB_CRON_ENABLED", "true").strip().lower() not in ("1", "true", "yes", "on"):
        logger.info("Scheduled job search disabled (JOB_CRON_ENABLED); skipping")
        return 0
    return asyncio.run(run_once())


if __name__ == "__main__":
    sys.exit(main())
