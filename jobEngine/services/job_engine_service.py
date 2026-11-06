"""Top-level orchestration entrypoint: one call runs the whole workflow.

The Flask API layer validates the request (``api/schemas.py``) and passes a
plain dict here — this module has no dependency on Flask or the API layer,
so it can also be driven from a script/test/future background worker
(section 37 — this is the seam a Celery/RQ task would call into).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..browser.manager import BrowserManager
from ..concurrency import warm_cpu_pool
from ..config.settings import get_settings
from ..config.sources import enabled_sources, load_job_sources
from ..graph.builder import build_graph
from ..graph.state import initial_state
from ..matching.fit_score import profile_fingerprint
from ..models.profile import CandidateProfile
from ..notifications.job_alerts import mail_high_score_jobs
from ..observability.logging import log_event, new_run_id
from ..persistence.rejected_jobs import save_rejected_jobs
from ..persistence.repository import save_run
from ..persistence.saved_jobs import save_jobs_from_run

logger = logging.getLogger(__name__)


def _resolve_sources(request: dict[str, Any]):
    settings = get_settings()
    sources = enabled_sources(load_job_sources())
    requested = request.get("sources")
    if requested:
        wanted = set(requested)
        sources = [s for s in sources if s.id in wanted]
    return sources[: settings.max_sources_per_run]


async def run_job_search(request: dict[str, Any], *, run_id: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    run_id = run_id or new_run_id()
    t0 = time.perf_counter()

    log_event(logger, logging.INFO, "job_search_started", run_id=run_id, extra_request=request)

    # Spawn the CPU workers now so their startup overlaps scraping, not normalization.
    warm_cpu_pool()

    resolved_sources = _resolve_sources(request)
    graph = build_graph(resolved_sources)

    state = initial_state(run_id=run_id, request=request)
    browser_manager = BrowserManager(settings)
    state["browser_manager"] = browser_manager

    try:
        final_state = await graph.ainvoke(state)
    finally:
        await browser_manager.close()

    result = final_state.get("final_result") or {
        "run_id": run_id,
        "status": "failed",
        "summary": {
            "sources_processed": 0,
            "jobs_discovered": 0,
            "jobs_validated": 0,
            "duplicates_removed": 0,
            "jobs_matched": 0,
            "jobs_recommended": 0,
        },
        "jobs": [],
        "source_errors": final_state.get("source_errors") or [],
        "metrics": None,
    }

    duration_ms = round((time.perf_counter() - t0) * 1000, 1)
    if result.get("metrics") is not None:
        result["metrics"]["workflow_duration_ms"] = duration_ms

    log_event(
        logger,
        logging.INFO,
        "job_search_completed",
        run_id=run_id,
        status=result.get("status"),
        duration_ms=duration_ms,
        jobs_recommended=result.get("summary", {}).get("jobs_recommended", 0),
    )

    if settings.persist_results and not settings.dry_run:
        # Three independent collections: write them concurrently in worker threads.
        await asyncio.gather(
            asyncio.to_thread(_persist_run, result, run_id),
            asyncio.to_thread(_persist_saved_jobs, result, run_id),
            asyncio.to_thread(_persist_rejected_jobs, final_state, run_id),
        )

    return result


def _persist_run(result: dict[str, Any], run_id: str) -> None:
    try:
        save_run(result)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist run %s (continuing; result still returned)", run_id)


def _persist_saved_jobs(result: dict[str, Any], run_id: str) -> None:
    try:
        save_jobs_from_run(result)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to update saved jobs store for run %s", run_id)
        return
    try:
        # After the save: the "already mailed" mark lives on the saved-job document.
        mail_high_score_jobs(result)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to email high-score jobs for run %s", run_id)


def _persist_rejected_jobs(final_state: dict[str, Any], run_id: str) -> None:
    try:
        # The result only carries recommended jobs; the rejected ones are in the graph state.
        profile = CandidateProfile.model_validate(final_state.get("candidate_profile") or {})
        save_rejected_jobs(
            final_state.get("matched_jobs") or [], run_id=run_id, profile_hash=profile_fingerprint(profile)
        )
    except Exception:  # noqa: BLE001
        logger.exception("Failed to update rejected jobs store for run %s", run_id)


async def test_source(source_id: str) -> dict[str, Any]:
    """Lightweight connectivity check for ``POST /api/jobs/sources/test``:
    runs the adapter directly (bypassing the matching pipeline) with a small
    job cap so it returns quickly.
    """
    from ..sources.registry import get_adapter_class
    from ..sources.base import SourceExecutionContext

    settings = get_settings()
    all_sources = load_job_sources()
    cfg = next((s for s in all_sources if s.id == source_id), None)
    if cfg is None:
        return {"ok": False, "error": f"unknown source_id {source_id!r}"}

    test_cfg = cfg.model_copy(update={"max_jobs": min(3, cfg.max_jobs), "max_pages": 1})
    adapter = get_adapter_class(test_cfg.adapter)()
    browser_manager = BrowserManager(settings)
    try:
        context = SourceExecutionContext(
            run_id="source-test", config=test_cfg, browser_manager=browser_manager, settings=settings
        )
        result = await adapter.discover_jobs(context)
    finally:
        await browser_manager.close()

    return {"ok": result.status != "failed", "result": result.model_dump(mode="json")}
