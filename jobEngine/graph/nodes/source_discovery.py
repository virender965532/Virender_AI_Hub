"""Dynamic per-source node factory (sections 6/7/31).

``graph/builder.py`` calls ``create_source_node`` once per enabled,
configured source and registers the result under that source's
``graph_node`` name — the core graph never hard-codes a source name. Each
node call is independent: a circuit-open or exhausted-retries source never
raises, it only ever contributes a ``SourceError`` to the shared state, so
one broken website can never fail the whole run (section 53).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...config.settings import get_settings
from ...config.sources import JobSourceConfig
from ...guardrails.content_limits import enforce_description_size
from ...guardrails.url_guardrails import SSRFViolation, validate_navigation_url
from ...models.errors import SourceError, SourceErrorStatus
from ...persistence.rejected_jobs import rejected_descriptions
from ...persistence.saved_jobs import list_saved_jobs
from ...sources import circuit_breaker
from ...sources.base import SourceExecutionContext
from ...sources.registry import get_adapter_class
from ...sources.search_keywords import with_keywords
from ...sources.title_filter import title_has_keyword
from ..builder_types import NodeFn

logger = logging.getLogger(__name__)


def _known_descriptions(source_id: str) -> dict[str, str]:
    """Saved, LLM-analyzed jobs and rejected jobs of this source: their descriptions by
    source_job_id. The adapter doesn't open these again; matching reuses their analysis."""
    known: dict[str, str] = {}
    try:
        known.update(rejected_descriptions(source_id))
    except Exception:  # noqa: BLE001
        logger.exception("source=%s: could not read rejected jobs; reading them again", source_id)
    try:
        saved = list_saved_jobs()
    except Exception:  # noqa: BLE001
        logger.exception("source=%s: could not read saved jobs; reading every job", source_id)
        return known
    for entry in saved:
        job = entry.get("job") or {}
        if job.get("source") == source_id and job.get("source_job_id") and job.get("description") and entry.get("llm_analysis"):
            known[str(job["source_job_id"])] = job["description"]
    return known


async def _fetch_job_detail(browser_manager, cfg, application_url: str) -> str | None:
    """Best-effort fetch of a job's detail page text via its ``application_url``."""
    if not application_url or browser_manager is None:
        return None
    try:
        url = validate_navigation_url(application_url, allowed_domains=cfg.allowed_domains)
    except SSRFViolation as exc:
        logger.info("source=%s: skipping detail fetch for %s: %s", cfg.id, application_url, exc)
        return None

    try:
        async with browser_manager.open_page(
            source_id=cfg.id, allowed_domains=cfg.allowed_domains
        ) as managed:
            page = managed.page
            await page.goto(url, wait_until="domcontentloaded", timeout=cfg.timeout_seconds * 1000)
            try:
                await page.wait_for_load_state("networkidle", timeout=cfg.timeout_seconds * 1000)
            except Exception:  # noqa: BLE001
                pass
            text = await page.inner_text("body")
            return text.strip() if text else None
    except Exception as exc:  # noqa: BLE001
        logger.info("source=%s: detail fetch failed for %s: %s", cfg.id, url, exc)
        return None


def create_source_node(source_config: JobSourceConfig) -> NodeFn:
    async def _node(state: dict[str, Any]) -> dict[str, Any]:
        cfg = source_config
        run_id = state.get("run_id", "")
        settings = get_settings()
        request = state.get("request") or {}
        max_jobs = request.get("max_jobs")
        if isinstance(max_jobs, int) and max_jobs > 0:
            # ``max_jobs`` jobs from *each* search URL; the source total is the sum.
            cfg = cfg.model_copy(
                update={"max_jobs_per_url": max_jobs, "max_jobs": max_jobs * max(1, len(cfg.search_urls))}
            )
        search_keywords = request.get("search_keywords")
        if search_keywords:
            cfg = cfg.model_copy(
                update={"search_urls": [with_keywords(u, search_keywords) for u in cfg.search_urls]}
            )

        if circuit_breaker.is_open(cfg.id):
            logger.warning("source=%s: circuit breaker open; skipping this run", cfg.id)
            err = SourceError(
                source_id=cfg.id,
                status=SourceErrorStatus.CIRCUIT_OPEN,
                message="circuit breaker open due to repeated recent failures",
                retryable=True,
                node=cfg.graph_node,
            )
            return {
                "source_errors": [err.model_dump(mode="json")],
                "source_results": {
                    cfg.id: {"source_id": cfg.id, "status": "failed", "jobs": [], "pages_processed": 0, "duration_ms": 0, "errors": [err.model_dump(mode="json")]}
                },
            }

        browser_manager = state.get("browser_manager")
        adapter_cls = get_adapter_class(cfg.adapter)
        adapter = adapter_cls()
        context = SourceExecutionContext(
            run_id=run_id,
            config=cfg,
            browser_manager=browser_manager,
            settings=settings,
            known_descriptions=_known_descriptions(cfg.id),
        )

        result = None
        max_attempts = 1 + settings.browser_max_retries
        for attempt in range(max_attempts):
            result = await adapter.discover_jobs(context)
            if result.status != "failed":
                break
            all_retryable = all(e.retryable for e in result.errors) if result.errors else False
            if not all_retryable or attempt == max_attempts - 1:
                break
            backoff = settings.browser_retry_backoff_seconds * (attempt + 1)
            logger.info(
                "source=%s: attempt %s/%s failed (retryable), backing off %.1fs",
                cfg.id, attempt + 1, max_attempts, backoff,
            )
            await asyncio.sleep(backoff)

        assert result is not None
        if result.status == "failed":
            circuit_breaker.record_failure(cfg.id)
        else:
            circuit_breaker.record_success(cfg.id)


        # Every adapter: drop excluded titles before any detail fetch or LLM call
        # (LinkedIn already skipped them on the results page, before opening them).
        if cfg.title_exclude_keywords:
            kept = [job for job in result.jobs if not title_has_keyword(job.title, cfg.title_exclude_keywords)]
            if len(kept) != len(result.jobs):
                logger.info(
                    "source=%s: %s jobs dropped by title_exclude_keywords", cfg.id, len(result.jobs) - len(kept)
                )
                result.jobs = kept

        # Only fetch detail pages for jobs the adapter returned without a description
        # (e.g. link-scan fallback); adapters like LinkedIn already read it in-session.
        missing = [job for job in result.jobs if not job.description]
        detail_texts = await asyncio.gather(
            *[_fetch_job_detail(browser_manager, cfg, job.application_url) for job in missing]
        )
        for job, detail_text in zip(missing, detail_texts):
            if detail_text:
                job.description = enforce_description_size(detail_text)
        for i, job in enumerate(result.jobs, 1):
            logger.debug(
                "source=%s job #%s: %s | %s | %s", cfg.id, i, job.title, job.company, job.application_url
            )
        logger.info(
            "source=%s: status=%s jobs=%s pages=%s duration_ms=%s errors=%s",
            cfg.id, result.status, len(result.jobs), result.pages_processed, result.duration_ms, len(result.errors),
        )

        return {
            "discovered_jobs": [j.model_dump(mode="json") for j in result.jobs],
            "source_results": {cfg.id: result.model_dump(mode="json")},
            "source_errors": [e.model_dump(mode="json") for e in result.errors],
        }

    _node.__name__ = f"source_node_{source_config.id}"
    return _node
