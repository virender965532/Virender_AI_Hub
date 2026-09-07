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
from ...guardrails.url_guardrails import SSRFViolation, validate_navigation_url
from ...models.errors import SourceError, SourceErrorStatus
from ...sources import circuit_breaker
from ...sources.base import SourceExecutionContext
from ...sources.registry import get_adapter_class
from ..builder_types import NodeFn

logger = logging.getLogger(__name__)


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
            run_id=run_id, config=cfg, browser_manager=browser_manager, settings=settings
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


        detail_texts = await asyncio.gather(
            *[_fetch_job_detail(browser_manager, cfg, job.application_url) for job in result.jobs]
        )
        for i, (job, detail_text) in enumerate(zip(result.jobs, detail_texts), 1):
            print(f"\nJob #{i}")
            print(f"  Title: {job.title}")
            print(f"  Company: {job.company}")
            print(f"  URL: {job.application_url}")
            if detail_text:
                print(f"  Detail: {detail_text}")
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
