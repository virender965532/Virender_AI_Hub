"""Config-driven adapter for plain career pages (section 33).

Handles the common case — a static or lightly-dynamic list of job cards with
CSS selectors declared in ``config/job_sources.yaml``'s ``selectors:`` block.
Not every site fits this shape (infinite scroll, SPA APIs, auth) — those get
a dedicated adapter (see ``linkedin/adapter.py``, ``instinctools/adapter.py``).
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urljoin

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from ...guardrails.content_limits import (
    enforce_description_size,
    enforce_job_count,
    pagination_depth_exceeded,
)
from ...guardrails.url_guardrails import SSRFViolation, validate_navigation_url
from ...models.errors import SourceError, SourceErrorStatus
from ...models.job import RawJob
from ..base import JobSourceAdapter, SourceDiscoveryResult, SourceExecutionContext, TimedDiscovery
from ..dom_utils import href_or_none as _href_or_none
from ..dom_utils import text_or_none as _text_or_none
from ..registry import register_adapter

logger = logging.getLogger(__name__)

_REQUIRED_SELECTORS = ("job_card", "title")


async def extract_jobs_from_current_page(page: Page, cfg, *, page_url: str, limit: int | None = None) -> list[RawJob]:
    """Extract job cards from whatever is currently loaded in ``page`` using
    ``cfg.selectors``. Deliberately takes no part in navigation, so it is
    directly unit-testable via ``page.set_content(fixture_html)`` — see
    ``tests/job_engine/test_generic_adapter.py``.
    """
    selectors = cfg.selectors
    cards = await page.query_selector_all(selectors["job_card"])
    jobs: list[RawJob] = []
    for card in cards:
        if limit is not None and len(jobs) >= limit:
            break
        title = await _text_or_none(card, selectors.get("title", ""))
        if not title:
            continue
        company = await _text_or_none(card, selectors.get("company", ""))
        location = await _text_or_none(card, selectors.get("location", ""))
        description = await _text_or_none(card, selectors.get("description", "")) or ""
        job_url = await _href_or_none(card, selectors.get("url", "a"), base_url=cfg.base_url)

        jobs.append(
            RawJob(
                source=cfg.id,
                title=title,
                company=company or "",
                location=location,
                description=enforce_description_size(description),
                application_url=job_url or page_url,
                source_url=page_url,
            )
        )
    return jobs


@register_adapter("generic")
class GenericAdapter(JobSourceAdapter):
    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        cfg = context.config
        timer = TimedDiscovery(cfg.id)
        selectors = cfg.selectors
        missing = [s for s in _REQUIRED_SELECTORS if not selectors.get(s)]
        if missing:
            timer.errors.append(
                SourceError(
                    source_id=cfg.id,
                    status=SourceErrorStatus.SELECTOR_MISSING,
                    message=f"selectors missing required keys: {missing}",
                    retryable=False,
                )
            )
            return timer.to_result()

        try:
            async with context.browser_manager.open_page(
                source_id=cfg.id, allowed_domains=cfg.allowed_domains
            ) as managed:
                for search_url in cfg.search_urls:
                    await self._discover_from_url(managed.page, cfg=cfg, url=search_url, timer=timer)
                    if len(timer.jobs) >= cfg.max_jobs:
                        break
        except SSRFViolation as exc:
            timer.errors.append(
                SourceError(source_id=cfg.id, status=SourceErrorStatus.NAVIGATION_ERROR, message=str(exc))
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("GenericAdapter failed for source=%s", cfg.id)
            timer.errors.append(
                SourceError(source_id=cfg.id, status=SourceErrorStatus.UNKNOWN, message=str(exc), retryable=True)
            )

        timer.jobs = enforce_job_count(timer.jobs, max_jobs=cfg.max_jobs, source=cfg.id)
        return timer.to_result()

    async def _discover_from_url(
        self, page: Page, *, cfg, url: str, timer: TimedDiscovery
    ) -> None:
        selectors = cfg.selectors
        start_count = len(timer.jobs)
        for page_num in range(1, cfg.max_pages + 1):
            if pagination_depth_exceeded(page_num, max_pages=cfg.max_pages):
                break
            if len(timer.jobs) >= cfg.max_jobs:
                return
            if cfg.max_jobs_per_url and len(timer.jobs) - start_count >= cfg.max_jobs_per_url:
                return

            page_url = validate_navigation_url(url, allowed_domains=cfg.allowed_domains)
            try:
                await page.goto(page_url, wait_until="domcontentloaded", timeout=cfg.timeout_seconds * 1000)
                await page.wait_for_selector(selectors["job_card"], timeout=cfg.timeout_seconds * 1000)
            except PlaywrightTimeoutError:
                timer.errors.append(
                    SourceError(
                        source_id=cfg.id,
                        status=SourceErrorStatus.TIMEOUT,
                        message=f"timed out waiting for job_card selector at {page_url}",
                        retryable=True,
                    )
                )
                return
            except Exception as exc:  # noqa: BLE001
                timer.errors.append(
                    SourceError(
                        source_id=cfg.id,
                        status=SourceErrorStatus.NAVIGATION_ERROR,
                        message=f"navigation failed for {page_url}: {exc}",
                        retryable=True,
                    )
                )
                return

            timer.pages_processed += 1
            remaining = cfg.max_jobs - len(timer.jobs)
            if cfg.max_jobs_per_url:
                remaining = min(remaining, cfg.max_jobs_per_url - (len(timer.jobs) - start_count))
            page_jobs = await extract_jobs_from_current_page(page, cfg, page_url=page_url, limit=remaining)
            if not page_jobs:
                return
            timer.jobs.extend(page_jobs)

            next_selector = selectors.get("next_page")
            if not next_selector:
                return
            next_link = await page.query_selector(next_selector)
            if next_link is None:
                return
            next_href = await next_link.get_attribute("href")
            if not next_href:
                return
            url = urljoin(page_url, next_href)
            await asyncio.sleep(cfg.rate_limit_delay_seconds)
