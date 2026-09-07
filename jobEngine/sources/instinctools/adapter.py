"""Instinctools (https://www.instinctools.eu/vacancies/) — public, no-login vacancies page.

Two-tier extraction, mirroring the resilience pattern already used by this
repo's Naukri scraper (``jobSearch/nodes/fetch_jobs_node.py``: try a primary
selector set, fall back to a link-scan if it finds nothing):

1. **Structured tier** — common "job card" CSS patterns
   (``.vacancy-card``/``.vacancies-item``/``[class*="vacan"]`` style
   containers) with a title/location/department extracted from within each.
2. **Link-scan fallback** — every anchor whose href matches ``/vacancies/<slug>``
   is treated as one vacancy (title = link text). This tier does not depend
   on any specific CSS class and degrades gracefully if Instinctools changes
   their markup.

Known limitation (see docs/job_engine/SOURCE_ADAPTER_GUIDE.md): the exact
structured-tier selectors are best-effort — this environment could not fetch
the live page to confirm markup, so the link-scan fallback is the primary
safety net and should be treated as the more trustworthy tier until the
structured selectors are verified against the real site.
"""

from __future__ import annotations

import logging

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from ...guardrails.content_limits import enforce_description_size, enforce_job_count
from ...guardrails.url_guardrails import SSRFViolation, validate_navigation_url
from ...models.errors import SourceError, SourceErrorStatus
from ...models.job import RawJob
from ..base import JobSourceAdapter, SourceDiscoveryResult, SourceExecutionContext, TimedDiscovery
from ..dom_utils import extract_link_cards, href_or_none, text_or_none
from ..registry import register_adapter

logger = logging.getLogger(__name__)

_CARD_SELECTORS = [".vacancy-card", ".vacancies-item", "[class*='vacancy']", "[class*='vacan']"]
_TITLE_SELECTORS = ["h3", "h2", ".title", "[class*='title']"]
_LOCATION_SELECTORS = ["[class*='location']", "[class*='city']"]
_DEPARTMENT_SELECTORS = ["[class*='department']", "[class*='category']", "[class*='team']"]


@register_adapter("instinctools")
class InstinctoolsAdapter(JobSourceAdapter):
    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        cfg = context.config
        timer = TimedDiscovery(cfg.id)

        try:
            async with context.browser_manager.open_page(
                source_id=cfg.id, allowed_domains=cfg.allowed_domains
            ) as managed:
                page = managed.page
                for search_url in cfg.search_urls:
                    if len(timer.jobs) >= cfg.max_jobs:
                        break
                    url = validate_navigation_url(search_url, allowed_domains=cfg.allowed_domains)
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=cfg.timeout_seconds * 1000)
                        await page.wait_for_load_state("networkidle", timeout=cfg.timeout_seconds * 1000)
                    except PlaywrightTimeoutError:
                        logger.info(
                            "instinctools: networkidle timeout at %s; continuing with current DOM", url
                        )
                    except Exception as exc:  # noqa: BLE001
                        timer.errors.append(
                            SourceError(
                                source_id=cfg.id,
                                status=SourceErrorStatus.NAVIGATION_ERROR,
                                message=f"navigation failed for {url}: {exc}",
                                retryable=True,
                            )
                        )
                        continue

                    timer.pages_processed += 1

                    found = await self._extract_structured(page, cfg=cfg)
                    if not found:
                        found = await self._extract_via_link_scan(page, cfg=cfg, listing_url=url)

                    if not found:
                        timer.errors.append(
                            SourceError(
                                source_id=cfg.id,
                                status=SourceErrorStatus.SELECTOR_MISSING,
                                message=(
                                    "no vacancies found via structured selectors or link-scan "
                                    f"fallback at {url}; markup may have changed"
                                ),
                                retryable=False,
                            )
                        )
                        continue

                    timer.jobs.extend(found)

        except SSRFViolation as exc:
            timer.errors.append(
                SourceError(source_id=cfg.id, status=SourceErrorStatus.NAVIGATION_ERROR, message=str(exc))
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("InstinctoolsAdapter failed for source=%s", cfg.id)
            timer.errors.append(
                SourceError(source_id=cfg.id, status=SourceErrorStatus.UNKNOWN, message=str(exc), retryable=True)
            )

        timer.jobs = enforce_job_count(timer.jobs, max_jobs=cfg.max_jobs, source=cfg.id)
        return timer.to_result()

    async def _extract_structured(self, page, *, cfg) -> list[RawJob]:
        for card_selector in _CARD_SELECTORS:
            try:
                cards = await page.query_selector_all(card_selector)
            except Exception:  # noqa: BLE001
                continue
            if not cards:
                continue

            jobs: list[RawJob] = []
            for card in cards[: cfg.max_jobs * 2]:
                title = None
                for sel in _TITLE_SELECTORS:
                    title = await text_or_none(card, sel)
                    if title:
                        break
                if not title:
                    continue

                location = None
                for sel in _LOCATION_SELECTORS:
                    location = await text_or_none(card, sel)
                    if location:
                        break

                department = None
                for sel in _DEPARTMENT_SELECTORS:
                    department = await text_or_none(card, sel)
                    if department:
                        break

                job_url = await href_or_none(card, "a", base_url=cfg.base_url)
                jobs.append(
                    RawJob(
                        source=cfg.id,
                        title=title,
                        company="Instinctools",
                        location=location,
                        employment_type=department,
                        description=enforce_description_size(department or ""),
                        application_url=job_url or cfg.base_url,
                        source_url=cfg.base_url,
                    )
                )
                if len(jobs) >= cfg.max_jobs:
                    break

            if jobs:
                return jobs
        return []

    async def _extract_via_link_scan(self, page, *, cfg, listing_url: str) -> list[RawJob]:
        candidates = await extract_link_cards(
            page,
            href_contains=("/vacancies/", "/vacancy/"),
            base_url=cfg.base_url,
            exclude_urls=tuple(cfg.search_urls),
            max_items=cfg.max_jobs,
        )
        return [
            RawJob(
                source=cfg.id,
                title=c["title"],
                company="Instinctools",
                description="",
                application_url=c["url"],
                source_url=listing_url,
            )
            for c in candidates
        ]
