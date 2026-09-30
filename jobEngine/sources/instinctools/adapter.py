"""Instinctools (https://www.instinctools.eu/vacancies/) — public, no-login vacancies page.

Two-tier extraction, mirroring the resilience pattern already used by this
repo's Naukri scraper (``jobSearch/nodes/fetch_jobs_node.py``: try a primary
selector set, fall back to a link-scan if it finds nothing):

1. **Structured tier** — ``.career-item`` job cards (confirmed against the
   live vacancies page), each with a title/location/description and a
   ``.career-item__expertises`` tag list (e.g. "#Office", "#Remote", "#AI",
   "#Python", ...). "#Office"/"#Remote" describe work arrangement rather than
   a skill and are split out into ``remote_type`` (remote wins when both are
   present); every other tag becomes a skill. The generic
   ``.vacancy-card``/``.vacancies-item``/``[class*="vacan"]`` patterns remain
   as fallback selectors in case Instinctools changes their markup.
2. **Link-scan fallback** — every anchor whose href matches ``/vacancies/<slug>``
   is treated as one vacancy (title = link text). Used only when the
   structured tier finds nothing (no location/description/tags in that case).
"""

from __future__ import annotations

import logging

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from ...guardrails.content_limits import enforce_description_size, enforce_job_count
from ...guardrails.url_guardrails import SSRFViolation, validate_navigation_url
from ...models.errors import SourceError, SourceErrorStatus
from ...models.job import RawJob
from ..base import JobSourceAdapter, SourceDiscoveryResult, SourceExecutionContext, TimedDiscovery
from ..dom_utils import extract_link_cards, href_or_none, text_or_none, texts_for_all
from ..registry import register_adapter

logger = logging.getLogger(__name__)

# ".career-item" etc. is the confirmed real markup (verified against the live
# vacancies page); the generic "[class*='vacan']" patterns are kept after it
# as a fallback in case Instinctools changes their markup.
_CARD_SELECTORS = [".career-item", ".vacancy-card", ".vacancies-item", "[class*='vacancy']", "[class*='vacan']"]
_TITLE_SELECTORS = [".career-item__title", "h3", "h2", ".title", "[class*='title']"]
_LOCATION_SELECTORS = [".career-item__locations", "[class*='location']", "[class*='city']"]
_DEPARTMENT_SELECTORS = ["[class*='department']", "[class*='category']", "[class*='team']"]
_DESCRIPTION_SELECTORS = [".career-item__text", "[class*='description']"]
_TAGS_CONTAINER_SELECTORS = [".career-item__expertises", "[class*='expertise']", "[class*='tag']"]

# Card tag values that describe work arrangement rather than a skill (see
# ".career-item__expertises" on the live page: "#Office", "#Remote", ...).
_REMOTE_TAG = "remote"
_OFFICE_TAG = "office"
_WORK_MODE_TAGS = {_REMOTE_TAG, _OFFICE_TAG}


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

                    if cfg.max_jobs_per_url:
                        found = found[: cfg.max_jobs_per_url]
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

                description = None
                for sel in _DESCRIPTION_SELECTORS:
                    description = await text_or_none(card, sel)
                    if description:
                        break

                tags: list[str] = []
                for sel in _TAGS_CONTAINER_SELECTORS:
                    tags = await texts_for_all(card, f"{sel} span")
                    if tags:
                        break
                remote_type, skills = self._split_tags(tags)

                job_url = await href_or_none(card, "a", base_url=cfg.base_url)
                jobs.append(
                    RawJob(
                        source=cfg.id,
                        title=title,
                        company="Instinctools",
                        location=location,
                        employment_type=department,
                        description=enforce_description_size(description or department or ""),
                        skills=skills,
                        remote_type=remote_type,
                        application_url=job_url or cfg.base_url,
                        source_url=cfg.base_url,
                    )
                )
                if len(jobs) >= cfg.max_jobs:
                    break

            if jobs:
                return jobs
        return []

    @staticmethod
    def _split_tags(tags: list[str]) -> tuple[str | None, list[str]]:
        """Split a card's "#Office #Remote #AI #AWS ..." tags into a
        (remote_type, skills) pair.

        "#Office"/"#Remote" describe work arrangement rather than a skill;
        when both are present on the same listing, remote wins (the role is
        at least partly remote-friendly, which is what a remote-seeking
        candidate cares about).
        """
        cleaned = [t.lstrip("#").strip() for t in tags]
        cleaned = [t for t in cleaned if t]

        work_modes = {t.lower() for t in cleaned if t.lower() in _WORK_MODE_TAGS}
        if _REMOTE_TAG in work_modes:
            remote_type = "remote"
        elif _OFFICE_TAG in work_modes:
            remote_type = "onsite"
        else:
            remote_type = None

        skills = [t for t in cleaned if t.lower() not in _WORK_MODE_TAGS]
        return remote_type, skills

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
