"""LinkedIn — treated as a special, auth-gated source (section 10).

No password automation, no CAPTCHA/anti-bot bypass is implemented here or
anywhere in this engine. The only supported authentication mechanism is a
Playwright ``storage_state`` JSON file the candidate generates *out of band*
(e.g. ``playwright codegen`` after logging in manually, or a one-off script
that logs in interactively and calls ``context.storage_state(path=...)``),
pointed to by the ``LINKEDIN_STORAGE_STATE_PATH`` env var. If that file is
absent, or LinkedIn redirects to a login/checkpoint page anyway, this
adapter reports a structured ``SourceError`` and the run continues with
other sources — it never falls back to attempting the login form itself.
"""

from __future__ import annotations

import logging
import os

from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from ...guardrails.content_limits import enforce_description_size, enforce_job_count
from ...guardrails.url_guardrails import SSRFViolation, validate_navigation_url
from ...models.errors import SourceError, SourceErrorStatus
from ...models.job import RawJob
from ..base import JobSourceAdapter, SourceDiscoveryResult, SourceExecutionContext, TimedDiscovery
from ..dom_utils import extract_link_cards, href_or_none, text_or_none
from ..registry import register_adapter

logger = logging.getLogger(__name__)

_CARD_SELECTORS = [
    "ul.jobs-search__results-list li",
    "div.jobs-search-results-list li.jobs-search-results__list-item",
    "li[data-occludable-job-id]",
]
_TITLE_SELECTORS = ["h3.base-search-card__title", "a.job-card-list__title", ".job-card-container__link"]
_COMPANY_SELECTORS = ["h4.base-search-card__subtitle", ".job-card-container__company-name"]
_LOCATION_SELECTORS = ["span.job-search-card__location", ".job-card-container__metadata-item"]

_BLOCKED_URL_MARKERS = ("/checkpoint/", "/authwall", "/login")


def _storage_state_path() -> str | None:
    path = (os.getenv("LINKEDIN_STORAGE_STATE_PATH") or "").strip()
    return path or None


@register_adapter("linkedin")
class LinkedInAdapter(JobSourceAdapter):
    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        cfg = context.config
        timer = TimedDiscovery(cfg.id)

        storage_state = _storage_state_path()
        if cfg.requires_auth and not storage_state:
            timer.errors.append(
                SourceError(
                    source_id=cfg.id,
                    status=SourceErrorStatus.AUTH_REQUIRED,
                    message=(
                        "LINKEDIN_STORAGE_STATE_PATH is not set — no authenticated session "
                        "available. Generate one by logging in manually and calling "
                        "context.storage_state(path=...); this engine does not automate "
                        "LinkedIn login."
                    ),
                    retryable=False,
                )
            )
            return timer.to_result()

        try:
            async with context.browser_manager.open_page(
                source_id=cfg.id,
                allowed_domains=cfg.allowed_domains,
                storage_state_path=storage_state,
            ) as managed:
                page = managed.page
                for search_url in cfg.search_urls:
                    if len(timer.jobs) >= cfg.max_jobs:
                        break
                    url = validate_navigation_url(search_url, allowed_domains=cfg.allowed_domains)
                    try:
                        await page.goto(url, wait_until="domcontentloaded", timeout=cfg.timeout_seconds * 1000)
                    except PlaywrightTimeoutError:
                        timer.errors.append(
                            SourceError(
                                source_id=cfg.id,
                                status=SourceErrorStatus.TIMEOUT,
                                message=f"timed out navigating to {url}",
                                retryable=True,
                            )
                        )
                        continue
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

                    current_url = page.url.lower()
                    if any(marker in current_url for marker in _BLOCKED_URL_MARKERS):
                        timer.errors.append(
                            SourceError(
                                source_id=cfg.id,
                                status=SourceErrorStatus.ANTI_BOT_BLOCKED
                                if "checkpoint" in current_url or "authwall" in current_url
                                else SourceErrorStatus.AUTH_REQUIRED,
                                message=f"LinkedIn redirected to {current_url}; session is not usable",
                                retryable=False,
                            )
                        )
                        continue

                    timer.pages_processed += 1
                    found = await self._extract_structured(page, cfg=cfg)
                    if not found:
                        found = await extract_link_cards(
                            page,
                            href_contains=("/jobs/view/",),
                            base_url=cfg.base_url,
                            max_items=cfg.max_jobs,
                        )
                        found = [
                            RawJob(
                                source=cfg.id,
                                title=c["title"],
                                company="",
                                description="",
                                application_url=c["url"],
                                source_url=url,
                            )
                            for c in found
                        ]

                    if not found:
                        timer.errors.append(
                            SourceError(
                                source_id=cfg.id,
                                status=SourceErrorStatus.SELECTOR_MISSING,
                                message=f"no job cards found at {url}; LinkedIn markup may have changed",
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
            logger.exception("LinkedInAdapter failed for source=%s", cfg.id)
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

                company = None
                for sel in _COMPANY_SELECTORS:
                    company = await text_or_none(card, sel)
                    if company:
                        break

                location = None
                for sel in _LOCATION_SELECTORS:
                    location = await text_or_none(card, sel)
                    if location:
                        break

                job_url = await href_or_none(card, "a", base_url=cfg.base_url)
                jobs.append(
                    RawJob(
                        source=cfg.id,
                        title=title,
                        company=company or "",
                        location=location,
                        description=enforce_description_size(""),
                        application_url=job_url or cfg.base_url,
                        source_url=cfg.base_url,
                    )
                )
                if len(jobs) >= cfg.max_jobs:
                    break

            if jobs:
                return jobs
        return []
