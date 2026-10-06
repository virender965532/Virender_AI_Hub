"""Shared LinkedIn plumbing: session handling, sign-in and the pagination loop.

Session handling:
  1. If a saved Playwright ``storage_state`` (cookies) exists in the MongoDB
     ``browser_sessions`` collection it is loaded and the search page is opened
     directly. A session file left by the pre-MongoDB version
     (``LINKEDIN_STORAGE_STATE_PATH``, default
     ``history/job_engine/linkedin_storage_state.json``) is imported once.
  2. If there is no session, or LinkedIn bounces the saved one to a login page,
     the adapter signs in via ``signin_url`` using the candidate's own
     ``LINKEDIN_EMAIL`` / ``LINKEDIN_PASSWORD`` and saves the new session to
     MongoDB so the next run skips the login step.
  3. Or sign in by hand: "Connect LinkedIn" on the Remote Job Finder page
     (``remote_login.py``, works on the headless server) or, on a machine with a
     display, ``python -m jobEngine.sources.linkedin.login``. Both save the
     session to MongoDB when the sign-in completes.

No CAPTCHA/anti-bot bypass is attempted. If LinkedIn raises a security
checkpoint after login, the adapter waits (``LINKEDIN_CHECKPOINT_WAIT_SECONDS``)
for the candidate to resolve it by hand — only useful with
``JOB_BROWSER_HEADLESS=false`` — and otherwise reports a structured
``SourceError`` so the run continues with other sources.

Card extraction is page-specific and lives in the subclasses:
``search_results_adapter.py`` (``/jobs/search-results/``) and
``search_adapter.py`` (``/jobs/search/``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from abc import abstractmethod
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from ...config.settings import JOB_ENGINE_HISTORY_DIR
from ...guardrails.content_limits import enforce_job_count, pagination_depth_exceeded
from ...guardrails.url_guardrails import SSRFViolation, validate_navigation_url
from ...models.errors import SourceError, SourceErrorStatus
from ...models.job import RawJob
from ...persistence.sessions import load_storage_state, mark_signed_out, save_storage_state
from ..base import JobSourceAdapter, SourceDiscoveryResult, SourceExecutionContext, TimedDiscovery
from ..dom_utils import extract_link_cards, text_or_none

logger = logging.getLogger(__name__)

_DEFAULT_SIGNIN_URL = "https://www.linkedin.com/login/"
_DEFAULT_STORAGE_STATE_PATH = JOB_ENGINE_HISTORY_DIR / "linkedin_storage_state.json"
_PAGE_SIZE = 25

_POSTED_TEXT_RE = re.compile(
    r"(\d+|an?)\s+(minute|min|hour|hr|day|week|month|year)s?\s+ago|just now|moments? ago",
    re.IGNORECASE,
)
_LOGIN_URL_MARKERS = ("/login", "/uas/login", "/authwall", "/signup")
_CHECKPOINT_URL_MARKERS = ("/checkpoint/", "/challenge")


def _legacy_storage_state_path() -> Path:
    raw = (os.getenv("LINKEDIN_STORAGE_STATE_PATH") or "").strip()
    return Path(raw) if raw else _DEFAULT_STORAGE_STATE_PATH


def _credentials() -> tuple[str, str] | None:
    email = (os.getenv("LINKEDIN_EMAIL") or "").strip()
    password = os.getenv("LINKEDIN_PASSWORD") or ""
    return (email, password) if email and password else None


def _checkpoint_wait_seconds() -> int:
    try:
        return max(0, int((os.getenv("LINKEDIN_CHECKPOINT_WAIT_SECONDS") or "120").strip()))
    except ValueError:
        return 120


def _is_login_url(url: str) -> bool:
    return any(marker in url.lower() for marker in _LOGIN_URL_MARKERS)


def _is_checkpoint_url(url: str) -> bool:
    return any(marker in url.lower() for marker in _CHECKPOINT_URL_MARKERS)


def _page_url(search_url: str, page_index: int) -> str:
    """LinkedIn paginates with ``start=<offset>`` (0, 25, 50, ...) on both the classic
    ``/jobs/search/`` and the new ``/jobs/search-results/`` page; ``currentJobId`` only
    pins the right-hand detail pane to one job, so it is dropped after the first page."""
    parsed = urlparse(search_url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    query["start"] = [str(page_index * _PAGE_SIZE)]
    if page_index > 0:
        query.pop("currentJobId", None)
    return urlunparse(parsed._replace(query=urlencode(query, doseq=True)))


class _AuthError(Exception):
    def __init__(self, status: SourceErrorStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class LinkedInBaseAdapter(JobSourceAdapter):
    """Signs in, walks every search URL page by page and hands each loaded page
    to ``_extract_page``, which subclasses implement for their page layout."""

    @abstractmethod
    async def _extract_page(
        self, page: Page, *, cfg, source_url: str, seen_ids: set[str], limit: int,
        known: dict[str, str] | None = None,
    ) -> tuple[list[RawJob], int]:
        """Returns the page's jobs and how many cards were dropped by
        ``cfg.title_keywords`` (they still count toward "this page had results").

        ``known``: descriptions of already-saved jobs by id — those jobs are not
        opened again (no click, no wait for the detail pane)."""
        raise NotImplementedError

    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        cfg = context.config
        timer = TimedDiscovery(cfg.id)
        try:
            saved_state = await asyncio.to_thread(
                load_storage_state, cfg.id, legacy_path=_legacy_storage_state_path()
            )
        except Exception:  # noqa: BLE001
            logger.exception("Could not load saved LinkedIn session from MongoDB; signing in fresh")
            saved_state = None

        if cfg.requires_auth and not saved_state and not _credentials():
            timer.errors.append(
                SourceError(
                    source_id=cfg.id,
                    status=SourceErrorStatus.AUTH_REQUIRED,
                    message=(
                        "No saved LinkedIn session and LINKEDIN_EMAIL / LINKEDIN_PASSWORD "
                        "are not set — cannot sign in."
                        " Use 'Connect LinkedIn' on the Remote Job Finder page (or run `python -m jobEngine.sources.linkedin.login`) to sign in by hand."
                    ),
                    retryable=False,
                )
            )
            return timer.to_result()

        try:
            async with context.browser_manager.open_page(
                source_id=cfg.id,
                allowed_domains=cfg.allowed_domains,
                storage_state=saved_state,
            ) as managed:
                page = managed.page
                seen_ids: set[str] = set()

                # LinkedIn serves unfiltered guest results instead of redirecting when
                # signed out, so check for the auth cookie up front.
                if not await self._has_session(managed.context):
                    await self._sign_in(page, cfg=cfg)
                    await self._save_session(managed.context, cfg.id)

                for search_url in cfg.search_urls:
                    url_jobs = 0
                    for page_index in range(cfg.max_pages):
                        if len(timer.jobs) >= cfg.max_jobs:
                            break
                        if cfg.max_jobs_per_url and url_jobs >= cfg.max_jobs_per_url:
                            break
                        if pagination_depth_exceeded(page_index + 1, max_pages=cfg.max_pages):
                            break
                        url = validate_navigation_url(
                            _page_url(search_url, page_index), allowed_domains=cfg.allowed_domains
                        )
                        if not await self._goto(page, url, cfg=cfg, timer=timer):
                            break

                        # A rejected session either redirects to login or silently drops
                        # li_at and serves guest results.
                        if (
                            _is_login_url(page.url)
                            or _is_checkpoint_url(page.url)
                            or not await self._has_session(managed.context)
                        ):
                            if saved_state:
                                await asyncio.to_thread(
                                    mark_signed_out, cfg.id, f"LinkedIn signed the session out at {page.url}"
                                )
                                saved_state = None
                            await self._sign_in(page, cfg=cfg)
                            await self._save_session(managed.context, cfg.id)
                            if not await self._goto(page, url, cfg=cfg, timer=timer):
                                break
                            if _is_login_url(page.url) or _is_checkpoint_url(page.url):
                                raise _AuthError(
                                    SourceErrorStatus.AUTH_REQUIRED,
                                    f"still redirected to {page.url} after sign-in",
                                )

                        timer.pages_processed += 1
                        limit = cfg.max_jobs - len(timer.jobs)
                        if cfg.max_jobs_per_url:
                            limit = min(limit, cfg.max_jobs_per_url - url_jobs)
                        found, filtered = await self._extract_page(
                            page, cfg=cfg, source_url=url, seen_ids=seen_ids, limit=limit,
                            known=context.known_descriptions,
                        )
                        if not found and not filtered:
                            if page_index == 0:
                                timer.errors.append(
                                    SourceError(
                                        source_id=cfg.id,
                                        status=SourceErrorStatus.SELECTOR_MISSING,
                                        message=f"no job cards found at {url}; LinkedIn markup may have changed",
                                        retryable=False,
                                    )
                                )
                            break  # past the last results page

                        timer.jobs.extend(found)
                        url_jobs += len(found)
                        already_saved = sum(1 for j in found if j.source_job_id in context.known_descriptions)
                        logger.info(
                            "LinkedIn page %s: %s new jobs (%s already saved, not opened; %s dropped by "
                            "title keywords; %s from this URL, total %s)",
                            page_index + 1, len(found), already_saved, filtered, url_jobs, len(timer.jobs),
                        )
                        if cfg.max_jobs_per_url and url_jobs >= cfg.max_jobs_per_url:
                            break
                        if len(found) + filtered < _PAGE_SIZE // 2:
                            break
                        await asyncio.sleep(cfg.rate_limit_delay_seconds)

                # Refresh the saved session (cookies rotate) for the next run; a
                # signed-out context is refused by save_storage_state.
                if await self._has_session(managed.context):
                    await self._save_session(managed.context, cfg.id)

        except _AuthError as exc:
            timer.errors.append(
                SourceError(source_id=cfg.id, status=exc.status, message=str(exc), retryable=False)
            )
        except SSRFViolation as exc:
            timer.errors.append(
                SourceError(source_id=cfg.id, status=SourceErrorStatus.NAVIGATION_ERROR, message=str(exc))
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed for source=%s", type(self).__name__, cfg.id)
            timer.errors.append(
                SourceError(source_id=cfg.id, status=SourceErrorStatus.UNKNOWN, message=str(exc), retryable=True)
            )

        timer.jobs = enforce_job_count(timer.jobs, max_jobs=cfg.max_jobs, source=cfg.id)
        return timer.to_result()

    async def _goto(self, page: Page, url: str, *, cfg, timer: TimedDiscovery) -> bool:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=cfg.timeout_seconds * 1000)
            return True
        except PlaywrightTimeoutError:
            timer.errors.append(
                SourceError(
                    source_id=cfg.id,
                    status=SourceErrorStatus.TIMEOUT,
                    message=f"timed out navigating to {url}",
                    retryable=True,
                )
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
        return False

    async def _has_session(self, context) -> bool:
        cookies = await context.cookies("https://www.linkedin.com")
        return any(c.get("name") == "li_at" and c.get("value") for c in cookies)

    async def _save_session(self, context, source_id: str) -> None:
        """Store the context's cookies in MongoDB; a failure only costs a sign-in next run."""
        try:
            state = await context.storage_state()
            await asyncio.to_thread(save_storage_state, source_id, state)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to save LinkedIn session to MongoDB")

    async def _sign_in(self, page: Page, *, cfg) -> None:
        creds = _credentials()
        if not creds:
            raise _AuthError(
                SourceErrorStatus.AUTH_REQUIRED,
                "saved LinkedIn session expired and LINKEDIN_EMAIL / LINKEDIN_PASSWORD are not set."
                " Use 'Connect LinkedIn' on the Remote Job Finder page (or run `python -m jobEngine.sources.linkedin.login`) to sign in by hand.",
            )
        email, password = creds
        signin_url = validate_navigation_url(
            cfg.signin_url or _DEFAULT_SIGNIN_URL, allowed_domains=cfg.allowed_domains
        )
        logger.info("LinkedIn session missing/expired — signing in as %s", email)

        if not _is_checkpoint_url(page.url):
            await page.goto(signin_url, wait_until="domcontentloaded", timeout=cfg.timeout_seconds * 1000)
            if not _is_login_url(page.url):
                return  # already signed in (login page redirected to the feed)
            try:
                # Attribute-based + :visible — LinkedIn's field ids are generated per
                # release and the page carries hidden duplicate forms.
                await page.locator(
                    "input[autocomplete='username']:visible, input[type='email']:visible, "
                    "input#username:visible, input[name='session_key']:visible"
                ).first.fill(email)
                await page.locator(
                    "input[type='password']:visible, input[name='session_password']:visible"
                ).first.fill(password)
                await page.get_by_role("button", name="Sign in", exact=True).locator("visible=true").first.click()
                await page.wait_for_url(
                    lambda u: not _is_login_url(u) or "/login-submit" in u.lower(),
                    timeout=cfg.timeout_seconds * 1000,
                )
                await page.wait_for_load_state("domcontentloaded")
            except PlaywrightTimeoutError as exc:
                error = await text_or_none(page, "#error-for-password, #error-for-username, .alert-content")
                if error:
                    raise _AuthError(SourceErrorStatus.AUTH_REQUIRED, f"LinkedIn sign-in failed: {error}") from exc
                raise _AuthError(SourceErrorStatus.TIMEOUT, f"LinkedIn sign-in timed out: {exc}") from exc

        if _is_checkpoint_url(page.url):
            wait_s = _checkpoint_wait_seconds()
            logger.warning(
                "LinkedIn security checkpoint at %s — waiting up to %ss for manual verification",
                page.url, wait_s,
            )
            try:
                await page.wait_for_url(
                    lambda u: not _is_checkpoint_url(u) and not _is_login_url(u),
                    timeout=wait_s * 1000,
                )
            except PlaywrightTimeoutError as exc:
                raise _AuthError(
                    SourceErrorStatus.ANTI_BOT_BLOCKED,
                    "LinkedIn asked for a security verification after sign-in. Sign in with "
                    "\"Connect LinkedIn\" on the Remote Job Finder page and complete it there; "
                    "the session is then saved for later runs.",
                ) from exc

        if _is_login_url(page.url):
            error = await text_or_none(page, "#error-for-password, #error-for-username, .alert-content")
            raise _AuthError(
                SourceErrorStatus.AUTH_REQUIRED,
                f"LinkedIn sign-in failed{': ' + error if error else ''}",
            )

        logger.info("LinkedIn sign-in succeeded; session will be saved to MongoDB")

    async def _first_text(self, card, selectors: list[str]) -> str | None:
        for sel in selectors:
            text = await text_or_none(card, sel)
            if text:
                return text
        return None

    async def _link_scan_fallback(
        self, page: Page, *, cfg, source_url: str, seen_ids: set[str], limit: int
    ) -> list[RawJob]:
        """Plain link scan for /jobs/view/ anchors. These jobs carry no company, so
        normalization drops them — surface it instead of silently reporting
        "success" with jobs that never make it through."""
        links = await extract_link_cards(
            page, href_contains=("/jobs/view/",), base_url=cfg.base_url, max_items=limit
        )
        if links:
            logger.warning(
                "LinkedIn: no job cards matched at %s (landed on %s); falling back to link scan — "
                "results will lack company/description", source_url, page.url,
            )
        fallback: list[RawJob] = []
        for c in links:
            if c["url"] in seen_ids:
                continue
            seen_ids.add(c["url"])
            fallback.append(
                RawJob(source=cfg.id, title=c["title"], application_url=c["url"], source_url=source_url)
            )
        return fallback
