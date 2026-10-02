"""LinkedIn — auth-gated source (section 10).

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
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from ...config.settings import JOB_ENGINE_HISTORY_DIR
from ...guardrails.content_limits import (
    enforce_description_size,
    enforce_job_count,
    pagination_depth_exceeded,
)
from ...guardrails.url_guardrails import SSRFViolation, validate_navigation_url
from ...models.errors import SourceError, SourceErrorStatus
from ...models.job import RawJob
from ...persistence.sessions import load_storage_state, mark_signed_out, save_storage_state
from ..base import JobSourceAdapter, SourceDiscoveryResult, SourceExecutionContext, TimedDiscovery
from ..dom_utils import extract_link_cards, text_or_none
from ..registry import register_adapter

logger = logging.getLogger(__name__)

_DEFAULT_SIGNIN_URL = "https://www.linkedin.com/login/"
_DEFAULT_STORAGE_STATE_PATH = JOB_ENGINE_HISTORY_DIR / "linkedin_storage_state.json"
_PAGE_SIZE = 25

# New (SDUI) search page, /jobs/search-results/: class names are obfuscated per
# release, so only these ``componentkey`` hooks (which embed the job id) are stable.
_NEW_CARD_ID_PREFIX = "job-card-component-ref-"
_NEW_CARD_SELECTOR = f"div[role=button][componentkey^='{_NEW_CARD_ID_PREFIX}']"
_NEW_ABOUT_SELECTOR = "[componentkey='JobDetails_AboutTheJob_{job_id}']"
# Text above "About the job" in the detail pane: walk up from that section until an
# ancestor also holds the top card ("India · 17 hours ago · Over 100 applicants").
_NEW_PANE_HEADER_JS = """s => {
  let e = document.querySelector(s);
  for (let i = 0; i < 8 && e; i++, e = e.parentElement) {
    const head = (e.innerText || '').split('About the job')[0];
    if (/\\bago\\b|just now/i.test(head)) return head;
  }
  return null;
}"""
# Read every card in one pass: clicking a card re-renders the list, which detaches
# any element handles taken beforehand. <p> order is title, company, location, ...
_NEW_CARDS_JS = """els => els.map(card => {
  const ps = [...card.querySelectorAll('p')]
    .map(p => (p.innerText || '').split('\\n')[0].trim())
    .filter(t => t && t !== '·');
  const text = card.innerText || '';
  // Cards show a bare "11 hours ago" / "Reposted 2 days ago" — no "Posted" prefix.
  const posted = (text.match(/(\\d+|an?)\\s+(minute|min|hour|hr|day|week|month|year)s?\\s+ago|just now|moments? ago/i) || [])[0] || null;
  const salary = ps.find(t => /(\\/(yr|month|hr)|INR|\\$|€|£|LPA)/i.test(t)) || null;
  return {
    job_id: (card.getAttribute('componentkey') || '').replace('job-card-component-ref-', ''),
    title: (ps[0] || '').replace(/\\s*\\(Verified job\\)$/, '') || null,
    company: ps[1] || null, location: ps[2] || null, posted, salary,
    easy_apply: /Easy Apply/.test(text),
  };
})"""

# Logged-in search page (left results list) first, guest page second.
_CARD_SELECTORS = [
    "li[data-occludable-job-id]",
    "div.job-card-container[data-job-id]",
    "ul.jobs-search__results-list li",
]
_TITLE_SELECTORS = [
    "a.job-card-list__title--link strong",
    "a.job-card-list__title--link",
    "a.job-card-list__title",
    ".job-card-container__link strong",
    ".job-card-container__link",
    "h3.base-search-card__title",
]
_COMPANY_SELECTORS = [
    ".artdeco-entity-lockup__subtitle",
    ".job-card-container__primary-description",
    ".job-card-container__company-name",
    "h4.base-search-card__subtitle",
]
_LOCATION_SELECTORS = [
    ".job-card-container__metadata-wrapper li",
    ".artdeco-entity-lockup__caption",
    ".job-card-container__metadata-item",
    "span.job-search-card__location",
]
_RESULTS_LIST_SELECTORS = [
    ".scaffold-layout__list > div",
    ".jobs-search-results-list",
    ".scaffold-layout__list",
]
_DESCRIPTION_SELECTORS = [
    ".jobs-search__job-details--container #job-details",
    "#job-details",
    ".jobs-description__content",
    ".jobs-box__html-content",
]
_POSTED_SELECTORS = [
    ".job-details-jobs-unified-top-card__tertiary-description-container",
    ".job-details-jobs-unified-top-card__primary-description-container",
]
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


@register_adapter("linkedin")
class LinkedInAdapter(JobSourceAdapter):
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
                        found = await self._extract_page(
                            page, cfg=cfg, source_url=url, seen_ids=seen_ids, limit=limit,
                            known=context.known_descriptions,
                        )
                        if not found:
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
                            "LinkedIn page %s: %s new jobs (%s already saved, not opened; %s from this URL, total %s)",
                            page_index + 1, len(found), already_saved, url_jobs, len(timer.jobs),
                        )
                        if cfg.max_jobs_per_url and url_jobs >= cfg.max_jobs_per_url:
                            break
                        if len(found) < _PAGE_SIZE // 2:
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
            logger.exception("LinkedInAdapter failed for source=%s", cfg.id)
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

    async def _load_all_cards(self, page: Page) -> None:
        """The results list lazy-renders cards as it scrolls; scroll it to the bottom."""
        card_selector = ", ".join(_CARD_SELECTORS)
        try:
            await page.wait_for_selector(f"{card_selector}, {_NEW_CARD_SELECTOR}", timeout=20_000)
        except PlaywrightTimeoutError:
            return
        if await page.query_selector(_NEW_CARD_SELECTOR):
            return  # the new page renders all 25 cards of a page up front
        # Scroll the last rendered card into view until no new cards appear; fixed
        # scrollBy steps on the list container stop short (only ~8 of 25 render).
        previous, stable_rounds = -1, 0
        for _ in range(30):
            cards = await page.query_selector_all(card_selector)
            if len(cards) >= _PAGE_SIZE:
                break
            if len(cards) == previous:
                stable_rounds += 1
                if stable_rounds >= 3:
                    break
            else:
                previous, stable_rounds = len(cards), 0
            try:
                await cards[-1].scroll_into_view_if_needed(timeout=3_000)
            except Exception:  # noqa: BLE001
                pass
            for selector in _RESULTS_LIST_SELECTORS:
                container = await page.query_selector(selector)
                if container is not None:
                    await container.evaluate("el => el.scrollBy(0, 800)")
            await page.wait_for_timeout(500)
        # Card bodies are placeholders until scrolled past; return to the top so
        # the per-card click/scroll in _read_details hydrates each in turn.
        for selector in _RESULTS_LIST_SELECTORS:
            container = await page.query_selector(selector)
            if container is not None:
                await container.evaluate("el => el.scrollTo(0, 0)")

    async def _extract_page(
        self, page: Page, *, cfg, source_url: str, seen_ids: set[str], limit: int,
        known: dict[str, str] | None = None,
    ) -> list[RawJob]:
        """``known``: descriptions of already-saved jobs by id — those jobs are not
        opened again (no click, no wait for the detail pane)."""
        known = known or {}
        await self._load_all_cards(page)

        if await page.query_selector(_NEW_CARD_SELECTOR):
            return await self._extract_new_layout(
                page, cfg=cfg, source_url=source_url, seen_ids=seen_ids, limit=limit, known=known
            )

        cards = []
        for selector in _CARD_SELECTORS:
            cards = await page.query_selector_all(selector)
            if cards:
                break

        jobs: list[RawJob] = []
        for card in cards:
            if len(jobs) >= limit:
                break
            job_id = await card.get_attribute("data-occludable-job-id") or await card.get_attribute(
                "data-job-id"
            )
            if not job_id:
                inner = await card.query_selector("[data-job-id]")
                job_id = await inner.get_attribute("data-job-id") if inner else None
            if job_id and job_id in seen_ids:
                continue

            try:
                await card.scroll_into_view_if_needed(timeout=3_000)
                await page.wait_for_timeout(150)
            except Exception:  # noqa: BLE001
                pass
            title = await self._first_text(card, _TITLE_SELECTORS)
            if not title:
                continue
            company = await self._first_text(card, _COMPANY_SELECTORS)
            location = await self._first_text(card, _LOCATION_SELECTORS)
            if job_id and job_id in known:
                description, posted = known[job_id], await self._card_posted(card)
                if not posted:  # card says "Viewed" — the pane still shows the age
                    posted = (await self._read_details(page, card))[1]
            else:
                description, posted = await self._read_details(page, card)
                posted = posted or await self._card_posted(card)

            if job_id:
                seen_ids.add(job_id)
            jobs.append(
                RawJob(
                    source=cfg.id,
                    source_job_id=job_id,
                    title=title.split("\n")[0].strip(),
                    company=company or "",
                    location=location,
                    description=enforce_description_size(description or ""),
                    posted_date_raw=posted,
                    remote_type="remote" if "f_WT=2" in source_url else None,
                    application_url=(
                        f"{cfg.base_url.rstrip('/')}/jobs/view/{job_id}/" if job_id else cfg.base_url
                    ),
                    source_url=source_url,
                    extra={"easy_apply": "f_AL=true" in source_url},
                )
            )

        if jobs:
            return jobs

        # Fallback: plain link scan for /jobs/view/ anchors. These jobs carry no
        # company, so normalization drops them — surface it instead of silently
        # reporting "success" with jobs that never make it through.
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

    async def _extract_new_layout(
        self, page: Page, *, cfg, source_url: str, seen_ids: set[str], limit: int,
        known: dict[str, str] | None = None,
    ) -> list[RawJob]:
        """Cards on /jobs/search-results/: fields come from the card's <p> order
        (title, company, location, ...); the description from the detail pane,
        which is keyed by job id so a stale pane is never read for the wrong job."""
        known = known or {}
        jobs: list[RawJob] = []
        for fields in await page.eval_on_selector_all(_NEW_CARD_SELECTOR, _NEW_CARDS_JS):
            if len(jobs) >= limit:
                break
            job_id = fields.get("job_id")
            title = fields.get("title")
            if not job_id or job_id in seen_ids or not title:
                continue
            seen_ids.add(job_id)
            posted = fields.get("posted")
            description = known.get(job_id)
            # A card you've opened shows "Viewed" instead of its age, so the pane is
            # the only place left to read it — open it even for already-saved jobs.
            if description is None or not posted:
                pane_description, pane_posted = await self._read_new_details(page, job_id)
                description = description if description is not None else pane_description
                posted = posted or pane_posted
            jobs.append(
                RawJob(
                    source=cfg.id,
                    source_job_id=job_id,
                    title=title,
                    company=fields.get("company") or "",
                    location=fields.get("location"),
                    salary=fields.get("salary"),
                    description=enforce_description_size(description),
                    posted_date_raw=posted,
                    remote_type="remote" if "f_WT=2" in source_url else None,
                    application_url=f"{cfg.base_url.rstrip('/')}/jobs/view/{job_id}/",
                    source_url=source_url,
                    extra={"easy_apply": bool(fields.get("easy_apply"))},
                )
            )
        return jobs

    async def _read_new_details(self, page: Page, job_id: str) -> tuple[str, str | None]:
        """Click the card and wait until *its* "About the job" section has text.
        The card is re-located by id on every attempt (the list re-renders on each
        click), and the first click after a page load is sometimes swallowed.
        Also returns the pane's age ("India · 17 hours ago · Over 100 applicants")."""
        selector = _NEW_ABOUT_SELECTOR.format(job_id=job_id)
        card = page.locator(f"div[role=button][componentkey='{_NEW_CARD_ID_PREFIX}{job_id}']").first
        for _ in range(2):
            try:
                await card.scroll_into_view_if_needed(timeout=5_000)
                await card.locator("p").first.click(timeout=5_000)
                await page.wait_for_function(
                    "s => { const e = document.querySelector(s); return !!e && e.innerText.trim().length > 50; }",
                    arg=selector,
                    timeout=10_000,
                )
                text = (await page.inner_text(selector)).strip()
                return text.removeprefix("About the job").strip(), await self._pane_posted(page, selector)
            except Exception:  # noqa: BLE001
                continue
        return "", None

    async def _pane_posted(self, page: Page, about_selector: str) -> str | None:
        """Age from the pane's top card, which sits a few ancestors above "About the job"."""
        try:
            header = await page.evaluate(_NEW_PANE_HEADER_JS, about_selector)
        except Exception:  # noqa: BLE001
            return None
        match = _POSTED_TEXT_RE.search(header or "")
        return match.group(0) if match else None

    async def _card_posted(self, card) -> str | None:
        """Relative age from the card's own footer ("11 hours ago"), for jobs whose
        detail pane was skipped (already saved) or failed to load."""
        try:
            text = await card.inner_text()
        except Exception:  # noqa: BLE001
            return None
        match = _POSTED_TEXT_RE.search(text or "")
        return match.group(0) if match else None

    async def _first_text(self, card, selectors: list[str]) -> str | None:
        for sel in selectors:
            text = await text_or_none(card, sel)
            if text:
                return text
        return None

    async def _read_details(self, page: Page, card) -> tuple[str, str | None]:
        """Click the card so the right-hand pane loads its full description."""
        try:
            await card.scroll_into_view_if_needed(timeout=5_000)
            link = await card.query_selector("a.job-card-list__title--link, a.job-card-container__link, a")
            await (link or card).click(timeout=5_000)
            await page.wait_for_timeout(700)
            await page.wait_for_selector(", ".join(_DESCRIPTION_SELECTORS), timeout=8_000)
        except Exception:  # noqa: BLE001
            return "", None
        description = await self._first_text(page, _DESCRIPTION_SELECTORS) or ""
        posted = await self._first_text(page, _POSTED_SELECTORS)
        if posted:
            # "India · Reposted 5 minutes ago · Over 100 applicants\n\nPromoted by hirer ..."
            parts = [p.strip() for p in posted.split("\n")[0].split("·")]
            posted = next((p for p in parts if "ago" in p.lower()), parts[0] if parts else None)
        return description, posted
