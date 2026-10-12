"""LinkedIn classic search page — ``/jobs/search/``.

Session handling and pagination live in ``base.py``; this module only reads the
left-hand results list and the right-hand detail pane of that page.
"""

from __future__ import annotations

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from ...config.settings import get_settings
from ...guardrails.content_limits import enforce_description_size
from ...models.job import RawJob
from ..registry import register_adapter
from ..title_filter import title_matches_keywords
from .base import _PAGE_SIZE, _POSTED_TEXT_RE, LinkedInBaseAdapter, note_relisted, parse_job_insights

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
# Workplace/employment chips on the detail pane ("Remote", "Full-time"), newest markup first.
_INSIGHT_SELECTORS = [
    ".job-details-fit-level-preferences",
    ".job-details-preferences-and-skills",
    ".job-details-jobs-unified-top-card__job-insight",
    ".jobs-unified-top-card__job-insight",
]
_POSTED_SELECTORS = [
    ".job-details-jobs-unified-top-card__tertiary-description-container",
    ".job-details-jobs-unified-top-card__primary-description-container",
]


@register_adapter("linkedin_search")
class LinkedInSearchAdapter(LinkedInBaseAdapter):
    async def _extract_page(
        self, page: Page, *, cfg, source_url: str, seen_ids: set[str], limit: int,
        known: dict[str, str] | None = None,
    ) -> tuple[list[RawJob], int]:
        known = known or {}
        await self._load_all_cards(page)

        cards = []
        for selector in _CARD_SELECTORS:
            cards = await page.query_selector_all(selector)
            if cards:
                break

        jobs: list[RawJob] = []
        filtered = 0
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
                note_relisted(seen_ids, job_id, source_url)
                continue
            # Claim the id before the first await: other search URLs run concurrently
            # in other tabs and must not scrape the same job.
            if job_id:
                seen_ids.add(job_id)

            try:
                await card.scroll_into_view_if_needed(timeout=3_000)
                await page.wait_for_timeout(150)
            except Exception:  # noqa: BLE001
                pass
            title = await self._first_text(card, _TITLE_SELECTORS)
            if not title:
                if job_id:
                    seen_ids.discard(job_id)  # card not rendered: another listing may still read it
                continue
            title = title.split("\n")[0].strip()
            if not title_matches_keywords(title, cfg.title_keywords):
                filtered += 1
                continue
            company = await self._first_text(card, _COMPANY_SELECTORS)
            location = await self._first_text(card, _LOCATION_SELECTORS)
            insights = ""
            if job_id and job_id in known:
                # Already saved: never opened again. A card that says "Viewed" has no
                # age; the saved posted date is reused for it (matching node / saved store).
                description, posted = known[job_id], await self._card_posted(card)
            else:
                description, posted, insights = await self._read_details(page, card)
                posted = posted or await self._card_posted(card)
            workplace, employment_type = parse_job_insights(insights)

            jobs.append(
                RawJob(
                    source=cfg.id,
                    source_job_id=job_id,
                    title=title,
                    company=company or "",
                    location=location,
                    description=enforce_description_size(description or ""),
                    posted_date_raw=posted,
                    employment_type=employment_type,
                    remote_type="remote" if "f_WT=2" in source_url else None,
                    application_url=(
                        f"{cfg.base_url.rstrip('/')}/jobs/view/{job_id}/" if job_id else cfg.base_url
                    ),
                    source_url=source_url,
                    extra={"easy_apply": "f_AL=true" in source_url, **({"workplace_type": workplace} if workplace else {})},
                )
            )

        if jobs or filtered:
            return jobs, filtered
        return await self._link_scan_fallback(
            page, cfg=cfg, source_url=source_url, seen_ids=seen_ids, limit=limit
        ), 0

    async def _load_all_cards(self, page: Page) -> None:
        """The results list lazy-renders cards as it scrolls; scroll it to the bottom."""
        card_selector = ", ".join(_CARD_SELECTORS)
        try:
            await page.wait_for_selector(card_selector, timeout=20_000)
        except PlaywrightTimeoutError:
            return
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

    async def _card_posted(self, card) -> str | None:
        """Relative age from the card's own footer ("11 hours ago"), for jobs whose
        detail pane was skipped (already saved) or failed to load."""
        try:
            text = await card.inner_text()
        except Exception:  # noqa: BLE001
            return None
        match = _POSTED_TEXT_RE.search(text or "")
        return match.group(0) if match else None

    async def _read_details(self, page: Page, card) -> tuple[str, str | None, str]:
        """Click the card so the right-hand pane loads its full description. Also
        returns the pane's workplace/employment chips text when
        JOB_WORLDWIDE_REMOTE_ENABLED is on ("" otherwise)."""
        try:
            await card.scroll_into_view_if_needed(timeout=5_000)
            link = await card.query_selector("a.job-card-list__title--link, a.job-card-container__link, a")
            await (link or card).click(timeout=5_000)
            await page.wait_for_timeout(700)
            await page.wait_for_selector(", ".join(_DESCRIPTION_SELECTORS), timeout=8_000)
        except Exception:  # noqa: BLE001
            return "", None, ""
        description = await self._first_text(page, _DESCRIPTION_SELECTORS) or ""
        posted = await self._first_text(page, _POSTED_SELECTORS)
        if posted:
            # "India · Reposted 5 minutes ago · Over 100 applicants\n\nPromoted by hirer ..."
            parts = [p.strip() for p in posted.split("\n")[0].split("·")]
            posted = next((p for p in parts if "ago" in p.lower()), parts[0] if parts else None)
        insights = ""
        if get_settings().worldwide_remote_enabled:
            for selector in _INSIGHT_SELECTORS:
                try:
                    insights = "\n".join(await page.locator(selector).all_inner_texts())
                except Exception:  # noqa: BLE001
                    insights = ""
                if insights.strip():
                    break
            # Older markup puts "Full-time · Mid-Senior level" on one line.
            insights = insights.replace("·", "\n")
        return description, posted, insights
