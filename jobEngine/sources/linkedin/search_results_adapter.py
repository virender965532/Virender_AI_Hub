"""LinkedIn new (SDUI) search page — ``/jobs/search-results/``.

Session handling and pagination live in ``base.py``; this module only reads the
job cards and the detail pane of that page.
"""

from __future__ import annotations

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from ...config.settings import get_settings
from ...guardrails.content_limits import enforce_description_size
from ...models.job import RawJob
from ..registry import register_adapter
from ..title_filter import title_passes
from .base import _POSTED_TEXT_RE, LinkedInBaseAdapter, note_relisted, parse_job_insights

# Class names are obfuscated per release, so only these ``componentkey`` hooks
# (which embed the job id) are stable.
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


@register_adapter("linkedin_search_results")
class LinkedInSearchResultsAdapter(LinkedInBaseAdapter):
    async def _extract_page(
        self, page: Page, *, cfg, source_url: str, seen_ids: set[str], limit: int,
        known: dict[str, str] | None = None,
    ) -> tuple[list[RawJob], int]:
        """Fields come from the card's <p> order (title, company, location, ...); the
        description from the detail pane, which is keyed by job id so a stale pane is
        never read for the wrong job."""
        known = known or {}
        try:
            # The page renders all 25 cards of a page up front — no scrolling needed.
            await page.wait_for_selector(_NEW_CARD_SELECTOR, timeout=20_000)
        except PlaywrightTimeoutError:
            return await self._link_scan_fallback(
                page, cfg=cfg, source_url=source_url, seen_ids=seen_ids, limit=limit
            ), 0

        jobs: list[RawJob] = []
        filtered = 0
        for fields in await page.eval_on_selector_all(_NEW_CARD_SELECTOR, _NEW_CARDS_JS):
            if len(jobs) >= limit:
                break
            job_id = fields.get("job_id")
            title = fields.get("title")
            if job_id and job_id in seen_ids:
                note_relisted(seen_ids, job_id, source_url)
                continue
            if not job_id or not title:
                continue
            seen_ids.add(job_id)
            if not title_passes(title, cfg.title_keywords, cfg.title_exclude_keywords):
                filtered += 1
                continue
            posted = fields.get("posted")
            description = known.get(job_id)
            header = ""
            # Already-saved jobs are never opened again. A card that says "Viewed" has no
            # age; the saved posted date is reused for it (matching node / saved store).
            if description is None:
                pane_description, pane_posted, header = await self._read_details(page, job_id)
                description = description if description is not None else pane_description
                posted = posted or pane_posted
            workplace, employment_type = (
                parse_job_insights(header) if get_settings().worldwide_remote_enabled else (None, None)
            )
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
                    employment_type=employment_type,
                    remote_type="remote" if "f_WT=2" in source_url else None,
                    application_url=f"{cfg.base_url.rstrip('/')}/jobs/view/{job_id}/",
                    source_url=source_url,
                    extra={
                        "easy_apply": bool(fields.get("easy_apply")),
                        **({"workplace_type": workplace} if workplace else {}),
                    },
                )
            )
        return jobs, filtered

    async def _read_details(self, page: Page, job_id: str) -> tuple[str, str | None, str]:
        """Click the card and wait until *its* "About the job" section has text.
        The card is re-located by id on every attempt (the list re-renders on each
        click), and the first click after a page load is sometimes swallowed.
        Also returns the pane's age ("India · 17 hours ago · Over 100 applicants") and
        the pane header text (workplace/employment chips are read from it)."""
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
                header = await self._pane_header(page, selector)
                match = _POSTED_TEXT_RE.search(header)
                return text.removeprefix("About the job").strip(), (match.group(0) if match else None), header
            except Exception:  # noqa: BLE001
                continue
        return "", None, ""

    async def _pane_header(self, page: Page, about_selector: str) -> str:
        """The pane's top card (age, workplace/employment chips), which sits a few
        ancestors above "About the job"."""
        try:
            return await page.evaluate(_NEW_PANE_HEADER_JS, about_selector) or ""
        except Exception:  # noqa: BLE001
            return ""
