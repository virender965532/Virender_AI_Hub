"""LinkedIn new (SDUI) search page — ``/jobs/search-results/``.

Session handling, pagination, the too-old check and the detail fetch live in
``base.py``; this module only reads a results page: its job cards (id, title,
company, location, age) and how many results pages the search has.
"""

from __future__ import annotations

import logging
import math
import re

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from ...config.settings import get_settings
from ...models.job import RawJob
from ..registry import register_adapter
from ..title_filter import title_passes
from .base import _PAGE_SIZE, LinkedInBaseAdapter, is_remote_filter, note_relisted

logger = logging.getLogger(__name__)

_WORKPLACE_IN_LOCATION_RE = re.compile(r"\((Remote|Hybrid|On-site)\)", re.IGNORECASE)

# Class names are obfuscated per release, so only these ``componentkey`` hooks
# (which embed the job id) are stable.
_NEW_CARD_ID_PREFIX = "job-card-component-ref-"
_NEW_CARD_SELECTOR = f"div[role=button][componentkey^='{_NEW_CARD_ID_PREFIX}']"
# Page count: the highest "Page N" pagination button, a "Page 1 of N" label, or an
# exact "123 results" header (LinkedIn writes "99+ results" when it won't say).
_PAGE_COUNT_JS = """() => {
  const nums = [];
  for (const el of document.querySelectorAll('[aria-label]')) {
    const m = (el.getAttribute('aria-label') || '').match(/^Page\\s+(\\d+)(?:\\s+of\\s+(\\d+))?/i);
    if (m) nums.push(+(m[2] || m[1]));
  }
  const text = document.body ? document.body.innerText : '';
  const of = text.match(/Page\\s+\\d+\\s+of\\s+(\\d+)/i);
  if (of) nums.push(+of[1]);
  const results = text.match(/(\\d[\\d,]*)\\s+results?\\b/i);
  return {pages: nums.length ? Math.max(...nums) : null,
          results: results ? +results[1].replace(/,/g, '') : null};
}"""
# Read every card in one pass. <p> order is title, company, location, ...
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
        skip_ids: set[str] | None = None,
    ) -> tuple[list[RawJob], int]:
        """Cards only — title, company, location, age — from the card's <p> order. Jobs
        already read this run, saved / rejected by an earlier run (``skip_ids``) or with
        an excluded title are skipped. Descriptions are fetched by ``base.py`` once
        every page is read; ages are checked there too."""
        skip_ids = skip_ids or set()
        try:
            # The page renders all 25 cards of a page up front — no scrolling needed.
            await page.wait_for_selector(_NEW_CARD_SELECTOR, timeout=20_000)
        except PlaywrightTimeoutError:
            return await self._link_scan_fallback(
                page, cfg=cfg, source_url=source_url, seen_ids=seen_ids, limit=limit
            ), 0

        worldwide = get_settings().worldwide_remote_enabled
        jobs: list[RawJob] = []
        skipped = 0
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
            if job_id in skip_ids:
                skipped += 1  # saved or rejected by an earlier run
                continue
            # Only the exclude list applies here; ``title_keywords`` is ignored on this page.
            if not title_passes(title, (), cfg.title_exclude_keywords):
                skipped += 1
                continue
            workplace = _workplace_from_location(fields.get("location")) if worldwide else None
            jobs.append(
                RawJob(
                    source=cfg.id,
                    source_job_id=job_id,
                    title=title,
                    company=fields.get("company") or "",
                    location=fields.get("location"),
                    salary=fields.get("salary"),
                    description="",
                    posted_date_raw=fields.get("posted"),
                    remote_type="remote" if is_remote_filter(source_url) or workplace == "Remote" else None,
                    application_url=f"{cfg.base_url.rstrip('/')}/jobs/view/{job_id}/",
                    source_url=source_url,
                    extra={
                        "easy_apply": bool(fields.get("easy_apply")),
                        **({"workplace_type": workplace} if workplace else {}),
                    },
                )
            )
        return jobs, skipped

    async def _page_count(self, page: Page) -> int | None:
        try:
            found = await page.evaluate(_PAGE_COUNT_JS) or {}
        except Exception:  # noqa: BLE001
            return None
        from_results = math.ceil(found["results"] / _PAGE_SIZE) if found.get("results") else None
        counts = [c for c in (found.get("pages"), from_results) if c]
        if not counts:
            logger.info("LinkedIn: the results page shows no page count; reading pages until a short one")
        return max(counts) if counts else None


def _workplace_from_location(location: str | None) -> str | None:
    """Cards say e.g. "Bengaluru, Karnataka, India (Hybrid)"."""
    match = _WORKPLACE_IN_LOCATION_RE.search(location or "")
    if not match:
        return None
    return {"remote": "Remote", "hybrid": "Hybrid", "on-site": "On-site"}[match.group(1).lower()]
