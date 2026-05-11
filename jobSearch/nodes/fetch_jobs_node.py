"""Fetch visible job cards from Naukri SRP after login."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any

from playwright.async_api import Page

from services.naukri_service import (
    NAUKRI_JD_PAGE_DELAY,
    NAUKRI_JD_RETRIES,
    NAUKRI_MAX_JD_ENRICH,
    _extract_posted_text,
    _split_csv_trim,
)

from ..state import JobRecord, WorkflowState

logger = logging.getLogger(__name__)

TARGET_SKILLS = frozenset({"react", "next", "node", "javascript", "typescript"})


def _srp_job_limit() -> int:
    """Max job cards to read from the search-results page (see NAUKRI_NO_OF_JOBS)."""
    raw = (os.getenv("NAUKRI_NO_OF_JOBS") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("Invalid NAUKRI_NO_OF_JOBS=%r; using 25", raw)
    return 25


NAUKRI_JS_JOBS_URL = (
    "https://www.naukri.com/javascript-jobs"
    "?k=javascript"
    "&ctcFilter=25to50&ctcFilter=50to75&ctcFilter=75to100&ctcFilter=100to500"
)

_BROWSER_SCRIPTS = Path(__file__).resolve().parent.parent / "browser_scripts"


def _load_browser_script(filename: str) -> str:
    return (_BROWSER_SCRIPTS / filename).read_text(encoding="utf-8")


_EXTRACT_TOPTIER_JS = _load_browser_script("extract_toptier.js")
_EXTRACT_LEGACY_JS = _load_browser_script("extract_legacy.js")
_DETAIL_SCRAPE_JS = _load_browser_script("detail_scrape.js")


def _location_is_remote(location: str) -> bool:
    s = (location or "").lower()
    return any(
        k in s
        for k in (
            "remote",
            "work from home",
            "wfh",
            "anywhere",
            "pan india remote",
        )
    )


def _skill_blob(skills: list[str]) -> str:
    return " ".join(str(s).strip() for s in skills if s and str(s).strip()).lower()


def _has_excluded_skill(blob: str) -> bool:
    """True if any forbidden stack keyword appears (java / c# / .net), not substring-only hacks."""
    if re.search(r"\bjava\b", blob):
        return True
    if "c#" in blob:
        return True
    if ".net" in blob or re.search(r"\bdotnet\b", blob) or re.search(r"\basp\.net\b", blob):
        return True
    if re.search(r"\bnet\b", blob):
        return True
    return False


def _matched_targets(blob: str) -> set[str]:
    """Which entries from TARGET_SKILLS appear in the normalized skill text blob."""
    found: set[str] = set()
    dotless = blob.replace(".", " ")
    for target in TARGET_SKILLS:
        if target == "javascript":
            if "javascript" in dotless or re.search(r"\bjs\b", dotless):
                found.add("javascript")
        elif target == "typescript":
            if "typescript" in dotless or re.search(r"\bts\b", dotless):
                found.add("typescript")
        elif target in dotless:
            found.add(target)
    return found


def compute_stack_relevance(skills: list[str]) -> tuple[bool, float]:
    """
    Relevant iff every TARGET_SKILLS token is reflected in the skill list and no TARGET_NOT
    skill appears (java, c#, net per user rules).
    Percentage = matched targets / |TARGET_SKILLS| * 100.
    """
    blob = _skill_blob(skills)
    if not blob:
        return False, 0.0

    excluded = _has_excluded_skill(blob)
    found = _matched_targets(blob)
    pct = round(100.0 * len(found) / len(TARGET_SKILLS), 1)
    ok = len(found) == len(TARGET_SKILLS) and not excluded
    return ok, pct


async def _click_read_more_if_present(page: Page) -> None:
    try:
        btn = page.get_by_role("button", name=re.compile(r"read\s*more", re.I))
        if await btn.count() > 0:
            await btn.first.click(timeout=3000)
            await page.wait_for_timeout(900)
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        lk = page.get_by_role("link", name=re.compile(r"read\s*more", re.I))
        if await lk.count() > 0:
            await lk.first.click(timeout=3000)
            await page.wait_for_timeout(900)
            return
    except Exception:  # noqa: BLE001
        pass
    try:
        rm = page.locator('[class*="read-more"]').first
        if await rm.count() > 0 and await rm.is_visible():
            await rm.click(timeout=2500)
            await page.wait_for_timeout(900)
    except Exception:  # noqa: BLE001
        pass


async def _scrape_job_detail_page(page: Page, url: str) -> dict[str, Any]:
    await page.goto(url, wait_until="domcontentloaded", timeout=90_000)
    await page.wait_for_selector("#jobs-desc", timeout=50_000)
    await _click_read_more_if_present(page)
    data = await page.evaluate(_DETAIL_SCRAPE_JS)
    if not isinstance(data, dict):
        return {}
    return {
        "description": str(data.get("description") or "").strip(),
        "skills": [str(s).strip() for s in (data.get("skills") or []) if str(s).strip()],
        "posted_raw": str(data.get("postedRaw") or "").strip(),
        "location": str(data.get("location") or "").strip(),
    }


async def _enrich_jobs_with_details(
    page: Page,
    jobs: list[JobRecord],
    *,
    listing_url: str,
) -> None:
    """Visit each job URL (optional); merge posted/location from JD page. Skills stay from SRP."""
    limit = NAUKRI_MAX_JD_ENRICH
    delay = NAUKRI_JD_PAGE_DELAY

    for idx, job in enumerate(jobs):
        link = str(job.get("link") or "").strip()

        if limit > 0 and idx >= limit:
            ok, pct = compute_stack_relevance(list(job.get("skills") or []))
            job["is_relevant"] = ok
            job["relevant_percentage"] = pct
            continue

        if not link:
            ok, pct = compute_stack_relevance(list(job.get("skills") or []))
            job["is_relevant"] = ok
            job["relevant_percentage"] = pct
            continue

        detail: dict[str, Any] | None = None
        for attempt in range(NAUKRI_JD_RETRIES):
            try:
                detail = await _scrape_job_detail_page(page, link)
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "JD scrape attempt %s/%s failed for %s: %s",
                    attempt + 1,
                    NAUKRI_JD_RETRIES,
                    link,
                    e,
                )
                await asyncio.sleep(delay * (attempt + 1))

        if detail:
            raw_posted = detail.get("posted_raw") or ""
            if raw_posted:
                job["uploaded_at"] = raw_posted
                ts = _extract_posted_text(raw_posted)
                if ts is not None:
                    job["posted"] = ts
            loc = detail.get("location") or ""
            if loc:
                job["location"] = loc

        job["is_remote"] = _location_is_remote(str(job.get("location") or ""))
        ok, pct = compute_stack_relevance(list(job.get("skills") or []))
        job["is_relevant"] = ok
        job["relevant_percentage"] = pct

        await asyncio.sleep(delay)

    try:
        await page.goto(listing_url, wait_until="domcontentloaded", timeout=90_000)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not return to listing URL after JD enrichment: %s", e)


async def _scroll_for_more_cards(page: Page, *, target_cards: int) -> None:
    """Scroll lazy-loaded SRP until we have at least ``target_cards`` cards or the list stops growing."""
    xpath = (
        "//div[contains(@class,'rounded-3xl') and contains(@class,'bg-n800') "
        "and contains(@class,'cursor-pointer')]"
    )
    max_rounds = max(24, min(80, target_cards * 3))
    last_count = -1
    stable = 0
    for _ in range(max_rounds):
        try:
            count = await page.locator(f"xpath={xpath}").count()
        except Exception:  # noqa: BLE001
            count = 0
        if count >= target_cards:
            logger.info("SRP shows %s job cards (need %s)", count, target_cards)
            return
        if count == last_count:
            stable += 1
            if stable >= 3:
                logger.info(
                    "SRP card count stable at %s before reaching target %s",
                    count,
                    target_cards,
                )
                break
        else:
            stable = 0
        last_count = count
        await page.mouse.wheel(0, 1400)
        try:
            await page.wait_for_load_state("networkidle", timeout=4000)
        except Exception:  # noqa: BLE001
            await page.wait_for_load_state("domcontentloaded", timeout=4000)


async def _extract_jobs(page: Page, *, limit: int) -> list[JobRecord]:
    primary: list[dict[str, Any]] = await page.evaluate(_EXTRACT_TOPTIER_JS)
    if not primary:
        primary = await page.evaluate(_EXTRACT_LEGACY_JS)

    out: list[JobRecord] = []
    seen: set[str] = set()
    for row in primary:
        if len(out) >= limit:
            break
        link = str(row.get("link") or "").strip()
        key = link or f"{row.get('title','')}|{row.get('company','')}"
        if key in seen:
            continue
        seen.add(key)
        salary = (row.get("salary") or "").strip() or "Not disclosed"
        loc = str(row.get("location") or "").strip() or "—"
        skills_raw = row.get("skills")
        if isinstance(skills_raw, list):
            sk_list = [str(s).strip() for s in skills_raw if str(s).strip()]
        else:
            sk_list = _split_csv_trim(str(skills_raw or ""))
        out.append(
            {
                "title": str(row.get("title") or "").strip() or "—",
                "company": str(row.get("company") or "").strip() or "—",
                "experience": str(row.get("experience") or "").strip() or "—",
                "location": loc,
                "salary": salary,
                "link": link,
                "description": "",
                "skills": sk_list,
                "is_remote": _location_is_remote(loc),
                "uploaded_at": "",
                "is_relevant": False,
                "relevant_percentage": 0.0,
            }
        )
    return out[:limit]


async def _fetch_with_page(page: Page, *, enrich_jd: bool) -> list[JobRecord]:
    listing_url = NAUKRI_JS_JOBS_URL
    cap = _srp_job_limit()
    await page.goto(listing_url, wait_until="domcontentloaded")

    try:
        await page.wait_for_selector(
            "#jobs-list-header, div.srp-jobtuple-wrapper, div.cust-job-tuple, "
            + "div[class*='rounded-3xl'][class*='bg-n800']",
            timeout=45_000,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Primary SRP markers not found; continuing with scroll/extract.")

    await _scroll_for_more_cards(page, target_cards=cap)
    jobs = await _extract_jobs(page, limit=cap)
    logger.info("Fetched %s job rows from SRP (cap=%s)", len(jobs), cap)

    if enrich_jd:
        await _enrich_jobs_with_details(page, jobs, listing_url=listing_url)
    else:
        for job in jobs:
            ok, pct = compute_stack_relevance(list(job.get("skills") or []))
            job["is_relevant"] = ok
            job["relevant_percentage"] = pct

    relevant_only = [j for j in jobs if j.get("is_relevant")]
    logger.info(
        "Keeping %s relevant jobs (dropped %s non-relevant)",
        len(relevant_only),
        len(jobs) - len(relevant_only),
    )
    return relevant_only


async def fetch_jobs_node(state: WorkflowState) -> dict[str, Any]:
    errs = list(state.get("errors") or [])
    session = state.get("session")
    if session is None or not state.get("login_complete"):
        errs.append("fetch_jobs_node: missing session or login not complete.")
        return {"jobs": [], "fetch_complete": False, "errors": errs}

    enrich_jd = bool(state.get("enrich_jd", False))
    page: Page = session.page
    try:
        jobs = await _fetch_with_page(page, enrich_jd=enrich_jd)
        return {"jobs": jobs, "fetch_complete": True, "errors": errs}
    except Exception as e:  # noqa: BLE001
        logger.exception("fetch_jobs_node failed")
        errs.append(str(e))
        return {"jobs": [], "fetch_complete": False, "errors": errs}
