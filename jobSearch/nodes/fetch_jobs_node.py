"""Fetch visible job cards from Naukri SRP after login."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from playwright.async_api import Page

from services.naukri_service import (
    NAUKRI_JD_PAGE_DELAY,
    NAUKRI_JD_RETRIES,
    NAUKRI_MAX_JD_ENRICH,
    _extract_posted_text,
    _split_csv_trim,
)

from ..state import JobRecord, WorkflowState
from ..utils.job_match_scoring import calculate_job_match

logger = logging.getLogger(__name__)


def _env_str(name: str, default: str) -> str:
    raw = (os.getenv(name) or "").strip()
    return raw if raw else default


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes")


NAUKRI_JOBS_BASE_URL = _env_str(
    "NAUKRI_JOBS_BASE_URL", "https://www.naukri.com/javascript-jobs"
)
NAUKRI_JOB_KEYWORD = _env_str("NAUKRI_JOB_KEYWORD", "javascript")
NAUKRI_JOB_AGE = _env_str("NAUKRI_JOB_AGE", "1")
NAUKRI_CTC_FILTERS = _env_str(
    "NAUKRI_CTC_FILTERS", "50to75,75to100,100to500"
)
RELEVANCE_MIN_PCT = _env_float("NAUKRI_JOB_RELEVANCE_MIN_PCT", 80.0)
_MAX_SRP_PAGES = _env_int("NAUKRI_JOB_MAX_PAGES", 100)
NAUKRI_IS_EMAIL_REQUIRED = _env_bool("NAUKRI_IS_EMAIL_REQUIRED", False)


def _build_srp_query() -> str:
    """Build SRP query string from NAUKRI_JOB_KEYWORD, NAUKRI_JOB_AGE, NAUKRI_CTC_FILTERS."""
    params: list[tuple[str, str]] = [
        ("k", NAUKRI_JOB_KEYWORD),
        ("jobAge", NAUKRI_JOB_AGE),
    ]
    for band in NAUKRI_CTC_FILTERS.split(","):
        band = band.strip()
        if band:
            params.append(("ctcFilter", band))
    return "?" + urlencode(params)


NAUKRI_JS_JOBS_QUERY = _build_srp_query()


def _passes_relevance_threshold(job: JobRecord) -> bool:
    """True when stack match score is at least RELEVANCE_MIN_PCT."""
    try:
        return float(job.get("relevant_percentage") or 0) >= RELEVANCE_MIN_PCT
    except (TypeError, ValueError):
        return False


def _relevant_job_cap() -> int:
    """Max relevant jobs to collect across paginated SRP pages (see NAUKRI_NO_OF_JOBS)."""
    raw = (os.getenv("NAUKRI_NO_OF_JOBS") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("Invalid NAUKRI_NO_OF_JOBS=%r; using 25", raw)
    return 25


def _srp_page_url(page_num: int) -> str:
    """Build paginated Naukri SRP URL; page 1 has no suffix, page 2+ uses ``-2``, ``-3``, etc."""
    if page_num <= 1:
        return NAUKRI_JOBS_BASE_URL + NAUKRI_JS_JOBS_QUERY
    return f"{NAUKRI_JOBS_BASE_URL}-{page_num}{NAUKRI_JS_JOBS_QUERY}"

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


def compute_stack_relevance(skills: list[str]) -> tuple[bool, float]:
    """Weighted profile match score with excluded-stack penalty (Java, C#, .NET, Golang)."""
    try:
        result = calculate_job_match(skills)
        pct = float(result["score"])
        excluded = bool(result.get("excluded_technologies"))
        ok = bool(result["matched_skills"]) and not excluded
        return ok, pct
    except Exception as e:
        logger.exception("compute_stack_relevance failed: %s", e)
        return False, 0.0

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


def _apply_posted_raw(job: JobRecord, raw_posted: str) -> None:
    """Set uploaded_at / posted timestamp from raw Naukri text such as ``5d ago``."""
    raw_posted = str(raw_posted or "").strip()
    if not raw_posted:
        return
    job["uploaded_at"] = raw_posted
    ts = _extract_posted_text(raw_posted)
    if ts is not None:
        job["posted"] = ts


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


async def _enrich_one_job(
    page: Page,
    job: JobRecord,
    *,
    listing_url: str,
    enrich_jd: bool,
    enrich_count: int,
) -> int:
    """Enrich a single job (optional JD visit) and set relevance fields. Returns updated enrich_count."""
    delay = NAUKRI_JD_PAGE_DELAY
    limit = NAUKRI_MAX_JD_ENRICH
    link = str(job.get("link") or "").strip()
    should_scrape = enrich_jd and link and (limit <= 0 or enrich_count < limit)

    if should_scrape:
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
            _apply_posted_raw(job, str(detail.get("posted_raw") or ""))
            loc = detail.get("location") or ""
            if loc:
                job["location"] = loc

        enrich_count += 1
        await asyncio.sleep(delay)

        try:
            await page.goto(listing_url, wait_until="domcontentloaded", timeout=90_000)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not return to listing URL after JD enrichment: %s", e)

    job["is_remote"] = _location_is_remote(str(job.get("location") or ""))
    ok, pct = compute_stack_relevance(list(job.get("skills") or []))
    job["is_relevant"] = ok
    job["relevant_percentage"] = pct
    return enrich_count


async def _scroll_for_more_cards(page: Page, *, target_cards: int | None = None) -> None:
    """Scroll lazy-loaded SRP until card count stabilizes or ``target_cards`` is reached."""
    xpath = (
        "//div[contains(@class,'rounded-3xl') and contains(@class,'bg-n800') "
        "and contains(@class,'cursor-pointer')]"
    )
    max_rounds = 80 if target_cards is None else max(24, min(80, target_cards * 3))
    last_count = -1
    stable = 0
    for _ in range(max_rounds):
        try:
            count = await page.locator(f"xpath={xpath}").count()
        except Exception:  # noqa: BLE001
            count = 0
        if target_cards is not None and count >= target_cards:
            logger.info("SRP shows %s job cards (need %s)", count, target_cards)
            return
        if count == last_count:
            stable += 1
            if stable >= 3:
                logger.info(
                    "SRP card count stable at %s%s",
                    count,
                    f" before reaching target {target_cards}" if target_cards is not None else "",
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


async def _extract_jobs(page: Page, *, limit: int | None = None) -> list[JobRecord]:
    primary: list[dict[str, Any]] = await page.evaluate(_EXTRACT_TOPTIER_JS)
    if not primary:
        primary = await page.evaluate(_EXTRACT_LEGACY_JS)

    out: list[JobRecord] = []
    seen: set[str] = set()
    for row in primary:
        if limit is not None and len(out) >= limit:
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
        record: JobRecord = {
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
        _apply_posted_raw(record, str(row.get("posted") or ""))
        out.append(record)
    return out[:limit] if limit is not None else out


async def _load_srp_page(page: Page, listing_url: str) -> None:
    await page.goto(listing_url, wait_until="domcontentloaded")
    try:
        await page.wait_for_selector(
            "#jobs-list-header, div.srp-jobtuple-wrapper, div.cust-job-tuple, "
            + "div[class*='rounded-3xl'][class*='bg-n800']",
            timeout=45_000,
        )
    except Exception:  # noqa: BLE001
        logger.warning("Primary SRP markers not found; continuing with scroll/extract.")


async def _fetch_with_page(page: Page, *, enrich_jd: bool) -> list[JobRecord]:
    cap = _relevant_job_cap()
    relevant_jobs: list[JobRecord] = []
    seen: set[str] = set()
    enrich_count = 0
    total_scanned = 0

    for page_num in range(1, _MAX_SRP_PAGES + 1):
        if len(relevant_jobs) >= cap:
            break

        listing_url = _srp_page_url(page_num)
        logger.info("Fetching SRP page %s: %s", page_num, listing_url)
        await _load_srp_page(page, listing_url)
        await _scroll_for_more_cards(page)
        page_jobs = await _extract_jobs(page)
        logger.info("Page %s: extracted %s job rows", page_num, len(page_jobs))

        new_jobs: list[JobRecord] = []
        for job in page_jobs:
            link = str(job.get("link") or "").strip()
            key = link or f"{job.get('title', '')}|{job.get('company', '')}"
            if key in seen:
                continue
            seen.add(key)
            new_jobs.append(job)

        if not new_jobs:
            logger.info("Page %s: no new jobs; stopping pagination", page_num)
            break

        for job in new_jobs:
            total_scanned += 1
            if enrich_jd:
                enrich_count = await _enrich_one_job(
                    page,
                    job,
                    listing_url=listing_url,
                    enrich_jd=True,
                    enrich_count=enrich_count,
                )
            else:
                ok, pct = compute_stack_relevance(list(job.get("skills") or []))
                job["is_relevant"] = ok
                job["relevant_percentage"] = pct

            if _passes_relevance_threshold(job):
                relevant_jobs.append(job)
                logger.info(
                    "Matched job %s/%s (%.1f%%): %s @ %s",
                    len(relevant_jobs),
                    cap,
                    float(job.get("relevant_percentage") or 0),
                    job.get("title"),
                    job.get("company"),
                )
                if len(relevant_jobs) >= cap:
                    break

        logger.info(
            "After page %s: %s relevant / %s cap (%s jobs scanned on page)",
            page_num,
            len(relevant_jobs),
            cap,
            len(new_jobs),
        )

    logger.info(
        "Pagination complete: kept %s jobs at %.0f%%+ relevance (scanned %s total, cap=%s)",
        len(relevant_jobs),
        RELEVANCE_MIN_PCT,
        total_scanned,
        cap,
    )
    return relevant_jobs


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
