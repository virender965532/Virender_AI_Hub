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
from ..utils.scrape_progress import update_progress

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


NAUKRI_JOB_KEYWORD_DEFAULT = _env_str("NAUKRI_JOB_KEYWORD", "javascript")
NAUKRI_JOB_AGE_DEFAULT = _env_str("NAUKRI_JOB_AGE", "3")
NAUKRI_CTC_FILTERS_DEFAULT = _env_str(
    "NAUKRI_CTC_FILTERS", "25to50,50to75,75to100,100to500"
)
RELEVANCE_MIN_PCT = _env_float("NAUKRI_JOB_RELEVANCE_MIN_PCT", 80.0)
_MAX_SRP_PAGES_DEFAULT = _env_int("NAUKRI_JOB_MAX_PAGES", 100)
NAUKRI_IS_EMAIL_REQUIRED = _env_bool("NAUKRI_IS_EMAIL_REQUIRED", False)

# Naukri ctcFilter query values with UI labels.
CTC_FILTER_OPTIONS: list[dict[str, str]] = [
    {"value": "0to3", "label": "0-3 Lakhs"},
    {"value": "3to6", "label": "3-6 Lakhs"},
    {"value": "6to10", "label": "6-10 Lakhs"},
    {"value": "10to15", "label": "10-15 Lakhs"},
    {"value": "15to25", "label": "15-25 Lakhs"},
    {"value": "25to50", "label": "25-50 Lakhs"},
    {"value": "50to75", "label": "50-75 Lakhs"},
    {"value": "75to100", "label": "75-100 Lakhs"},
    {"value": "100to500", "label": "1-5 Cr"},
]
_CTC_FILTER_VALUES = {opt["value"] for opt in CTC_FILTER_OPTIONS}


def parse_ctc_filters(raw: str | list[str] | None) -> list[str]:
    """Normalize CTC filter values from CSV string or list; drop unknowns."""
    if isinstance(raw, list):
        parts = [str(x).strip() for x in raw]
    else:
        parts = [p.strip() for p in str(raw or "").split(",")]
    out: list[str] = []
    for band in parts:
        if band and band in _CTC_FILTER_VALUES and band not in out:
            out.append(band)
    return out


def _env_no_of_jobs_default() -> int:
    raw = (os.getenv("NAUKRI_NO_OF_JOBS") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            logger.warning("Invalid NAUKRI_NO_OF_JOBS=%r; using 25", raw)
    return 25


def get_naukri_search_defaults() -> dict[str, Any]:
    """Defaults for the job-search UI (sourced from .env)."""
    return {
        "keyword": NAUKRI_JOB_KEYWORD_DEFAULT,
        "job_age": NAUKRI_JOB_AGE_DEFAULT,
        "ctc_filters": parse_ctc_filters(NAUKRI_CTC_FILTERS_DEFAULT),
        "no_of_jobs": _env_no_of_jobs_default(),
        "max_pages": _MAX_SRP_PAGES_DEFAULT,
        "relevance_min_pct": RELEVANCE_MIN_PCT,
        "ctc_options": CTC_FILTER_OPTIONS,
    }


def keyword_to_naukri_slug(keyword: str) -> str:
    """
    Map a free-text keyword to Naukri's jobs path slug.

    Examples:
      javascript -> javascript
      node.js    -> node-dot-js
      node js    -> node-js
      nodejs     -> nodejs
    """
    raw = (keyword or "").strip().lower()
    if not raw:
        raw = NAUKRI_JOB_KEYWORD_DEFAULT
    # Dots become the literal token "-dot-" (Naukri convention).
    slug = raw.replace(".", "-dot-")
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or NAUKRI_JOB_KEYWORD_DEFAULT


def build_jobs_base_url(keyword: str) -> str:
    """Build ``https://www.naukri.com/{slug}-jobs`` for the given keyword."""
    return f"https://www.naukri.com/{keyword_to_naukri_slug(keyword)}-jobs"


def _build_srp_query(
    keyword: str, *, job_age: str, ctc_filters: list[str]
) -> str:
    """Build SRP query string from keyword, job age, and CTC bands."""
    k = (keyword or "").strip() or NAUKRI_JOB_KEYWORD_DEFAULT
    age = (job_age or "").strip() or NAUKRI_JOB_AGE_DEFAULT
    params: list[tuple[str, str]] = [
        ("k", k),
        ("jobAge", age),
    ]
    for band in ctc_filters:
        band = band.strip()
        if band:
            params.append(("ctcFilter", band))
    return "?" + urlencode(params)


def _passes_relevance_threshold(job: JobRecord, *, min_pct: float) -> bool:
    """True when stack match score is at least ``min_pct``."""
    try:
        return float(job.get("relevant_percentage") or 0) >= min_pct
    except (TypeError, ValueError):
        return False


def _srp_page_url(
    page_num: int,
    *,
    keyword: str,
    job_age: str,
    ctc_filters: list[str],
) -> str:
    """Build paginated Naukri SRP URL; page 1 has no suffix, page 2+ uses ``-2``, ``-3``, etc."""
    base = build_jobs_base_url(keyword)
    query = _build_srp_query(keyword, job_age=job_age, ctc_filters=ctc_filters)
    if page_num <= 1:
        return base + query
    return f"{base}-{page_num}{query}"

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


async def _fetch_with_page(
    page: Page,
    *,
    enrich_jd: bool,
    keyword: str,
    job_age: str,
    ctc_filters: list[str],
    no_of_jobs: int,
    max_pages: int,
    relevance_min_pct: float,
    progress_id: str | None = None,
) -> list[JobRecord]:
    cap = max(1, int(no_of_jobs))
    pages_limit = max(1, int(max_pages))
    min_pct = float(relevance_min_pct)
    relevant_jobs: list[JobRecord] = []
    seen: set[str] = set()
    enrich_count = 0
    total_scanned = 0
    search_keyword = (keyword or "").strip() or NAUKRI_JOB_KEYWORD_DEFAULT
    age = (job_age or "").strip() or NAUKRI_JOB_AGE_DEFAULT
    bands = list(ctc_filters or [])

    update_progress(
        progress_id,
        status="running",
        phase="fetch",
        message="Scanning Naukri listings…",
        found=0,
        target=cap,
        scanned=0,
        page=0,
        keyword=search_keyword,
    )

    for page_num in range(1, pages_limit + 1):
        if len(relevant_jobs) >= cap:
            break

        listing_url = _srp_page_url(
            page_num,
            keyword=search_keyword,
            job_age=age,
            ctc_filters=bands,
        )
        logger.info("Fetching SRP page %s: %s", page_num, listing_url)
        update_progress(
            progress_id,
            status="running",
            phase="fetch",
            page=page_num,
            found=len(relevant_jobs),
            scanned=total_scanned,
            message=(
                f"Page {page_num}: found {len(relevant_jobs)} of {cap} matching jobs…"
            ),
        )
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

            if _passes_relevance_threshold(job, min_pct=min_pct):
                relevant_jobs.append(job)
                logger.info(
                    "Matched job %s/%s (%.1f%%): %s @ %s",
                    len(relevant_jobs),
                    cap,
                    float(job.get("relevant_percentage") or 0),
                    job.get("title"),
                    job.get("company"),
                )
                update_progress(
                    progress_id,
                    status="running",
                    phase="fetch",
                    page=page_num,
                    found=len(relevant_jobs),
                    scanned=total_scanned,
                    message=(
                        f"Found {len(relevant_jobs)} of {cap} matching jobs "
                        f"(page {page_num})…"
                    ),
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
        update_progress(
            progress_id,
            found=len(relevant_jobs),
            scanned=total_scanned,
            page=page_num,
            message=(
                f"Page {page_num} done: {len(relevant_jobs)} of {cap} matching jobs"
            ),
        )

    logger.info(
        "Pagination complete: kept %s jobs at %.0f%%+ relevance (scanned %s total, cap=%s)",
        len(relevant_jobs),
        min_pct,
        total_scanned,
        cap,
    )
    return relevant_jobs


def _resolve_fetch_config(state: WorkflowState) -> dict[str, Any]:
    """Merge workflow state overrides with .env defaults."""
    defaults = get_naukri_search_defaults()
    keyword = str(state.get("job_keyword") or defaults["keyword"]).strip()
    if not keyword:
        keyword = defaults["keyword"]

    job_age = str(state.get("job_age") or defaults["job_age"]).strip()
    if not job_age:
        job_age = defaults["job_age"]

    raw_ctc = state.get("ctc_filters")
    if raw_ctc is None:
        ctc_filters = list(defaults["ctc_filters"])
    else:
        # Empty list is valid (no CTC filter applied on Naukri SRP).
        ctc_filters = parse_ctc_filters(raw_ctc)

    try:
        raw_jobs = state.get("no_of_jobs")
        no_of_jobs = (
            int(defaults["no_of_jobs"])
            if raw_jobs is None
            else int(raw_jobs)
        )
    except (TypeError, ValueError):
        no_of_jobs = int(defaults["no_of_jobs"])
    no_of_jobs = max(1, no_of_jobs)

    try:
        raw_pages = state.get("max_pages")
        max_pages = (
            int(defaults["max_pages"])
            if raw_pages is None
            else int(raw_pages)
        )
    except (TypeError, ValueError):
        max_pages = int(defaults["max_pages"])
    max_pages = max(1, max_pages)

    try:
        relevance_min_pct = float(
            state.get("relevance_min_pct")
            if state.get("relevance_min_pct") is not None
            else defaults["relevance_min_pct"]
        )
    except (TypeError, ValueError):
        relevance_min_pct = float(defaults["relevance_min_pct"])

    return {
        "keyword": keyword,
        "job_age": job_age,
        "ctc_filters": ctc_filters,
        "no_of_jobs": no_of_jobs,
        "max_pages": max_pages,
        "relevance_min_pct": relevance_min_pct,
    }


async def fetch_jobs_node(state: WorkflowState) -> dict[str, Any]:
    errs = list(state.get("errors") or [])
    session = state.get("session")
    progress_id = str(state.get("progress_id") or "").strip() or None
    if session is None or not state.get("login_complete"):
        errs.append("fetch_jobs_node: missing session or login not complete.")
        update_progress(
            progress_id,
            status="error",
            phase="fetch",
            message=errs[-1],
            error=errs[-1],
        )
        return {"jobs": [], "fetch_complete": False, "errors": errs}

    enrich_jd = bool(state.get("enrich_jd", False))
    cfg = _resolve_fetch_config(state)
    logger.info(
        "Fetch config: keyword=%r age=%s ctc=%s no_of_jobs=%s max_pages=%s relevance_min=%s enrich_jd=%s",
        cfg["keyword"],
        cfg["job_age"],
        cfg["ctc_filters"],
        cfg["no_of_jobs"],
        cfg["max_pages"],
        cfg["relevance_min_pct"],
        enrich_jd,
    )
    page: Page = session.page
    try:
        jobs = await _fetch_with_page(
            page,
            enrich_jd=enrich_jd,
            keyword=cfg["keyword"],
            job_age=cfg["job_age"],
            ctc_filters=cfg["ctc_filters"],
            no_of_jobs=cfg["no_of_jobs"],
            max_pages=cfg["max_pages"],
            relevance_min_pct=cfg["relevance_min_pct"],
            progress_id=progress_id,
        )
        update_progress(
            progress_id,
            status="running",
            phase="display",
            found=len(jobs),
            message=f"Found {len(jobs)} matching jobs. Finishing up…",
        )
        return {
            "jobs": jobs,
            "fetch_complete": True,
            "errors": errs,
            "relevance_min_pct": cfg["relevance_min_pct"],
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("fetch_jobs_node failed")
        errs.append(str(e))
        update_progress(
            progress_id,
            status="error",
            phase="fetch",
            message=str(e),
            error=str(e),
        )
        return {"jobs": [], "fetch_complete": False, "errors": errs}
