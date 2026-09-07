"""Guardrail 5 — content size limits (section 14/62): HTML size, description
length, job counts, and pagination depth. Every adapter and graph node that
touches raw external content goes through these before it moves further.
"""

from __future__ import annotations

import logging

from ..config.settings import get_settings

logger = logging.getLogger(__name__)


def enforce_html_size(html: str, *, source: str) -> str:
    settings = get_settings()
    limit = settings.max_html_bytes
    size = len(html.encode("utf-8", errors="ignore"))
    if size <= limit:
        return html
    logger.warning("Truncating oversized HTML from %s: %s bytes > %s limit", source, size, limit)
    # Truncate by characters (close enough for a hard safety cap; avoids re-encoding cost).
    return html[: limit // 2]


def enforce_description_size(description: str) -> str:
    settings = get_settings()
    limit = settings.max_description_chars
    if len(description) <= limit:
        return description
    return description[:limit]


def enforce_job_count(jobs: list, *, max_jobs: int, source: str) -> list:
    if len(jobs) <= max_jobs:
        return jobs
    logger.info("Capping %s jobs from %s to configured max_jobs=%s", len(jobs), source, max_jobs)
    return jobs[:max_jobs]


def pagination_depth_exceeded(page_num: int, *, max_pages: int) -> bool:
    settings = get_settings()
    hard_cap = min(max_pages, settings.max_pagination_depth)
    return page_num > hard_cap
