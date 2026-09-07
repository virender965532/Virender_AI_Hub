"""DeduplicateJobs node (section 23).

Keys checked in priority order — the first one that matches an already-seen
job marks it a duplicate: (1) source + source_job_id, (2) canonical
application URL, (3) content hash, (4) normalized company+title+location.
First occurrence wins; source order in ``config/job_sources.yaml`` expresses
priority (list a company's own career page before an aggregator to prefer it).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _canonical_url(url: str) -> str:
    return (url or "").strip().rstrip("/").split("?")[0].lower()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _dedupe_keys(job: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    source_job_id = job.get("source_job_id")
    if source_job_id:
        keys.append(f"id:{job.get('source')}:{source_job_id}")
    app_url = _canonical_url(job.get("application_url") or "")
    if app_url:
        keys.append(f"url:{app_url}")
    content_hash = job.get("raw_text_hash")
    if content_hash:
        keys.append(f"hash:{content_hash}")
    combo = f"{_normalize(job.get('company', ''))}|{_normalize(job.get('title', ''))}|{_normalize(job.get('location') or '')}"
    keys.append(f"combo:{combo}")
    return keys


async def deduplication_node(state: dict[str, Any]) -> dict[str, Any]:
    validated = state.get("validated_jobs") or []
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []

    for job in validated:
        keys = _dedupe_keys(job)
        if any(k in seen for k in keys):
            duplicates.append(job)
            continue
        seen.update(keys)
        unique.append(job)

    logger.info(
        "deduplication_node: %s unique / %s duplicates removed (of %s)",
        len(unique), len(duplicates), len(validated),
    )
    return {"validated_jobs": unique, "duplicate_jobs": duplicates}
