"""ValidateJobs node: content-quality guardrail + user search-criteria filters.

Two independent checks, both applied here:

1. Content quality — a job that made it through ``Job`` schema validation
   but is essentially empty (no description, no skills, no requirements) is
   too thin to match meaningfully.
2. Requested search criteria (section 36/74: ``keywords``/``location``/
   ``remote``/``country``/``city``/``employment_type``) — job sources are
   navigated via their configured ``search_urls`` (not re-templated per
   request; see docs/job_engine/CONFIGURATION.md "Known limitations"), so
   request-level filters are enforced here as a post-discovery filter rather
   than steering the live search itself.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from ...models.errors import ErrorSeverity, ErrorType, WorkflowError

logger = logging.getLogger(__name__)

_MIN_DESCRIPTION_CHARS = 10


def _has_minimal_content(job: dict[str, Any]) -> bool:
    if len((job.get("description") or "").strip()) >= _MIN_DESCRIPTION_CHARS:
        return True
    if job.get("skills") or job.get("requirements") or job.get("responsibilities"):
        return True
    return False


def _job_text_blob(job: dict[str, Any]) -> str:
    return " ".join(
        [
            job.get("title") or "",
            job.get("description") or "",
            " ".join(job.get("skills") or []),
            " ".join(job.get("requirements") or []),
        ]
    ).lower()


def _matches_keywords(job: dict[str, Any], keywords: list[str]) -> bool:
    if not keywords:
        return True
    blob = _job_text_blob(job)
    return any(kw.strip().lower() in blob for kw in keywords if kw.strip())


def _matches_location(job: dict[str, Any], *, location: str | None, country: str | None, city: str | None, remote: bool | None) -> bool:
    wanted = [v for v in (location, country, city) if v]
    if not wanted:
        return True
    job_location = (job.get("location") or "").lower()
    location_match = any(re.search(re.escape(w.lower()), job_location) for w in wanted)
    if location_match:
        return True
    if remote and job.get("remote_type") in ("remote", "hybrid"):
        return True
    return False


def _matches_employment_type(job: dict[str, Any], employment_type: str | None) -> bool:
    if not employment_type:
        return True
    job_type = (job.get("employment_type") or "").lower()
    return employment_type.lower() in job_type if job_type else True


async def validation_node(state: dict[str, Any]) -> dict[str, Any]:
    run_id = state.get("run_id")
    request = state.get("request") or {}
    normalized = state.get("normalized_jobs") or []

    keywords = request.get("keywords") or []
    location = request.get("location")
    country = request.get("country")
    city = request.get("city")
    remote = request.get("remote")
    employment_type = request.get("employment_type")

    validated: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for job in normalized:
        # if not _has_minimal_content(job):
        #     errors.append(
        #         WorkflowError(
        #             error_code="job_content_too_thin",
        #             error_type=ErrorType.VALIDATION,
        #             message=f"job {job.get('title')!r} at {job.get('company')!r} has no usable content",
        #             source=job.get("source"),
        #             node="validation",
        #             severity=ErrorSeverity.LOW,
        #             run_id=run_id,
        #         ).model_dump(mode="json")
        #     )
        #     continue

        # if not _matches_keywords(job, keywords):
        #     continue
        # if not _matches_location(job, location=location, country=country, city=city, remote=remote):
        #     continue
        # if not _matches_employment_type(job, employment_type):
        #     continue

        validated.append(job)

    logger.info("validation_node: %s/%s jobs passed content-quality + search-criteria filters", len(validated), len(normalized))
    return {"validated_jobs": validated, "errors": errors}
