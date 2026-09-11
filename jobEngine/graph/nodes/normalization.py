"""NormalizeJobs node: RawJob -> validated Job (section 11).

Every job that fails ``Job`` schema validation (missing title/company,
invalid application URL, disallowed domain) is dropped here with a recorded
``WorkflowError`` rather than propagated as a loosely-typed dict.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import ValidationError

from ...guardrails.url_guardrails import check_url
from ...models.errors import ErrorSeverity, ErrorType, WorkflowError
from ...models.job import Job, RemoteType, content_hash

logger = logging.getLogger(__name__)

_REMOTE_RE = re.compile(r"\b(remote|work from home|wfh|anywhere)\b", re.IGNORECASE)
_HYBRID_RE = re.compile(r"\bhybrid\b", re.IGNORECASE)


def _infer_remote_type(location: str | None, description: str, *, adapter_hint: str | None = None) -> RemoteType:
    """Resolve a job's remote type.

    An adapter can supply an authoritative hint (e.g. Instinctools' explicit
    "#Office"/"#Remote" card tags, see ``jobEngine/sources/instinctools/adapter.py``)
    which is trusted over regex inference from free-text location/description —
    text-based guessing is a fallback for sources with no such explicit signal.
    """
    if adapter_hint:
        try:
            return RemoteType(adapter_hint)
        except ValueError:
            pass

    blob = f"{location or ''} {description}"
    if _HYBRID_RE.search(blob):
        return RemoteType.HYBRID
    if _REMOTE_RE.search(blob):
        return RemoteType.REMOTE
    if location:
        return RemoteType.ONSITE
    return RemoteType.UNKNOWN


async def normalization_node(state: dict[str, Any]) -> dict[str, Any]:
    run_id = state.get("run_id")
    raw_jobs = state.get("discovered_jobs") or []
    source_configs = {c["id"]: c for c in state.get("source_configs") or []}

    normalized: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for raw in raw_jobs:
        source_id = raw.get("source", "")
        allowed_domains = (source_configs.get(source_id) or {}).get("allowed_domains") or []

        app_url = raw.get("application_url") or ""
        src_url = raw.get("source_url") or app_url
        if allowed_domains:
            for url in (app_url, src_url):
                result = check_url(url, allowed_domains=allowed_domains, resolve_dns=False)
                if not result.ok:
                    errors.append(
                        WorkflowError(
                            error_code="job_url_rejected",
                            error_type=ErrorType.GUARDRAIL,
                            message=f"job URL {url!r} rejected: {result.reason}",
                            source=source_id,
                            node="normalization",
                            severity=ErrorSeverity.LOW,
                            run_id=run_id,
                        ).model_dump(mode="json")
                    )
                    app_url = None
                    break

        if not app_url:
            continue

        description = raw.get("description") or ""
        job_id = content_hash(
            source=source_id,
            title=raw.get("title") or "",
            company=raw.get("company") or "",
            description=description,
        )

        try:
            job = Job(
                job_id=job_id,
                source=source_id,
                source_job_id=raw.get("source_job_id"),
                title=raw.get("title") or "",
                company=raw.get("company") or "",
                location=raw.get("location"),
                employment_type=raw.get("employment_type"),
                experience_required=raw.get("experience_required"),
                description=description,
                requirements=raw.get("requirements") or [],
                responsibilities=raw.get("responsibilities") or [],
                skills=raw.get("skills") or [],
                salary=raw.get("salary"),
                posted_date=None,
                application_url=app_url,
                source_url=src_url,
                remote_type=_infer_remote_type(
                    raw.get("location"), description, adapter_hint=raw.get("remote_type")
                ),
                raw_text_hash=job_id,
                discovered_at=raw.get("discovered_at"),
            )
        except ValidationError as exc:
            errors.append(
                WorkflowError(
                    error_code="job_normalization_failed",
                    error_type=ErrorType.VALIDATION,
                    message=str(exc),
                    source=source_id,
                    node="normalization",
                    severity=ErrorSeverity.LOW,
                    run_id=run_id,
                ).model_dump(mode="json")
            )
            continue

        normalized.append(job.model_dump(mode="json"))

    logger.info("normalization_node: %s/%s raw jobs normalized", len(normalized), len(raw_jobs))
    return {"normalized_jobs": normalized, "errors": errors}
