"""JSON-safe serialization for job records (API responses and browser panel)."""

from __future__ import annotations

from typing import Any

from ..state import JobRecord


def _as_text(value: object, *, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, list):
        return ", ".join(str(item).strip() for item in value if str(item).strip())
    text = str(value).strip()
    return text if text else default


def job_record_to_dict(job: JobRecord) -> dict[str, Any]:
    """Convert a JobRecord to a JSON-serializable dict."""
    skills = job.get("skills") or []
    if not isinstance(skills, list):
        skills = [str(skills)]

    return {
        "title": _as_text(job.get("title"), default="—"),
        "company": _as_text(job.get("company"), default="—"),
        "experience": _as_text(job.get("experience"), default="—"),
        "location": _as_text(job.get("location"), default="—"),
        "salary": _as_text(job.get("salary"), default="Not disclosed"),
        "link": _as_text(job.get("link")),
        "skills": [str(s).strip() for s in skills if str(s).strip()],
        "is_remote": bool(job.get("is_remote")),
        "is_relevant": bool(job.get("is_relevant")),
        "relevant_percentage": float(job.get("relevant_percentage") or 0),
        "uploaded_at": _as_text(job.get("uploaded_at")),
        "posted": job.get("posted"),
    }
