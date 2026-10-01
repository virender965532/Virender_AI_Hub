"""Pydantic request/response schemas for the Flask API layer (section 36).

Every field is validated before a job search is ever started — see Guardrail
1 (schema validation) in ``docs/job_engine/SECURITY_REVIEW.md``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

_MAX_KEYWORDS = 20
_MAX_KEYWORD_CHARS = 200


class JobSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sources: list[str] | None = Field(default=None, max_length=20)
    keywords: list[str] = Field(default_factory=list)
    location: str | None = Field(default=None, max_length=200)
    remote: bool | None = None
    country: str | None = Field(default=None, max_length=100)
    city: str | None = Field(default=None, max_length=100)
    employment_type: str | None = Field(default=None, max_length=100)
    minimum_match_score: float | None = Field(default=None, ge=0, le=100)
    max_results: int = Field(default=50, ge=1, le=500)
    # How many jobs to scrape from *each* ``search_urls`` entry of every
    # selected source; all of them are combined, matched, shown and saved.
    # -1 or omitted = all (still bounded by each source's ``max_jobs`` in
    # config/job_sources.yaml).
    max_jobs: int | None = Field(default=None, ge=-1, le=1000)
    # Replaces the ``keywords=`` parameter of each source's search URL (e.g.
    # LinkedIn's boolean search). Not used as a filter.
    search_keywords: str | None = Field(default=None, max_length=2000)
    # Strict remote filter: keep only remote/hybrid jobs. (``remote`` only
    # widens a location filter to also accept remote jobs.)
    remote_only: bool = False

    @field_validator("max_jobs")
    @classmethod
    def _validate_max_jobs(cls, value: int | None) -> int | None:
        if value is None or value == -1:
            return None
        if value == 0:
            raise ValueError("max_jobs must be -1 (all) or a positive integer")
        return value

    @field_validator("search_keywords")
    @classmethod
    def _validate_search_keywords(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return value.strip()

    @field_validator("sources")
    @classmethod
    def _validate_sources(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = [v.strip().lower() for v in values if v and v.strip()]
        return cleaned or None

    @field_validator("keywords")
    @classmethod
    def _validate_keywords(cls, values: list[str]) -> list[str]:
        cleaned = [v.strip()[:_MAX_KEYWORD_CHARS] for v in values if v and v.strip()]
        if len(cleaned) > _MAX_KEYWORDS:
            raise ValueError(f"at most {_MAX_KEYWORDS} keywords are allowed")
        return cleaned


class SourceTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=64)
