"""Config-driven job source definitions (``config/job_sources.yaml``).

Adding a new source is: add one entry here, implement/point to an adapter,
register it — the LangGraph builder (``jobEngine/graph/builder.py``) creates
the node dynamically from this list. Nothing about the core graph changes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .settings import JOB_SOURCES_CONFIG_PATH

_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")
_ALLOWED_SCHEMES = {"http", "https"}


class JobSourceConfig(BaseModel):
    """One entry in ``config/job_sources.yaml``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=2, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    enabled: bool = True
    base_url: str
    search_urls: list[str] = Field(default_factory=list)
    adapter: str = Field(min_length=1, max_length=64)
    graph_node: str | None = None
    requires_auth: bool = False
    signin_url: str | None = None
    browser_profile: str = "default"
    timeout_seconds: int = Field(default=60, ge=1, le=600)
    max_jobs: int = Field(default=100, ge=1, le=1000)
    # Optional cap per ``search_urls`` entry (set per run from the request's
    # ``max_jobs``); ``max_jobs`` stays the cap for the whole source.
    max_jobs_per_url: int | None = Field(default=None, ge=1, le=1000)
    max_pages: int = Field(default=10, ge=1, le=200)
    rate_limit_delay_seconds: float = Field(default=1.0, ge=0.0, le=60.0)
    allowed_domains: list[str] = Field(default_factory=list)
    selectors: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not _ID_PATTERN.match(value):
            raise ValueError(
                f"source id {value!r} must be lowercase alnum/_/- and start with a letter"
            )
        return value

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        return _validate_url(value, field_name="base_url")

    @field_validator("search_urls")
    @classmethod
    def _validate_search_urls(cls, values: list[str]) -> list[str]:
        return [_validate_url(v, field_name="search_urls") for v in values]

    @field_validator("signin_url")
    @classmethod
    def _validate_signin_url(cls, value: str | None) -> str | None:
        return _validate_url(value, field_name="signin_url") if value else value

    @model_validator(mode="after")
    def _fill_defaults(self) -> "JobSourceConfig":
        if not self.graph_node:
            object.__setattr__(self, "graph_node", f"{self.id}_job_node")
        if not self.allowed_domains:
            domains = {urlparse(self.base_url).netloc.lower()}
            domains.update(urlparse(u).netloc.lower() for u in self.search_urls)
            object.__setattr__(self, "allowed_domains", sorted(d for d in domains if d))
        if self.enabled and not self.search_urls:
            raise ValueError(f"source {self.id!r} is enabled but has no search_urls")
        return self


def _validate_url(value: str, *, field_name: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"{field_name} {value!r} must use http(s), got scheme {parsed.scheme!r}"
        )
    if not parsed.netloc:
        raise ValueError(f"{field_name} {value!r} is missing a host")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Job source config not found at {path}. Copy/create it from the documented "
            "example in docs/job_engine/CONFIGURATION.md."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping at the top level")
    return data


def load_job_sources(path: Path | None = None) -> list[JobSourceConfig]:
    """Parse and validate ``config/job_sources.yaml``.

    Raises on malformed configuration (duplicate ids, bad URLs, missing
    fields) — configuration errors fail fast rather than surfacing later as a
    confusing graph-build error.
    """
    cfg_path = path or JOB_SOURCES_CONFIG_PATH
    data = _load_yaml(cfg_path)
    raw_sources = data.get("job_sources") or []
    if not isinstance(raw_sources, list):
        raise ValueError(f"{cfg_path}: 'job_sources' must be a list")

    sources: list[JobSourceConfig] = []
    seen_ids: set[str] = set()
    for i, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise ValueError(f"{cfg_path}: job_sources[{i}] must be a mapping")
        source = JobSourceConfig.model_validate(raw)
        if source.id in seen_ids:
            raise ValueError(f"{cfg_path}: duplicate source id {source.id!r}")
        seen_ids.add(source.id)
        sources.append(source)
    return sources


def enabled_sources(sources: list[JobSourceConfig]) -> list[JobSourceConfig]:
    return [s for s in sources if s.enabled]
