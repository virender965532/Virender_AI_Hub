"""The contract every job source adapter implements (section 32).

The core graph only ever talks to ``JobSourceAdapter.discover_jobs`` and only
ever receives a ``SourceDiscoveryResult`` back — website-specific navigation,
selectors, retries, and pagination all live inside the adapter.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field

from ..browser.manager import BrowserManager
from ..config.settings import EngineSettings
from ..config.sources import JobSourceConfig
from ..models.errors import SourceError, SourceRunStatus
from ..models.job import RawJob


@dataclass
class SourceExecutionContext:
    run_id: str
    config: JobSourceConfig
    browser_manager: BrowserManager
    settings: EngineSettings
    # source_job_id -> description of jobs already in the saved-jobs store with
    # an LLM analysis. Adapters may use it instead of opening the job again;
    # the matching node reuses the stored analysis for these jobs anyway.
    known_descriptions: dict[str, str] = field(default_factory=dict)


class SourceDiscoveryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    status: SourceRunStatus
    jobs: list[RawJob] = Field(default_factory=list)
    pages_processed: int = 0
    duration_ms: int = 0
    errors: list[SourceError] = Field(default_factory=list)


class JobSourceAdapter(ABC):
    """Base class for all job source adapters. Subclasses must not raise out
    of ``discover_jobs`` — every failure becomes a ``SourceError`` inside the
    returned result so one broken source never fails the whole run.
    """

    @abstractmethod
    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        raise NotImplementedError


class TimedDiscovery:
    """Small helper so adapters don't hand-roll perf_counter bookkeeping."""

    def __init__(self, source_id: str) -> None:
        self.source_id = source_id
        self._t0 = time.perf_counter()
        self.pages_processed = 0
        self.jobs: list[RawJob] = []
        self.errors: list[SourceError] = []

    def elapsed_ms(self) -> int:
        return int((time.perf_counter() - self._t0) * 1000)

    def to_result(self) -> SourceDiscoveryResult:
        if self.jobs and self.errors:
            status: SourceRunStatus = "partial"
        elif self.jobs:
            status = "success"
        else:
            status = "failed"
        return SourceDiscoveryResult(
            source_id=self.source_id,
            status=status,
            jobs=self.jobs,
            pages_processed=self.pages_processed,
            duration_ms=self.elapsed_ms(),
            errors=self.errors,
        )
