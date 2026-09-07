"""Typed error records used across browser, network, parsing, LLM, config, and DB failures."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ErrorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(str, Enum):
    BROWSER = "browser"
    NETWORK = "network"
    PARSING = "parsing"
    LLM = "llm"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    GUARDRAIL = "guardrail"
    UNKNOWN = "unknown"


class WorkflowError(BaseModel):
    """A single structured error surfaced from anywhere in the graph."""

    model_config = ConfigDict(extra="forbid")

    error_code: str
    error_type: ErrorType
    message: str
    source: str | None = None
    node: str | None = None
    retryable: bool = False
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    run_id: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SourceErrorStatus(str, Enum):
    TIMEOUT = "timeout"
    NAVIGATION_ERROR = "navigation_error"
    AUTH_REQUIRED = "auth_required"
    ANTI_BOT_BLOCKED = "anti_bot_blocked"
    SELECTOR_MISSING = "selector_missing"
    RATE_LIMITED = "rate_limited"
    HTTP_ERROR = "http_error"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


class SourceError(BaseModel):
    """A source-scoped failure; never fails the overall run (source isolation)."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    status: SourceErrorStatus
    message: str
    retryable: bool = False
    node: str | None = None
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


SourceRunStatus = Literal["success", "partial", "failed"]
