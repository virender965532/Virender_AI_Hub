"""Structured logging helper (section 26).

The app's root logger is configured once in ``app.py`` with a plain-text
formatter shared by every module in the project — this module deliberately
does not change that global configuration. Instead, ``log_event`` emits one
JSON object per log line through the normal logger, so every important job
-engine event is machine-parseable (run_id/node/source/job_id/event/
duration_ms/status/error_type correlated) without touching how any other
module logs.

Never pass secrets/PII: this function does not redact anything itself —
callers must not pass passwords, cookies, tokens, session ids, or candidate
phone/email (section 26 "Do NOT log").
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any


def new_run_id() -> str:
    return f"job-search-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    run_id: str | None = None,
    node: str | None = None,
    source: str | None = None,
    job_id: str | None = None,
    duration_ms: float | None = None,
    status: str | None = None,
    error_type: str | None = None,
    trace_id: str | None = None,
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": logging.getLevelName(level),
        "event": event,
        "run_id": run_id,
        "node": node,
        "source": source,
        "job_id": job_id,
        "duration_ms": duration_ms,
        "status": status,
        "error_type": error_type,
        "trace_id": trace_id,
    }
    payload.update(extra)
    payload = {k: v for k, v in payload.items() if v is not None}
    logger.log(level, json.dumps(payload, default=str))
