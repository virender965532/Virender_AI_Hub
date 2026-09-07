"""Per-source circuit breaker (section 30).

A source with too many consecutive failures is skipped for a cooldown window
instead of being retried forever on every run. State is process-local
in-memory (mirrors the existing ``jobSearch/utils/scrape_progress.py``
in-memory-dict convention) — good enough for a single-process Flask app and
explicitly documented as a known limitation for multi-process deployments.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ..config.settings import get_settings


@dataclass
class _BreakerState:
    consecutive_failures: int = 0
    opened_until: float = 0.0


_lock = threading.Lock()
_STATE: dict[str, _BreakerState] = {}


def is_open(source_id: str) -> bool:
    with _lock:
        state = _STATE.get(source_id)
        if state is None:
            return False
        return time.time() < state.opened_until


def record_success(source_id: str) -> None:
    with _lock:
        _STATE[source_id] = _BreakerState()


def record_failure(source_id: str) -> None:
    settings = get_settings()
    with _lock:
        state = _STATE.setdefault(source_id, _BreakerState())
        state.consecutive_failures += 1
        if state.consecutive_failures >= settings.circuit_breaker_failure_threshold:
            state.opened_until = time.time() + settings.circuit_breaker_cooldown_seconds


def reset(source_id: str | None = None) -> None:
    with _lock:
        if source_id is None:
            _STATE.clear()
        else:
            _STATE.pop(source_id, None)
