"""Retry-with-backoff and circuit breakers shared by the LLM client and the data providers.

Every external dependency (OpenAI, each market-data provider, each news
provider) gets its own ``CircuitBreaker``: after N consecutive failures it
opens and calls fail fast for a cooldown, then one trial call is allowed.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0


class CircuitOpenError(RuntimeError):
    """The circuit breaker for a dependency is open; calls fail fast until the cooldown passes."""


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures; half-opens after ``cooldown`` seconds."""

    def __init__(
        self, name: str, threshold: int, cooldown: float, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.name = name
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None and self._clock() - self._opened_at < self.cooldown

    @property
    def state(self) -> str:
        return "open" if self.is_open else "closed"

    def before_call(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            remaining = self.cooldown - (self._clock() - self._opened_at)
            if remaining > 0:
                raise CircuitOpenError(f"{self.name} circuit breaker open; retry in {remaining:.0f}s")
            # Half-open: allow a trial call; one more failure re-opens immediately.
            self._opened_at = None
            self._failures = self.threshold - 1

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.threshold and self._opened_at is None:
                self._opened_at = self._clock()
                logger.error("%s circuit breaker opened after %d consecutive failures", self.name, self._failures)


def backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter for retry ``attempt`` (1-based)."""
    delay = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
    return delay + random.uniform(0, delay / 2)


def call_with_retry(
    fn: Callable[[], T],
    *,
    breaker: CircuitBreaker,
    is_transient: Callable[[Exception], bool],
    max_retries: int,
    sleep: Callable[[float], None] = time.sleep,
    label: str = "",
) -> T:
    """Run ``fn`` through ``breaker``, retrying only transient errors with backoff + jitter."""
    attempt = 0
    while True:
        attempt += 1
        breaker.before_call()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - classified below
            breaker.record_failure()
            if not is_transient(exc) or attempt > max_retries:
                raise
            delay = backoff_delay(attempt)
            logger.warning("%s transient error %s (attempt %d), retrying in %.2fs",
                           label or breaker.name, type(exc).__name__, attempt, delay)
            sleep(delay)
            continue
        breaker.record_success()
        return result
