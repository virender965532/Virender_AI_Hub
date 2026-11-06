"""Tracing, structured logging, per-request metrics and the per-request LLM budget.

Every request gets a ``Trace``. Route decisions, data fetches (provider,
latency, cache hit), LLM calls (model, tokens, cost), worker results, critic
verdicts and the final status are logged as one JSON line each, tagged with the
trace_id. Events can also be forwarded to a listener (SSE stream / Gradio UI).

The trace of the running request is kept in a ``ContextVar`` so the data layer
can log fetches without threading a trace through every call; ``asyncio.to_thread``
copies the context into worker threads. Strings are redacted before logging so
an API key can never reach the logs.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import threading
import time
import uuid
from collections import Counter
from typing import Any, Callable

from .prompts import PROMPT_VERSION

logger = logging.getLogger("stock_finder.trace")

Listener = Callable[[str, dict[str, Any]], None]

# USD per 1M tokens (input, output). Unknown models are costed at 0.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

_MAX_LOGGED_FIELD = 2000
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|apikey|token|password|secret)=([^&\s]+)"),
    re.compile(r"mongodb(\+srv)?://[^@\s]+@"),
)

_current: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("stock_finder_trace", default=None)


class BudgetExceededError(RuntimeError):
    """The request hit SF_MAX_LLM_CALLS."""


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def redact(text: str) -> str:
    """Mask API keys, ``key=...`` query parameters and Mongo credentials."""
    text = _SECRET_PATTERNS[0].sub("sk-***", text)
    text = _SECRET_PATTERNS[1].sub(lambda m: f"{m.group(1)}=***", text)
    return _SECRET_PATTERNS[2].sub("mongodb://***@", text)


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Best-effort USD cost; dated model ids (``gpt-4o-2024-08-06``) use their base price."""
    price = MODEL_PRICES.get(model)
    if price is None:
        base = max((name for name in MODEL_PRICES if model.startswith(name)), key=len, default=None)
        price = MODEL_PRICES.get(base or "", (0.0, 0.0))
    return (prompt_tokens * price[0] + completion_tokens * price[1]) / 1_000_000


def _clip(value: Any) -> Any:
    if isinstance(value, str):
        value = redact(value)
        return value if len(value) <= _MAX_LOGGED_FIELD else value[:_MAX_LOGGED_FIELD] + "...[truncated]"
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clip(v) for v in value[:50]]
    return value


def current_trace() -> "Trace | None":
    return _current.get()


def log_data_fetch(*, provider: str, kind: str, key: str, latency_ms: int, cache_hit: bool, ok: bool = True,
                   error: str | None = None) -> None:
    """Log a data fetch on the current trace (or plainly when no request is active)."""
    trace = _current.get()
    fields = {"provider": provider, "kind": kind, "key": key, "latency_ms": latency_ms, "cache_hit": cache_hit, "ok": ok}
    if error:
        fields["error"] = error
    if trace is not None:
        trace.record_fetch(cache_hit=cache_hit, ok=ok)
        trace.log("data_fetch", **fields)
    else:
        logger.debug(json.dumps({"event": "data_fetch", **_clip(fields)}, default=str))


class Trace:
    """Per-request trace: id, metrics, budget and event fan-out. Thread-safe."""

    def __init__(
        self,
        *,
        max_llm_calls: int,
        listener: Listener | None = None,
        trace_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.trace_id = trace_id or new_trace_id()
        self.max_llm_calls = max_llm_calls
        self.listener = listener
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self.models_used: Counter[str] = Counter()
        self.fetches = 0
        self.cache_hits = 0
        self.fetch_errors = 0
        self.degraded = False
        self.budget_exceeded = False
        self.status: str | None = None
        self._token: contextvars.Token | None = None

    # --- context ------------------------------------------------------------------------------

    def __enter__(self) -> "Trace":
        self._token = _current.set(self)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None

    # --- budget -------------------------------------------------------------------------------

    def check_budget(self) -> None:
        """Raise ``BudgetExceededError`` if another LLM call would exceed the request budget."""
        with self._lock:
            if self.llm_calls >= self.max_llm_calls:
                self.budget_exceeded = True
                raise BudgetExceededError(f"LLM call budget reached ({self.max_llm_calls} calls)")

    @property
    def remaining_calls(self) -> int:
        with self._lock:
            return max(0, self.max_llm_calls - self.llm_calls)

    def record_llm_call(self, *, model: str, usage: dict[str, int], purpose: str, attempts: int, degraded: bool) -> None:
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        cost = estimate_cost(model, prompt, completion)
        with self._lock:
            self.llm_calls += 1
            self.tokens_in += prompt
            self.tokens_out += completion
            self.cost_usd += cost
            self.models_used[model] += 1
            if degraded:
                self.degraded = True
        self.log(
            "llm_call", model=model, purpose=purpose, prompt_version=PROMPT_VERSION, prompt_tokens=prompt,
            completion_tokens=completion, cost_usd=round(cost, 6), attempts=attempts, degraded=degraded,
        )

    def record_embedding(self, *, model: str, tokens: int) -> None:
        cost = estimate_cost(model, tokens, 0)
        with self._lock:
            self.tokens_in += tokens
            self.cost_usd += cost
        self.log("embedding", model=model, tokens=tokens, cost_usd=round(cost, 6))

    def record_fetch(self, *, cache_hit: bool, ok: bool) -> None:
        with self._lock:
            self.fetches += 1
            self.cache_hits += int(cache_hit)
            self.fetch_errors += int(not ok)

    # --- events -------------------------------------------------------------------------------

    def log(self, event: str, **fields: Any) -> None:
        """Write one structured JSON log line (redacted, size-capped)."""
        record = {"trace_id": self.trace_id, "event": event, **_clip(fields)}
        logger.info(json.dumps(record, default=str, ensure_ascii=False))

    def emit(self, event: str, data: dict[str, Any], *, log: bool = True) -> None:
        """Log an event and forward it to the listener. Listener errors never propagate."""
        if log:
            self.log(event, **data)
        if self.listener is not None:
            try:
                self.listener(event, data)
            except Exception:  # noqa: BLE001 - a broken client must not break the request
                logger.debug("trace_id=%s listener failed for %s", self.trace_id, event, exc_info=True)

    def progress(self, stage: str, message: str, **extra: Any) -> None:
        """Human-readable progress event for the SSE stream / UI."""
        self.emit("progress", {"stage": stage, "message": message, **extra})

    # --- summary ------------------------------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "trace_id": self.trace_id,
                "latency_ms": int((self._clock() - self._started) * 1000),
                "llm_calls": self.llm_calls,
                "max_llm_calls": self.max_llm_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "estimated_cost_usd": round(self.cost_usd, 5),
                "models_used": dict(self.models_used),
                "data_fetches": self.fetches,
                "cache_hits": self.cache_hits,
                "fetch_errors": self.fetch_errors,
                "degraded": self.degraded,
                "budget_exceeded": self.budget_exceeded,
                "status": self.status,
            }

    def finish(self, status: str) -> dict[str, Any]:
        """Record the final status and log the request metrics."""
        self.status = status
        metrics = self.metrics()
        self.log("request_complete", **metrics)
        return metrics
