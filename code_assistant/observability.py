"""Tracing, structured logging, per-request metrics and the per-request budget.

Every request gets a ``Trace``. Each LLM call, tool call, route decision, plan,
test run, review verdict and approval is logged as one JSON line through the
app's standard ``logging`` setup, tagged with the trace_id. Events can also be
forwarded to a listener (the SSE stream / Gradio UI). Payloads are redacted
before logging so no secret ever reaches the logs.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import Counter
from typing import Any, Callable

from .prompts import PROMPT_VERSION
from .workspace.sandbox import redact

logger = logging.getLogger("code_assistant.trace")

Listener = Callable[[str, dict[str, Any]], None]

# USD per 1M tokens (input, output). Unknown models are costed at 0 and logged as such.
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


class BudgetExceededError(RuntimeError):
    """The request hit CA_MAX_LLM_CALLS or CA_MAX_TOKENS_REQUEST."""


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


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


class Trace:
    """Per-request trace: id, metrics, budget and event fan-out. Thread-safe."""

    def __init__(
        self,
        *,
        max_llm_calls: int,
        max_tokens: int,
        listener: Listener | None = None,
        trace_id: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.trace_id = trace_id or new_trace_id()
        self.max_llm_calls = max_llm_calls
        self.max_tokens = max_tokens
        self.listener = listener
        self._clock = clock
        self._started = clock()
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.cost_usd = 0.0
        self.tools_used: Counter[str] = Counter()
        self.fix_attempts = 0
        self.review_verdict: str | None = None
        self.degraded = False
        self.status: str | None = None

    # --- budget -------------------------------------------------------------------------------

    def check_budget(self) -> None:
        """Raise ``BudgetExceededError`` if another LLM call would exceed the request budget."""
        with self._lock:
            if self.llm_calls >= self.max_llm_calls:
                raise BudgetExceededError(f"LLM call budget reached ({self.max_llm_calls} calls)")
            if self.tokens_in + self.tokens_out >= self.max_tokens:
                raise BudgetExceededError(f"token budget reached ({self.max_tokens} tokens)")

    def record_llm_call(self, *, model: str, usage: dict[str, int], purpose: str, attempts: int, degraded: bool) -> None:
        prompt = int(usage.get("prompt_tokens", 0))
        completion = int(usage.get("completion_tokens", 0))
        cost = estimate_cost(model, prompt, completion)
        with self._lock:
            self.llm_calls += 1
            self.tokens_in += prompt
            self.tokens_out += completion
            self.cost_usd += cost
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

    def record_tool(self, name: str) -> None:
        with self._lock:
            self.tools_used[name] += 1

    # --- events -------------------------------------------------------------------------------

    def log(self, event: str, **fields: Any) -> None:
        """Write one structured JSON log line (redacted, size-capped)."""
        record = {"trace_id": self.trace_id, "event": event, **_clip(fields)}
        logger.info(json.dumps(record, default=str, ensure_ascii=False))

    def emit(self, event: str, data: dict[str, Any], *, log: bool = True) -> None:
        """Log an event and forward it to the listener (SSE / UI). Listener errors never propagate."""
        if log:
            self.log(event, **data)
        if self.listener is not None:
            try:
                self.listener(event, _clip(data) if event != "diff" else data)
            except Exception:  # noqa: BLE001 - a broken client must not break the request
                logger.debug("trace_id=%s listener failed for %s", self.trace_id, event, exc_info=True)

    # --- summary ------------------------------------------------------------------------------

    def metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "trace_id": self.trace_id,
                "latency_ms": int((self._clock() - self._started) * 1000),
                "llm_calls": self.llm_calls,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "estimated_cost_usd": round(self.cost_usd, 5),
                "tools_used": dict(self.tools_used),
                "fix_attempts": self.fix_attempts,
                "review_verdict": self.review_verdict,
                "degraded": self.degraded,
                "status": self.status,
            }

    def finish(self, status: str) -> dict[str, Any]:
        """Record the final status and log the request metrics."""
        self.status = status
        metrics = self.metrics()
        self.log("request_complete", **metrics)
        return metrics
