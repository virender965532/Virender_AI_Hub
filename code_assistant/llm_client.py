"""OpenAI wrapper: retries, backoff, timeouts, model fallback, circuit breaker, token accounting.

- One shared ``OpenAI`` client (SDK-level retries disabled; retries happen here).
- Only transient errors are retried, with exponential backoff + jitter.
  Auth / bad-request errors fail immediately.
- A context-length error calls the caller's ``on_context_overflow`` to shrink
  the messages and retries once.
- A structured response that does not parse into the schema is retried once
  with the validation error appended to the prompt.
- If the primary model keeps failing, ``CA_FALLBACK_MODEL`` is tried and the
  result is marked ``degraded``.
- A circuit breaker opens after N consecutive failed API calls and fails fast
  for a cooldown period.
- Every call checks and charges the per-request ``Trace`` budget.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from .config import CodeAssistantSettings, get_settings
from .observability import Trace
from .prompts import PARSE_REPAIR_TEMPLATE

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Messages = list[dict[str, Any]]

TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0
_EMBED_BATCH = 96
_EMBED_MAX_CHARS = 8000


class LLMUnavailableError(RuntimeError):
    """The model could not produce a valid response (after retries and fallback)."""


class CircuitOpenError(LLMUnavailableError):
    """The circuit breaker is open; calls fail fast until the cooldown passes."""


class ContextLengthError(LLMUnavailableError):
    """The prompt is too long for the model, even after one trim."""


class _ParseFailure(Exception):
    """Internal: the response did not match the schema."""


class _ContextOverflow(Exception):
    """Internal: the API rejected the prompt as too long."""


@dataclass(frozen=True)
class LLMResult:
    """A parsed structured response plus call metadata."""

    output: BaseModel
    model: str
    attempts: int
    usage: dict[str, int]
    degraded: bool = False


@dataclass(frozen=True)
class ChatResult:
    """A tool-calling chat response."""

    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    degraded: bool = False


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures; half-opens after ``cooldown`` seconds."""

    def __init__(self, threshold: int, cooldown: float, clock: Callable[[], float] = time.monotonic) -> None:
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

    def before_call(self) -> None:
        with self._lock:
            if self._opened_at is None:
                return
            remaining = self.cooldown - (self._clock() - self._opened_at)
            if remaining > 0:
                raise CircuitOpenError(f"OpenAI circuit breaker open; retry in {remaining:.0f}s")
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
                logger.error("OpenAI circuit breaker opened after %d consecutive failures", self._failures)


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        key: int(getattr(usage, key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _is_context_length_error(exc: openai.BadRequestError) -> bool:
    code = getattr(exc, "code", None) or ""
    return code == "context_length_exceeded" or "maximum context length" in str(exc).lower()


def supports_temperature(model: str) -> bool:
    """Reasoning models (o1/o3/o4, gpt-5) reject a custom temperature."""
    name = model.lower()
    return not (name.startswith(("o1", "o3", "o4", "gpt-5")))


class LLMClient:
    """Structured-output, tool-calling and embedding calls with bounded, transient-only retries."""

    def __init__(
        self,
        settings: CodeAssistantSettings | None = None,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._sleep = sleep
        self._lock = threading.Lock()
        self.breaker = breaker or CircuitBreaker(
            self.settings.circuit_failure_threshold, self.settings.circuit_cooldown_seconds
        )

    @property
    def available(self) -> bool:
        return self._client is not None or bool(self.settings.openai_api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            with self._lock:
                if self._client is None:
                    if not self.settings.openai_api_key:
                        raise LLMUnavailableError("OPENAI_API_KEY is not set")
                    self._client = openai.OpenAI(
                        api_key=self.settings.openai_api_key,
                        timeout=self.settings.request_timeout,
                        max_retries=0,
                    )
        return self._client

    def _backoff(self, attempt: int) -> float:
        delay = min(_BACKOFF_MAX_SECONDS, _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)))
        return delay + random.uniform(0, delay / 2)

    # --- core retry loop ----------------------------------------------------------------------

    def _with_retries(self, fn: Callable[[], Any], *, model: str, trace_id: str) -> tuple[Any, int]:
        """Run one API call with transient-only retries. Returns ``(response, attempts)``."""
        attempt = 0
        while True:
            attempt += 1
            self.breaker.before_call()
            try:
                response = fn()
            except TRANSIENT_ERRORS as exc:
                self.breaker.record_failure()
                if attempt > self.settings.max_retries:
                    logger.error(
                        "trace_id=%s model=%s transient error, giving up after %d attempts: %s",
                        trace_id, model, attempt, type(exc).__name__,
                    )
                    raise LLMUnavailableError(f"{type(exc).__name__} after {attempt} attempts") from exc
                delay = self._backoff(attempt)
                logger.warning(
                    "trace_id=%s model=%s transient error %s (attempt %d), retrying in %.2fs",
                    trace_id, model, type(exc).__name__, attempt, delay,
                )
                self._sleep(delay)
                continue
            except openai.BadRequestError as exc:
                if _is_context_length_error(exc):
                    raise _ContextOverflow(str(exc)) from exc
                self.breaker.record_failure()
                logger.error("trace_id=%s model=%s bad request: %s", trace_id, model, exc)
                raise LLMUnavailableError("BadRequestError") from exc
            except openai.APIError as exc:
                # Auth, permission, not found, ...: retrying cannot help.
                self.breaker.record_failure()
                logger.error("trace_id=%s model=%s non-retryable error: %s", trace_id, model, type(exc).__name__)
                raise LLMUnavailableError(type(exc).__name__) from exc
            self.breaker.record_success()
            return response, attempt

    def _with_overflow_retry(
        self,
        call: Callable[[Messages], Any],
        messages: Messages,
        on_context_overflow: Callable[[Messages], Messages] | None,
        *,
        model: str,
        trace_id: str,
    ) -> tuple[Any, int]:
        try:
            return self._with_retries(lambda: call(messages), model=model, trace_id=trace_id)
        except _ContextOverflow as exc:
            if on_context_overflow is None:
                raise ContextLengthError(str(exc)) from exc
            logger.warning("trace_id=%s model=%s context too long; trimming and retrying once", trace_id, model)
            trimmed = on_context_overflow(messages)
            try:
                return self._with_retries(lambda: call(trimmed), model=model, trace_id=trace_id)
            except _ContextOverflow as again:
                raise ContextLengthError(str(again)) from again

    def _with_fallback(self, run: Callable[[str], Any], *, model: str, trace: Trace, purpose: str) -> tuple[Any, str, bool]:
        """Run ``run(model)``; on terminal failure try the fallback model once. Returns ``(result, model, degraded)``."""
        try:
            return run(model), model, False
        except CircuitOpenError:
            raise
        except LLMUnavailableError as exc:
            fallback = self.settings.fallback_model
            if not fallback or fallback == model:
                raise
            trace.log("model_fallback", purpose=purpose, primary=model, fallback=fallback, error=str(exc))
            return run(fallback), fallback, True

    # --- public API ---------------------------------------------------------------------------

    def parse(
        self,
        *,
        model: str,
        messages: Messages,
        schema: type[T],
        trace: Trace,
        purpose: str,
        temperature: float = 0.0,
    ) -> LLMResult:
        """Call ``model`` and return its response parsed into ``schema``."""

        def call_once(use_model: str, msgs: Messages) -> tuple[T, dict[str, int], int]:
            kwargs: dict[str, Any] = {"model": use_model, "messages": msgs, "response_format": schema}
            if supports_temperature(use_model):
                kwargs["temperature"] = temperature

            def api_call(m: Messages) -> Any:
                try:
                    return self._get_client().chat.completions.parse(**{**kwargs, "messages": m})
                except (ValidationError, openai.LengthFinishReasonError) as exc:
                    raise _ParseFailure(str(exc)) from exc

            trace.check_budget()
            completion, attempts = self._with_overflow_retry(api_call, msgs, None, model=use_model, trace_id=trace.trace_id)
            usage = _usage_dict(completion)
            message = completion.choices[0].message
            if getattr(message, "refusal", None):
                raise _ParseFailure(f"model refused: {message.refusal}")
            parsed = message.parsed
            if parsed is None:
                raise _ParseFailure("empty parsed output")
            if not isinstance(parsed, schema):
                try:
                    parsed = schema.model_validate(parsed)
                except ValidationError as exc:
                    raise _ParseFailure(str(exc)) from exc
            return parsed, usage, attempts

        def run(use_model: str) -> tuple[T, dict[str, int], int]:
            try:
                return call_once(use_model, messages)
            except _ParseFailure as first:
                trace.log("parse_repair", model=use_model, purpose=purpose, error=str(first))
                repair = {"role": "user", "content": PARSE_REPAIR_TEMPLATE.format(error=str(first)[:1000])}
                try:
                    output, usage, attempts = call_once(use_model, [*messages, repair])
                except _ParseFailure as second:
                    raise LLMUnavailableError(f"structured output invalid after repair: {second}") from second
                return output, usage, attempts + 1

        (output, usage, attempts), used, degraded = self._with_fallback(run, model=model, trace=trace, purpose=purpose)
        trace.record_llm_call(model=used, usage=usage, purpose=purpose, attempts=attempts, degraded=degraded)
        return LLMResult(output=output, model=used, attempts=attempts, usage=usage, degraded=degraded)

    def chat(
        self,
        *,
        model: str,
        messages: Messages,
        trace: Trace,
        purpose: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.1,
        on_context_overflow: Callable[[Messages], Messages] | None = None,
    ) -> ChatResult:
        """Free-text / tool-calling completion."""

        def run(use_model: str) -> tuple[Any, int]:
            kwargs: dict[str, Any] = {"model": use_model}
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            if supports_temperature(use_model):
                kwargs["temperature"] = temperature
            trace.check_budget()
            return self._with_overflow_retry(
                lambda m: self._get_client().chat.completions.create(messages=m, **kwargs),
                messages, on_context_overflow, model=use_model, trace_id=trace.trace_id,
            )

        (completion, attempts), used, degraded = self._with_fallback(run, model=model, trace=trace, purpose=purpose)
        usage = _usage_dict(completion)
        trace.record_llm_call(model=used, usage=usage, purpose=purpose, attempts=attempts, degraded=degraded)
        message = completion.choices[0].message
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments or "{}"}
            for tc in (getattr(message, "tool_calls", None) or [])
        ]
        return ChatResult(content=message.content or "", tool_calls=tool_calls, model=used, usage=usage, degraded=degraded)

    def embed(self, texts: list[str], *, trace: Trace | None = None) -> list[list[float]]:
        """Embed ``texts`` in batches. Raises ``LLMUnavailableError`` on failure."""
        model = self.settings.embedding_model
        trace_id = trace.trace_id if trace else "-"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _EMBED_BATCH):
            batch = [t[:_EMBED_MAX_CHARS] or " " for t in texts[start:start + _EMBED_BATCH]]
            response, _ = self._with_retries(
                lambda b=batch: self._get_client().embeddings.create(model=model, input=b),
                model=model, trace_id=trace_id,
            )
            vectors.extend(item.embedding for item in response.data)
            if trace is not None:
                trace.record_embedding(model=model, tokens=_usage_dict(response).get("prompt_tokens", 0))
        return vectors


_shared_client: LLMClient | None = None
_shared_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """Process-wide LLM client (a single underlying OpenAI client and circuit breaker)."""
    global _shared_client
    if _shared_client is None:
        with _shared_lock:
            if _shared_client is None:
                _shared_client = LLMClient()
    return _shared_client
