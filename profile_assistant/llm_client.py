"""Thin OpenAI wrapper: structured outputs, retries, backoff and timeouts.

- One shared ``OpenAI`` client (SDK-level retries disabled; retries happen here).
- Only transient errors are retried, with exponential backoff + jitter.
- Auth / bad-request errors fail immediately.
- A response that does not parse into the schema is retried once with the
  validation error appended to the prompt.
- Any terminal failure is raised as ``LLMUnavailableError``.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from .config import ProfileAssistantSettings, get_settings
from .prompts import PARSE_REPAIR_TEMPLATE

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 8.0


class LLMUnavailableError(RuntimeError):
    """The model could not produce a valid response (after retries, if any)."""


class _ParseFailure(Exception):
    """Internal: the response did not match the schema."""


@dataclass(frozen=True)
class LLMResult:
    """A parsed model response plus call metadata for logging."""

    output: BaseModel
    model: str
    attempts: int
    usage: dict[str, int]


def _usage_dict(completion: Any) -> dict[str, int]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    return {
        key: int(getattr(usage, key, 0) or 0)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


class LLMClient:
    """Structured-output calls against OpenAI with bounded, transient-only retries."""

    def __init__(
        self,
        settings: ProfileAssistantSettings | None = None,
        *,
        client: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client
        self._sleep = sleep
        self._lock = threading.Lock()

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

    def _call_once(self, model: str, messages: list[dict[str, str]], schema: type[T]) -> tuple[T, dict[str, int]]:
        try:
            completion = self._get_client().chat.completions.parse(
                model=model,
                messages=messages,
                response_format=schema,
                temperature=0,
                timeout=self.settings.request_timeout,
            )
        except (ValidationError, openai.LengthFinishReasonError) as exc:
            raise _ParseFailure(str(exc)) from exc

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
        return parsed, _usage_dict(completion)

    def _call_with_retries(
        self, model: str, messages: list[dict[str, str]], schema: type[T], *, trace_id: str
    ) -> tuple[T, dict[str, int], int]:
        attempt = 0
        while True:
            attempt += 1
            try:
                output, usage = self._call_once(model, messages, schema)
                return output, usage, attempt
            except TRANSIENT_ERRORS as exc:
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
            except openai.APIError as exc:
                # Auth, permission, bad request, not found, ...: retrying cannot help.
                logger.error("trace_id=%s model=%s non-retryable error: %s", trace_id, model, type(exc).__name__)
                raise LLMUnavailableError(type(exc).__name__) from exc

    def parse(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: type[T],
        trace_id: str,
    ) -> LLMResult:
        """Call ``model`` and return its response parsed into ``schema``."""
        try:
            output, usage, attempts = self._call_with_retries(model, messages, schema, trace_id=trace_id)
        except _ParseFailure as first:
            logger.warning("trace_id=%s model=%s structured output invalid, repairing once", trace_id, model)
            repair = {"role": "user", "content": PARSE_REPAIR_TEMPLATE.format(error=str(first)[:1000])}
            try:
                output, usage, attempts = self._call_with_retries(
                    model, [*messages, repair], schema, trace_id=trace_id
                )
            except _ParseFailure as second:
                raise LLMUnavailableError(f"structured output invalid after repair: {second}") from second
            attempts += 1
        return LLMResult(output=output, model=model, attempts=attempts, usage=usage)


_shared_client: LLMClient | None = None
_shared_lock = threading.Lock()


def get_llm_client() -> LLMClient:
    """Process-wide LLM client (a single underlying OpenAI client instance)."""
    global _shared_client
    if _shared_client is None:
        with _shared_lock:
            if _shared_client is None:
                _shared_client = LLMClient()
    return _shared_client
