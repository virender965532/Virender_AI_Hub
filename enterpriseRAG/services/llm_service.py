from __future__ import annotations

import json
import logging
import time
from typing import Any

from openai import OpenAI

from enterpriseRAG.config.settings import get_settings
from enterpriseRAG.services.observability import ObservabilityService, get_trace_id, utc_now_iso

logger = logging.getLogger(__name__)


class LLMService:
    """Centralized OpenAI client with observability hooks."""

    def __init__(self, observability: ObservabilityService | None = None) -> None:
        self.settings = get_settings()
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is missing in environment.")
        self.client = OpenAI(api_key=self.settings.openai_api_key)
        self.obs = observability

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        stage: str,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        started_at = utc_now_iso()
        t0 = time.perf_counter()
        trace_id = get_trace_id() or ""

        kwargs: dict[str, Any] = {
            "model": self.settings.model_name,
            "messages": messages,
            "temperature": temperature,
            "store": True,
            "extra_headers": {"X-Client-Trace-Id": trace_id},
            "metadata": {"trace_id": trace_id, "stage": stage},
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        content = (response.choices[0].message.content or "").strip()

        usage = response.usage
        log: dict[str, Any] = {
            "stage": stage,
            "model": self.settings.model_name,
            "elapsed_ms": elapsed_ms,
            "request_started_at": started_at,
            "response_received_at": utc_now_iso(),
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        if self.obs:
            self.obs.record_api_log(log)
        return content, log

    def chat_json(
        self,
        *,
        messages: list[dict[str, str]],
        stage: str,
        temperature: float = 0.1,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        content, log = self.chat(
            messages=messages, stage=stage, temperature=temperature, json_mode=True
        )
        try:
            return json.loads(content), log
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM at stage %s", stage)
            return {}, log

    def embed(self, text: str, stage: str = "embedding") -> tuple[list[float], dict[str, Any]]:
        started_at = utc_now_iso()
        t0 = time.perf_counter()
        trace_id = get_trace_id() or ""

        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=text,
            extra_headers={"X-Client-Trace-Id": trace_id},
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        vector = response.data[0].embedding
        usage = response.usage
        log: dict[str, Any] = {
            "stage": stage,
            "model": self.settings.embedding_model,
            "elapsed_ms": elapsed_ms,
            "request_started_at": started_at,
            "response_received_at": utc_now_iso(),
            "prompt_tokens": usage.total_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }
        if self.obs:
            self.obs.record_api_log(log)
        return vector, log
