"""LLM-based grounded matching (sections 16/20/21/22).

Uses ``ChatOpenAI.with_structured_output`` so the model's response is always
parsed into ``JobMatchAnalysis`` or rejected — there is no free-form output
path. On a malformed/invalid response the call is retried once with an
explicit repair instruction; if it still fails, the caller treats this job
as "LLM analysis unavailable" (rule + embedding scores still apply) rather
than crashing the run (section 13: parse -> validate -> repair -> reject).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from ..config.settings import EngineSettings, get_settings
from ..guardrails.prompt_injection import detect_suspicious_patterns, wrap_untrusted_content
from ..models.job import Job, JobMatchAnalysis
from ..models.profile import CandidateProfile

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


class LLMCallMetrics:
    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.retries = 0
        self.calls = 0
        self.estimated_cost_usd = 0.0

    def merge(self, other: "LLMCallMetrics") -> None:
        """Add another call's counters (concurrent calls each get their own instance)."""
        for name in ("prompt_tokens", "completion_tokens", "total_tokens", "retries", "calls", "estimated_cost_usd"):
            setattr(self, name, getattr(self, name) + getattr(other, name))


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def _extract_usage(raw_message) -> tuple[int, int, int]:
    usage = getattr(raw_message, "usage_metadata", None)
    if usage:
        return (
            int(usage.get("input_tokens", 0) or 0),
            int(usage.get("output_tokens", 0) or 0),
            int(usage.get("total_tokens", 0) or 0),
        )
    meta = getattr(raw_message, "response_metadata", {}) or {}
    token_usage = meta.get("token_usage") or {}
    return (
        int(token_usage.get("prompt_tokens", 0) or 0),
        int(token_usage.get("completion_tokens", 0) or 0),
        int(token_usage.get("total_tokens", 0) or 0),
    )


class LLMMatcher:
    def __init__(self, settings: EngineSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._structured_llm = None
        self._system_template = _load_prompt("job_matching_system.txt")
        self._user_template = _load_prompt("job_matching_user.txt")

    def available(self) -> bool:
        return self.settings.llm_enabled and bool(self.settings.openai_api_key)

    def _get_structured_llm(self):
        if self._structured_llm is None:
            from langchain_openai import ChatOpenAI

            base = ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout,
                temperature=0.1,
            )
            self._structured_llm = base.with_structured_output(JobMatchAnalysis, include_raw=True)
        return self._structured_llm

    def _build_messages(self, *, profile: CandidateProfile, job: Job, repair_hint: str = "") -> list[dict[str, str]]:
        system = self._system_template.format(prompt_version=self.settings.prompt_version)
        if repair_hint:
            system = f"{system}\n\nIMPORTANT: {repair_hint}"

        job_text = "\n".join(
            [
                job.description,
                "Requirements: " + "; ".join(job.requirements),
                "Responsibilities: " + "; ".join(job.responsibilities),
                "Skills listed: " + ", ".join(job.skills),
                f"Experience required: {job.experience_required or 'not specified'}",
            ]
        )
        user = self._user_template.format(
            candidate_profile=profile.to_llm_context(),
            job_title=job.title,
            job_company=job.company,
            job_location=job.location or "not specified",
            wrapped_job_content=wrap_untrusted_content(job_text),
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def analyze(
        self, *, profile: CandidateProfile, job: Job, metrics: LLMCallMetrics
    ) -> JobMatchAnalysis | None:
        if not self.available():
            return None

        suspicious = detect_suspicious_patterns(job.description)
        if suspicious:
            logger.info(
                "job=%s: prompt-injection-like pattern(s) detected in job description "
                "(treated as inert data): %s",
                job.job_id if hasattr(job, "job_id") else job.title,
                suspicious,
            )

        repair_hint = ""
        for attempt in range(1 + self.settings.openai_max_retries):
            messages = self._build_messages(profile=profile, job=job, repair_hint=repair_hint)
            metrics.calls += 1
            t0 = time.perf_counter()
            try:
                result = self._get_structured_llm().invoke(messages)
            except Exception:  # noqa: BLE001
                logger.exception("LLM matching call failed (attempt %s)", attempt + 1)
                metrics.retries += 1
                repair_hint = "Your previous response could not be processed. Return only the requested structured fields."
                continue
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.debug("LLM matching call took %.1fms (attempt %s)", elapsed_ms, attempt + 1)

            raw_message = result.get("raw") if isinstance(result, dict) else None
            parsed = result.get("parsed") if isinstance(result, dict) else None
            parsing_error = result.get("parsing_error") if isinstance(result, dict) else None

            if raw_message is not None:
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(raw_message)
                metrics.prompt_tokens += prompt_tokens
                metrics.completion_tokens += completion_tokens
                metrics.total_tokens += total_tokens
                metrics.estimated_cost_usd += (
                    prompt_tokens / 1000.0 * self.settings.llm_price_per_1k_prompt_usd
                    + completion_tokens / 1000.0 * self.settings.llm_price_per_1k_completion_usd
                )

            if parsed is not None and parsing_error is None:
                return parsed

            metrics.retries += 1
            repair_hint = (
                "Your previous response did not match the required structured schema "
                f"(error: {parsing_error}). Return ONLY the structured fields, correctly typed."
            )
            logger.warning("LLM matching output failed validation (attempt %s): %s", attempt + 1, parsing_error)

        logger.error("LLM matching gave up after %s attempts for job %r", 1 + self.settings.openai_max_retries, job.title)
        return None
