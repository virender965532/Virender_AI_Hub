"""LLM second opinion for jobs the rules leave at "needs verification".

Reuses the job engine's LLM integration (same provider, model, timeout and cost
metrics as ``llm_matcher.py``) — no new service. The model must quote the sentences it
relied on; every quote is checked against the job text and dropped if it isn't
there verbatim. A verdict of A/B/D with no surviving quote is ignored, so the job
stays at "needs verification" rather than being upgraded on invented evidence.
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_provider import get_chat_model, llm_configured

from ..config.settings import EngineSettings, get_settings
from ..guardrails.prompt_injection import wrap_untrusted_content
from ..models.job import Job, WorldwideCategory, WorldwideClassification, WorldwideEvidence
from .llm_matcher import LLMCallMetrics, _extract_usage, _load_prompt

logger = logging.getLogger(__name__)

_DECISIVE = {
    WorldwideCategory.WORLDWIDE_CONFIRMED: "positive",
    WorldwideCategory.INDIA_ELIGIBLE: "positive",
    WorldwideCategory.COUNTRY_RESTRICTED: "negative",
}


class WorldwideReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal[
        "worldwide_confirmed", "india_eligible", "needs_verification", "country_restricted", "insufficient_info"
    ]
    evidence_quotes: list[str] = Field(
        default_factory=list,
        description="Sentences copied verbatim from the job text that support the category. Empty if none.",
    )
    reasoning: str = Field(max_length=600)


def _norm(text: str) -> str:
    return " ".join((text or "").split()).lower()


def verified_quotes(quotes: list[str], job_text: str) -> list[str]:
    """The quotes that really occur in the job text (whitespace/case-insensitive)."""
    haystack = _norm(job_text)
    return [q.strip() for q in quotes if len(_norm(q)) >= 12 and _norm(q) in haystack]


def apply_review(current: WorldwideClassification, review: WorldwideReview, job_text: str) -> WorldwideClassification:
    """Merge an LLM review into the rules' result; only grounded, decisive verdicts change it."""
    category = WorldwideCategory(review.category)
    signal = _DECISIVE.get(category)
    quotes = verified_quotes(review.evidence_quotes, job_text)
    if signal is None or not quotes:
        return current.model_copy(
            update={"reason": f"{current.reason} LLM review: still unclear — {review.reasoning}".strip()}
        )
    evidence = [
        WorldwideEvidence(text=q[:600], found_in="description", signal=signal, rule="llm_quote") for q in quotes
    ]
    return current.model_copy(
        update={
            "category": category,
            "india_eligible": signal == "positive",
            "evidence": evidence + list(current.evidence),
            "reason": f"LLM review: {review.reasoning}",
            "method": "llm",
        }
    )


class WorldwideReviewer:
    def __init__(self, settings: EngineSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._llm = None
        self._system = _load_prompt("worldwide_remote_system.txt")

    def available(self) -> bool:
        s = self.settings
        return s.worldwide_remote_enabled and s.worldwide_llm_review and s.llm_enabled and llm_configured(s.openai_api_key)

    def _structured_llm(self):
        if self._llm is None:
            self._llm = get_chat_model(
                self.settings.openai_model,
                openai_api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout,
                temperature=0,
            ).with_structured_output(WorldwideReview, include_raw=True)
        return self._llm

    def review(self, job: Job, current: WorldwideClassification, metrics: LLMCallMetrics) -> WorldwideClassification:
        if not self.available():
            return current
        job_text = job.description
        user = "\n".join(
            [
                f"Title: {job.title}",
                f"Company: {job.company}",
                f"LinkedIn location: {job.location or 'not specified'}",
                f"Employment type: {job.employment_type or 'not specified'}",
                f"Detected arrangement: {current.arrangement.value}",
                "Job text:",
                wrap_untrusted_content(job_text),
            ]
        )
        metrics.calls += 1
        try:
            result = self._structured_llm().invoke([{"role": "system", "content": self._system}, {"role": "user", "content": user}])
        except Exception:  # noqa: BLE001
            logger.exception("worldwide LLM review failed for %r", job.title)
            return current
        raw = result.get("raw") if isinstance(result, dict) else None
        if raw is not None:
            prompt_tokens, completion_tokens, total_tokens = _extract_usage(raw)
            metrics.prompt_tokens += prompt_tokens
            metrics.completion_tokens += completion_tokens
            metrics.total_tokens += total_tokens
            metrics.estimated_cost_usd += (
                prompt_tokens / 1000.0 * self.settings.llm_price_per_1k_prompt_usd
                + completion_tokens / 1000.0 * self.settings.llm_price_per_1k_completion_usd
            )
        parsed = result.get("parsed") if isinstance(result, dict) else None
        if parsed is None:
            return current
        return apply_review(current, parsed, job_text)
