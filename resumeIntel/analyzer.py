"""LLM-based resume-vs-JD gap analysis.

Uses ``with_structured_output`` on the configured provider's chat model
(``llm_provider.get_chat_model``: OpenAI or Ollama) so the model's response is always
parsed into ``ResumeGapAnalysis`` or rejected — no free-form output path. On
a malformed response, retries once with an explicit repair instruction; if
it still fails, the caller surfaces a clear error rather than fabricating
a result (mirrors ``jobEngine/matching/llm_matcher.py``).
"""

from __future__ import annotations

import logging
from pathlib import Path

from llm_provider import get_chat_model, llm_configured

from .config.settings import ResumeIntelSettings, get_settings
from .guardrails import clamp_text, detect_suspicious_patterns, wrap_untrusted_content
from .models import ResumeGapAnalysis

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


class ResumeGapAnalysisError(RuntimeError):
    """Raised when the LLM analysis is unavailable or fails after retries."""


class ResumeGapAnalyzer:
    def __init__(self, settings: ResumeIntelSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self._structured_llm = None
        self._system_template = _load_prompt("gap_analysis_system.txt")
        self._user_template = _load_prompt("gap_analysis_user.txt")

    def available(self) -> bool:
        return self.settings.llm_enabled and llm_configured(self.settings.openai_api_key)

    def _get_structured_llm(self):
        if self._structured_llm is None:
            base = get_chat_model(
                self.settings.openai_model,
                openai_api_key=self.settings.openai_api_key,
                timeout=self.settings.openai_timeout,
                temperature=0.1,
            )
            self._structured_llm = base.with_structured_output(ResumeGapAnalysis, include_raw=True)
        return self._structured_llm

    def _build_messages(self, *, resume_text: str, jd_text: str, repair_hint: str = "") -> list[dict[str, str]]:
        system = self._system_template
        if repair_hint:
            system = f"{system}\n\nIMPORTANT: {repair_hint}"

        resume_text = clamp_text(resume_text, self.settings.max_resume_chars)
        jd_text = clamp_text(jd_text, self.settings.max_jd_chars)

        for label, text in (("resume", resume_text), ("job description", jd_text)):
            suspicious = detect_suspicious_patterns(text)
            if suspicious:
                logger.info(
                    "Prompt-injection-like pattern(s) detected in %s (treated as inert data): %s",
                    label,
                    suspicious,
                )

        user = self._user_template.format(
            wrapped_resume=wrap_untrusted_content(resume_text),
            wrapped_job_description=wrap_untrusted_content(jd_text),
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def analyze(self, *, resume_text: str, jd_text: str) -> ResumeGapAnalysis:
        if not self.available():
            raise ResumeGapAnalysisError(
                "Resume Intelligence LLM analysis is not configured (LLM_PROVIDER=openai without OPENAI_API_KEY, or disabled)."
            )

        repair_hint = ""
        last_error: str | None = None
        for attempt in range(1 + self.settings.openai_max_retries):
            messages = self._build_messages(resume_text=resume_text, jd_text=jd_text, repair_hint=repair_hint)
            try:
                result = self._get_structured_llm().invoke(messages)
            except Exception:  # noqa: BLE001
                logger.exception("Resume gap analysis LLM call failed (attempt %s)", attempt + 1)
                repair_hint = "Your previous response could not be processed. Return only the requested structured fields."
                last_error = "LLM call failed"
                continue

            parsed = result.get("parsed") if isinstance(result, dict) else None
            parsing_error = result.get("parsing_error") if isinstance(result, dict) else None
            if parsed is not None and parsing_error is None:
                return parsed

            last_error = str(parsing_error) if parsing_error else "empty response"
            repair_hint = (
                f"Your previous response did not match the required structured schema (error: {last_error}). "
                "Return ONLY the structured fields, correctly typed."
            )
            logger.warning("Resume gap analysis output failed validation (attempt %s): %s", attempt + 1, last_error)

        raise ResumeGapAnalysisError(f"Resume gap analysis failed after retries: {last_error}")
