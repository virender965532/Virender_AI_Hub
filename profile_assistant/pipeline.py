"""Orchestration: input guard -> answer -> critic -> revise loop -> fallback -> output guard.

Every step is logged as one JSON line carrying the request's ``trace_id``
(model, attempts, verdict, scores, latency, token usage). The API key and
profile text are never logged.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from typing import Any

from .answer_agent import AnswerAgent
from .config import ProfileAssistantSettings, get_settings
from .critic_agent import CriticAgent
from .guardrails import check_input, check_output
from .llm_client import LLMClient, LLMUnavailableError, get_llm_client
from .profile_loader import get_profile
from .prompts import I_DONT_KNOW, PROMPT_VERSION, UNAVAILABLE_MESSAGE
from .schemas import AnswerOutput, CriticVerdict

logger = logging.getLogger(__name__)

STATUS_PASSED = "passed"
STATUS_REVISED = "revised"
STATUS_FALLBACK = "fallback"
STATUS_BLOCKED = "blocked"
STATUS_ERROR = "error"


def _log(event: str, trace_id: str, level: int = logging.INFO, **fields: Any) -> None:
    payload = {"event": event, "trace_id": trace_id, "module": "profile_assistant", **fields}
    logger.log(level, json.dumps({k: v for k, v in payload.items() if v is not None}, default=str))


class ProfileAssistantPipeline:
    """Runs one question through the guarded answer/critic reflection loop."""

    def __init__(
        self,
        settings: ProfileAssistantSettings | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        llm = llm or get_llm_client()
        self.answer_agent = AnswerAgent(llm, self.settings)
        self.critic_agent = CriticAgent(llm, self.settings)

    def answer_question(self, question: str, history: list | None = None) -> dict[str, Any]:
        """Answer ``question`` about the profile; see module docstring for the flow."""
        trace_id = uuid.uuid4().hex
        started = time.perf_counter()
        tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        def finish(answer: str, status: str, revisions: int, evidence: list[str], **extra: Any) -> dict[str, Any]:
            latency_ms = int((time.perf_counter() - started) * 1000)
            _log(
                "pa.request.completed", trace_id, status=status, revisions=revisions,
                latency_ms=latency_ms, tokens=tokens, prompt_version=PROMPT_VERSION, **extra,
            )
            return {
                "answer": answer,
                "status": status,
                "revisions": revisions,
                "evidence": evidence,
                "trace_id": trace_id,
                "latency_ms": latency_ms,
            }

        def add_usage(usage: dict[str, int]) -> None:
            for key in tokens:
                tokens[key] += usage.get(key, 0)

        question = (question or "").strip()
        _log("pa.request.received", trace_id, route="answer_question", question_chars=len(question))

        guard = check_input(question, max_chars=self.settings.max_question_chars)
        if not guard.allowed:
            _log("pa.guard.input_blocked", trace_id, logging.WARNING, reason=guard.reason)
            return finish(guard.message, STATUS_BLOCKED, 0, [], reason=guard.reason)

        profile_text = get_profile().text
        revisions = 0
        try:
            candidate = self._run_answer(trace_id, question, profile_text, history, add_usage, revision=0)
            while True:
                verdict = self._run_critic(trace_id, question, profile_text, candidate, add_usage, revision=revisions)
                if verdict.verdict == "PASS":
                    final, status = candidate, (STATUS_PASSED if revisions == 0 else STATUS_REVISED)
                    break
                if revisions >= self.settings.max_revisions:
                    _log("pa.fallback", trace_id, logging.WARNING, revisions=revisions)
                    final = AnswerOutput(
                        reasoning="fallback", question_type=candidate.question_type, in_scope=candidate.in_scope,
                        found_in_profile=False, answer=I_DONT_KNOW, evidence=[],
                    )
                    status = STATUS_FALLBACK
                    break
                revisions += 1
                candidate = self._run_answer(
                    trace_id, question, profile_text, history, add_usage,
                    revision=revisions, feedback=verdict.feedback, previous=candidate,
                )
        except LLMUnavailableError as exc:
            _log("pa.llm_unavailable", trace_id, logging.ERROR, error=str(exc)[:300])
            return finish(UNAVAILABLE_MESSAGE, STATUS_ERROR, revisions, [])

        for text in (final.answer, *final.evidence):
            out_guard = check_output(text, api_key=self.settings.openai_api_key)
            if not out_guard.allowed:
                _log("pa.guard.output_blocked", trace_id, logging.WARNING, reason=out_guard.reason)
                return finish(I_DONT_KNOW, STATUS_BLOCKED, revisions, [], reason=out_guard.reason)

        return finish(final.answer, status, revisions, list(final.evidence))

    def _run_answer(
        self,
        trace_id: str,
        question: str,
        profile_text: str,
        history: list | None,
        add_usage: Any,
        *,
        revision: int,
        feedback: str | None = None,
        previous: AnswerOutput | None = None,
    ) -> AnswerOutput:
        started = time.perf_counter()
        result = self.answer_agent.answer(
            question=question, profile_text=profile_text, history=history, trace_id=trace_id,
            critic_feedback=feedback, previous_answer=previous,
        )
        add_usage(result.usage)
        output: AnswerOutput = result.output  # type: ignore[assignment]
        _log(
            "pa.answer", trace_id, agent="answer", model=result.model, attempts=result.attempts,
            revision=revision, in_scope=output.in_scope, found_in_profile=output.found_in_profile,
            evidence_count=len(output.evidence), latency_ms=int((time.perf_counter() - started) * 1000),
            usage=result.usage,
        )
        return output

    def _run_critic(
        self,
        trace_id: str,
        question: str,
        profile_text: str,
        candidate: AnswerOutput,
        add_usage: Any,
        *,
        revision: int,
    ) -> CriticVerdict:
        started = time.perf_counter()
        result = self.critic_agent.review(
            question=question, profile_text=profile_text, candidate=candidate, trace_id=trace_id
        )
        add_usage(result.usage)
        verdict: CriticVerdict = result.output  # type: ignore[assignment]
        _log(
            "pa.critic", trace_id, agent="critic", model=result.model, attempts=result.attempts,
            revision=revision, verdict=verdict.verdict, groundedness=verdict.groundedness,
            relevance=verdict.relevance, scope_correct=verdict.scope_correct,
            latency_ms=int((time.perf_counter() - started) * 1000), usage=result.usage,
        )
        return verdict


_pipeline: ProfileAssistantPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> ProfileAssistantPipeline:
    """Process-wide pipeline instance."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = ProfileAssistantPipeline()
    return _pipeline


def set_pipeline(pipeline: ProfileAssistantPipeline | None) -> None:
    """Replace (or reset with ``None``) the process-wide pipeline; used by tests."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = pipeline


def answer_question(question: str, history: list | None = None) -> dict[str, Any]:
    """Answer a question about the profile.

    Returns ``{"answer", "status", "revisions", "evidence", "trace_id", "latency_ms"}``
    where status is one of passed / revised / fallback / blocked / error.
    """
    return get_pipeline().answer_question(question, history or [])
