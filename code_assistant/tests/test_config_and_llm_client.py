"""Settings validation, LLM retries, repair, fallback, circuit breaker and budget."""

from __future__ import annotations

import httpx
import openai
import pytest

from code_assistant.config import CodeAssistantConfigError, get_settings, load_settings
from code_assistant.llm_client import CircuitBreaker, ContextLengthError, LLMUnavailableError
from code_assistant.observability import BudgetExceededError, Trace
from code_assistant.schemas import RouteDecision

from .conftest import CODER, FALLBACK, ROUTER, reply

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def rate_limit() -> openai.RateLimitError:
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=_REQUEST), body=None)


def auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError("bad key", response=httpx.Response(401, request=_REQUEST), body=None)


def context_error() -> openai.BadRequestError:
    return openai.BadRequestError(
        "This model's maximum context length is 128000 tokens",
        response=httpx.Response(400, request=_REQUEST),
        body={"code": "context_length_exceeded"},
    )


def route(intent: str = "explain", confidence: float = 0.9) -> RouteDecision:
    return RouteDecision(reasoning="r", intent=intent, difficulty="easy", confidence=confidence)


def test_reviewer_equal_to_coder_is_a_startup_error(monkeypatch):
    monkeypatch.setenv("CA_CODER_MODEL", "gpt-4o")
    monkeypatch.setenv("CA_REVIEWER_MODEL", "GPT-4o")
    with pytest.raises(CodeAssistantConfigError, match="must be different"):
        load_settings()


def test_defaults_and_allowlist_parsing(monkeypatch):
    monkeypatch.setenv("CA_COMMAND_ALLOWLIST", "pytest, python -m pytest")
    settings = load_settings()
    assert settings.coder_model == "gpt-4o" and settings.reviewer_model == "o3-mini"
    assert settings.command_allowlist == (("pytest",), ("python", "-m", "pytest"))
    assert settings.url_prefix == "/code-assistant"
    assert "sk-test" not in repr(settings)


def test_root_prefix_rejected(monkeypatch):
    monkeypatch.setenv("CA_URL_PREFIX", "/")
    with pytest.raises(CodeAssistantConfigError):
        load_settings()


def test_transient_error_retried_then_succeeds(make_llm, trace):
    llm, fake = make_llm({ROUTER: [rate_limit(), route()]})
    result = llm.parse(model=ROUTER, messages=[{"role": "user", "content": "x"}], schema=RouteDecision, trace=trace, purpose="t")
    assert result.output.intent == "explain"
    assert result.attempts == 2 and not result.degraded
    assert trace.llm_calls == 1


def test_auth_error_not_retried_but_falls_back(make_llm, trace):
    llm, fake = make_llm(chat_scripts={CODER: [auth_error()], FALLBACK: [reply("hi")]})
    result = llm.chat(model=CODER, messages=[{"role": "user", "content": "x"}], trace=trace, purpose="t")
    assert result.content == "hi" and result.degraded and result.model == FALLBACK
    assert len(fake.calls_for(CODER)) == 1  # never retried
    assert trace.degraded


def test_fallback_failure_raises(make_llm, trace):
    llm, _ = make_llm(chat_scripts={CODER: [auth_error()], FALLBACK: [auth_error()]})
    with pytest.raises(LLMUnavailableError):
        llm.chat(model=CODER, messages=[], trace=trace, purpose="t")


def test_parse_failure_repaired_once(make_llm, trace):
    llm, fake = make_llm({ROUTER: [None, route("fix_bug")]})
    result = llm.parse(model=ROUTER, messages=[{"role": "user", "content": "x"}], schema=RouteDecision, trace=trace, purpose="t")
    assert result.output.intent == "fix_bug"
    assert "did not match" in fake.calls[-1]["messages"][-1]["content"]


def test_context_overflow_trims_and_retries_once(make_llm, trace):
    llm, fake = make_llm(chat_scripts={CODER: [context_error(), reply("ok")]})
    trimmed = []

    def shrink(messages):
        trimmed.append(True)
        return messages[-1:]

    result = llm.chat(model=CODER, messages=[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
                      trace=trace, purpose="t", on_context_overflow=shrink)
    assert result.content == "ok" and trimmed
    assert len(fake.calls[-1]["messages"]) == 1


def test_context_overflow_without_trimmer(make_llm, trace):
    llm, _ = make_llm(chat_scripts={CODER: [context_error()], FALLBACK: [context_error()]})
    with pytest.raises(ContextLengthError):
        llm.chat(model=CODER, messages=[], trace=trace, purpose="t")


def test_reasoning_model_gets_no_temperature(make_llm, trace):
    llm, fake = make_llm({"o3-mini": [route()]})
    llm.parse(model="o3-mini", messages=[], schema=RouteDecision, trace=trace, purpose="t")
    assert "temperature" not in fake.calls[-1]


def test_circuit_breaker_opens_and_cools_down():
    now = [0.0]
    breaker = CircuitBreaker(threshold=5, cooldown=30, clock=lambda: now[0])
    for _ in range(5):
        breaker.record_failure()
    assert breaker.is_open
    with pytest.raises(LLMUnavailableError, match="circuit breaker"):
        breaker.before_call()
    now[0] = 31
    breaker.before_call()  # half-open trial allowed
    breaker.record_failure()
    assert breaker.is_open


def test_open_circuit_skips_api_and_fallback(make_llm, trace):
    llm, fake = make_llm(chat_scripts={CODER: [], FALLBACK: []})
    for _ in range(get_settings().circuit_failure_threshold):
        llm.breaker.record_failure()
    with pytest.raises(LLMUnavailableError):
        llm.chat(model=CODER, messages=[], trace=trace, purpose="t")
    assert fake.calls == []


def test_budget_stops_calls(make_llm):
    tiny = Trace(max_llm_calls=1, max_tokens=10_000)
    llm, _ = make_llm(chat_scripts={CODER: [reply("one"), reply("two")]})
    llm.chat(model=CODER, messages=[], trace=tiny, purpose="t")
    with pytest.raises(BudgetExceededError):
        llm.chat(model=CODER, messages=[], trace=tiny, purpose="t")


def test_trace_metrics_and_redacted_logs(caplog):
    events = []
    trace = Trace(max_llm_calls=5, max_tokens=1000, listener=lambda e, d: events.append((e, d)))
    trace.record_llm_call(model="gpt-4o", usage={"prompt_tokens": 1000, "completion_tokens": 100}, purpose="t", attempts=1, degraded=False)
    with caplog.at_level("INFO", logger="code_assistant.trace"):
        trace.emit("tool_result", {"output": "key=sk-abcdefghijklmnopqrstuvwxyz0123"})
    assert "sk-abcdefghijklmnopqrstuvwxyz0123" not in caplog.text
    assert events[0][1]["output"] == "key=[REDACTED]"
    metrics = trace.finish("answered")
    assert metrics["llm_calls"] == 1 and metrics["estimated_cost_usd"] > 0 and metrics["status"] == "answered"
