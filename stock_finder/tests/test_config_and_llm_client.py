"""Config validation, LLM client retry / parse-repair / fallback / budget, and the circuit breaker."""

from __future__ import annotations

import httpx
import openai
import pytest

from stock_finder.config import StockFinderConfigError, get_settings, load_settings
from stock_finder.llm_client import LLMClient, LLMUnavailableError
from stock_finder.observability import BudgetExceededError, Trace, redact
from stock_finder.resilience import CircuitBreaker, CircuitOpenError, call_with_retry
from stock_finder.schemas import SummaryDraft

from .conftest import ANALYST, FALLBACK, FakeOpenAI


def _trace(calls: int = 10) -> Trace:
    return Trace(max_llm_calls=calls)


def _rate_limit() -> openai.RateLimitError:
    request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    return openai.RateLimitError("slow down", response=httpx.Response(429, request=request), body=None)


# --- config ---------------------------------------------------------------------------------------

def test_defaults(settings):
    assert settings.market == "IN" and settings.exchange_suffix == ".NS"
    assert settings.critic_model == "gpt-4.1" and settings.analyst_model == "gpt-4o"
    assert settings.url_prefix == "/stock-finder"
    assert settings.mongodb_db_name == "stock_research"
    assert "SEBI-registered" in settings.disclaimer
    assert abs(sum(settings.score_weights["moderate"].values()) - 1.0) < 1e-6


def test_startup_error_when_critic_equals_analyst(monkeypatch):
    monkeypatch.setenv("SF_CRITIC_MODEL", "gpt-4o")
    monkeypatch.setenv("SF_ANALYST_MODEL", "GPT-4o")
    with pytest.raises(StockFinderConfigError, match="must be different"):
        load_settings()


def test_register_raises_when_critic_equals_analyst(monkeypatch):
    from flask import Flask

    from stock_finder import register_stock_finder

    monkeypatch.setenv("SF_CRITIC_MODEL", "gpt-4o")
    get_settings.cache_clear()
    with pytest.raises(StockFinderConfigError):
        register_stock_finder(Flask(__name__))


def test_us_market_disclaimer(monkeypatch):
    monkeypatch.setenv("SF_MARKET", "US")
    settings = load_settings()
    assert settings.exchange_suffix == "" and "SEC-registered" in settings.disclaimer


def test_invalid_market_and_weights(monkeypatch):
    monkeypatch.setenv("SF_MARKET", "UK")
    with pytest.raises(StockFinderConfigError):
        load_settings()
    monkeypatch.setenv("SF_MARKET", "IN")
    monkeypatch.setenv("SF_SCORE_WEIGHTS", '{"moderate": {"momentum": -1}}')
    with pytest.raises(StockFinderConfigError):
        load_settings()


def test_score_weight_override_is_normalized(monkeypatch):
    monkeypatch.setenv("SF_SCORE_WEIGHTS", '{"aggressive": {"momentum": 1, "fundamentals": 1, "valuation": 0, "news_sentiment": 0, "risk": 0}}')
    weights = load_settings().score_weights["aggressive"]
    assert weights["momentum"] == 0.5 and weights["valuation"] == 0.0


def test_api_key_not_in_repr_and_redacted(settings):
    assert "sk-test" not in repr(settings)
    assert "sk-test" not in redact("key sk-test-not-a-real-key-000000000000 and ?apikey=abc123")
    assert "abc123" not in redact("https://x/y?apikey=abc123")
    assert "user:pw" not in redact("mongodb+srv://user:pw@cluster/db")


# --- LLM client -----------------------------------------------------------------------------------

def test_parse_success_records_usage(settings):
    fake = FakeOpenAI({ANALYST: [{"executive_summary": "ok"}]})
    trace = _trace()
    result = LLMClient(settings, client=fake, sleep=lambda s: None).parse(
        model=ANALYST, messages=[{"role": "user", "content": "x"}], schema=SummaryDraft, trace=trace, purpose="t")
    assert result.output.executive_summary == "ok" and not result.degraded
    assert trace.llm_calls == 1 and trace.tokens_in == 100


def test_transient_errors_are_retried(settings):
    fake = FakeOpenAI({ANALYST: [_rate_limit(), _rate_limit(), {"executive_summary": "ok"}]})
    sleeps: list[float] = []
    result = LLMClient(settings, client=fake, sleep=sleeps.append).parse(
        model=ANALYST, messages=[], schema=SummaryDraft, trace=_trace(), purpose="t")
    assert result.output.executive_summary == "ok" and result.attempts == 3 and len(sleeps) == 2


def test_parse_failure_is_repaired_once(settings):
    from pydantic import ValidationError

    try:
        SummaryDraft.model_validate({})
    except ValidationError as exc:
        bad = exc
    fake = FakeOpenAI({ANALYST: [bad, {"executive_summary": "fixed"}]})
    result = LLMClient(settings, client=fake, sleep=lambda s: None).parse(
        model=ANALYST, messages=[{"role": "user", "content": "x"}], schema=SummaryDraft, trace=_trace(), purpose="t")
    assert result.output.executive_summary == "fixed"
    assert "did not match" in fake.calls[-1]["messages"][-1]["content"]


def test_fallback_model_marks_degraded(settings):
    fake = FakeOpenAI({ANALYST: [_rate_limit()] * 10, FALLBACK: [{"executive_summary": "fb"}]})
    trace = _trace()
    result = LLMClient(settings, client=fake, sleep=lambda s: None).parse(
        model=ANALYST, messages=[], schema=SummaryDraft, trace=trace, purpose="t")
    assert result.model == FALLBACK and result.degraded and trace.degraded


def test_non_transient_error_not_retried_and_fallback_fails(settings):
    request = httpx.Request("POST", "https://api.openai.com")
    auth = openai.AuthenticationError("bad key", response=httpx.Response(401, request=request), body=None)
    fake = FakeOpenAI({ANALYST: [auth], FALLBACK: [auth]})
    with pytest.raises(LLMUnavailableError):
        LLMClient(settings, client=fake, sleep=lambda s: None).parse(
            model=ANALYST, messages=[], schema=SummaryDraft, trace=_trace(), purpose="t")
    assert len(fake.calls) == 2


def test_budget_exceeded(settings):
    fake = FakeOpenAI({ANALYST: [{"executive_summary": "a"}]})
    trace = _trace(calls=1)
    client = LLMClient(settings, client=fake, sleep=lambda s: None)
    client.parse(model=ANALYST, messages=[], schema=SummaryDraft, trace=trace, purpose="t")
    with pytest.raises(BudgetExceededError):
        client.parse(model=ANALYST, messages=[], schema=SummaryDraft, trace=trace, purpose="t")
    assert trace.budget_exceeded


# --- resilience -----------------------------------------------------------------------------------

def test_circuit_breaker_opens_and_half_opens():
    now = [0.0]
    breaker = CircuitBreaker("p", threshold=2, cooldown=10, clock=lambda: now[0])

    def boom() -> None:
        raise ConnectionError("down")

    for _ in range(2):
        with pytest.raises(ConnectionError):
            call_with_retry(boom, breaker=breaker, is_transient=lambda e: False, max_retries=0)
    with pytest.raises(CircuitOpenError):
        call_with_retry(lambda: 1, breaker=breaker, is_transient=lambda e: False, max_retries=0)
    now[0] = 11
    assert call_with_retry(lambda: 1, breaker=breaker, is_transient=lambda e: False, max_retries=0) == 1
    assert breaker.state == "closed"
