"""LLM client retry policy, structured-output repair, config and profile validation."""

from __future__ import annotations

import json

import httpx
import openai
import pytest

from profile_assistant.config import ProfileAssistantConfigError, get_settings, load_settings
from profile_assistant.llm_client import LLMClient, LLMUnavailableError
from profile_assistant.profile_loader import ProfileLoadError, get_profile, reload_profile
from profile_assistant.schemas import AnswerOutput

from .conftest import FakeOpenAI, answer

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls: type[openai.APIStatusError], code: int) -> openai.APIStatusError:
    return cls("boom", response=httpx.Response(code, request=_REQUEST), body=None)


def _client(script: list) -> tuple[LLMClient, FakeOpenAI, list[float]]:
    sleeps: list[float] = []
    fake = FakeOpenAI({"gpt-4o-mini": script})
    return LLMClient(get_settings(), client=fake, sleep=sleeps.append), fake, sleeps


def _call(llm: LLMClient):
    return llm.parse(model="gpt-4o-mini", messages=[{"role": "user", "content": "q"}], schema=AnswerOutput, trace_id="t")


def test_transient_error_is_retried_with_backoff():
    llm, fake, sleeps = _client([_status_error(openai.RateLimitError, 429), openai.APITimeoutError(_REQUEST), answer("ok")])
    result = _call(llm)

    assert result.output.answer == "ok"
    assert result.attempts == 3
    assert len(fake.calls) == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] * 0.9  # exponential (with jitter)


def test_transient_error_gives_up_after_max_retries():
    llm, fake, _ = _client([_status_error(openai.InternalServerError, 500)] * 10)
    with pytest.raises(LLMUnavailableError):
        _call(llm)
    assert len(fake.calls) == 1 + get_settings().max_retries


def test_auth_error_is_not_retried():
    llm, fake, sleeps = _client([_status_error(openai.AuthenticationError, 401), answer("never")])
    with pytest.raises(LLMUnavailableError):
        _call(llm)
    assert len(fake.calls) == 1
    assert sleeps == []


def test_bad_request_is_not_retried():
    llm, fake, _ = _client([_status_error(openai.BadRequestError, 400), answer("never")])
    with pytest.raises(LLMUnavailableError):
        _call(llm)
    assert len(fake.calls) == 1


def test_parse_failure_retried_once_with_validation_error():
    llm, fake, _ = _client([None, answer("fixed")])
    result = _call(llm)

    assert result.output.answer == "fixed"
    repair_message = fake.calls[1]["messages"][-1]["content"]
    assert "could not be parsed" in repair_message


def test_parse_failure_twice_raises():
    llm, _, _ = _client([None, None])
    with pytest.raises(LLMUnavailableError):
        _call(llm)


def test_startup_error_when_models_are_equal(monkeypatch):
    monkeypatch.setenv("PA_ANSWER_MODEL", "gpt-4o")
    monkeypatch.setenv("PA_CRITIC_MODEL", "GPT-4o")
    with pytest.raises(ProfileAssistantConfigError, match="must be different"):
        load_settings()


def test_register_fails_fast_when_models_are_equal(monkeypatch):
    from flask import Flask

    from profile_assistant import register_profile_assistant

    monkeypatch.setenv("PA_CRITIC_MODEL", "gpt-4o-mini")
    with pytest.raises(ProfileAssistantConfigError):
        register_profile_assistant(Flask(__name__))


def test_sample_profile_loads_and_renders():
    profile = get_profile()
    assert profile.name == "Virender Kumar"
    assert "EXPERIENCE:" in profile.text and "Impressico Business Solutions" in profile.text


def test_invalid_profile_fails_fast(tmp_path, monkeypatch):
    bad = tmp_path / "profile.json"
    bad.write_text(json.dumps({"headline": "no name"}), encoding="utf-8")
    monkeypatch.setenv("PA_PROFILE_PATH", str(bad))
    get_settings.cache_clear()
    with pytest.raises(ProfileLoadError, match="name"):
        get_profile()


def test_text_profile_and_hot_reload(tmp_path, monkeypatch):
    path = tmp_path / "profile.md"
    path.write_text("# Jane Doe\nPython developer.", encoding="utf-8")
    monkeypatch.setenv("PA_PROFILE_PATH", str(path))
    get_settings.cache_clear()
    assert get_profile().name == "Jane Doe"

    path.write_text("# Jane Smith\nPython developer.", encoding="utf-8")
    assert get_profile().name == "Jane Doe"  # cached
    assert reload_profile().name == "Jane Smith"
    assert get_profile().name == "Jane Smith"


def test_redact_contact_details_keeps_years_and_urls():
    from profile_assistant.profile_loader import redact_contact_details

    text = "Call +91 98215 92146 or mail jane.doe@example.com; 2013 - 2017; www.linkedin.com/in/jane-doe"
    redacted = redact_contact_details(text)
    assert "98215" not in redacted and "example.com" not in redacted
    assert "2013 - 2017" in redacted and "linkedin.com/in/jane-doe" in redacted


def test_extra_pdf_source_is_merged_cleaned_and_redacted(tmp_path, monkeypatch):
    from profile_assistant import profile_loader

    pdf = tmp_path / "Profile.pdf"
    pdf.write_bytes(b"%PDF-stub")
    monkeypatch.setattr(
        profile_loader,
        "_read_pdf_text",
        lambda _p: "Contact\n9821592146 (Mobile)\nI�m a developer\nAugust 2021�-�August 2024\nPage 1 of 3",
    )
    monkeypatch.setenv("PA_EXTRA_PROFILE_PATHS", str(pdf))
    get_settings.cache_clear()

    text = get_profile().text
    assert "=== SOURCE 1 (primary): profile.json ===" in text
    assert "=== SOURCE 2 (supplementary): Profile.pdf ===" in text
    assert "I'm a developer" in text and "August 2021 - August 2024" in text
    assert "9821592146" not in text and "Page 1 of 3" not in text


def test_missing_extra_source_fails_fast(tmp_path, monkeypatch):
    monkeypatch.setenv("PA_EXTRA_PROFILE_PATHS", str(tmp_path / "missing.pdf"))
    get_settings.cache_clear()
    with pytest.raises(ProfileLoadError, match="not found"):
        get_profile()


def test_extra_sources_can_be_disabled(monkeypatch):
    monkeypatch.setenv("PA_EXTRA_PROFILE_PATHS", "")
    get_settings.cache_clear()
    assert "SOURCE 2" not in get_profile().text
