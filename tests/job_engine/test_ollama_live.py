"""Live checks against the local Ollama server (real model, real inference).

Opt-in, since a call takes 10-60s on CPU:

    $env:RUN_OLLAMA_TESTS = "1"; python -m pytest tests/job_engine/test_ollama_live.py -v

Skipped (with the reason) when the server or the configured model is missing.
Uses OLLAMA_HOST / OLLAMA_MODEL from the environment, else the defaults.
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
from pydantic import BaseModel

from jobEngine.config.settings import EngineSettings
from jobEngine.matching.llm_matcher import LLMCallMetrics, LLMMatcher
from jobEngine.models.job import Job, JobMatchAnalysis
from llm_provider import (
    LLMProviderError,
    check_ollama,
    get_chat_model,
    get_openai_client,
    get_provider_settings,
    reset_caches,
)

_LIVE_ENV = {name: os.environ.get(name) for name in ("OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_NUM_CTX", "OLLAMA_TIMEOUT")}

pytestmark = pytest.mark.skipif(os.environ.get("RUN_OLLAMA_TESTS") != "1", reason="set RUN_OLLAMA_TESTS=1 to run live Ollama tests")

_JOB_TEXT = (
    "We are hiring a Senior AI Engineer to build LangGraph agents in Python. "
    "We provide full visa sponsorship for this role. "
    "This position is fully remote and open to candidates anywhere in the world."
)


class Evidence(BaseModel):
    visa_status: Literal["POSITIVE_EVIDENCE", "NEGATIVE_EVIDENCE", "MENTIONED_UNCLEAR", "NOT_MENTIONED"]
    remote_status: Literal["GLOBAL_REMOTE_EVIDENCE", "RESTRICTED_REMOTE", "REMOTE_UNCLEAR", "NOT_MENTIONED"]


@pytest.fixture(autouse=True)
def live_ollama(monkeypatch):
    # The repo conftest resets provider env vars; restore the developer's Ollama settings.
    for name, value in _LIVE_ENV.items():
        if value:
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_caches()
    try:
        check_ollama(include_embeddings=False)
    except LLMProviderError as exc:
        pytest.skip(str(exc))


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "Classify the job posting's visa sponsorship and remote-work evidence."},
        {"role": "user", "content": _JOB_TEXT},
    ]


def test_openai_sdk_path_returns_structured_output():
    completion = get_openai_client(max_retries=0).chat.completions.parse(
        model="gpt-4o-mini", messages=_messages(), response_format=Evidence, temperature=0,
    )
    parsed = completion.choices[0].message.parsed
    assert isinstance(parsed, Evidence)
    assert completion.model.startswith(get_provider_settings().ollama_model.split(":")[0])


def test_langchain_path_returns_structured_output():
    result = get_chat_model("gpt-4o-mini", temperature=0).with_structured_output(Evidence, include_raw=True).invoke(_messages())
    assert result["parsing_error"] is None and isinstance(result["parsed"], Evidence)
    assert result["raw"].usage_metadata["total_tokens"] > 0
    assert "<think>" not in (result["raw"].content or "")


def test_job_matcher_end_to_end(candidate_profile):
    """The real job-engine classifier prompt and JobMatchAnalysis schema on the local model."""
    job = Job(
        job_id="live1", source="test", title="Senior AI Engineer", company="Acme", location="Remote",
        description=_JOB_TEXT, skills=["Python", "LangGraph"],
        application_url="https://example.com/apply", source_url="https://example.com/job", raw_text_hash="live1",
    )
    metrics = LLMCallMetrics()
    analysis = LLMMatcher(EngineSettings(openai_api_key="", llm_enabled=True)).analyze(
        profile=candidate_profile, job=job, metrics=metrics,
    )
    assert isinstance(analysis, JobMatchAnalysis), f"no valid analysis after {metrics.calls} call(s)"
    assert analysis.eligibility is not None
    print("\nlive analysis:", analysis.recommendation, analysis.eligibility.visa_status,
          analysis.eligibility.remote_status, f"tokens={metrics.total_tokens} calls={metrics.calls}")
