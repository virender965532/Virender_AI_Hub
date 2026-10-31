"""The job engine's LLM classifiers and LangGraph workflow on LLM_PROVIDER=ollama.

No network: ChatOllama is pointed at an in-memory HTTP transport, so the real
request building, JSON-schema output parsing and usage accounting are exercised.
"""

from __future__ import annotations

import json

import httpx
import pytest

from jobEngine.config import settings as settings_module
from jobEngine.config.settings import EngineSettings
from jobEngine.matching.llm_matcher import LLMCallMetrics, LLMMatcher
from jobEngine.matching.rejection_analyzer import RejectionAnalyzer
from jobEngine.matching.worldwide_llm import WorldwideReviewer
from jobEngine.models.job import Job, JobEligibility, JobMatchAnalysis
from llm_provider import get_chat_model, reset_caches

from .test_graph_integration import _run, _source_cfg

_VISA_SENTENCE = "We provide full visa sponsorship for this role."


@pytest.fixture
def ollama(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    reset_caches()
    settings_module.get_settings.cache_clear()
    yield
    settings_module.get_settings.cache_clear()


def _analysis(visa_status="POSITIVE_EVIDENCE", visa_quotes=(_VISA_SENTENCE,)) -> JobMatchAnalysis:
    return JobMatchAnalysis(
        matched_skills=["Python", "LangGraph"], recommendation="strong_match", explanation="Strong LangGraph fit.",
        eligibility=JobEligibility(
            visa_status=visa_status, visa_quotes=list(visa_quotes), relocation_support="NOT_MENTIONED",
            remote_status="REMOTE_UNCLEAR", india_category="needs_verification",
        ),
    )


def _serve(analysis: JobMatchAnalysis, seen: list[dict]):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = {
            "model": "qwen3:8b", "created_at": "2026-01-01T00:00:00Z", "done": True, "done_reason": "stop",
            "message": {"role": "assistant", "content": analysis.model_dump_json()},
            "prompt_eval_count": 900, "eval_count": 300,
        }
        return httpx.Response(200, content=(json.dumps(body) + "\n").encode())

    return handler


def _install(settings: EngineSettings, handler) -> None:
    """Mock the exact (cached) ChatOllama the matcher will get from the provider factory."""
    import ollama as ollama_pkg

    llm = get_chat_model(settings.openai_model, temperature=0, seed=7)
    llm._client = ollama_pkg.Client(host=llm.base_url, transport=httpx.MockTransport(handler))


def _job() -> Job:
    return Job(
        job_id="j1", source="test", title="Senior AI Engineer", company="Acme",
        description=f"Build LLM systems with LangGraph. {_VISA_SENTENCE}",
        application_url="https://example.com/apply", source_url="https://example.com/job/1", raw_text_hash="h1",
    )


def test_classifiers_are_available_on_ollama_without_an_openai_key(ollama):
    settings = EngineSettings(openai_api_key="", llm_enabled=True, worldwide_remote_enabled=True)
    assert LLMMatcher(settings).available()
    assert RejectionAnalyzer(settings).available()
    assert WorldwideReviewer(settings).available()
    assert not LLMMatcher(EngineSettings(openai_api_key="", llm_enabled=False)).available()


def test_resume_intel_is_available_on_ollama_without_a_key(ollama):
    from resumeIntel.analyzer import ResumeGapAnalyzer
    from resumeIntel.config.settings import ResumeIntelSettings

    assert ResumeGapAnalyzer(ResumeIntelSettings(openai_api_key="", llm_enabled=True)).available()


def test_matcher_structured_output_on_ollama(ollama, candidate_profile):
    settings = EngineSettings(openai_api_key="", llm_enabled=True)
    seen: list[dict] = []
    _install(settings, _serve(_analysis(), seen))

    metrics = LLMCallMetrics()
    result = LLMMatcher(settings).analyze(profile=candidate_profile, job=_job(), metrics=metrics)

    assert isinstance(result, JobMatchAnalysis)
    assert result.eligibility.visa_status == "POSITIVE_EVIDENCE"  # schema/labels unchanged
    assert metrics.calls == 1 and metrics.prompt_tokens == 900 and metrics.completion_tokens == 300
    request = seen[0]
    assert request["model"] == "qwen3:8b" and request["think"] is False
    assert request["options"]["temperature"] == 0 and request["options"]["seed"] == 7
    assert "eligibility" in request["format"]["properties"]  # the JobMatchAnalysis JSON schema
    assert request["messages"][0]["role"] == "system"  # same prompts as on OpenAI


def test_matcher_repairs_invalid_output_on_ollama(ollama, candidate_profile):
    settings = EngineSettings(openai_api_key="", llm_enabled=True, openai_max_retries=1)
    seen: list[dict] = []
    good = _serve(_analysis(), seen)

    def handler(request: httpx.Request) -> httpx.Response:
        if not seen:
            seen.append(json.loads(request.content))
            bad = {"model": "qwen3:8b", "created_at": "2026-01-01T00:00:00Z", "done": True,
                   "message": {"role": "assistant", "content": '{"recommendation": "maybe"}'}}
            return httpx.Response(200, content=(json.dumps(bad) + "\n").encode())
        return good(request)

    _install(settings, handler)
    metrics = LLMCallMetrics()
    result = LLMMatcher(settings).analyze(profile=candidate_profile, job=_job(), metrics=metrics)
    assert result is not None and metrics.calls == 2 and metrics.retries == 1
    assert "did not match the required structured schema" in seen[1]["messages"][0]["content"]


def test_analysis_cache_version_separates_providers(monkeypatch, ollama):
    settings = EngineSettings(openai_api_key="sk-x", llm_enabled=True)
    on_ollama = LLMMatcher(settings).version
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    reset_caches()
    assert LLMMatcher(settings).version != on_ollama  # a gpt-4o-mini analysis is never served as a qwen3 one


async def test_langgraph_workflow_runs_llm_matching_on_ollama(monkeypatch, ollama):
    monkeypatch.setenv("JOB_ENGINE_LLM_ENABLED", "true")
    settings_module.get_settings.cache_clear()
    settings = settings_module.get_settings()
    assert settings.openai_api_key == "" and settings.llm_enabled

    seen: list[dict] = []
    _install(settings, _serve(_analysis(visa_status="NOT_MENTIONED", visa_quotes=()), seen))

    result = (await _run([_source_cfg("good_source", "_test_good_adapter")],
                         request={"max_results": 10, "minimum_match_score": 0}))["final_result"]

    assert seen, "the graph's matching node should have called the Ollama model"
    assert result["status"] != "failed"
    job = result["jobs"][0]
    assert job["llm_analysis"]["recommendation"] == "strong_match"
    assert job["llm_analysis"]["eligibility"]["visa_status"] == "NOT_MENTIONED"
    assert job["job"]["visa_remote"]["visa_status"] and job["job"]["visa_remote"]["remote_status"]
