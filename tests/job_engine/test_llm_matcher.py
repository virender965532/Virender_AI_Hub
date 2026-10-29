from __future__ import annotations

from jobEngine.config.settings import EngineSettings
from jobEngine.matching.llm_matcher import LLMCallMetrics, LLMMatcher
from jobEngine.models.job import Job, JobMatchAnalysis


class _FakeRaw:
    def __init__(self, prompt_tokens=100, completion_tokens=50):
        self.usage_metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }


def _job() -> Job:
    return Job(
        job_id="j1", source="test", title="Senior AI Engineer", company="Acme",
        description="Build LLM systems with LangGraph.",
        application_url="https://example.com/apply", source_url="https://example.com/job/1",
        raw_text_hash="hash1",
    )


def _valid_analysis() -> JobMatchAnalysis:
    return JobMatchAnalysis(
        matched_skills=["Python"], recommendation="strong_match", explanation="Great fit.",
    )


def test_unavailable_without_api_key_never_calls_llm(candidate_profile):
    matcher = LLMMatcher(EngineSettings(openai_api_key="", llm_enabled=True))
    metrics = LLMCallMetrics()
    result = matcher.analyze(profile=candidate_profile, job=_job(), metrics=metrics)
    assert result is None
    assert metrics.calls == 0


def test_recovers_after_one_bad_response_then_good_response(monkeypatch, candidate_profile):
    settings = EngineSettings(openai_api_key="sk-fake", llm_enabled=True, openai_max_retries=2)
    matcher = LLMMatcher(settings)

    responses = [
        {"raw": _FakeRaw(), "parsed": None, "parsing_error": ValueError("bad json")},
        {"raw": _FakeRaw(), "parsed": _valid_analysis(), "parsing_error": None},
    ]

    class _FakeStructuredLLM:
        def invoke(self, messages):
            return responses.pop(0)

    monkeypatch.setattr(matcher, "_get_structured_llm", lambda: _FakeStructuredLLM())

    metrics = LLMCallMetrics()
    result = matcher.analyze(profile=candidate_profile, job=_job(), metrics=metrics)

    assert result is not None
    assert result.recommendation == "strong_match"
    assert metrics.calls == 2
    assert metrics.retries == 1
    assert metrics.total_tokens == 300  # two calls x 150 tokens each


def test_gives_up_after_exhausting_retries(monkeypatch, candidate_profile):
    settings = EngineSettings(openai_api_key="sk-fake", llm_enabled=True, openai_max_retries=1)
    matcher = LLMMatcher(settings)

    class _AlwaysBadLLM:
        def invoke(self, messages):
            return {"raw": _FakeRaw(), "parsed": None, "parsing_error": ValueError("still bad")}

    monkeypatch.setattr(matcher, "_get_structured_llm", lambda: _AlwaysBadLLM())

    metrics = LLMCallMetrics()
    result = matcher.analyze(profile=candidate_profile, job=_job(), metrics=metrics)

    assert result is None
    assert metrics.calls == 2  # 1 + openai_max_retries(1)
    assert metrics.retries == 2


def test_prompt_wraps_job_content_as_untrusted(monkeypatch, candidate_profile):
    from jobEngine.guardrails.prompt_injection import UNTRUSTED_CLOSE, UNTRUSTED_OPEN

    settings = EngineSettings(openai_api_key="sk-fake", llm_enabled=True)
    matcher = LLMMatcher(settings)
    messages = matcher._build_messages(profile=candidate_profile, job=_job())
    user_message = messages[1]["content"]
    assert UNTRUSTED_OPEN in user_message
    assert UNTRUSTED_CLOSE in user_message

    system_message = messages[0]["content"]
    assert candidate_profile.personal_information.phone not in system_message
    assert candidate_profile.personal_information.phone not in user_message
