"""Shared fixtures: a scripted fake OpenAI client; no test ever calls the real API."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

os.environ["OPENAI_API_KEY"] = "sk-test-not-a-real-key-000000000000"
os.environ["PA_ANSWER_MODEL"] = "gpt-4o-mini"
os.environ["PA_CRITIC_MODEL"] = "gpt-4o"
os.environ["PA_UI_ENABLED"] = "false"
os.environ["PA_ADMIN_TOKEN"] = "test-admin-token"
for _var in ("PA_PROFILE_PATH", "PA_MAX_REVISIONS", "PA_MAX_RETRIES", "PA_URL_PREFIX"):
    os.environ.pop(_var, None)

from profile_assistant import pipeline as pipeline_module  # noqa: E402
from profile_assistant.config import get_settings  # noqa: E402
from profile_assistant.llm_client import LLMClient  # noqa: E402
from profile_assistant.profile_loader import clear_profile_cache  # noqa: E402
from profile_assistant.schemas import AnswerOutput, CriticVerdict  # noqa: E402


def completion(parsed: Any) -> SimpleNamespace:
    """Build an object shaped like an OpenAI ``ParsedChatCompletion``."""
    message = SimpleNamespace(parsed=parsed, refusal=None)
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def answer(
    text: str,
    *,
    question_type: str = "about_person",
    found: bool = True,
    evidence: list[str] | None = None,
) -> AnswerOutput:
    return AnswerOutput(
        reasoning="r", question_type=question_type, in_scope=question_type != "out_of_scope",
        found_in_profile=found, answer=text, evidence=evidence or [],
    )


def verdict(passed: bool, feedback: str = "") -> CriticVerdict:
    score = 5 if passed else 2
    return CriticVerdict(
        reasoning="r", groundedness=score, relevance=score, scope_correct=passed,
        verdict="PASS" if passed else "FAIL", feedback=feedback,
    )


class FakeOpenAI:
    """Mimics ``client.chat.completions.parse``; responses are scripted per model.

    Each scripted item is either a parsed pydantic object or an exception to raise.
    """

    def __init__(self, scripts: dict[str, list[Any]]) -> None:
        self.scripts = {model: list(items) for model, items in scripts.items()}
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        item = self.scripts[kwargs["model"]].pop(0)
        if isinstance(item, BaseException):
            raise item
        return completion(item)

    def calls_for(self, model: str) -> list[dict[str, Any]]:
        return [c for c in self.calls if c["model"] == model]


@pytest.fixture(autouse=True)
def _fresh_state():
    get_settings.cache_clear()
    clear_profile_cache()
    pipeline_module.set_pipeline(None)
    yield
    pipeline_module.set_pipeline(None)
    clear_profile_cache()
    get_settings.cache_clear()


@pytest.fixture
def make_pipeline():
    """Install a pipeline backed by a ``FakeOpenAI`` with the given scripts."""

    def _make(answer_script: list[Any], critic_script: list[Any]) -> FakeOpenAI:
        settings = get_settings()
        fake = FakeOpenAI({settings.answer_model: answer_script, settings.critic_model: critic_script})
        llm = LLMClient(settings, client=fake, sleep=lambda _s: None)
        pipeline_module.set_pipeline(pipeline_module.ProfileAssistantPipeline(settings, llm))
        return fake

    return _make
