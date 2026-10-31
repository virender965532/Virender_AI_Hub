"""Repo-wide test setup: every suite starts on the default OpenAI provider.

A developer's ``.env`` may set ``LLM_PROVIDER=ollama``; tests that need Ollama
set it themselves (see tests/llm_provider/).
"""

import pytest

from llm_provider import reset_caches

_PROVIDER_ENV = (
    "LLM_PROVIDER", "EMBEDDING_PROVIDER", "OLLAMA_HOST", "OLLAMA_MODEL", "OLLAMA_MODEL_MAP",
    "OLLAMA_EMBEDDING_MODEL", "OLLAMA_THINK", "OLLAMA_NUM_CTX", "OLLAMA_TIMEOUT",
)


@pytest.fixture(autouse=True)
def _default_llm_provider(monkeypatch):
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    reset_caches()
    yield
    reset_caches()
