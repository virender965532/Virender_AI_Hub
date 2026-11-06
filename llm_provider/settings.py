"""Environment-driven settings for the global LLM provider switch.

``LLM_PROVIDER`` picks the backend for every LLM call in the hub:

* ``openai`` (default) — the existing OpenAI integration; each module keeps
  its own model env vars (``JOB_ENGINE_OPENAI_MODEL``, ``PA_ANSWER_MODEL``...)
  and ``OPENAI_API_KEY``.
* ``ollama`` — a local Ollama server (``OLLAMA_HOST``). Every module's OpenAI
  model name is mapped to an Ollama model: ``OLLAMA_MODEL_MAP`` entries first,
  then names that already look like Ollama tags (``qwen3:14b``), else
  ``OLLAMA_MODEL``. Embedding models map the same way to
  ``OLLAMA_EMBEDDING_MODEL``. No OpenAI key is needed.

``EMBEDDING_PROVIDER`` (default: same as ``LLM_PROVIDER``) lets embeddings stay
on one backend while chat runs on the other.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

from dotenv import load_dotenv

from .errors import LLMProviderConfigError

load_dotenv()

PROVIDERS = ("openai", "ollama")

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:8b"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


def _env_str(name: str, default: str) -> str:
    raw = (os.getenv(name) or "").strip()
    return raw if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _provider(name: str, default: str) -> str:
    value = _env_str(name, default).lower()
    if value not in PROVIDERS:
        raise LLMProviderConfigError(
            f"{name}={value!r} is not supported; use one of: {', '.join(PROVIDERS)}."
        )
    return value


def parse_model_map(raw: str) -> dict[str, str]:
    """``"gpt-4o=qwen3:14b, gpt-4o-mini=qwen3:8b"`` -> ``{"gpt-4o": "qwen3:14b", ...}``."""
    mapping: dict[str, str] = {}
    for item in (raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        source, sep, target = item.partition("=")
        if not sep or not source.strip() or not target.strip():
            raise LLMProviderConfigError(
                f"OLLAMA_MODEL_MAP entry {item!r} is invalid; expected 'openai-model=ollama-model'."
            )
        mapping[source.strip().lower()] = target.strip()
    return mapping


def _optional_positive_int(name: str) -> int | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise LLMProviderConfigError(f"{name}={raw!r} must be a positive integer.") from exc
    if value <= 0:
        raise LLMProviderConfigError(f"{name}={raw!r} must be a positive integer.")
    return value


@dataclass(frozen=True)
class ProviderSettings:
    provider: str = field(default_factory=lambda: _provider("LLM_PROVIDER", "openai"))
    embedding_provider: str = field(
        default_factory=lambda: _provider("EMBEDDING_PROVIDER", _env_str("LLM_PROVIDER", "openai"))
    )
    ollama_host: str = field(
        default_factory=lambda: _env_str("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).rstrip("/")
    )
    ollama_model: str = field(default_factory=lambda: _env_str("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL))
    ollama_embedding_model: str = field(
        default_factory=lambda: _env_str("OLLAMA_EMBEDDING_MODEL", DEFAULT_OLLAMA_EMBEDDING_MODEL)
    )
    ollama_model_map: dict[str, str] = field(
        default_factory=lambda: parse_model_map(os.getenv("OLLAMA_MODEL_MAP", ""))
    )
    # qwen3 "thinks" before answering by default; measured ~16x slower for the
    # same classification, so it is off unless explicitly enabled.
    ollama_think: bool = field(default_factory=lambda: _env_bool("OLLAMA_THINK", False))
    # Context window for LangChain (ChatOllama) calls. The OpenAI-compatible
    # endpoint ignores per-request num_ctx, so also set OLLAMA_CONTEXT_LENGTH on
    # the Ollama server to the same value (see docs/LLM_PROVIDER.md).
    ollama_num_ctx: int = field(
        default_factory=lambda: _optional_positive_int("OLLAMA_NUM_CTX") or 16384
    )
    # Replaces the modules' OpenAI timeouts (30-60s) on Ollama: local inference,
    # especially on CPU, is much slower than the hosted API.
    ollama_timeout: float = field(
        default_factory=lambda: float(_optional_positive_int("OLLAMA_TIMEOUT") or 300)
    )

    def __post_init__(self) -> None:
        if not self.ollama_host.startswith(("http://", "https://")):
            raise LLMProviderConfigError(
                f"OLLAMA_HOST={self.ollama_host!r} must start with http:// or https://."
            )

    @property
    def is_ollama(self) -> bool:
        return self.provider == "ollama"

    @property
    def embeddings_on_ollama(self) -> bool:
        return self.embedding_provider == "ollama"


@lru_cache(maxsize=1)
def get_provider_settings() -> ProviderSettings:
    return ProviderSettings()
