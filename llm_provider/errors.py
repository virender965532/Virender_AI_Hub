"""Errors raised by the LLM provider layer (configuration and local-service checks)."""

from __future__ import annotations


class LLMProviderError(RuntimeError):
    """Base class for provider-layer errors."""


class LLMProviderConfigError(LLMProviderError, ValueError):
    """An invalid provider setting (unknown LLM_PROVIDER, bad OLLAMA_HOST...)."""


class OllamaUnavailableError(LLMProviderError):
    """The Ollama server at OLLAMA_HOST cannot be reached."""


class OllamaModelNotFoundError(LLMProviderError):
    """A configured model has not been pulled into the local Ollama server."""
