"""Global LLM provider switch (``LLM_PROVIDER=openai|ollama``) for every module in the hub.

See ``llm_provider/factory.py`` for the entry points and docs/LLM_PROVIDER.md
for configuration and Windows setup.
"""

from .errors import LLMProviderConfigError, LLMProviderError, OllamaModelNotFoundError, OllamaUnavailableError
from .factory import (
    describe,
    embeddings_configured,
    get_agents_model,
    get_async_openai_client,
    get_chat_model,
    get_embeddings,
    get_openai_client,
    llm_configured,
    missing_credentials_message,
    provider_name,
    reset_caches,
    resolve_embedding_model,
    resolve_model,
    warn_if_shared_model,
)
from .health import check_ollama, log_startup_status, provider_health
from .settings import ProviderSettings, get_provider_settings

__all__ = [
    "LLMProviderConfigError",
    "LLMProviderError",
    "OllamaModelNotFoundError",
    "OllamaUnavailableError",
    "ProviderSettings",
    "check_ollama",
    "describe",
    "embeddings_configured",
    "get_agents_model",
    "get_async_openai_client",
    "get_chat_model",
    "get_embeddings",
    "get_openai_client",
    "get_provider_settings",
    "llm_configured",
    "log_startup_status",
    "missing_credentials_message",
    "provider_health",
    "provider_name",
    "reset_caches",
    "resolve_embedding_model",
    "resolve_model",
    "warn_if_shared_model",
]
