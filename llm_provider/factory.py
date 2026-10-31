"""The one place that decides which LLM backend serves a call.

Consumers ask for a capability, never a provider:

* ``get_chat_model`` — a LangChain chat model (``ChatOpenAI`` / ``ChatOllama``);
  ``with_structured_output(..., include_raw=True)`` behaves the same on both.
* ``get_embeddings`` — a LangChain embeddings object (``OpenAIEmbeddings`` /
  ``OllamaEmbeddings``).
* ``get_openai_client`` / ``get_async_openai_client`` — an OpenAI-SDK client.
  On OpenAI this is the plain ``openai.OpenAI``. On Ollama it is an
  ``openai.OpenAI`` pointed at Ollama's OpenAI-compatible ``/v1`` endpoint,
  behind a small adapter that maps model names, turns qwen3 "thinking" off
  (unless ``OLLAMA_THINK=true``) and drops OpenAI-only request fields. The
  same SDK call shapes keep working (``chat.completions.create/parse`` incl.
  tool calls and ``response_format``, ``embeddings.create``) and errors keep
  their ``openai.*`` types, so each module's retry/fallback logic is unchanged.
* ``get_agents_model`` — a model for the OpenAI Agents SDK (``agents.Agent``).

Model names passed in are each module's configured OpenAI names; on Ollama
they are mapped with ``resolve_model`` / ``resolve_embedding_model``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import openai

from .settings import ProviderSettings, get_provider_settings

logger = logging.getLogger(__name__)

_OLLAMA_PLACEHOLDER_KEY = "ollama"  # the SDK requires a key; Ollama ignores it
# Accepted by OpenAI, meaningless to Ollama (stored completions / dashboard metadata).
_OPENAI_ONLY_CHAT_FIELDS = ("store", "metadata", "service_tier", "prediction")

_instances: dict[tuple[Any, ...], Any] = {}
_instances_lock = threading.Lock()
_warned_shared: set[tuple[Any, ...]] = set()  # each shared-model warning is logged once per process


def _cached(key: tuple[Any, ...], build):
    """Reuse model/embedding instances across modules (keyed by their full config)."""
    with _instances_lock:
        if key not in _instances:
            _instances[key] = build()
        return _instances[key]


def reset_caches() -> None:
    """Forget cached settings and instances (tests, or after changing env vars)."""
    with _instances_lock:
        _instances.clear()
        _warned_shared.clear()
    get_provider_settings.cache_clear()


# --- naming --------------------------------------------------------------------------------


def provider_name() -> str:
    return get_provider_settings().provider


def _map_name(model: str, default: str, settings: ProviderSettings) -> str:
    name = (model or "").strip()
    mapped = settings.ollama_model_map.get(name.lower())
    if mapped:
        return mapped
    if ":" in name:  # already an Ollama tag such as "qwen3:14b"
        return name
    return default


def resolve_model(model: str, settings: ProviderSettings | None = None) -> str:
    """The model that will actually serve a call configured for ``model``."""
    settings = settings or get_provider_settings()
    if not settings.is_ollama:
        return model
    return _map_name(model, settings.ollama_model, settings)


def resolve_embedding_model(model: str, settings: ProviderSettings | None = None) -> str:
    settings = settings or get_provider_settings()
    if not settings.embeddings_on_ollama:
        return model
    return _map_name(model, settings.ollama_embedding_model, settings)


def warn_if_shared_model(module: str, roles: dict[str, str]) -> None:
    """Warn when roles that must be independent (answer vs critic) resolve to one Ollama model.

    Their configured (OpenAI) names are still validated as different by each
    module; on a single local model the critic is a second pass of the same model.
    """
    s = get_provider_settings()
    if not s.is_ollama:
        return
    by_model: dict[str, list[str]] = {}
    for role, model in roles.items():
        by_model.setdefault(resolve_model(model, s), []).append(role)
    for model, shared in by_model.items():
        key = (module, model, tuple(shared))
        if len(shared) > 1 and key not in _warned_shared:
            _warned_shared.add(key)
            logger.warning(
                "%s: %s all run on Ollama model %r, so the critic is not an independent model. "
                "Map one of them to another pulled model with OLLAMA_MODEL_MAP (e.g. gpt-4o=llama3.1:8b).",
                module, " and ".join(shared), model,
            )


def llm_configured(openai_api_key: str | None) -> bool:
    """Whether chat calls can be made: Ollama needs no key; OpenAI needs ``openai_api_key``."""
    return get_provider_settings().is_ollama or bool(openai_api_key)


def embeddings_configured(openai_api_key: str | None) -> bool:
    return get_provider_settings().embeddings_on_ollama or bool(openai_api_key)


def missing_credentials_message() -> str:
    return "OPENAI_API_KEY is not set (LLM_PROVIDER=openai; set LLM_PROVIDER=ollama to use a local model)."


def describe() -> dict[str, Any]:
    """Non-secret summary of the active configuration (health endpoints, startup log)."""
    s = get_provider_settings()
    info: dict[str, Any] = {"provider": s.provider, "embedding_provider": s.embedding_provider}
    if s.is_ollama or s.embeddings_on_ollama:
        info.update(
            ollama_host=s.ollama_host,
            ollama_model=s.ollama_model,
            ollama_embedding_model=s.ollama_embedding_model,
            ollama_model_map=dict(s.ollama_model_map),
            ollama_think=s.ollama_think,
        )
    return info


# --- LangChain -----------------------------------------------------------------------------


def get_chat_model(
    model: str,
    *,
    openai_api_key: str | None = None,
    timeout: float | None = None,
    temperature: float | None = None,
    seed: int | None = None,
):
    """A LangChain chat model for ``model`` (an OpenAI model name) on the active provider."""
    s = get_provider_settings()
    if s.is_ollama:
        name = resolve_model(model, s)

        def build():
            from langchain_ollama import ChatOllama

            kwargs: dict[str, Any] = {
                "model": name,
                "base_url": s.ollama_host,
                "num_ctx": s.ollama_num_ctx,
                # False = no thinking, and no <think> text inside the answer.
                "reasoning": s.ollama_think,
                "client_kwargs": {"timeout": s.ollama_timeout},
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if seed is not None:
                kwargs["seed"] = seed
            logger.debug("ChatOllama model=%s (configured %s) host=%s", name, model, s.ollama_host)
            return ChatOllama(**kwargs)

        key = ("chat", "ollama", name, temperature, seed, s.ollama_host, s.ollama_num_ctx, s.ollama_think,
               s.ollama_timeout)
        return _cached(key, build)

    def build_openai():
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {"model": model, "api_key": openai_api_key, "timeout": timeout}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        return ChatOpenAI(**kwargs)

    return _cached(("chat", "openai", model, openai_api_key, timeout, temperature, seed), build_openai)


def get_embeddings(model: str, *, openai_api_key: str | None = None):
    """A LangChain embeddings object for ``model`` (an OpenAI embedding model name)."""
    s = get_provider_settings()
    if s.embeddings_on_ollama:
        name = resolve_embedding_model(model, s)

        def build():
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(model=name, base_url=s.ollama_host)

        return _cached(("embed", "ollama", name, s.ollama_host), build)

    def build_openai():
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {"model": model}
        if openai_api_key:
            kwargs["api_key"] = openai_api_key
        return OpenAIEmbeddings(**kwargs)

    return _cached(("embed", "openai", model, openai_api_key), build_openai)


# --- OpenAI SDK ----------------------------------------------------------------------------


def _unset(value: Any) -> bool:
    return value is None or isinstance(value, (openai.NotGiven, openai.Omit))


def _log_ollama_error(exc: Exception, model: str, s: ProviderSettings, *, embedding: bool = False) -> None:
    if isinstance(exc, openai.NotFoundError):
        logger.error(
            "Ollama model %r is not available at %s. Download it first with: ollama pull %s",
            model, s.ollama_host, model,
        )
    elif isinstance(exc, openai.APIConnectionError):
        logger.error(
            "Ollama is not reachable at %s (%s). Start the Ollama app (or run `ollama serve`) "
            "or fix OLLAMA_HOST.", s.ollama_host, type(exc).__name__,
        )
    elif embedding and isinstance(exc, openai.APIStatusError) and exc.status_code == 501:
        logger.error(
            "Ollama model %r cannot create embeddings. Set OLLAMA_EMBEDDING_MODEL to an embedding "
            "model (default nomic-embed-text) and run: ollama pull <model>", model,
        )


def _chat_kwargs(kwargs: dict[str, Any], s: ProviderSettings) -> dict[str, Any]:
    out = {k: v for k, v in kwargs.items() if k not in _OPENAI_ONLY_CHAT_FIELDS}
    out["model"] = resolve_model(kwargs.get("model", ""), s)
    if not s.ollama_think and _unset(out.get("reasoning_effort")):
        out["reasoning_effort"] = "none"  # Ollama maps this to think=false
    if "timeout" in out and not _unset(out["timeout"]):
        out["timeout"] = s.ollama_timeout
    return out


def _embed_kwargs(kwargs: dict[str, Any], s: ProviderSettings) -> dict[str, Any]:
    return {**kwargs, "model": resolve_embedding_model(kwargs.get("model", ""), s)}


class _Completions:
    def __init__(self, inner: Any, s: ProviderSettings) -> None:
        self._inner, self._s = inner, s

    def _call(self, method: str, kwargs: dict[str, Any]) -> Any:
        adapted = _chat_kwargs(kwargs, self._s)
        try:
            return getattr(self._inner, method)(**adapted)
        except Exception as exc:
            _log_ollama_error(exc, adapted["model"], self._s)
            raise

    def create(self, **kwargs: Any) -> Any:
        return self._call("create", kwargs)

    def parse(self, **kwargs: Any) -> Any:
        return self._call("parse", kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _AsyncCompletions(_Completions):
    async def _acall(self, method: str, kwargs: dict[str, Any]) -> Any:
        adapted = _chat_kwargs(kwargs, self._s)
        try:
            return await getattr(self._inner, method)(**adapted)
        except Exception as exc:
            _log_ollama_error(exc, adapted["model"], self._s)
            raise

    def create(self, **kwargs: Any) -> Any:
        return self._acall("create", kwargs)

    def parse(self, **kwargs: Any) -> Any:
        return self._acall("parse", kwargs)


class _Embeddings:
    def __init__(self, inner: Any, s: ProviderSettings) -> None:
        self._inner, self._s = inner, s

    def create(self, **kwargs: Any) -> Any:
        if not self._s.embeddings_on_ollama:
            return self._inner.create(**kwargs)
        adapted = _embed_kwargs(kwargs, self._s)
        try:
            return self._inner.create(**adapted)
        except Exception as exc:
            _log_ollama_error(exc, adapted["model"], self._s, embedding=True)
            raise

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _Chat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


class ProviderClient:
    """OpenAI-SDK-shaped client whose chat and embedding calls go to the configured providers.

    Everything other than ``chat.completions`` and ``embeddings`` (``base_url``,
    ``with_options``...) is delegated to the chat client.
    """

    def __init__(self, chat_client: Any, embed_factory, s: ProviderSettings, *, is_async: bool = False) -> None:
        self._chat_client = chat_client
        self._embed_factory = embed_factory
        self._embed_client: Any | None = None
        self._s = s
        if s.is_ollama:
            self.chat = _Chat((_AsyncCompletions if is_async else _Completions)(chat_client.chat.completions, s))
        else:
            self.chat = chat_client.chat

    @property
    def embeddings(self) -> _Embeddings:
        # Built lazily: an OpenAI embedding client needs a key that an Ollama-only setup may not have.
        if self._embed_client is None:
            self._embed_client = self._embed_factory()
        return _Embeddings(self._embed_client.embeddings, self._s)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chat_client, name)


def _sdk_client(cls, provider: str, *, api_key: str | None, timeout: float | None, max_retries: int | None,
                s: ProviderSettings):
    kwargs: dict[str, Any] = {}
    if max_retries is not None:
        kwargs["max_retries"] = max_retries
    if provider == "ollama":
        return cls(base_url=f"{s.ollama_host}/v1", api_key=_OLLAMA_PLACEHOLDER_KEY, timeout=s.ollama_timeout, **kwargs)
    if timeout is not None:
        kwargs["timeout"] = timeout
    if api_key:
        kwargs["api_key"] = api_key
    return cls(**kwargs)


def _build_client(cls, *, api_key, timeout, max_retries, is_async: bool):
    s = get_provider_settings()
    if not s.is_ollama and not s.embeddings_on_ollama:
        # Unchanged OpenAI path: a plain SDK client, exactly as before the switch existed.
        return _sdk_client(cls, "openai", api_key=api_key, timeout=timeout, max_retries=max_retries, s=s)
    chat = _sdk_client(cls, s.provider, api_key=api_key, timeout=timeout, max_retries=max_retries, s=s)
    if s.embedding_provider == s.provider:
        embed_factory = lambda: chat  # noqa: E731
    else:
        embed_factory = lambda: _sdk_client(  # noqa: E731
            cls, s.embedding_provider, api_key=api_key, timeout=timeout, max_retries=max_retries, s=s
        )
    return ProviderClient(chat, embed_factory, s, is_async=is_async)


def get_openai_client(*, api_key: str | None = None, timeout: float | None = None,
                      max_retries: int | None = None) -> Any:
    """A sync OpenAI-SDK client for the active provider (see module docstring).

    On OpenAI, ``api_key`` (or ``OPENAI_API_KEY``) is required by the SDK as before.
    """
    return _build_client(openai.OpenAI, api_key=api_key, timeout=timeout, max_retries=max_retries, is_async=False)


def get_async_openai_client(*, api_key: str | None = None, timeout: float | None = None,
                            max_retries: int | None = None) -> Any:
    return _build_client(openai.AsyncOpenAI, api_key=api_key, timeout=timeout, max_retries=max_retries, is_async=True)


def get_agents_model(model: str) -> Any:
    """``model`` for ``agents.Agent(model=...)`` (OpenAI Agents SDK).

    OpenAI: the model name, unchanged. Ollama: a Chat Completions model over the
    local server; the SDK's trace upload (which needs an OpenAI key) is disabled.
    """
    s = get_provider_settings()
    if not s.is_ollama:
        return model
    from agents import OpenAIChatCompletionsModel, set_tracing_disabled

    set_tracing_disabled(True)
    return OpenAIChatCompletionsModel(model=resolve_model(model, s), openai_client=get_async_openai_client())
