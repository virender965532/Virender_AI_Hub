"""Explicit checks of the local Ollama setup (service reachable, models pulled).

Nothing here downloads a model: a missing one is reported with the
``ollama pull`` command to run. Used by ``python -m llm_provider check``, the
job engine health endpoint and the app's startup log.
"""

from __future__ import annotations

import logging
from typing import Any

from .errors import OllamaModelNotFoundError, OllamaUnavailableError
from .settings import ProviderSettings, get_provider_settings

logger = logging.getLogger(__name__)


def _model_names(listing: Any) -> set[str]:
    models = getattr(listing, "models", None)
    if models is None and isinstance(listing, dict):
        models = listing.get("models", [])
    names: set[str] = set()
    for m in models or []:
        name = getattr(m, "model", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else None)
        if name:
            names.add(name)
            if name.endswith(":latest"):
                names.add(name[: -len(":latest")])
    return names


def _with_latest(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def installed_models(settings: ProviderSettings | None = None, *, client: Any = None) -> set[str]:
    """Names of the models pulled into the Ollama server. Raises ``OllamaUnavailableError``."""
    s = settings or get_provider_settings()
    try:
        if client is None:
            import ollama

            client = ollama.Client(host=s.ollama_host, timeout=10)
        return _model_names(client.list())
    except ImportError as exc:
        raise OllamaUnavailableError(
            "The 'ollama' Python package is not installed; run: pip install -r requirements.txt"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - httpx/connection errors vary by platform
        raise OllamaUnavailableError(
            f"Ollama is not reachable at {s.ollama_host} ({type(exc).__name__}: {exc}). "
            "Start the Ollama app (or run `ollama serve`) or fix OLLAMA_HOST."
        ) from exc


def required_models(settings: ProviderSettings | None = None, *, include_embeddings: bool = True) -> list[str]:
    s = settings or get_provider_settings()
    models: list[str] = []
    if s.is_ollama:
        models.append(s.ollama_model)
        models.extend(v for v in s.ollama_model_map.values() if v not in models)
    if include_embeddings and s.embeddings_on_ollama and s.ollama_embedding_model not in models:
        models.append(s.ollama_embedding_model)
    return models


def check_ollama(
    settings: ProviderSettings | None = None, *, include_embeddings: bool = True, client: Any = None
) -> dict[str, Any]:
    """Verify the server and every configured model; raises a provider error with a fix-it hint."""
    s = settings or get_provider_settings()
    installed = installed_models(s, client=client)
    needed = required_models(s, include_embeddings=include_embeddings)
    missing = [m for m in needed if m not in installed and _with_latest(m) not in installed]
    if missing:
        commands = "; ".join(f"ollama pull {m}" for m in missing)
        raise OllamaModelNotFoundError(
            f"Ollama at {s.ollama_host} is missing model(s) {', '.join(missing)}. Download with: {commands}"
        )
    return {"ok": True, "host": s.ollama_host, "models": needed}


def provider_health(settings: ProviderSettings | None = None, *, include_embeddings: bool = True) -> dict[str, Any]:
    """Never raises; ``{"ok": False, "error": ...}`` when Ollama or a model is missing."""
    s = settings or get_provider_settings()
    if not (s.is_ollama or (include_embeddings and s.embeddings_on_ollama)):
        return {"ok": True, "provider": s.provider}
    try:
        return {"provider": s.provider, **check_ollama(s, include_embeddings=include_embeddings)}
    except (OllamaUnavailableError, OllamaModelNotFoundError) as exc:
        return {"ok": False, "provider": s.provider, "host": s.ollama_host, "error": str(exc)}


def log_startup_status() -> None:
    """One log line about the active provider; warns (does not fail) if Ollama isn't ready."""
    try:
        s = get_provider_settings()
    except Exception as exc:  # noqa: BLE001
        logger.error("Invalid LLM provider configuration: %s", exc)
        return
    if not (s.is_ollama or s.embeddings_on_ollama):
        logger.info("LLM provider: openai")
        return
    status = provider_health(s)
    if status["ok"]:
        logger.info("LLM provider: %s (Ollama %s, models %s)", s.provider, s.ollama_host, ", ".join(status["models"]))
    else:
        logger.warning("LLM provider: %s — %s", s.provider, status["error"])
