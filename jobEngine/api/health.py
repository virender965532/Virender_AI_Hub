"""Health checks (section 51). Never returns secrets — only booleans/strings
about availability.
"""

from __future__ import annotations

import importlib.util
from typing import Any

from llm_provider import llm_configured, provider_health, provider_name, resolve_model

from ..config.settings import JOB_SOURCES_CONFIG_PATH, get_settings
from ..config.sources import enabled_sources, load_job_sources


def check_configuration() -> dict[str, Any]:
    try:
        sources = load_job_sources()
        return {"ok": True, "path": str(JOB_SOURCES_CONFIG_PATH), "source_count": len(sources)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def check_llm_provider() -> dict[str, Any]:
    settings = get_settings()
    configured = llm_configured(settings.openai_api_key)
    status = {
        "provider": provider_name(),
        "configured": configured,
        "enabled": settings.llm_enabled,
        "model": resolve_model(settings.openai_model),
    }
    service = provider_health(include_embeddings=False)  # the job engine embeds nothing
    if not service["ok"]:
        status["error"] = service["error"]
    return {"ok": configured and settings.llm_enabled and service["ok"], **status}


def check_browser_available() -> dict[str, Any]:
    available = importlib.util.find_spec("playwright") is not None
    return {"ok": available}


def check_database() -> dict[str, Any]:
    from ..persistence.mongo import ping

    try:
        ping()
        return {"ok": True, "backend": "mongodb"}
    except Exception as exc:  # noqa: BLE001
        # Exception text can include the host; never the URI's credentials.
        return {"ok": False, "backend": "mongodb", "error": type(exc).__name__}


def build_health_report() -> dict[str, Any]:
    config = check_configuration()
    checks = {
        "configuration": config,
        "database": check_database(),
        "llm_provider": check_llm_provider(),
        "browser": check_browser_available(),
    }
    enabled = []
    if config.get("ok"):
        try:
            enabled = [s.id for s in enabled_sources(load_job_sources())]
        except Exception:  # noqa: BLE001
            enabled = []
    checks["enabled_sources"] = enabled
    checks["ok"] = all(bool(c.get("ok", True)) for c in checks.values() if isinstance(c, dict))
    return checks
