"""Health checks (section 51). Never returns secrets — only booleans/strings
about availability.
"""

from __future__ import annotations

import importlib.util
from typing import Any

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
    configured = bool(settings.openai_api_key)
    return {
        "ok": configured and settings.llm_enabled,
        "configured": configured,
        "enabled": settings.llm_enabled,
        "model": settings.openai_model,
    }


def check_embedding_provider() -> dict[str, Any]:
    settings = get_settings()
    configured = bool(settings.openai_api_key)
    return {
        "ok": configured and settings.embeddings_enabled,
        "configured": configured,
        "enabled": settings.embeddings_enabled,
        "model": settings.embedding_model,
    }


def check_browser_available() -> dict[str, Any]:
    available = importlib.util.find_spec("playwright") is not None
    return {"ok": available}


def check_database() -> dict[str, Any]:
    from ..config.settings import JOB_ENGINE_OUTPUT_DIR

    try:
        JOB_ENGINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        probe = JOB_ENGINE_OUTPUT_DIR / ".health_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"ok": True, "backend": "flat-file (output/job_engine)"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def build_health_report() -> dict[str, Any]:
    config = check_configuration()
    checks = {
        "configuration": config,
        "database": check_database(),
        "llm_provider": check_llm_provider(),
        "embedding_provider": check_embedding_provider(),
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
