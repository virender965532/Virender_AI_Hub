"""LoadJobSourceConfiguration node.

Loads ``config/job_sources.yaml``, filters to enabled sources (and, if the
caller asked for specific ``request["sources"]``, to that subset), and caps
the count at ``JOB_MAX_SOURCES`` (section 62 — performance/backpressure).
"""

from __future__ import annotations

import logging
from typing import Any

from ...config.settings import get_settings
from ...config.sources import enabled_sources, load_job_sources
from ...models.errors import ErrorSeverity, ErrorType, WorkflowError

logger = logging.getLogger(__name__)


async def source_configuration_node(state: dict[str, Any]) -> dict[str, Any]:
    run_id = state.get("run_id")
    settings = get_settings()
    request = state.get("request") or {}
    requested_ids = request.get("sources")

    try:
        all_sources = load_job_sources()
    except Exception as exc:  # noqa: BLE001
        logger.exception("source_configuration_node: failed to load job source config")
        return {
            "errors": [
                WorkflowError(
                    error_code="source_config_load_failed",
                    error_type=ErrorType.CONFIGURATION,
                    message=str(exc),
                    node="source_configuration",
                    severity=ErrorSeverity.CRITICAL,
                    retryable=False,
                    run_id=run_id,
                ).model_dump(mode="json")
            ],
            "source_configs": [],
        }

    sources = enabled_sources(all_sources)
    if requested_ids:
        requested = set(requested_ids)
        sources = [s for s in sources if s.id in requested]

    if len(sources) > settings.max_sources_per_run:
        logger.warning(
            "Capping %s enabled sources to configured max_sources_per_run=%s",
            len(sources),
            settings.max_sources_per_run,
        )
        sources = sources[: settings.max_sources_per_run]

    logger.info("source_configuration_node: %s source(s) selected: %s", len(sources), [s.id for s in sources])
    return {"source_configs": [s.model_dump(mode="json") for s in sources]}
