"""First graph node: defensive re-validation of the search request.

The Flask API layer (``api/schemas.py``/``JobSearchRequest``) already rejects
malformed requests before a graph run is even started; this node is a second,
independent checkpoint (defense in depth) that records a ``WorkflowError``
rather than crashing if something unexpected still made it through.
"""

from __future__ import annotations

import logging
from typing import Any

from ...models.errors import ErrorSeverity, ErrorType, WorkflowError

logger = logging.getLogger(__name__)


async def input_validation_node(state: dict[str, Any]) -> dict[str, Any]:
    request = state.get("request") or {}
    errors: list[dict[str, Any]] = []

    max_results = request.get("max_results")
    if max_results is not None and (not isinstance(max_results, int) or max_results <= 0):
        errors.append(
            WorkflowError(
                error_code="invalid_max_results",
                error_type=ErrorType.VALIDATION,
                message=f"max_results must be a positive integer, got {max_results!r}",
                node="input_validation",
                severity=ErrorSeverity.LOW,
                run_id=state.get("run_id"),
            ).model_dump(mode="json")
        )

    sources = request.get("sources")
    if sources is not None and not isinstance(sources, list):
        errors.append(
            WorkflowError(
                error_code="invalid_sources",
                error_type=ErrorType.VALIDATION,
                message=f"sources must be a list, got {type(sources).__name__}",
                node="input_validation",
                severity=ErrorSeverity.LOW,
                run_id=state.get("run_id"),
            ).model_dump(mode="json")
        )

    if errors:
        logger.warning("input_validation_node: %s issue(s) found: %s", len(errors), errors)
    return {"errors": errors}
