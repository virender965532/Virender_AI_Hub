"""FinalOutput node: assembles the ``JobSearchResult`` the API returns.

``workflow_duration_ms`` is filled in by the service layer after
``graph.ainvoke`` returns (wall-clock timing belongs outside the graph, not
inside one of its nodes) — see ``services/job_engine_service.py``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _determine_status(state: dict[str, Any]) -> str:
    errors = state.get("errors") or []
    critical = [e for e in errors if e.get("severity") == "critical"]
    validated = state.get("validated_jobs") or []

    if critical and not validated:
        return "failed"
    if critical or state.get("degraded") or state.get("source_errors"):
        return "partial"
    return "completed"


async def final_output_node(state: dict[str, Any]) -> dict[str, Any]:
    run_id = state.get("run_id", "")
    source_results = state.get("source_results") or {}
    discovered = state.get("discovered_jobs") or []
    validated = state.get("validated_jobs") or []
    duplicates = state.get("duplicate_jobs") or []
    matched = state.get("matched_jobs") or []
    ranked = state.get("ranked_jobs") or []

    metrics = dict(state.get("metrics") or {})
    metrics.update(
        {
            "run_id": run_id,
            "jobs_discovered": len(discovered),
            "jobs_validated": len(validated),
            "jobs_deduplicated": len(duplicates),
            "source_failures": sum(1 for r in source_results.values() if r.get("status") == "failed"),
            "browser_failures": sum(
                1 for e in (state.get("source_errors") or []) if e.get("status") in ("timeout", "navigation_error")
            ),
        }
    )

    status = _determine_status(state)
    summary = {
        "sources_processed": len(source_results),
        "jobs_discovered": len(discovered),
        "jobs_validated": len(validated),
        "duplicates_removed": len(duplicates),
        "jobs_matched": len(matched),
        "jobs_recommended": len(ranked),
        "degraded": bool(state.get("degraded")),
        "degraded_reason": state.get("degraded_reason"),
    }

    final_result = {
        "run_id": run_id,
        "status": status,
        "summary": summary,
        "jobs": ranked,
        "source_errors": state.get("source_errors") or [],
        "metrics": metrics,
    }

    logger.info(
        "final_output_node: run=%s status=%s sources=%s discovered=%s recommended=%s",
        run_id, status, summary["sources_processed"], summary["jobs_discovered"], summary["jobs_recommended"],
    )
    return {"final_result": final_result, "metrics": metrics}
