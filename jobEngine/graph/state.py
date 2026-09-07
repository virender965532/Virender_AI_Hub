"""LangGraph state for the job discovery workflow (section 8).

Source nodes run as real parallel graph branches (one node per enabled,
configured source — see ``graph/builder.py``), so every field a source node
writes to must merge across concurrent branches instead of being clobbered by
"last write wins". ``list``/``dict`` fields a source node touches use
``Annotated`` reducers for that reason; everything else (sequential,
single-writer nodes) is a plain field.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge_dicts(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    return {**a, **b}


class JobDiscoveryState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]

    # Live BrowserManager for this run only — intentionally NOT serializable,
    # mirroring the existing ``jobSearch`` graph's ``WorkflowState.session``
    # convention (a live Playwright session held directly in state). Created
    # once per run by the service layer, read-only for every source node.
    browser_manager: Any

    candidate_profile: dict[str, Any]
    source_configs: list[dict[str, Any]]

    # --- written concurrently by per-source nodes; must use reducers ---
    discovered_jobs: Annotated[list[dict[str, Any]], operator.add]
    source_results: Annotated[dict[str, Any], _merge_dicts]
    source_errors: Annotated[list[dict[str, Any]], operator.add]

    # --- sequential, single-writer from here on ---
    normalized_jobs: list[dict[str, Any]]
    validated_jobs: list[dict[str, Any]]
    duplicate_jobs: list[dict[str, Any]]
    prefiltered_jobs: list[dict[str, Any]]
    matched_jobs: list[dict[str, Any]]
    ranked_jobs: list[dict[str, Any]]

    guardrail_results: list[dict[str, Any]]
    metrics: dict[str, Any]
    errors: Annotated[list[dict[str, Any]], operator.add]

    degraded: bool
    degraded_reason: str | None
    final_result: dict[str, Any]


def initial_state(*, run_id: str, request: dict[str, Any]) -> JobDiscoveryState:
    return {
        "run_id": run_id,
        "request": request,
        "source_configs": [],
        "discovered_jobs": [],
        "source_results": {},
        "source_errors": [],
        "normalized_jobs": [],
        "validated_jobs": [],
        "duplicate_jobs": [],
        "prefiltered_jobs": [],
        "matched_jobs": [],
        "ranked_jobs": [],
        "guardrail_results": [],
        "metrics": {},
        "errors": [],
        "degraded": False,
        "degraded_reason": None,
    }
