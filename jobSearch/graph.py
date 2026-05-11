"""LangGraph StateGraph: login → fetch → display."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes.display_jobs_node import display_jobs_node
from .nodes.fetch_jobs_node import fetch_jobs_node
from .nodes.login_node import login_node
from .state import WorkflowState, initial_workflow_state

logger = logging.getLogger(__name__)


async def _login_wrapper(state: WorkflowState) -> dict[str, Any]:
    return await login_node(state)


async def _fetch_wrapper(state: WorkflowState) -> dict[str, Any]:
    return await fetch_jobs_node(state)


async def _display_wrapper(state: WorkflowState) -> dict[str, Any]:
    return await display_jobs_node(state)


def build_graph() -> Any:
    graph = StateGraph(WorkflowState)
    graph.add_node("login_node", _login_wrapper)
    graph.add_node("fetch_jobs_node", _fetch_wrapper)
    graph.add_node("display_jobs_node", _display_wrapper)

    graph.add_edge(START, "login_node")
    graph.add_edge("login_node", "fetch_jobs_node")
    graph.add_edge("fetch_jobs_node", "display_jobs_node")
    graph.add_edge("display_jobs_node", END)

    compiled = graph.compile()
    logger.debug("Compiled job search LangGraph.")
    return compiled


async def run_job_search_workflow(
    *,
    initial_state: WorkflowState | None = None,
) -> WorkflowState:
    """Run full workflow; returns final merged state."""
    app = build_graph()
    base = initial_state if initial_state is not None else initial_workflow_state()
    result: WorkflowState = await app.ainvoke(base)
    return result
