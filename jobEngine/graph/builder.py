"""Builds the LangGraph ``StateGraph`` for one run (section 6/31).

Unlike this repo's other two LangGraph usages (``jobSearch``, ``enterpriseRAG``,
both compiled once as module-level singletons with a fixed node set), this
graph's node set depends on which sources are enabled/requested for a given
run — so it is deliberately rebuilt (cheap: this is graph *wiring*, not
browser/LLM work) per run from the resolved ``JobSourceConfig`` list, rather
than cached as a singleton. See docs/job_engine/ARCHITECTURE.md.

::

    START -> input_validation -> profile_loader -> source_configuration
           -> {source_node_1, source_node_2, ...}   (fan-out, parallel)
           -> normalization                          (fan-in)
           -> validation -> matching
           -> guardrail_check -> ranking -> final_output -> END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from ..config.sources import JobSourceConfig
from .nodes.final_output import final_output_node
from .nodes.guardrail_check import guardrail_check_node
from .nodes.input_validation import input_validation_node
from .nodes.matching import matching_node
from .nodes.normalization import normalization_node
from .nodes.profile_loader import profile_loader_node
from .nodes.ranking import ranking_node
from .nodes.source_configuration import source_configuration_node
from .nodes.source_discovery import create_source_node
from .nodes.validation import validation_node
from .state import JobDiscoveryState


def build_graph(source_configs: list[JobSourceConfig]) -> Any:
    graph = StateGraph(JobDiscoveryState)

    graph.add_node("input_validation", input_validation_node)
    graph.add_node("profile_loader", profile_loader_node)
    graph.add_node("source_configuration", source_configuration_node)
    graph.add_node("normalization", normalization_node)
    graph.add_node("validation", validation_node)
    graph.add_node("matching", matching_node)
    graph.add_node("guardrail_check", guardrail_check_node)
    graph.add_node("ranking", ranking_node)
    graph.add_node("final_output", final_output_node)

    graph.add_edge(START, "input_validation")
    graph.add_edge("input_validation", "profile_loader")
    graph.add_edge("profile_loader", "source_configuration")

    if source_configs:
        print(source_configs)
        for cfg in source_configs:
            graph.add_node(cfg.graph_node, create_source_node(cfg))
            graph.add_edge("source_configuration", cfg.graph_node)
            graph.add_edge(cfg.graph_node, "normalization")
    else:
        graph.add_edge("source_configuration", "normalization")

    graph.add_edge("normalization", "validation")
    graph.add_edge("validation", "matching")
    graph.add_edge("matching", "guardrail_check")
    graph.add_edge("guardrail_check", "ranking")
    graph.add_edge("ranking", "final_output")
    graph.add_edge("final_output", END)

    return graph.compile()


def draw_graph_png(source_configs: list[JobSourceConfig]) -> bytes:
    """PNG of the graph wiring for these sources (rendered via mermaid.ink, needs network)."""
    return build_graph(source_configs).get_graph().draw_mermaid_png()


def display_graph(source_configs: list[JobSourceConfig]) -> None:
    """Show the graph inline in a Jupyter notebook."""
    from IPython.display import Image, display

    graph = build_graph(source_configs)
    display(
        Image(graph.get_graph().draw_mermaid_png())
    )
