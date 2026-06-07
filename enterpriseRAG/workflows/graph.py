from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from enterpriseRAG.agents.orchestrator import AgentOrchestrator
from enterpriseRAG.guardrails.pipeline import GuardrailPipeline
from enterpriseRAG.memory.store import MemoryStore
from enterpriseRAG.retrievers.engine import (
    BM25Retriever,
    CompressionRetriever,
    HybridRetriever,
    MultiQueryRetriever,
    ParentChildRetriever,
    VectorRetriever,
)
from enterpriseRAG.services.document_service import DocumentService
from enterpriseRAG.services.llm_service import LLMService
from enterpriseRAG.services.observability import ObservabilityService, new_trace_id
from enterpriseRAG.services.reranker_service import RerankerService
from enterpriseRAG.tools.registry import ToolRegistry
from enterpriseRAG.workflows.state import EnterpriseRAGState, initial_state

logger = logging.getLogger(__name__)

_compiled_graph = None
_shared_obs: ObservabilityService | None = None


def _get_shared_obs() -> ObservabilityService:
    global _shared_obs
    if _shared_obs is None:
        _shared_obs = ObservabilityService()
    return _shared_obs


def _build_services() -> tuple[
    ObservabilityService,
    LLMService,
    DocumentService,
    RerankerService,
    GuardrailPipeline,
    MemoryStore,
    ToolRegistry,
    AgentOrchestrator,
    dict[str, Any],
]:
    obs = _get_shared_obs()
    llm = LLMService(observability=obs)
    doc_service = DocumentService(observability=obs)
    reranker = RerankerService(observability=obs)
    guardrails = GuardrailPipeline(observability=obs)
    memory = MemoryStore()

    vector = VectorRetriever(doc_service, llm, obs)
    bm25 = BM25Retriever(doc_service, obs)
    hybrid = HybridRetriever(vector, bm25, doc_service, obs)
    parent_child = ParentChildRetriever(vector, doc_service, obs)
    multi_query = MultiQueryRetriever(hybrid, llm, obs)
    compression = CompressionRetriever(llm, obs)

    retrievers: dict[str, Any] = {
        "vector": vector,
        "bm25": bm25,
        "hybrid": hybrid,
        "parent_child": parent_child,
        "multi_query": multi_query,
        "compression": compression,
    }

    tools = ToolRegistry(doc_service, llm, reranker, obs, retrievers)
    orchestrator = AgentOrchestrator(
        llm, doc_service, reranker, tools, guardrails, memory, obs, retrievers
    )
    return obs, llm, doc_service, reranker, guardrails, memory, tools, orchestrator, retrievers


def _route_after_input(state: EnterpriseRAGState) -> Literal["intent_agent", "__end__"]:
    if state.get("blocked"):
        return END
    return "intent_agent"


def _route_after_verification(state: EnterpriseRAGState) -> Literal["retrieval_agent", "rerank_agent"]:
    from enterpriseRAG.config.settings import get_settings

    verification = state.get("verification", {})
    loop_count = state.get("retrieval_loop_count", 0)
    max_loops = get_settings().max_retrieval_loops
    if verification.get("needs_reretrieval") and loop_count < max_loops:
        return "retrieval_agent"
    return "rerank_agent"


def _route_after_reflection(state: EnterpriseRAGState) -> Literal["answer_generation_agent", "hallucination_detection_agent"]:
    from enterpriseRAG.config.settings import get_settings

    reflection = state.get("reflection", {})
    loop_count = state.get("reflection_loop_count", 0)
    max_loops = get_settings().max_reflection_loops
    if reflection.get("action") == "regenerate" and loop_count < max_loops:
        return "answer_generation_agent"
    return "hallucination_detection_agent"


def build_graph() -> Any:
    global _compiled_graph
    if _compiled_graph is not None:
        return _compiled_graph

    _, _, _, _, _, _, _, orchestrator, _ = _build_services()

    graph = StateGraph(EnterpriseRAGState)

    # Register all agent nodes
    agents = [
        ("input_guardrails", orchestrator.input_guardrails_agent),
        ("intent_agent", orchestrator.intent_agent),
        ("role_intelligence_agent", orchestrator.role_intelligence_agent),
        ("query_planning_agent", orchestrator.query_planning_agent),
        ("multi_query_agent", orchestrator.multi_query_agent),
        ("retrieval_strategy_agent", orchestrator.retrieval_strategy_agent),
        ("retrieval_agent", orchestrator.retrieval_agent),
        ("context_compression_agent", orchestrator.context_compression_agent),
        ("evidence_aggregation_agent", orchestrator.evidence_aggregation_agent),
        ("verification_agent", orchestrator.verification_agent),
        ("rerank_agent", orchestrator.rerank_agent),
        ("citation_agent", orchestrator.citation_agent),
        ("answer_generation_agent", orchestrator.answer_generation_agent),
        ("critic_agent", orchestrator.critic_agent),
        ("reflection_agent", orchestrator.reflection_agent),
        ("hallucination_detection_agent", orchestrator.hallucination_detection_agent),
        ("governance_agent", orchestrator.governance_agent),
        ("response_formatting_agent", orchestrator.response_formatting_agent),
    ]

    for name, handler in agents:
        graph.add_node(name, lambda s, h=handler, n=name: orchestrator._run_agent(n, h, s))

    # Workflow edges
    graph.add_edge(START, "input_guardrails")
    graph.add_conditional_edges("input_guardrails", _route_after_input)
    graph.add_edge("intent_agent", "role_intelligence_agent")
    graph.add_edge("role_intelligence_agent", "query_planning_agent")
    graph.add_edge("query_planning_agent", "multi_query_agent")
    graph.add_edge("multi_query_agent", "retrieval_strategy_agent")
    graph.add_edge("retrieval_strategy_agent", "retrieval_agent")
    graph.add_edge("retrieval_agent", "context_compression_agent")
    graph.add_edge("context_compression_agent", "evidence_aggregation_agent")
    graph.add_edge("evidence_aggregation_agent", "verification_agent")
    graph.add_conditional_edges("verification_agent", _route_after_verification)
    graph.add_edge("rerank_agent", "citation_agent")
    graph.add_edge("citation_agent", "answer_generation_agent")
    graph.add_edge("answer_generation_agent", "critic_agent")
    graph.add_edge("critic_agent", "reflection_agent")
    graph.add_conditional_edges("reflection_agent", _route_after_reflection)
    graph.add_edge("hallucination_detection_agent", "governance_agent")
    graph.add_edge("governance_agent", "response_formatting_agent")
    graph.add_edge("response_formatting_agent", END)

    _compiled_graph = graph.compile()
    logger.info("Enterprise RAG LangGraph compiled with 18 agents.")
    return _compiled_graph


def run_enterprise_rag_workflow(
    query: str,
    *,
    session_id: str = "default",
    role: str = "Enterprise Architect",
    trace_id: str | None = None,
) -> tuple[EnterpriseRAGState, ObservabilityService]:
    obs = _get_shared_obs()
    tid = trace_id or new_trace_id()
    obs.start_workflow(tid)

    state = initial_state(query, session_id=session_id, role=role, trace_id=tid)
    app = build_graph()
    result: EnterpriseRAGState = app.invoke(state)
    return result, obs
