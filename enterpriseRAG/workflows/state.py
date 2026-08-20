from __future__ import annotations

from typing import Any, TypedDict


class EnterpriseRAGState(TypedDict, total=False):
    # Input
    query: str
    session_id: str
    role: str

    # Intent & planning
    intent: dict[str, Any]
    answer_profile: dict[str, Any]
    quality_validation: dict[str, Any]
    complexity: str
    question_type: str
    role_config: dict[str, Any]
    subqueries: list[str]
    query_variations: list[str]
    retrieval_strategy: str

    # Retrieval
    raw_chunks: list[dict[str, Any]]
    compressed_chunks: list[dict[str, Any]]
    aggregated_evidence: list[dict[str, Any]]
    reranked_chunks: list[dict[str, Any]]

    # Verification & generation
    verification: dict[str, Any]
    confidence: float
    retrieval_loop_count: int
    draft_answer: str
    final_answer: str
    citations: list[dict[str, str]]

    # Reflection
    critic_feedback: dict[str, Any]
    reflection: dict[str, Any]
    reflection_loop_count: int
    hallucination_check: dict[str, Any]
    governance_check: dict[str, Any]

    # Guardrails
    input_guardrails: dict[str, Any]
    retrieval_guardrails: dict[str, Any]
    generation_guardrails: dict[str, Any]
    output_guardrails: dict[str, Any]
    guardrail_status: dict[str, Any]

    # Memory & entities
    entities: dict[str, list[str]]
    conversation_history: list[dict[str, str]]

    # Observability
    trace_id: str
    document_name: str
    tools_used: list[str]
    blocked: bool
    blocked_reason: str
    error: str


def initial_state(
    query: str,
    *,
    session_id: str = "default",
    role: str = "Enterprise Architect",
    trace_id: str = "",
) -> EnterpriseRAGState:
    return {
        "query": query,
        "session_id": session_id,
        "role": role,
        "trace_id": trace_id,
        "subqueries": [],
        "query_variations": [],
        "raw_chunks": [],
        "compressed_chunks": [],
        "aggregated_evidence": [],
        "reranked_chunks": [],
        "retrieval_loop_count": 0,
        "reflection_loop_count": 0,
        "confidence": 0.0,
        "tools_used": [],
        "blocked": False,
        "citations": [],
        "entities": {},
        "conversation_history": [],
    }
