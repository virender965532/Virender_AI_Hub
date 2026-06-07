from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage

from enterpriseRAG.config.roles import list_roles
from enterpriseRAG.config.settings import SUPPORTED_EXTENSIONS, UPLOADS_DIR
from enterpriseRAG.services.document_service import DocumentService
from enterpriseRAG.services.observability import utc_now_iso
from enterpriseRAG.workflows.graph import run_enterprise_rag_workflow

logger = logging.getLogger(__name__)

_doc_service = DocumentService()


def save_uploaded_file(file_storage: FileStorage) -> Path:
    return _doc_service.save_upload(file_storage)


def get_uploaded_file() -> Path | None:
    return _doc_service.get_uploaded_file()


def get_uploaded_file_info() -> dict[str, Any] | None:
    uploaded = get_uploaded_file()
    if not uploaded:
        return None
    return {
        "name": uploaded.name,
        "extension": uploaded.suffix.lower(),
        "size_bytes": uploaded.stat().st_size,
    }


def get_document_preview_text(max_chars: int = 50000) -> str:
    return _doc_service.preview_text(max_chars=max_chars)


def get_supported_roles() -> list[dict[str, Any]]:
    return list_roles()


def ask_enterprise_rag_question(
    query: str,
    *,
    role: str = "Enterprise Architect",
    session_id: str = "default",
) -> dict[str, Any]:
    """Main entry point for the Enterprise Autonomous Multi-Agent RAG Platform."""
    cleaned = (query or "").strip()
    if not cleaned:
        raise ValueError("Question is required.")

    if not get_uploaded_file():
        raise FileNotFoundError(
            "No document uploaded. Please upload a PDF, TXT, Word, or Excel file first."
        )

    workflow_started = utc_now_iso()
    t0 = time.perf_counter()

    final_state, obs = run_enterprise_rag_workflow(
        cleaned,
        session_id=session_id,
        role=role,
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    answer = final_state.get("final_answer") or final_state.get("draft_answer", "")

    chunks = final_state.get("reranked_chunks") or final_state.get("aggregated_evidence", [])

    obs.record_api_log(
        {
            "stage": "workflow",
            "query": cleaned,
            "role": final_state.get("role", role),
            "elapsed_ms": elapsed_ms,
            "request_started_at": workflow_started,
            "response_received_at": utc_now_iso(),
        }
    )

    return {
        "answer": answer,
        "trace_id": final_state.get("trace_id", obs.trace_id),
        "role": final_state.get("role", role),
        "confidence": final_state.get("confidence", 0.0),
        "hallucination_risk_pct": final_state.get("hallucination_check", {}).get("risk_pct", 0),
        "retrieved_chunks": [
            {
                "section": c.get("section"),
                "page": c.get("page"),
                "score": c.get("rerank_score", c.get("score", 0)),
            }
            for c in chunks
        ],
        "citations": final_state.get("citations", []),
        "intent": final_state.get("intent", {}),
        "retrieval_strategy": final_state.get("retrieval_strategy", ""),
        "subqueries": final_state.get("subqueries", []),
        "agent_traces": obs.to_dict()["agent_traces"],
        "tool_usage": obs.to_dict()["tool_usage"],
        "token_usage": obs.to_dict()["token_usage"],
        "guardrail_status": final_state.get("guardrail_status", obs.to_dict().get("guardrail_events")),
        "guardrail_events": obs.guardrail_events,
        "hallucination_check": final_state.get("hallucination_check", {}),
        "answer_profile": final_state.get("answer_profile", {}),
        "quality_validation": final_state.get("quality_validation", {}),
        "answer_mode": final_state.get("answer_profile", {}).get("answer_mode", ""),
        "query_category": final_state.get("answer_profile", {}).get("category", ""),
        "verification": final_state.get("verification", {}),
        "critic_feedback": final_state.get("critic_feedback", {}),
        "reflection": final_state.get("reflection", {}),
        "entities": final_state.get("entities", {}),
        "workflow_elapsed_ms": elapsed_ms,
        "api_logs": obs.api_logs,
        "document_name": final_state.get("document_name", ""),
        "blocked": final_state.get("blocked", False),
    }


_ensure = UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
