from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K = 4
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

BASE_DIR = Path(__file__).resolve().parents[1]
PDF_PATH = (
    BASE_DIR
    / "Data"
    / "StudyMaterial"
    / "Complete AI"
    / "AI Agents guidebook.pdf"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

_trace_id_ctx: ContextVar[str | None] = ContextVar("simple_rag_trace_id", default=None)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _setup_openai_tracing() -> bool:
    """Enable OpenTelemetry OpenAI instrumentation when available."""
    try:
        from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

        OpenAIInstrumentor().instrument()
        logger.info("OpenAI OpenTelemetry tracing enabled for Simple RAG.")
        return True
    except ImportError:
        logger.info(
            "OpenAI OpenTelemetry tracing not installed; using structured trace logs only."
        )
        return False


OPENAI_TRACING_ENABLED = _setup_openai_tracing()


def _extract_token_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def _log_api_call(api_log: dict[str, Any]) -> None:
    logger.info("simple_rag_openai_call %s", json.dumps(api_log, default=str))


def _chunk_section_label(content: str) -> str:
    for line in content.splitlines():
        cleaned = line.strip().lstrip("#").strip()
        if cleaned and len(cleaned) > 3:
            return cleaned[:100]
    return "Document"


@lru_cache(maxsize=1)
def _get_vector_db() -> FAISS:
    if not PDF_PATH.exists():
        raise RuntimeError(f"PDF file not found: {PDF_PATH}")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def _embed_query(query: str, trace_id: str) -> tuple[list[float], dict[str, Any]]:
    if not OPENAI_CLIENT:
        raise RuntimeError("OPENAI_API_KEY is missing in environment.")

    started_at = _utc_now_iso()
    t0 = time.perf_counter()

    response = OPENAI_CLIENT.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
        extra_headers={"X-Client-Trace-Id": trace_id},
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    ended_at = _utc_now_iso()
    usage = _extract_token_usage(response)

    api_log = {
        "stage": "embedding",
        "model": EMBEDDING_MODEL,
        "query": query,
        "trace_id": trace_id,
        "openai_tracing_enabled": OPENAI_TRACING_ENABLED,
        "request_started_at": started_at,
        "response_received_at": ended_at,
        "elapsed_ms": elapsed_ms,
        **usage,
    }
    _log_api_call(api_log)
    return response.data[0].embedding, api_log


def _retrieve_chunks(
    query: str,
    trace_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    vector_db = _get_vector_db()
    query_vector, embed_log = _embed_query(query, trace_id)

    started_at = _utc_now_iso()
    t0 = time.perf_counter()

    docs = vector_db.similarity_search_by_vector(query_vector, k=TOP_K)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    ended_at = _utc_now_iso()

    retrieved: list[dict[str, Any]] = []
    for doc in docs:
        page_index = int(doc.metadata.get("page", 0)) + 1
        section = _chunk_section_label(doc.page_content)
        retrieved.append(
            {
                "section": section,
                "page": page_index,
                "content": doc.page_content,
            }
        )

    retrieval_log = {
        "stage": "vector_retrieval",
        "query": query,
        "trace_id": trace_id,
        "openai_tracing_enabled": OPENAI_TRACING_ENABLED,
        "request_started_at": started_at,
        "response_received_at": ended_at,
        "elapsed_ms": elapsed_ms,
        "top_k": TOP_K,
        "retrieved_sections": [
            {"section": item["section"], "page": item["page"]} for item in retrieved
        ],
    }
    _log_api_call(retrieval_log)

    return retrieved, embed_log, retrieval_log


def _generate_answer(
    query: str,
    chunks: list[dict[str, Any]],
    trace_id: str,
) -> tuple[str, dict[str, Any]]:
    if not OPENAI_CLIENT:
        raise RuntimeError("OPENAI_API_KEY is missing in environment.")

    if not chunks:
        return "No relevant information found in the document.", {}

    context_parts = []
    for chunk in chunks:
        context_parts.append(
            f"""
SECTION: {chunk['section']}
PAGE: {chunk['page']}

CONTENT:
{chunk['content']}
"""
        )

    context_str = "\n\n".join(context_parts)

    prompt = f"""
Answer the question using only the provided context from the AI Agents guidebook.

Rules:
- Be clear and practical.
- Every document-derived claim MUST include a citation in this exact format:
  (Section: <section title>, Page: <page number>)
- Use the SECTION and PAGE values from the context blocks.
- If the context is insufficient, say so briefly.

Context:
{context_str}

Question:
{query}
"""

    started_at = _utc_now_iso()
    t0 = time.perf_counter()

    response = OPENAI_CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        store=True,
        extra_headers={"X-Client-Trace-Id": trace_id},
        metadata={"trace_id": trace_id, "stage": "answer_generation", "workflow": "simple_rag"},
    )

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    ended_at = _utc_now_iso()
    usage = _extract_token_usage(response)

    api_log = {
        "stage": "answer_generation",
        "model": MODEL_NAME,
        "query": query,
        "trace_id": trace_id,
        "openai_tracing_enabled": OPENAI_TRACING_ENABLED,
        "request_started_at": started_at,
        "response_received_at": ended_at,
        "elapsed_ms": elapsed_ms,
        **usage,
    }
    _log_api_call(api_log)

    answer = (response.choices[0].message.content or "").strip()
    return answer, api_log


def ask_simple_rag_question(query: str) -> dict[str, Any]:
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise ValueError("Question is required.")

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing in environment.")

    trace_id = str(uuid.uuid4())
    _trace_id_ctx.set(trace_id)

    workflow_started_at = _utc_now_iso()
    t0 = time.perf_counter()

    chunks, embed_log, retrieval_log = _retrieve_chunks(cleaned_query, trace_id)
    answer, answer_log = _generate_answer(cleaned_query, chunks, trace_id)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    workflow_log = {
        "stage": "workflow",
        "query": cleaned_query,
        "trace_id": trace_id,
        "openai_tracing_enabled": OPENAI_TRACING_ENABLED,
        "request_started_at": workflow_started_at,
        "response_received_at": _utc_now_iso(),
        "elapsed_ms": elapsed_ms,
        "pdf_path": str(PDF_PATH),
    }
    _log_api_call(workflow_log)

    api_logs = [embed_log, retrieval_log, answer_log, workflow_log]

    return {
        "answer": answer,
        "trace_id": trace_id,
        "retrieved_chunks": [
            {"section": chunk["section"], "page": chunk["page"]} for chunk in chunks
        ],
        "api_logs": api_logs,
    }
