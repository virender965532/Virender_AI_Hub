from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.documents import Document

from enterpriseRAG.config.settings import get_settings
from enterpriseRAG.services.document_service import DocumentService
from enterpriseRAG.services.llm_service import LLMService
from enterpriseRAG.services.observability import ObservabilityService

logger = logging.getLogger(__name__)


class VectorRetriever:
    def __init__(
        self,
        doc_service: DocumentService,
        llm: LLMService,
        observability: ObservabilityService | None = None,
    ) -> None:
        self.doc_service = doc_service
        self.llm = llm
        self.obs = observability
        self.settings = get_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.settings.top_k_retrieve
        t0 = time.perf_counter()
        indexes = self.doc_service.get_active_indexes()
        vector_store = indexes["vector_store"]

        query_vector, _ = self.llm.embed(query, stage="vector_retrieval_embed")
        docs = vector_store.similarity_search_by_vector(query_vector, k=top_k)

        results = []
        for i, doc in enumerate(docs):
            score = 1.0 - (i / max(len(docs), 1))
            results.append(self.doc_service.chunk_to_dict(doc, score=score))

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if self.obs:
            self.obs.record_tool("vector_search", elapsed_ms, result_count=len(results))
        return results


class BM25Retriever:
    def __init__(
        self,
        doc_service: DocumentService,
        observability: ObservabilityService | None = None,
    ) -> None:
        self.doc_service = doc_service
        self.obs = observability
        self.settings = get_settings()
        self._bm25 = None
        self._corpus: list[Document] = []

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi

        indexes = self.doc_service.get_active_indexes()
        self._corpus = indexes["children"]
        tokenized = [doc.page_content.lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.settings.top_k_retrieve
        t0 = time.perf_counter()
        self._ensure_bm25()

        scores = self._bm25.get_scores(query.lower().split())
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        max_score = max(scores) if len(scores) else 1.0
        for idx in ranked_indices:
            doc = self._corpus[idx]
            norm_score = scores[idx] / max_score if max_score else 0
            results.append(self.doc_service.chunk_to_dict(doc, score=float(norm_score)))

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if self.obs:
            self.obs.record_tool("bm25_search", elapsed_ms, result_count=len(results))
        return results


class HybridRetriever:
    def __init__(
        self,
        vector: VectorRetriever,
        bm25: BM25Retriever,
        doc_service: DocumentService,
        observability: ObservabilityService | None = None,
    ) -> None:
        self.vector = vector
        self.bm25 = bm25
        self.doc_service = doc_service
        self.obs = observability
        self.settings = get_settings()

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        top_k = top_k or self.settings.top_k_retrieve
        t0 = time.perf_counter()
        alpha = self.settings.hybrid_alpha

        vector_results = self.vector.retrieve(query, top_k=top_k)
        bm25_results = self.bm25.retrieve(query, top_k=top_k)

        combined: dict[str, dict[str, Any]] = {}
        for chunk in vector_results:
            key = chunk.get("chunk_id") or chunk["content"][:100]
            chunk["score"] = chunk.get("score", 0) * alpha
            combined[key] = chunk

        for chunk in bm25_results:
            key = chunk.get("chunk_id") or chunk["content"][:100]
            if key in combined:
                combined[key]["score"] = combined[key].get("score", 0) + chunk.get("score", 0) * (1 - alpha)
            else:
                chunk["score"] = chunk.get("score", 0) * (1 - alpha)
                combined[key] = chunk

        results = sorted(combined.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if self.obs:
            self.obs.record_tool("hybrid_search", elapsed_ms, result_count=len(results))
        return results


class ParentChildRetriever:
    """Retrieve child chunks, expand to parent context."""

    def __init__(
        self,
        vector: VectorRetriever,
        doc_service: DocumentService,
        observability: ObservabilityService | None = None,
    ) -> None:
        self.vector = vector
        self.doc_service = doc_service
        self.obs = observability

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        settings = get_settings()
        top_k = top_k or settings.top_k_retrieve
        t0 = time.perf_counter()

        child_results = self.vector.retrieve(query, top_k=top_k)
        indexes = self.doc_service.get_active_indexes()
        parents = indexes["parents"]

        expanded: list[dict[str, Any]] = []
        for child in child_results[: settings.top_k_rerank * 2]:
            page = child.get("page", 0)
            parent_match = next(
                (p for p in parents if p.metadata.get("page") == page),
                None,
            )
            if parent_match:
                expanded_chunk = self.doc_service.chunk_to_dict(
                    parent_match, score=child.get("score", 0)
                )
                expanded_chunk["matched_child"] = child.get("chunk_id", "")
            else:
                expanded_chunk = child
            expanded.append(expanded_chunk)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if self.obs:
            self.obs.record_tool("parent_child_retrieval", elapsed_ms, result_count=len(expanded))
        return expanded or child_results


class MultiQueryRetriever:
    def __init__(
        self,
        base_retriever: HybridRetriever,
        llm: LLMService,
        observability: ObservabilityService | None = None,
    ) -> None:
        self.base = base_retriever
        self.llm = llm
        self.obs = observability
        self.settings = get_settings()

    def generate_variations(self, query: str, role_priorities: tuple[str, ...]) -> list[str]:
        priorities = ", ".join(role_priorities[:3])
        json_hint = '{"queries": ["...", "...", "..."]}'
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate 3 alternative search queries for document retrieval. "
                    f"Prioritize: {priorities}. Return JSON: {json_hint}"
                ),
            },
            {"role": "user", "content": query},
        ]
        data, _ = self.llm.chat_json(messages=messages, stage="multi_query_generation")
        queries = data.get("queries", [])
        return [query] + [q for q in queries if isinstance(q, str)][:3]

    def retrieve(
        self, query: str, role_priorities: tuple[str, ...] = (), top_k: int | None = None
    ) -> list[dict[str, Any]]:
        top_k = top_k or self.settings.top_k_retrieve
        t0 = time.perf_counter()
        variations = self.generate_variations(query, role_priorities)

        combined: dict[str, dict[str, Any]] = {}
        per_query_k = max(top_k // len(variations), 5)
        for var_query in variations:
            results = self.base.retrieve(var_query, top_k=per_query_k)
            for chunk in results:
                key = chunk.get("chunk_id") or chunk["content"][:100]
                if key not in combined or chunk.get("score", 0) > combined[key].get("score", 0):
                    combined[key] = chunk

        results = sorted(combined.values(), key=lambda x: x.get("score", 0), reverse=True)[:top_k]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if self.obs:
            self.obs.record_tool(
                "multi_query_retrieval",
                elapsed_ms,
                variations=len(variations),
                result_count=len(results),
            )
        return results


class CompressionRetriever:
    """Contextual compression of retrieved chunks."""

    def __init__(
        self,
        llm: LLMService,
        observability: ObservabilityService | None = None,
    ) -> None:
        self.llm = llm
        self.obs = observability

    def compress(
        self, query: str, chunks: list[dict[str, Any]], max_chunks: int = 10
    ) -> list[dict[str, Any]]:
        if len(chunks) <= max_chunks:
            return chunks

        t0 = time.perf_counter()
        compressed: list[dict[str, Any]] = []
        for chunk in chunks[: max_chunks * 2]:
            content = chunk.get("content", "")
            if len(content) <= 600:
                compressed.append(chunk)
                continue

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Extract only the sentences relevant to the query. "
                        "Keep factual content. Return compressed text only."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nContent:\n{content[:2000]}",
                },
            ]
            result, _ = self.llm.chat(messages=messages, stage="context_compression", temperature=0)
            compressed_chunk = dict(chunk)
            compressed_chunk["content"] = result
            compressed_chunk["compressed"] = True
            compressed.append(compressed_chunk)

        result = compressed[:max_chunks]
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if self.obs:
            self.obs.record_tool("context_compression", elapsed_ms, output_count=len(result))
        return result
