from __future__ import annotations

import ast
import json
import operator
import re
import time
from typing import Any, Callable

from enterpriseRAG.services.document_service import DocumentService
from enterpriseRAG.services.llm_service import LLMService
from enterpriseRAG.services.observability import ObservabilityService
from enterpriseRAG.services.reranker_service import RerankerService


class ToolRegistry:
    """Isolated enterprise toolset for agent workflows."""

    def __init__(
        self,
        doc_service: DocumentService,
        llm: LLMService,
        reranker: RerankerService,
        observability: ObservabilityService | None = None,
        retrievers: dict[str, Any] | None = None,
    ) -> None:
        self.doc_service = doc_service
        self.llm = llm
        self.reranker = reranker
        self.obs = observability
        self.retrievers = retrievers or {}
        self._knowledge_graph: dict[str, list[str]] = {}

    def _timed(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        if self.obs:
            self.obs.record_tool(name, elapsed_ms)
        return result

    def vector_search(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        retriever = self.retrievers.get("vector")
        if not retriever:
            return []
        return self._timed("vector_search_tool", retriever.retrieve, query, top_k)

    def bm25_search(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        retriever = self.retrievers.get("bm25")
        if not retriever:
            return []
        return self._timed("bm25_search_tool", retriever.retrieve, query, top_k)

    def hybrid_search(self, query: str, top_k: int = 30) -> list[dict[str, Any]]:
        retriever = self.retrievers.get("hybrid")
        if not retriever:
            return []
        return self._timed("hybrid_search_tool", retriever.retrieve, query, top_k)

    def rerank(self, query: str, chunks: list[dict[str, Any]], top_n: int = 5) -> list[dict[str, Any]]:
        return self._timed("reranker_tool", self.reranker.rerank, query, chunks, top_n)

    def python_analysis(self, code: str) -> dict[str, Any]:
        def _analyze(c: str) -> dict[str, Any]:
            try:
                tree = ast.parse(c)
                functions = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
                classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                imports = [
                    n.names[0].name
                    for n in ast.walk(tree)
                    if isinstance(n, ast.Import)
                ]
                return {
                    "valid": True,
                    "functions": functions,
                    "classes": classes,
                    "imports": imports,
                    "line_count": len(c.splitlines()),
                }
            except SyntaxError as e:
                return {"valid": False, "error": str(e)}

        return self._timed("python_analysis_tool", _analyze, code)

    def calculator(self, expression: str) -> dict[str, Any]:
        def _calc(expr: str) -> dict[str, Any]:
            allowed = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.Pow: operator.pow,
                ast.USub: operator.neg,
            }

            def _eval(node: ast.AST) -> float:
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                    return float(node.value)
                if isinstance(node, ast.BinOp):
                    return allowed[type(node.op)](_eval(node.left), _eval(node.right))
                if isinstance(node, ast.UnaryOp):
                    return allowed[type(node.op)](_eval(node.operand))
                raise ValueError("Unsupported expression")

            tree = ast.parse(expr.strip(), mode="eval")
            result = _eval(tree.body)
            return {"expression": expr, "result": result}

        try:
            return self._timed("calculator_tool", _calc, expression)
        except Exception as e:
            return {"expression": expression, "error": str(e)}

    def document_statistics(self) -> dict[str, Any]:
        return self._timed("document_statistics_tool", self.doc_service.document_stats)

    def pdf_analysis(self) -> dict[str, Any]:
        stats = self.doc_service.document_stats()
        preview = self.doc_service.preview_text(max_chars=2000)
        return self._timed(
            "pdf_analysis_tool",
            lambda: {
                **stats,
                "preview_excerpt": preview[:500],
                "has_tables": bool(re.search(r"\|.*\|", preview)),
            },
        )

    def table_extraction(self) -> list[dict[str, Any]]:
        def _extract() -> list[dict[str, Any]]:
            preview = self.doc_service.preview_text(max_chars=30000)
            tables: list[dict[str, Any]] = []
            for i, block in enumerate(preview.split("\n\n")):
                if "|" in block and block.count("|") >= 3:
                    rows = [r.strip() for r in block.split("\n") if r.strip()]
                    tables.append({"table_id": i, "rows": rows[:20]})
            return tables

        return self._timed("table_extraction_tool", _extract)

    def citation_formatter(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        citations = []
        for chunk in chunks:
            citations.append(
                {
                    "citation": f"(Section: {chunk.get('section', 'Unknown')}, Page: {chunk.get('page', '?')})",
                    "section": chunk.get("section", ""),
                    "page": str(chunk.get("page", "")),
                }
            )
        return self._timed("citation_tool", lambda: citations)

    def knowledge_graph_builder(self, text: str) -> dict[str, Any]:
        def _build(t: str) -> dict[str, Any]:
            messages = [
                {
                    "role": "system",
                    "content": (
                        'Extract entities and relationships. Return JSON: '
                        '{"entities": ["..."], "relationships": [{"from":"", "to":"", "type":""}]}'
                    ),
                },
                {"role": "user", "content": t[:3000]},
            ]
            data, _ = self.llm.chat_json(messages=messages, stage="knowledge_graph_builder")
            for entity in data.get("entities", []):
                self._knowledge_graph.setdefault("entities", []).append(entity)
            return data

        return self._timed("knowledge_graph_builder_tool", _build, text)

    def list_tools(self) -> list[str]:
        return [
            "vector_search",
            "bm25_search",
            "hybrid_search",
            "reranker",
            "python_analysis",
            "calculator",
            "document_statistics",
            "pdf_analysis",
            "table_extraction",
            "citation",
            "knowledge_graph_builder",
        ]
