from __future__ import annotations

import logging
import re
from typing import Any

from enterpriseRAG.agents.intent_profile import (
    classify_query_intent,
    profile_from_dict,
    profile_to_dict,
    resolve_answer_profile,
)
from enterpriseRAG.agents.query_utils import (
    analyze_query_requirements,
    answer_includes_required_code,
    build_answer_user_prompt,
    build_regeneration_prompt,
    build_subqueries,
)
from enterpriseRAG.config.roles import get_role_config
from enterpriseRAG.config.settings import get_settings
from enterpriseRAG.guardrails.pipeline import GuardrailPipeline
from enterpriseRAG.memory.store import MemoryStore
from enterpriseRAG.services.document_service import DocumentService
from enterpriseRAG.services.llm_service import LLMService
from enterpriseRAG.services.observability import ObservabilityService
from enterpriseRAG.services.answer_quality import validate_answer_quality
from enterpriseRAG.services.confidence_service import compute_confidence_score
from enterpriseRAG.services.reranker_service import RerankerService
from enterpriseRAG.tools.registry import ToolRegistry
from enterpriseRAG.workflows.state import EnterpriseRAGState

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """All 18 enterprise agents as executable node handlers."""

    def __init__(
        self,
        llm: LLMService,
        doc_service: DocumentService,
        reranker: RerankerService,
        tools: ToolRegistry,
        guardrails: GuardrailPipeline,
        memory: MemoryStore,
        observability: ObservabilityService,
        retrievers: dict[str, Any],
    ) -> None:
        self.llm = llm
        self.doc_service = doc_service
        self.reranker = reranker
        self.tools = tools
        self.guardrails = guardrails
        self.memory = memory
        self.obs = observability
        self.retrievers = retrievers
        self.settings = get_settings()

    def _run_agent(self, name: str, fn: Any, state: EnterpriseRAGState) -> dict[str, Any]:
        trace, t0 = self.obs.start_agent(name)
        try:
            result = fn(state)
            self.obs.finish_agent(trace, t0, **result.get("_agent_details", {}))
            result.pop("_agent_details", None)
            return result
        except Exception as e:
            logger.exception("Agent %s failed", name)
            self.obs.finish_agent(trace, t0, status="error", error=str(e))
            return {"error": str(e)}

    # ── 1. Intent Agent ──────────────────────────────────────────────
    def intent_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        query = state["query"]
        requirements = analyze_query_requirements(query)
        messages = [
            {
                "role": "system",
                "content": (
                    'Analyze user intent. Return JSON: {"intent": "", "complexity": "low|medium|high", '
                    '"question_type": "factual|analytical|comparative|procedural|strategic|implementation", '
                    '"topics": [], "requires_calculation": false, "requires_code": false, '
                    '"requires_guardrail_example": false, "excluded_output_terms": []}'
                ),
            },
            {"role": "user", "content": query},
        ]
        data, _ = self.llm.chat_json(messages=messages, stage="intent_agent")

        # Merge LLM intent with deterministic heuristics (heuristics win for code detection)
        merged_requirements = {
            **requirements,
            "requires_code": requirements["requires_code"] or data.get("requires_code", False),
            "requires_guardrail_example": (
                requirements["requires_guardrail_example"]
                or data.get("requires_guardrail_example", False)
            ),
            "excluded_terms": requirements["excluded_terms"]
            or data.get("excluded_output_terms", []),
        }
        if merged_requirements["requires_code"]:
            data["question_type"] = "implementation"
            data["complexity"] = "medium" if data.get("complexity") == "low" else data.get("complexity", "medium")

        intent_classification = classify_query_intent(query, merged_requirements)

        return {
            "intent": {
                **data,
                "query_requirements": merged_requirements,
                "classification": intent_classification,
            },
            "complexity": data.get("complexity", "medium"),
            "question_type": data.get("question_type", "factual"),
            "_agent_details": {
                "complexity": data.get("complexity"),
                "requires_code": merged_requirements["requires_code"],
                "category": intent_classification.get("category"),
                "answer_mode": intent_classification.get("answer_mode"),
            },
        }

    # ── 2. Role Intelligence Agent ───────────────────────────────────
    def role_intelligence_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        session_id = state.get("session_id", "default")
        role = state.get("role") or self.memory.session.get_role(
            session_id, self.settings.default_role
        )
        role_cfg = get_role_config(role)
        self.memory.session.set_preferences(session_id, {"role": role_cfg.id})

        intent = state.get("intent", {})
        requirements = intent.get("query_requirements") or analyze_query_requirements(state["query"])
        classification = intent.get("classification") or classify_query_intent(state["query"], requirements)

        role_config_dict = {
            "label": role_cfg.label,
            "retrieval_priorities": list(role_cfg.retrieval_priorities),
            "answer_sections": list(role_cfg.answer_sections),
            "terminology": role_cfg.terminology,
            "depth": role_cfg.depth,
            "system_prompt": role_cfg.system_prompt,
        }

        profile = resolve_answer_profile(
            query=state["query"],
            requirements=requirements,
            intent_classification=classification,
            role_config=role_config_dict,
        )

        return {
            "role": role_cfg.id,
            "role_config": role_config_dict,
            "answer_profile": profile_to_dict(profile),
            "_agent_details": {
                "role": role_cfg.id,
                "answer_mode": profile.answer_mode,
                "category": profile.category,
                "sections": list(profile.sections),
            },
        }

    # ── 3. Query Planning Agent ───────────────────────────────────────
    def query_planning_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        query = state["query"]
        intent = state.get("intent", {})
        requirements = intent.get("query_requirements") or analyze_query_requirements(query)
        classification = intent.get("classification", {})
        category = classification.get("category", "")

        if requirements.get("multi_part") or requirements.get("requires_code"):
            subqueries = build_subqueries(query, requirements)
            return {"subqueries": subqueries, "_agent_details": {"subquery_count": len(subqueries)}}

        if category in {"multi_hop_retrieval", "research_report", "agent_architecture"}:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Decompose this query into 4-8 retrieval subqueries spanning entities, tools, "
                        "frameworks, patterns, and project examples. Return JSON: {\"subqueries\": [\"...\"]}"
                    ),
                },
                {"role": "user", "content": query},
            ]
            data, _ = self.llm.chat_json(messages=messages, stage="query_planning_agent_multihop")
            subqueries = data.get("subqueries", [query])
            return {"subqueries": subqueries[:8], "_agent_details": {"subquery_count": len(subqueries)}}

        complexity = state.get("complexity", "medium")
        if complexity == "low":
            return {"subqueries": [query], "_agent_details": {"subquery_count": 1}}

        messages = [
            {
                "role": "system",
                "content": (
                    'Decompose complex questions into 2-5 subqueries for retrieval. '
                    'Return JSON: {"subqueries": ["..."]}'
                ),
            },
            {"role": "user", "content": query},
        ]
        data, _ = self.llm.chat_json(messages=messages, stage="query_planning_agent")
        subqueries = data.get("subqueries", [query])
        return {"subqueries": subqueries[:5], "_agent_details": {"subquery_count": len(subqueries)}}

    # ── 4. Multi Query Agent ──────────────────────────────────────────
    def multi_query_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        mq = self.retrievers.get("multi_query")
        profile = profile_from_dict(state.get("answer_profile", {}))
        priorities = profile.retrieval_priorities or tuple(
            state.get("role_config", {}).get("retrieval_priorities", ())
        )
        variations = [state["query"]]
        if mq:
            variations = mq.generate_variations(state["query"], priorities)
        return {"query_variations": variations, "_agent_details": {"variation_count": len(variations)}}

    # ── 5. Retrieval Strategy Agent ─────────────────────────────────
    def retrieval_strategy_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        qtype = state.get("question_type", "factual")
        complexity = state.get("complexity", "medium")
        category = state.get("intent", {}).get("classification", {}).get("category", "")

        if category in {"multi_hop_retrieval", "research_report", "agent_architecture"}:
            strategy = "multi_query"
        elif category == "verification_question":
            strategy = "hybrid"
        elif qtype == "factual":
            strategy = "hybrid"
        elif qtype == "procedural":
            strategy = "parent_child"
        elif complexity == "high":
            strategy = "multi_query"
        else:
            strategy = "hybrid"

        return {"retrieval_strategy": strategy, "_agent_details": {"strategy": strategy}}

    # ── 6-8. Retrieval Agents ─────────────────────────────────────────
    def retrieval_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        strategy = state.get("retrieval_strategy", "hybrid")
        subqueries = state.get("subqueries") or [state["query"]]
        profile = profile_from_dict(state.get("answer_profile", {}))
        priorities = profile.retrieval_priorities or tuple(
            state.get("role_config", {}).get("retrieval_priorities", ())
        )
        top_k = self.settings.top_k_retrieve

        all_chunks: list[dict[str, Any]] = []
        tools_used = list(state.get("tools_used", []))

        for subq in subqueries:
            if strategy == "vector":
                chunks = self.tools.vector_search(subq, top_k)
                tools_used.append("vector_search")
            elif strategy == "bm25":
                chunks = self.tools.bm25_search(subq, top_k)
                tools_used.append("bm25_search")
            elif strategy == "parent_child":
                pc = self.retrievers.get("parent_child")
                chunks = pc.retrieve(subq, top_k) if pc else self.tools.hybrid_search(subq, top_k)
                tools_used.append("parent_child_retrieval")
            elif strategy == "multi_query":
                mq = self.retrievers.get("multi_query")
                chunks = mq.retrieve(subq, priorities, top_k) if mq else self.tools.hybrid_search(subq, top_k)
                tools_used.append("multi_query_retrieval")
            else:
                chunks = self.tools.hybrid_search(subq, top_k)
                tools_used.append("hybrid_search")
            all_chunks.extend(chunks)

        deduped = self.doc_service.dedupe_chunks(all_chunks)
        indexes = self.doc_service.get_active_indexes()

        return {
            "raw_chunks": deduped,
            "document_name": indexes["document_name"],
            "tools_used": tools_used,
            "_agent_details": {"chunk_count": len(deduped), "strategy": strategy},
        }

    # ── 9. Context Compression Agent ──────────────────────────────────
    def context_compression_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        compressor = self.retrievers.get("compression")
        chunks = state.get("raw_chunks", [])
        if compressor:
            compressed = compressor.compress(state["query"], chunks)
        else:
            compressed = chunks[: self.settings.top_k_rerank * 2]
        return {
            "compressed_chunks": compressed,
            "_agent_details": {"compressed_count": len(compressed)},
        }

    # ── 10. Evidence Aggregation Agent ────────────────────────────────
    def evidence_aggregation_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        chunks = state.get("compressed_chunks") or state.get("raw_chunks", [])
        filtered, retrieval_guard = self.guardrails.check_retrieval(chunks)

        messages = [
            {
                "role": "system",
                "content": (
                    'Summarize key evidence points from chunks for the query. '
                    'Return JSON: {"evidence_points": [{"point": "", "section": "", "page": 0}], '
                    '"contradictions": [], "gaps": []}'
                ),
            },
            {
                "role": "user",
                "content": f"Query: {state['query']}\n\nChunks:\n{filtered[:8]}",
            },
        ]
        data, _ = self.llm.chat_json(messages=messages, stage="evidence_aggregation_agent")

        return {
            "aggregated_evidence": filtered,
            "retrieval_guardrails": retrieval_guard,
            "_agent_details": {
                "evidence_count": len(data.get("evidence_points", [])),
                "gaps": len(data.get("gaps", [])),
            },
        }

    # ── 11. Verification Agent (Self-RAG) ─────────────────────────────
    def verification_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        chunks = state.get("aggregated_evidence", [])
        loop_count = state.get("retrieval_loop_count", 0)

        messages = [
            {
                "role": "system",
                "content": (
                    'Verify if retrieved evidence is sufficient. Return JSON: '
                    '{"sufficient": true, "confidence": 0.0-1.0, "missing_context": [], '
                    '"contradictions": [], "needs_reretrieval": false, "reason": ""}'
                ),
            },
            {
                "role": "user",
                "content": f"Query: {state['query']}\nEvidence chunks: {len(chunks)}\nSample: {chunks[:3]}",
            },
        ]
        data, _ = self.llm.chat_json(messages=messages, stage="verification_agent")

        confidence = float(data.get("confidence", 0.5))
        needs_reretrieval = (
            data.get("needs_reretrieval", False)
            or confidence < self.settings.confidence_threshold
        ) and loop_count < self.settings.max_retrieval_loops

        return {
            "verification": data,
            "confidence": confidence,
            "retrieval_loop_count": loop_count + (1 if needs_reretrieval else 0),
            "_agent_details": {"confidence": confidence, "needs_reretrieval": needs_reretrieval},
            "_needs_reretrieval": needs_reretrieval,
        }

    # ── Reranking (between verification and generation) ─────────────────
    def rerank_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        chunks = state.get("aggregated_evidence", [])
        reranked = self.tools.rerank(state["query"], chunks, self.settings.top_k_rerank)
        tools_used = list(state.get("tools_used", []))
        tools_used.append("reranker")
        return {
            "reranked_chunks": reranked,
            "tools_used": tools_used,
            "_agent_details": {"reranked_count": len(reranked)},
        }

    # ── 12. Citation Agent ────────────────────────────────────────────
    def citation_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        chunks = state.get("reranked_chunks") or state.get("aggregated_evidence", [])
        citations = self.tools.citation_formatter(chunks)
        return {"citations": citations, "_agent_details": {"citation_count": len(citations)}}

    # ── 13. Answer Generation Agent ───────────────────────────────────
    def answer_generation_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        chunks = state.get("reranked_chunks") or state.get("aggregated_evidence", [])
        profile = profile_from_dict(state.get("answer_profile", {}))
        intent = state.get("intent", {})
        requirements = intent.get("query_requirements") or analyze_query_requirements(state["query"])

        context_parts = []
        for chunk in chunks:
            section = chunk.get("section", "Section")
            page = chunk.get("page", "?")
            context_parts.append(
                f"[Chunk | Section: {section}, Page: {page}]\n{chunk.get('content', '')}"
            )
        context_str = "\n\n---\n\n".join(context_parts) or "No context available."

        quality = state.get("quality_validation", {})
        is_regeneration = (
            state.get("reflection_loop_count", 0) > 0
            or quality.get("needs_regeneration")
        )

        if is_regeneration and quality.get("issues"):
            user_prompt = build_regeneration_prompt(
                query=state["query"],
                context_str=context_str,
                document_name=state.get("document_name", "uploaded document"),
                profile=profile,
                requirements=requirements,
                previous_answer=state.get("draft_answer", ""),
                quality_issues=quality.get("issues", []),
            )
        else:
            user_prompt = build_answer_user_prompt(
                query=state["query"],
                context_str=context_str,
                document_name=state.get("document_name", "uploaded document"),
                profile=profile,
                requirements=requirements,
            )

        messages = [
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        answer, _ = self.llm.chat(
            messages=messages,
            stage="answer_generation_agent",
            temperature=0.15 if is_regeneration else 0.2,
        )

        gen_guard = self.guardrails.check_generation(answer, chunks)
        if requirements.get("requires_code") and not answer_includes_required_code(answer, requirements):
            gen_guard["code_missing"] = True
            gen_guard["passed"] = False

        return {
            "draft_answer": answer,
            "generation_guardrails": gen_guard,
            "confidence": min(state.get("confidence", 0.5), gen_guard.get("confidence", 0.5)),
            "_agent_details": {
                "answer_length": len(answer),
                "has_code": answer_includes_required_code(answer, requirements),
                "answer_mode": profile.answer_mode,
                "regenerated": is_regeneration,
            },
        }

    # ── 14. Critic Agent ────────────────────────────────────────────────
    def critic_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        intent = state.get("intent", {})
        requirements = intent.get("query_requirements") or analyze_query_requirements(state["query"])
        answer = state.get("draft_answer", "")
        chunks = state.get("reranked_chunks") or state.get("aggregated_evidence", [])
        answer_profile = state.get("answer_profile", {})

        quality = validate_answer_quality(
            query=state["query"],
            answer=answer,
            chunks=chunks,
            requirements=requirements,
            answer_profile=answer_profile,
        )

        has_code = answer_includes_required_code(answer, requirements)
        if requirements.get("requires_code") and not has_code:
            quality["needs_regeneration"] = True
            quality["issues"] = list(quality.get("issues", [])) + [
                "Missing required ```python code block."
            ]

        critic_payload = {
            "accuracy_score": quality["grounding_score"],
            "completeness_score": quality["intent_alignment"],
            "issues": quality["issues"],
            "missing_details": [],
            "needs_regeneration": quality["needs_regeneration"],
            "code_example_present": has_code,
            "quality_validation": quality,
        }

        return {
            "critic_feedback": critic_payload,
            "quality_validation": quality,
            "_agent_details": quality,
        }

    # ── 15. Reflection Agent ──────────────────────────────────────────
    def reflection_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        critic = state.get("critic_feedback", {})
        quality = state.get("quality_validation", {})
        loop_count = state.get("reflection_loop_count", 0)
        needs_regen = (
            critic.get("needs_regeneration", False)
            or quality.get("needs_regeneration", False)
            or quality.get("overall_score", 1.0) < 0.75
        ) and loop_count < self.settings.max_reflection_loops

        reflection_text = (
            f"Quality score {quality.get('overall_score', 0):.2f}. "
            f"Issues: {quality.get('issues', [])[:5]}"
        )

        return {
            "reflection": {
                "reflection": reflection_text,
                "action": "regenerate" if needs_regen else "accept",
                "improvements": quality.get("issues", [])[:5],
            },
            "reflection_loop_count": loop_count + (1 if needs_regen else 0),
            "_agent_details": {"action": "regenerate" if needs_regen else "accept", "needs_regen": needs_regen},
        }

    # ── 16. Hallucination Detection Agent ─────────────────────────────
    def hallucination_detection_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        chunks = state.get("reranked_chunks") or state.get("aggregated_evidence", [])
        answer = state.get("draft_answer", "")
        check = self.guardrails.detect_hallucination_risk(
            answer,
            chunks,
            verification=state.get("verification", {}),
            critic=state.get("critic_feedback", {}),
        )
        confidence = compute_confidence_score(
            hallucination_check=check,
            verification=state.get("verification"),
            critic_feedback=state.get("critic_feedback"),
            generation_guardrails=state.get("generation_guardrails"),
        )
        return {
            "hallucination_check": check,
            "confidence": confidence,
            "_agent_details": {**check, "confidence": confidence},
        }

    # ── 17. Governance Agent ──────────────────────────────────────────
    def governance_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    'Check answer for enterprise policy compliance. '
                    'Return JSON: {"compliant": true, "violations": [], "recommendations": []}'
                ),
            },
            {"role": "user", "content": state.get("draft_answer", "")[:2000]},
        ]
        data, _ = self.llm.chat_json(messages=messages, stage="governance_agent")
        return {"governance_check": data, "_agent_details": data}

    # ── 18. Response Formatting Agent ─────────────────────────────────
    def response_formatting_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        answer = state.get("draft_answer", "")
        profile = profile_from_dict(state.get("answer_profile", {}))

        if profile.suppress_executive:
            answer = self._strip_executive_sections(answer)

        masked, output_guard = self.guardrails.check_output(answer)
        guardrail_status = self.guardrails.status_summary()

        # Extract entities for memory
        messages = [
            {
                "role": "system",
                "content": (
                    'Extract entities. Return JSON: {"topics": [], "technologies": [], '
                    '"companies": [], "products": []}'
                ),
            },
            {"role": "user", "content": f"Query: {state['query']}\nAnswer excerpt: {masked[:1000]}"},
        ]
        entities, _ = self.llm.chat_json(messages=messages, stage="entity_extraction")
        self.memory.entity.update(entities)

        session_id = state.get("session_id", "default")
        self.memory.short_term.add_message(session_id, "user", state["query"])
        self.memory.short_term.add_message(session_id, "assistant", masked)
        self.memory.short_term.set_retrieval_context(
            session_id, state.get("reranked_chunks", [])
        )

        return {
            "final_answer": masked,
            "output_guardrails": output_guard,
            "guardrail_status": guardrail_status,
            "entities": entities,
            "_agent_details": {"formatted": True},
        }

    @staticmethod
    def _strip_executive_sections(answer: str) -> str:
        executive_headers = (
            "Executive Summary",
            "Business Impact",
            "Expected ROI",
        )
        result = answer
        for header in executive_headers:
            result = re.sub(
                rf"(?is)(^|\n)(?:##+\s*)?{re.escape(header)}\s*\n.*?(?=\n(?:##+\s*[A-Z]|\Z))",
                "\n",
                result,
            )
        return result.strip()

    # ── Input Guardrails (pre-workflow) ───────────────────────────────
    def input_guardrails_agent(self, state: EnterpriseRAGState) -> dict[str, Any]:
        result = self.guardrails.check_input(state["query"])
        if not result["passed"]:
            return {
                "input_guardrails": result,
                "blocked": True,
                "blocked_reason": result.get("blocked_reason"),
                "final_answer": f"I cannot process this request. {result.get('blocked_reason')}",
            }
        return {"input_guardrails": result, "blocked": False}
