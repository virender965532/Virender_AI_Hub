from __future__ import annotations

import re
from typing import Any

from enterpriseRAG.agents.intent_profile import AnswerProfile, PRODUCTION_SYSTEM_PROMPT


def focused_retrieval_query(query: str) -> str:
    """Use the topic-defining part of multi-part questions for vector search."""
    lines = [line.strip() for line in query.splitlines() if line.strip()]
    if len(lines) <= 1:
        return query

    for line in lines:
        if "?" in line:
            return line

    return lines[0]


def analyze_query_requirements(query: str) -> dict[str, Any]:
    """Detect code, guardrail examples, and output constraints from the user query."""
    lower = query.lower()

    requires_code = bool(
        re.search(
            r"\b(python|code|snippet|implementation|runnable|script)\b",
            lower,
        )
        or re.search(r"example.*\b(in python|python)\b", lower)
        or re.search(r"custom guardrail", lower)
    )
    requires_guardrail_example = bool(re.search(r"\bguardrail", lower))

    excluded_terms: list[str] = []
    for pattern in (
        r"don'?t have (\w+)",
        r"without (\w+)",
        r"exclude (\w+)",
        r"no (\w+) in the output",
        r"not contain (\w+)",
    ):
        for match in re.finditer(pattern, lower):
            term = match.group(1).strip()
            if term and term not in {"the", "a", "an", "in", "output", "response"}:
                excluded_terms.append(term)

    preserved: list[str] = []
    for term in excluded_terms:
        case_match = re.search(re.escape(term), query, re.IGNORECASE)
        preserved.append(case_match.group(0) if case_match else term)

    multi_part = bool(
        requires_code and requires_guardrail_example
    ) or query.count("?") > 1 or "\n" in query.strip()

    return {
        "requires_code": requires_code,
        "requires_guardrail_example": requires_guardrail_example,
        "excluded_terms": list(dict.fromkeys(preserved)),
        "multi_part": multi_part,
    }


def build_subqueries(query: str, requirements: dict[str, Any]) -> list[str]:
    """Split multi-part questions into retrieval-friendly subqueries."""
    if not requirements.get("multi_part"):
        return [focused_retrieval_query(query)]

    subqueries = [focused_retrieval_query(query)]

    if requirements.get("requires_guardrail_example"):
        subqueries.append(
            "guardrails AI agents validation checkpoints output filtering fallback mechanisms"
        )

    if requirements.get("requires_code"):
        subqueries.append(
            "custom guardrails output filtering validation checkpoint agent response"
        )

    for term in requirements.get("excluded_terms", []):
        subqueries.append(f"output filtering exclude terms response validation {term}")

    return list(dict.fromkeys(subqueries))[:5]


def build_answer_user_prompt(
    *,
    query: str,
    context_str: str,
    document_name: str,
    profile: AnswerProfile,
    requirements: dict[str, Any],
) -> str:
    section_guide = "\n".join(f"- {s}" for s in profile.sections)
    excluded = requirements.get("excluded_terms") or []

    code_rules = ""
    if requirements.get("requires_code"):
        exclude_note = ""
        if excluded:
            exclude_note = (
                f"\n- Forbidden output terms: {', '.join(excluded)}. "
                "Use case-insensitive validation (.lower()). "
                "Valid demo strings must NOT contain these terms. "
                "Show one passing example and one blocked example."
            )
        code_rules = f"""
CODE REQUIREMENTS:{exclude_note}
- Include a complete ```python block with: validate function, process/wrap function, and demo usage.
- Apply document concepts: validation checkpoints, output filtering, fallback on failure.
- Do NOT cite the document inside code blocks.
"""

    anti_hallucination = """
GROUNDING REQUIREMENTS:
- **From Document** section: ONLY facts supported by the context below, each with (Section: X, Page: Y).
- If the context lacks detail, say "The document does not specify this" — never invent ROI, legal risk, or business metrics.
- Do NOT include Business Impact, Risks, Expected ROI, or Executive Summary unless the user explicitly asked for business analysis.
- Separate clearly:
  1) retrieved information from the PDF (must be cited)
  2) model reasoning/recommendation (only when requested; do not present as document fact)
"""

    verification_rules = ""
    if profile.category == "verification_question":
        verification_rules = """
VERIFICATION MODE:
- The first line of Direct Answer MUST start with exactly one of: "Yes." or "No."
- If "Yes", provide 1-3 evidence bullets with citations.
- If "No", write "The document does not specify this." and include the closest related evidence (if any).
"""

    return f"""Document: "{document_name}"

RETRIEVED CONTEXT:
{context_str}

USER QUESTION:
{query}

ANSWER MODE: {profile.answer_mode} | CATEGORY: {profile.category}

Use ONLY these sections (no others):
{section_guide}
{anti_hallucination}
{verification_rules}
{code_rules}
Be direct and concise. Answer every part of the question."""


def build_regeneration_prompt(
    *,
    query: str,
    context_str: str,
    document_name: str,
    profile: AnswerProfile,
    requirements: dict[str, Any],
    previous_answer: str,
    quality_issues: list[str],
) -> str:
    base = build_answer_user_prompt(
        query=query,
        context_str=context_str,
        document_name=document_name,
        profile=profile,
        requirements=requirements,
    )
    issues_text = "\n".join(f"- {issue}" for issue in quality_issues[:10])
    return f"""REGENERATE the answer. The previous attempt failed quality validation.

ISSUES TO FIX:
{issues_text}

PREVIOUS ANSWER (do not repeat mistakes):
{previous_answer[:2000]}

{base}
"""


def answer_includes_required_code(answer: str, requirements: dict[str, Any]) -> bool:
    if not requirements.get("requires_code"):
        return True
    return bool(re.search(r"```python", answer, re.IGNORECASE))


# Backward-compatible export used by orchestrator imports
_ANSWER_GENERATION_RULES = PRODUCTION_SYSTEM_PROMPT
