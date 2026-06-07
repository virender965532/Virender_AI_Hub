from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Categories used for intent-aware generation.
CATEGORY_DEFINITION = "definition"
CATEGORY_TECHNICAL = "technical_question"
CATEGORY_CODE = "code_generation"
CATEGORY_BUSINESS = "business_analysis"
CATEGORY_EXECUTIVE = "executive_briefing"
CATEGORY_RESEARCH = "research_report"
CATEGORY_SUMMARY = "summarization"
CATEGORY_MULTI_HOP = "multi_hop_retrieval"
CATEGORY_VERIFICATION = "verification_question"
CATEGORY_ARCHITECTURE = "agent_architecture"

EXECUTIVE_SECTIONS = frozenset({
    "Executive Summary",
    "Business Impact",
    "Risks",
    "Recommendations",
    "Expected ROI",
    "Strategic Alignment",
    "Workforce Impact",
})

TECHNICAL_RETRIEVAL_PRIORITIES = (
    "guardrails",
    "validation checkpoints",
    "output filtering",
    "fallback mechanisms",
    "agent safety",
    "implementation",
)

SUMMARY_RETRIEVAL_PRIORITIES = (
    "design patterns",
    "planning pattern",
    "reflection pattern",
    "components",
    "lifecycle",
    "agent workflow",
)

MULTI_HOP_RETRIEVAL_PRIORITIES = (
    "projects",
    "frameworks",
    "tools",
    "llm models",
    "agent count",
    "use cases",
    "compare",
)

PRODUCTION_SYSTEM_PROMPT = """You are a precise enterprise RAG assistant. Answer ONLY from the provided document context.

STRICT RULES:
1. **Grounding first** — Every factual claim about the document MUST cite (Section: <title>, Page: <number>).
2. **No fabrication** — If information is NOT in the retrieved context, write:
   "The document does not specify this." Do NOT invent ROI, business impact, legal risk, or statistics.
3. **No unsolicited business reports** — Do NOT add Executive Summary, Business Impact, Risks, or Expected ROI
   unless the user explicitly requested a business or executive analysis.
4. **Concise** — Answer only what was asked. No filler paragraphs.
5. **Separate sections**:
   - Document-grounded facts (with citations)
   - Code examples (no citations inside code blocks)
6. **Code quality** — When writing Python:
   - Use case-insensitive term checks (.lower())
   - Valid demo strings must NOT contain forbidden terms
   - Show both a passing example and a blocked example
   - Implement validation checkpoint + fallback message pattern from the document
7. **Tone** — Clear, direct, technical when code is requested; avoid board-report language unless requested.
"""


@dataclass(frozen=True)
class AnswerProfile:
    category: str
    answer_mode: str  # technical | executive | concise | report
    sections: tuple[str, ...]
    system_prompt: str
    retrieval_priorities: tuple[str, ...]
    suppress_executive: bool
    require_citations: bool
    allow_synthesis: bool  # code section only


def classify_query_intent(query: str, requirements: dict[str, Any]) -> dict[str, Any]:
    """Classify query before generation. Heuristics-first for reliability."""
    lower = query.lower()

    wants_executive = bool(
        re.search(
            r"\b(executive summary|business impact|roi|return on investment|"
            r"board report|strategic briefing|cost savings|competitive advantage)\b",
            lower,
        )
    )
    wants_business = bool(
        re.search(r"\b(business analysis|business case|market impact|revenue)\b", lower)
    )
    wants_report = bool(re.search(r"\b(full report|detailed report|research report)\b", lower))
    wants_summary = bool(re.search(r"\b(summarize|summary of|overview of)\b", lower))
    is_definition = bool(re.search(r"\b(what is|what are|define|explain)\b", lower))
    is_verification = bool(
        re.search(
            r"\b(does the document mention|does the pdf mention|is .* mentioned|which chapter discusses)\b",
            lower,
        )
    )
    is_multi_hop = bool(
        re.search(
            r"\b(compare|across all projects|list all projects|which projects use|"
            r"create a comparison table|identify every use|consolidated architecture diagram)\b",
            lower,
        )
    )
    is_architecture = bool(
        re.search(
            r"\b(design a multi-agent architecture|redesign|how would you add|"
            r"most suitable pattern|using patterns from)\b",
            lower,
        )
    )

    if requirements.get("requires_code"):
        category = CATEGORY_CODE
        if is_definition:
            category = CATEGORY_CODE  # definition + code
    elif is_verification:
        category = CATEGORY_VERIFICATION
    elif is_multi_hop:
        category = CATEGORY_MULTI_HOP
    elif is_architecture:
        category = CATEGORY_ARCHITECTURE
    elif wants_executive:
        category = CATEGORY_EXECUTIVE
    elif wants_business:
        category = CATEGORY_BUSINESS
    elif wants_report:
        category = CATEGORY_RESEARCH
    elif wants_summary:
        category = CATEGORY_SUMMARY
    elif is_definition:
        category = CATEGORY_DEFINITION
    elif re.search(r"\b(how to|implement|architecture|design|pattern)\b", lower):
        category = CATEGORY_TECHNICAL
    else:
        category = CATEGORY_DEFINITION

    # Technical/code questions must NEVER use executive mode unless explicitly requested.
    if category == CATEGORY_CODE and not wants_executive:
        answer_mode = "technical"
    elif category == CATEGORY_VERIFICATION:
        answer_mode = "concise"
    elif category in (CATEGORY_MULTI_HOP, CATEGORY_ARCHITECTURE):
        answer_mode = "report"
    elif category in (CATEGORY_EXECUTIVE, CATEGORY_BUSINESS) and wants_executive:
        answer_mode = "executive"
    elif category == CATEGORY_RESEARCH:
        answer_mode = "report"
    elif wants_summary:
        answer_mode = "concise"
    else:
        answer_mode = "concise"

    return {
        "category": category,
        "answer_mode": answer_mode,
        "wants_executive": wants_executive,
        "wants_business": wants_business,
        "is_multi_hop": is_multi_hop,
        "is_verification": is_verification,
    }


def resolve_answer_profile(
    *,
    query: str,
    requirements: dict[str, Any],
    intent_classification: dict[str, Any],
    role_config: dict[str, Any],
) -> AnswerProfile:
    """Resolve sections and prompts from query intent — role is secondary."""
    category = intent_classification.get("category", CATEGORY_DEFINITION)
    answer_mode = intent_classification.get("answer_mode", "concise")
    wants_executive = intent_classification.get("wants_executive", False)

    role_sections = tuple(role_config.get("answer_sections", ("Direct Answer",)))
    role_prompt = role_config.get("system_prompt", PRODUCTION_SYSTEM_PROMPT)
    role_priorities = tuple(role_config.get("retrieval_priorities", ()))

    # ── Code / technical: override CEO template entirely ─────────────────
    if requirements.get("requires_code") or category == CATEGORY_CODE:
        sections: tuple[str, ...] = ("Direct Answer", "From Document", "Code Example")
        return AnswerProfile(
            category=category,
            answer_mode="technical",
            sections=sections,
            system_prompt=PRODUCTION_SYSTEM_PROMPT,
            retrieval_priorities=TECHNICAL_RETRIEVAL_PRIORITIES,
            suppress_executive=True,
            require_citations=True,
            allow_synthesis=True,
        )

    if category == CATEGORY_TECHNICAL:
        return AnswerProfile(
            category=category,
            answer_mode="technical",
            sections=("Direct Answer", "From Document", "Implementation Notes"),
            system_prompt=PRODUCTION_SYSTEM_PROMPT,
            retrieval_priorities=TECHNICAL_RETRIEVAL_PRIORITIES,
            suppress_executive=True,
            require_citations=True,
            allow_synthesis=True,
        )

    if category == CATEGORY_SUMMARY:
        return AnswerProfile(
            category=category,
            answer_mode="concise",
            sections=("Summary", "From Document"),
            system_prompt=PRODUCTION_SYSTEM_PROMPT,
            retrieval_priorities=SUMMARY_RETRIEVAL_PRIORITIES,
            suppress_executive=True,
            require_citations=True,
            allow_synthesis=False,
        )

    if category == CATEGORY_VERIFICATION:
        return AnswerProfile(
            category=category,
            answer_mode="concise",
            sections=("Direct Answer", "Evidence"),
            system_prompt=PRODUCTION_SYSTEM_PROMPT,
            retrieval_priorities=SUMMARY_RETRIEVAL_PRIORITIES,
            suppress_executive=True,
            require_citations=True,
            allow_synthesis=False,
        )

    if category == CATEGORY_DEFINITION or answer_mode == "concise":
        return AnswerProfile(
            category=category,
            answer_mode="concise",
            sections=("Direct Answer", "From Document"),
            system_prompt=PRODUCTION_SYSTEM_PROMPT,
            retrieval_priorities=SUMMARY_RETRIEVAL_PRIORITIES,
            suppress_executive=True,
            require_citations=True,
            allow_synthesis=False,
        )

    if category == CATEGORY_MULTI_HOP:
        return AnswerProfile(
            category=category,
            answer_mode="report",
            sections=("Direct Answer", "Findings", "From Document"),
            system_prompt=PRODUCTION_SYSTEM_PROMPT,
            retrieval_priorities=MULTI_HOP_RETRIEVAL_PRIORITIES,
            suppress_executive=True,
            require_citations=True,
            allow_synthesis=False,
        )

    if category == CATEGORY_ARCHITECTURE:
        return AnswerProfile(
            category=category,
            answer_mode="report",
            sections=("Recommended Architecture", "Rationale", "From Document"),
            system_prompt=PRODUCTION_SYSTEM_PROMPT,
            retrieval_priorities=MULTI_HOP_RETRIEVAL_PRIORITIES,
            suppress_executive=True,
            require_citations=True,
            allow_synthesis=True,
        )

    # Executive / business ONLY when explicitly requested
    if wants_executive and category in (CATEGORY_EXECUTIVE, CATEGORY_BUSINESS):
        return AnswerProfile(
            category=category,
            answer_mode="executive",
            sections=role_sections,
            system_prompt=role_prompt + "\n\n" + PRODUCTION_SYSTEM_PROMPT,
            retrieval_priorities=role_priorities,
            suppress_executive=False,
            require_citations=True,
            allow_synthesis=False,
        )

    # Default: concise grounded answer — never full CEO template by default
    return AnswerProfile(
        category=category,
        answer_mode="concise",
        sections=("Direct Answer", "From Document"),
        system_prompt=PRODUCTION_SYSTEM_PROMPT,
        retrieval_priorities=role_priorities or TECHNICAL_RETRIEVAL_PRIORITIES,
        suppress_executive=True,
        require_citations=True,
        allow_synthesis=False,
    )


def profile_to_dict(profile: AnswerProfile) -> dict[str, Any]:
    return {
        "category": profile.category,
        "answer_mode": profile.answer_mode,
        "sections": list(profile.sections),
        "system_prompt": profile.system_prompt,
        "retrieval_priorities": list(profile.retrieval_priorities),
        "suppress_executive": profile.suppress_executive,
        "require_citations": profile.require_citations,
        "allow_synthesis": profile.allow_synthesis,
    }


def profile_from_dict(data: dict[str, Any]) -> AnswerProfile:
    return AnswerProfile(
        category=data.get("category", CATEGORY_DEFINITION),
        answer_mode=data.get("answer_mode", "concise"),
        sections=tuple(data.get("sections", ("Direct Answer",))),
        system_prompt=data.get("system_prompt", PRODUCTION_SYSTEM_PROMPT),
        retrieval_priorities=tuple(data.get("retrieval_priorities", ())),
        suppress_executive=data.get("suppress_executive", True),
        require_citations=data.get("require_citations", True),
        allow_synthesis=data.get("allow_synthesis", False),
    )
