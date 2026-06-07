from __future__ import annotations

import re
from typing import Any

from enterpriseRAG.agents.intent_profile import EXECUTIVE_SECTIONS

# Minimum overall score to accept without regeneration.
QUALITY_THRESHOLD = 0.80

_FORBIDDEN_UNCITED_PATTERNS = [
    r"\bROI\b",
    r"\breturn on investment\b",
    r"\bcustomer satisfaction\b",
    r"\brevenue growth\b",
    r"\blegal ramifications\b",
    r"\bcompetitive advantage\b",
    r"\boperational costs\b",
    r"\bbrand reputation\b",
    r"\blong-term cost savings\b",
]


def _extract_python_blocks(answer: str) -> list[str]:
    return re.findall(r"```python\s*\n([\s\S]*?)```", answer, re.IGNORECASE)


def _strip_code(answer: str) -> str:
    return re.sub(r"```[\s\S]*?```", " ", answer)


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p and len(p.strip()) > 20]


def validate_code_examples(answer: str, excluded_terms: list[str] | None = None) -> list[str]:
    """Detect logical bugs in Python examples."""
    errors: list[str] = []
    excluded = [t.lower() for t in (excluded_terms or []) if t]

    for i, block in enumerate(_extract_python_blocks(answer), start=1):
        if excluded and ".lower()" not in block and "casefold" not in block:
            errors.append(f"Code block {i}: use case-insensitive matching for excluded terms.")

        if excluded:
            for term in excluded:
                if re.search(rf"if\s+{re.escape(term)}\s+in\s+response", block, re.I):
                    errors.append(f"Code block {i}: case-sensitive check for '{term}' — use .lower().")

        # Valid example strings must not contain forbidden terms.
        for match in re.finditer(
            r"(?:llm_response|valid_response|sample_response|response)\s*=\s*[\"']([^\"']+)[\"']",
            block,
            re.I,
        ):
            value = match.group(1)
            if any(term in value.lower() for term in excluded):
                errors.append(
                    f"Code block {i}: valid example string contains forbidden term: '{value[:60]}'"
                )

        # Comments claiming "should not mention X" while string contains X.
        for term in excluded:
            if re.search(rf"should not mention {re.escape(term)}", block, re.I):
                window = block[max(0, block.lower().find("should not") - 20) : block.lower().find("should not") + 120]
                if term in window.lower() and '"' in window:
                    errors.append(f"Code block {i}: comment contradicts example containing '{term}'.")

        if "def " in block and "if __name__" not in block and "print(" not in block:
            errors.append(f"Code block {i}: add runnable demo (print or __main__ block).")

    return errors


def detect_executive_template_leak(answer: str, suppress_executive: bool) -> list[str]:
    """Flag unsolicited business-report sections."""
    if not suppress_executive:
        return []

    issues: list[str] = []
    for section in EXECUTIVE_SECTIONS:
        if re.search(rf"^#{{1,3}}\s*{re.escape(section)}\b", answer, re.I | re.M):
            issues.append(f"Unsolicited executive section: '{section}'")
        if re.search(rf"\b{re.escape(section)}\s*\n", answer, re.I):
            issues.append(f"Unsolicited executive section: '{section}'")
    return issues


def detect_uncited_business_claims(answer: str) -> list[str]:
    """Flag common hallucinated business phrases without citations."""
    text = _strip_code(answer)
    issues: list[str] = []
    for pattern in _FORBIDDEN_UNCITED_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            # Check if citation nearby (same paragraph / within 200 chars)
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 150)
            window = text[start:end]
            if not re.search(r"\(Section:", window, re.I):
                issues.append(f"Uncited business claim: '{match.group(0)}'")
    return issues


def score_conciseness(answer: str, answer_mode: str) -> float:
    word_count = len(answer.split())
    if answer_mode == "technical":
        if word_count <= 350:
            return 1.0
        if word_count <= 500:
            return 0.8
        return max(0.3, 1.0 - (word_count - 500) / 500)
    if word_count <= 200:
        return 1.0
    if word_count <= 350:
        return 0.85
    return max(0.4, 1.0 - (word_count - 350) / 400)


def score_grounding(answer: str, chunks: list[dict[str, Any]]) -> float:
    if not chunks:
        return 0.0
    text = _strip_code(answer)
    citation_count = len(re.findall(r"\(Section:", text, re.I))
    uncited = detect_uncited_business_claims(answer)
    base = min(1.0, citation_count * 0.25)
    penalty = min(0.5, len(uncited) * 0.1)
    return round(max(0.0, base - penalty + (0.3 if citation_count >= 2 else 0)), 3)


def detect_unsupported_statements(answer: str, chunks: list[dict[str, Any]]) -> list[str]:
    """Heuristic unsupported-statement detection against retrieved context."""
    if not chunks:
        return ["No retrieved chunks available for grounding."]

    text = _strip_code(answer)
    chunk_text = " ".join(c.get("content", "") for c in chunks).lower()
    issues: list[str] = []

    for sentence in _sentence_split(text):
        if re.search(r"\(Section:", sentence, re.I):
            continue
        # Allow explicit uncertainty statements.
        if re.search(r"document does not specify|not explicitly stated", sentence, re.I):
            continue
        words = [w for w in re.findall(r"\b[a-z]{4,}\b", sentence.lower())]
        if len(words) < 5:
            continue
        overlap = sum(1 for w in set(words) if w in chunk_text) / max(len(set(words)), 1)
        # factual-sounding claim with poor overlap and no citation
        if overlap < 0.18 and re.search(r"\b(is|are|will|can|leads to|results in|improves)\b", sentence, re.I):
            issues.append(f"Unsupported statement: '{sentence[:140]}'")

    return issues


def score_intent_alignment(
    answer: str,
    requirements: dict[str, Any],
    suppress_executive: bool,
) -> float:
    score = 1.0
    if requirements.get("requires_code"):
        if not re.search(r"```python", answer, re.I):
            score -= 0.5
    leaks = detect_executive_template_leak(answer, suppress_executive)
    score -= min(0.6, len(leaks) * 0.2)
    return round(max(0.0, score), 3)


def validate_answer_quality(
    *,
    query: str,
    answer: str,
    chunks: list[dict[str, Any]],
    requirements: dict[str, Any],
    answer_profile: dict[str, Any],
) -> dict[str, Any]:
    """Score answer quality; recommend regeneration if below threshold."""
    suppress_executive = answer_profile.get("suppress_executive", True)
    answer_mode = answer_profile.get("answer_mode", "concise")
    excluded = requirements.get("excluded_terms", [])

    code_errors = validate_code_examples(answer, excluded)
    executive_leaks = detect_executive_template_leak(answer, suppress_executive)
    uncited_claims = detect_uncited_business_claims(answer)
    unsupported = detect_unsupported_statements(answer, chunks)

    intent_score = score_intent_alignment(answer, requirements, suppress_executive)
    grounding_score = score_grounding(answer, chunks)
    conciseness_score = score_conciseness(answer, answer_mode)
    code_score = 1.0 if not code_errors else max(0.0, 1.0 - len(code_errors) * 0.25)
    hallucination_score = max(0.0, 1.0 - min(0.8, len(unsupported) * 0.12 + len(uncited_claims) * 0.1))

    # Verification questions should be yes/no-first.
    if answer_profile.get("category") == "verification_question":
        first_nonempty = next((ln.strip() for ln in answer.splitlines() if ln.strip()), "")
        if not re.match(r"^(Yes\.|No\.)", first_nonempty):
            intent_score = max(0.0, intent_score - 0.35)
            unsupported.append("Verification question must start with 'Yes.' or 'No.'")

    overall = round(
        0.25 * intent_score
        + 0.25 * grounding_score
        + 0.20 * hallucination_score
        + 0.15 * code_score
        + 0.15 * conciseness_score,
        3,
    )

    issues = code_errors + executive_leaks + uncited_claims + unsupported
    needs_regeneration = overall < QUALITY_THRESHOLD or bool(executive_leaks) or (
        requirements.get("requires_code") and code_errors
    ) or bool(unsupported)

    # Hard cap quality if there are substantial unsupported claims.
    if len(unsupported) >= 2:
        overall = min(overall, 0.65)
        needs_regeneration = True

    return {
        "overall_score": overall,
        "intent_alignment": intent_score,
        "grounding_score": grounding_score,
        "hallucination_score": round(hallucination_score, 3),
        "code_score": code_score,
        "conciseness_score": conciseness_score,
        "unsupported_statement_count": len(unsupported),
        "issues": issues,
        "needs_regeneration": needs_regeneration,
        "passed": overall >= QUALITY_THRESHOLD and not executive_leaks,
    }
