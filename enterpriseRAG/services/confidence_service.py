from __future__ import annotations

from typing import Any

# Common words that inflate false "hallucination" overlap scores.
_STOPWORDS = frozenset(
    {
        "about", "after", "also", "answer", "based", "been", "being", "below",
        "business", "could", "document", "example", "from", "have", "help",
        "here", "implementation", "include", "into", "more", "other", "provide",
        "response", "section", "should", "such", "that", "their", "them",
        "these", "they", "this", "through", "using", "what", "when", "which",
        "will", "with", "would", "your",
    }
)


def compute_confidence_score(
    *,
    hallucination_check: dict[str, Any],
    verification: dict[str, Any] | None = None,
    critic_feedback: dict[str, Any] | None = None,
    generation_guardrails: dict[str, Any] | None = None,
) -> float:
    """Unified confidence score (0-1) shown in the dashboard ring."""
    verification = verification or {}
    critic_feedback = critic_feedback or {}
    generation_guardrails = generation_guardrails or {}

    risk_pct = float(hallucination_check.get("risk_pct", 50))
    grounded = 1.0 - (risk_pct / 100.0)

    verification_conf = float(verification.get("confidence", 0.85))
    critic_acc = float(critic_feedback.get("accuracy_score", 0.85))
    gen_conf = float(generation_guardrails.get("confidence", 0.85))

    confidence = (
        0.40 * grounded
        + 0.30 * verification_conf
        + 0.20 * critic_acc
        + 0.10 * gen_conf
    )

    # Well-grounded answers with citations should present high confidence (82%+).
    if risk_pct <= 20 and verification.get("sufficient", True):
        confidence = max(confidence, 0.82)

    return round(min(0.98, max(0.0, confidence)), 3)
