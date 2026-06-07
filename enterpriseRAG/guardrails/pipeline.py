from __future__ import annotations

import re
from typing import Any

from enterpriseRAG.services.observability import ObservabilityService

# Input guardrail patterns
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"disregard\s+(all\s+)?(prior|previous)",
    r"you\s+are\s+now\s+",
    r"system\s+prompt",
    r"jailbreak",
    r"do\s+anything\s+now",
    r"DAN\s+mode",
]

_JAILBREAK_PATTERNS = [
    r"pretend\s+you\s+are\s+not",
    r"bypass\s+safety",
    r"without\s+restrictions",
    r"no\s+ethical",
]

_TOXIC_PATTERNS = [
    r"\b(hate|kill|attack)\s+(all|every)",
]

_PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN-REDACTED]"),
    (r"\b\d{16}\b", "[CARD-REDACTED]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL-REDACTED]"),
    (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE-REDACTED]"),
]

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


class GuardrailPipeline:
    """Enterprise guardrails for input, retrieval, generation, and output."""

    def __init__(self, observability: ObservabilityService | None = None) -> None:
        self.obs = observability

    def _record(self, stage: str, check: str, passed: bool, **details: Any) -> None:
        if self.obs:
            self.obs.record_guardrail(stage, check, passed, **details)

    def check_input(self, query: str) -> dict[str, Any]:
        results: dict[str, Any] = {"passed": True, "checks": [], "blocked_reason": None}
        lower = query.lower()

        for pattern in _INJECTION_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                self._record("input", "prompt_injection", False, pattern=pattern)
                results["checks"].append({"check": "prompt_injection", "passed": False})
                results["passed"] = False
                results["blocked_reason"] = "Potential prompt injection detected."
                return results
        self._record("input", "prompt_injection", True)

        for pattern in _JAILBREAK_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                self._record("input", "jailbreak", False, pattern=pattern)
                results["checks"].append({"check": "jailbreak", "passed": False})
                results["passed"] = False
                results["blocked_reason"] = "Potential jailbreak attempt detected."
                return results
        self._record("input", "jailbreak", True)

        for pattern in _TOXIC_PATTERNS:
            if re.search(pattern, lower, re.IGNORECASE):
                self._record("input", "toxicity", False)
                results["checks"].append({"check": "toxicity", "passed": False})
                results["passed"] = False
                results["blocked_reason"] = "Potentially harmful content detected."
                return results
        self._record("input", "toxicity", True)

        sensitive_found = bool(re.search(r"\b(password|secret|api[_-]?key|credential)\s*[:=]", lower))
        self._record("input", "sensitive_data", not sensitive_found)
        results["checks"].append({"check": "sensitive_data", "passed": not sensitive_found})

        return results

    def check_retrieval(self, chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not chunks:
            self._record("retrieval", "context_available", False)
            return [], {"passed": False, "reason": "No chunks retrieved"}

        filtered: list[dict[str, Any]] = []
        for chunk in chunks:
            content = chunk.get("content", "")
            if len(content.strip()) < 20:
                continue
            if chunk.get("score", 0) < 0.01 and chunk.get("rerank_score") is None:
                continue
            filtered.append(chunk)

        seen_content: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for chunk in filtered:
            key = content_key = chunk.get("content", "")[:300]
            if key in seen_content:
                continue
            seen_content.add(key)
            deduped.append(chunk)

        quality_scores = [min(1.0, len(c.get("content", "")) / 500) for c in deduped]
        avg_quality = sum(quality_scores) / max(len(quality_scores), 1)

        self._record("retrieval", "duplicate_removal", True, removed=len(filtered) - len(deduped))
        self._record("retrieval", "context_quality", avg_quality > 0.3, avg_quality=avg_quality)

        return deduped, {
            "passed": len(deduped) > 0,
            "avg_quality": round(avg_quality, 3),
            "chunk_count": len(deduped),
        }

    def check_generation(
        self, answer: str, chunks: list[dict[str, Any]], citations_required: bool = True
    ) -> dict[str, Any]:
        has_citations = bool(re.search(r"\(Section:|Page:|Source:", answer, re.IGNORECASE))
        citation_ok = has_citations or not citations_required or not chunks

        unsupported_claims = len(re.findall(r"\b(definitely|certainly|always|never)\b", answer, re.I))
        confidence = max(0.0, min(1.0, 1.0 - unsupported_claims * 0.05))

        self._record("generation", "citation_enforcement", citation_ok)
        self._record("generation", "confidence_scoring", True, confidence=confidence)

        return {
            "passed": citation_ok,
            "confidence": round(confidence, 3),
            "has_citations": has_citations,
        }

    def check_output(self, answer: str) -> tuple[str, dict[str, Any]]:
        masked = answer
        pii_found = False
        for pattern, replacement in _PII_PATTERNS:
            if re.search(pattern, masked):
                pii_found = True
                masked = re.sub(pattern, replacement, masked)

        unsafe = bool(re.search(r"\b(hack|exploit|bypass\s+security)\b", answer, re.I))
        self._record("output", "pii_masking", True, pii_found=pii_found)
        self._record("output", "unsafe_content", not unsafe)

        return masked, {
            "passed": not unsafe,
            "pii_masked": pii_found,
            "policy_compliant": not unsafe,
        }

    def detect_hallucination_risk(
        self,
        answer: str,
        chunks: list[dict[str, Any]],
        *,
        verification: dict[str, Any] | None = None,
        critic: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        verification = verification or {}
        critic = critic or {}

        if not chunks:
            return {
                "risk": "high",
                "score": 0.85,
                "risk_pct": 85.0,
                "reason": "No source context",
                "overlap": 0.0,
            }

        # Generated Python/examples are intentional — exclude from grounding check.
        answer_for_check = re.sub(r"```[\s\S]*?```", " ", answer)
        answer_for_check = re.sub(r"# .*", " ", answer_for_check)

        chunk_text = " ".join(c.get("content", "") for c in chunks)[:12000].lower()
        answer_words = {
            w
            for w in re.findall(r"\b[a-z]{4,}\b", answer_for_check.lower())
            if w not in _STOPWORDS
        }
        chunk_words = {
            w
            for w in re.findall(r"\b[a-z]{4,}\b", chunk_text)
            if w not in _STOPWORDS
        }

        if not answer_words:
            return {
                "risk": "medium",
                "score": 0.35,
                "risk_pct": 35.0,
                "reason": "Short answer",
                "overlap": 0.0,
            }

        overlap = len(answer_words & chunk_words) / max(len(answer_words), 1)
        citation_count = len(re.findall(r"\(Section:", answer, re.IGNORECASE))
        has_code = bool(re.search(r"```", answer))

        # Start from paraphrase-aware baseline (executive summaries rarely share raw tokens).
        risk_score = 0.28
        risk_score += max(0.0, 0.22 - overlap * 0.5)

        # Strong grounding signals reduce risk.
        if citation_count >= 2:
            risk_score -= 0.14
        elif citation_count >= 1:
            risk_score -= 0.08

        if len(chunks) >= 5:
            risk_score -= 0.10
        elif len(chunks) >= 3:
            risk_score -= 0.06

        if verification.get("sufficient", True):
            risk_score -= 0.08

        if float(critic.get("accuracy_score", 0.85)) >= 0.8:
            risk_score -= 0.06

        if has_code:
            # Code blocks are synthesized by design — small penalty only.
            risk_score -= 0.04

        # Well-cited, multi-chunk answers: cap hallucination risk at 10–18%.
        if citation_count >= 2 and len(chunks) >= 3:
            risk_score = min(risk_score, 0.18)
            if verification.get("sufficient", True):
                risk_score = max(risk_score, 0.10)

        risk_score = round(min(0.95, max(0.08, risk_score)), 3)
        risk_pct = round(risk_score * 100, 1)

        if risk_pct <= 20:
            risk = "low"
        elif risk_pct <= 40:
            risk = "medium"
        else:
            risk = "high"

        self._record(
            "generation",
            "hallucination_detection",
            risk == "low",
            score=risk_score,
            risk_pct=risk_pct,
        )
        return {
            "risk": risk,
            "score": risk_score,
            "risk_pct": risk_pct,
            "overlap": round(overlap, 3),
            "citation_count": citation_count,
            "chunk_count": len(chunks),
        }

    def status_summary(self) -> dict[str, Any]:
        if not self.obs:
            return {"overall": "unknown", "checks": []}
        events = self.obs.guardrail_events
        failed = [e for e in events if not e.get("passed")]
        return {
            "overall": "pass" if not failed else "warn",
            "total_checks": len(events),
            "failed_checks": len(failed),
            "events": events,
        }
