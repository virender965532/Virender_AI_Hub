"""Prompt-injection defense (Guardrail 4, section 46).

Job descriptions are untrusted external content. The real defense is
*architectural*: the untrusted text is always wrapped in explicit delimiters
inside a prompt whose system instructions tell the model to treat everything
between them as inert data, never as instructions. ``detect_suspicious_patterns``
additionally flags likely injection attempts for logging/guardrail metrics —
it never silently "fixes" the text, and a detection never blocks the job by
itself (the match/reject decision belongs to the matching pipeline).
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(the\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(the\s+)?(prior|previous|above)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?(system|hidden|internal)\s+(prompt|instructions)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you\s+are\s+)?", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"print\s+(your|the)\s+(instructions|prompt)", re.IGNORECASE),
]

UNTRUSTED_OPEN = "<<<UNTRUSTED_JOB_CONTENT_START>>>"
UNTRUSTED_CLOSE = "<<<UNTRUSTED_JOB_CONTENT_END>>>"


def detect_suspicious_patterns(text: str) -> list[str]:
    if not text:
        return []
    hits: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


def wrap_untrusted_content(text: str) -> str:
    """Delimit untrusted job text for inclusion in an LLM prompt."""
    safe_text = (text or "").replace(UNTRUSTED_OPEN, "").replace(UNTRUSTED_CLOSE, "")
    return f"{UNTRUSTED_OPEN}\n{safe_text}\n{UNTRUSTED_CLOSE}"
