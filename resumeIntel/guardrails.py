"""Guardrails for Resume Intelligence.

Both the resume text and the pasted job description are untrusted external
content. They are always wrapped in explicit delimiters inside the prompt,
with system instructions telling the model to treat everything between them
as inert data, never as instructions (see jobEngine's equivalent guardrail
for the same architectural pattern).
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(the\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(the\s+)?(prior|previous|above)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?(system|hidden|internal)\s+(prompt|instructions)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
]

UNTRUSTED_OPEN = "<<<UNTRUSTED_CONTENT_START>>>"
UNTRUSTED_CLOSE = "<<<UNTRUSTED_CONTENT_END>>>"


def detect_suspicious_patterns(text: str) -> list[str]:
    if not text:
        return []
    return [pattern.pattern for pattern in _INJECTION_PATTERNS if pattern.search(text)]


def wrap_untrusted_content(text: str) -> str:
    safe_text = (text or "").replace(UNTRUSTED_OPEN, "").replace(UNTRUSTED_CLOSE, "")
    return f"{UNTRUSTED_OPEN}\n{safe_text}\n{UNTRUSTED_CLOSE}"


def clamp_text(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    return text[:max_chars]
