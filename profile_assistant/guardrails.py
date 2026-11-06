"""Input and output guardrails for the Profile Assistant.

Input: reject empty, overly long, and obvious prompt-injection questions
before any model call. Output: block answers that leak system-prompt text or
look like secrets. These are cheap deterministic checks; the system prompts
themselves remain the primary defence against injection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .prompts import CONFIDENTIAL_PROMPTS, EMPTY_INPUT_MESSAGE, INPUT_REFUSAL, TOO_LONG_MESSAGE

_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|earlier|all|your|the|these)\b.{0,20}\b(instructions?|rules?|prompts?|directives?|guidelines?)\b",
        r"\b(reveal|show|print|repeat|output|display|leak|tell me)\b.{0,40}\b(system|hidden|initial|original)\s+(prompt|instructions?|message)\b",
        r"\bsystem\s+prompt\b",
        r"\byou\s+are\s+now\b",
        r"\bpretend\s+(to\s+be|you\s+are)\b",
        r"\b(developer|dan|god)\s+mode\b",
        r"\bjailbreak",
        r"\bnew\s+instructions?\s*:",
        r"</?\s*(system|profile|instructions?)\s*>",
        r"^\s*(system|assistant)\s*:",
    )
)

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p)
    for p in (
        r"\bsk-[A-Za-z0-9_\-]{20,}",  # OpenAI-style keys
        r"\bAKIA[0-9A-Z]{16}\b",  # AWS access key id
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"\bgh[pousr]_[A-Za-z0-9]{30,}\b",  # GitHub tokens
        r"\bxox[abpr]-[A-Za-z0-9\-]{10,}",  # Slack tokens
        r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT
    )
)

_MIN_LEAK_FRAGMENT_CHARS = 40


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _prompt_fragments() -> tuple[str, ...]:
    """Distinctive lines of the system prompts (template placeholders excluded)."""
    fragments: list[str] = []
    for prompt in CONFIDENTIAL_PROMPTS:
        for line in prompt.splitlines():
            line = re.sub(r"^\s*(\d+\.|-)\s*", "", line)
            if "{" in line or len(line) < _MIN_LEAK_FRAGMENT_CHARS:
                continue
            fragments.append(_normalize(line))
    return tuple(dict.fromkeys(fragments))


_PROMPT_FRAGMENTS = _prompt_fragments()


@dataclass(frozen=True)
class GuardResult:
    """Outcome of a guardrail check."""

    allowed: bool
    reason: str = ""
    message: str = ""


def check_input(question: str, *, max_chars: int) -> GuardResult:
    """Validate a user question before it reaches any model."""
    text = (question or "").strip()
    if not text:
        return GuardResult(False, "empty", EMPTY_INPUT_MESSAGE)
    if len(text) > max_chars:
        return GuardResult(False, "too_long", TOO_LONG_MESSAGE.format(limit=max_chars))
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return GuardResult(False, "prompt_injection", INPUT_REFUSAL)
    return GuardResult(True)


def check_output(text: str, *, api_key: str = "") -> GuardResult:
    """Block model output that leaks system-prompt text or secrets."""
    if api_key and api_key in text:
        return GuardResult(False, "secret_leak")
    for pattern in _SECRET_PATTERNS:
        if pattern.search(text):
            return GuardResult(False, "secret_leak")
    normalized = _normalize(text)
    for fragment in _PROMPT_FRAGMENTS:
        if fragment in normalized:
            return GuardResult(False, "system_prompt_leak")
    return GuardResult(True)
