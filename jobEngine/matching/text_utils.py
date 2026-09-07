"""Tokenization helpers shared by the rule-based filter and the scorer.

Deliberately simple (regex tokenization, no NLP dependency) — the point is
consistent, explainable overlap counting, not semantic understanding (that's
what the embedding + LLM stages are for).
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
_STOPWORDS = frozenset(
    {
        "and", "the", "for", "with", "you", "our", "are", "will", "have",
        "this", "that", "your", "role", "team", "work", "years", "year",
        "experience", "strong", "good", "knowledge", "ability", "skills",
    }
)


def normalize_tokens(text: str, *, min_len: int = 2) -> set[str]:
    if not text:
        return set()
    raw = _TOKEN_RE.findall(text.lower())
    return {t for t in raw if len(t) >= min_len and t not in _STOPWORDS}


def skill_token_set(skills: list[str]) -> set[str]:
    tokens: set[str] = set()
    for skill in skills or []:
        tokens |= normalize_tokens(skill, min_len=1)
    return tokens


def overlap_score(needle_tokens: set[str], haystack_tokens: set[str]) -> float:
    """Percentage of ``needle_tokens`` present in ``haystack_tokens``."""
    if not needle_tokens:
        return 0.0
    return round(100.0 * len(needle_tokens & haystack_tokens) / len(needle_tokens), 1)
