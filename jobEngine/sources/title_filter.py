"""Title keyword filter for sources whose search URL can't narrow results itself
(e.g. LinkedIn ``/jobs/search/`` without ``keywords=``). Configured per source
as ``title_keywords`` in ``config/job_sources.yaml``; an empty list keeps every job.

Matching is case-insensitive on whole words, and separators are flexible:
"full-stack" matches "Full Stack" / "Fullstack", "nodejs" matches "Node.js",
"cicd" matches "CI/CD". Short keywords such as "ai" do not match inside other
words ("Maintenance").
"""

from __future__ import annotations

import re
from functools import lru_cache

_SEPARATORS_RE = re.compile(r"[./\-_]")


@lru_cache(maxsize=32)
def _compile(keywords: tuple[str, ...]) -> re.Pattern[str] | None:
    alternatives: set[str] = set()
    for keyword in keywords:
        k = keyword.strip().lower()
        if not k:
            continue
        alternatives.add(re.escape(k.replace("-", " ")))
        alternatives.add(re.escape(k.replace("-", "")))
    if not alternatives:
        return None
    # Longest first so "ai-engineer" wins over "ai" (only matters for readability of logs).
    body = "|".join(sorted(alternatives, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])")


def title_matches_keywords(title: str | None, keywords: list[str] | tuple[str, ...]) -> bool:
    """True when ``title`` contains at least one keyword, or no keywords are configured."""
    pattern = _compile(tuple(keywords))
    if pattern is None:
        return True
    lowered = (title or "").lower()
    variants = (lowered, _SEPARATORS_RE.sub(" ", lowered), _SEPARATORS_RE.sub("", lowered))
    return any(pattern.search(v) for v in variants)
