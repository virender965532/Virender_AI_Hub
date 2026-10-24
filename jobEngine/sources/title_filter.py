"""Title keyword filter for sources whose search URL can't narrow results itself
(e.g. LinkedIn ``/jobs/search/`` without ``keywords=``). Configured per source
as ``title_keywords`` in ``config/job_sources.yaml``; an empty list keeps every job.

Matching is on whole words, and separators are flexible: "full-stack" matches
"Full Stack" / "Fullstack", "nodejs" matches "Node.js", "cicd" matches "CI/CD".
Short keywords such as "ai" do not match inside other words ("Maintenance").

Case: an all-lowercase keyword is matched case-insensitively (title lowercased).
A keyword with any uppercase letter (e.g. "AI") is used as it is and matched
case-sensitively against the original title, so "AI" matches "AI Engineer" but
not "Ai" / "ai".
"""

from __future__ import annotations

import re
from functools import lru_cache

_SEPARATORS_RE = re.compile(r"[./\-_]")


def _alternatives(keywords: list[str]) -> str | None:
    alternatives: set[str] = set()
    for k in keywords:
        alternatives.add(re.escape(k.replace("-", " ")))
        alternatives.add(re.escape(k.replace("-", "")))
    if not alternatives:
        return None
    # Longest first so "ai-engineer" wins over "ai" (only matters for readability of logs).
    return "|".join(sorted(alternatives, key=len, reverse=True))


@lru_cache(maxsize=32)
def _compile(keywords: tuple[str, ...]) -> tuple[re.Pattern[str] | None, re.Pattern[str] | None]:
    """(pattern for the lowercased title, pattern for the title as it is)."""
    lower: list[str] = []
    exact: list[str] = []
    for keyword in keywords:
        k = keyword.strip()
        if not k:
            continue
        if k == k.lower():
            lower.append(k)
        else:
            exact.append(k)
    lower_body = _alternatives(lower)
    exact_body = _alternatives(exact)
    return (
        re.compile(rf"(?<![a-z0-9])(?:{lower_body})(?![a-z0-9])") if lower_body else None,
        re.compile(rf"(?<![A-Za-z0-9])(?:{exact_body})(?![A-Za-z0-9])") if exact_body else None,
    )


def _variants(text: str) -> tuple[str, str, str]:
    return text, _SEPARATORS_RE.sub(" ", text), _SEPARATORS_RE.sub("", text)


def title_has_keyword(title: str | None, keywords: list[str] | tuple[str, ...]) -> bool:
    """True when ``title`` contains at least one keyword (never when none are configured)."""
    lower_pattern, exact_pattern = _compile(tuple(keywords))
    title = title or ""
    if lower_pattern and any(lower_pattern.search(v) for v in _variants(title.lower())):
        return True
    return bool(exact_pattern) and any(exact_pattern.search(v) for v in _variants(title))


def title_matches_keywords(title: str | None, keywords: list[str] | tuple[str, ...]) -> bool:
    """True when ``title`` contains at least one keyword, or no keywords are configured."""
    return _compile(tuple(keywords)) == (None, None) or title_has_keyword(title, keywords)


def title_passes(
    title: str | None,
    keywords: list[str] | tuple[str, ...],
    exclude_keywords: list[str] | tuple[str, ...] = (),
) -> bool:
    """``title_keywords`` allow-list passed and no ``title_exclude_keywords`` word in it."""
    return title_matches_keywords(title, keywords) and not title_has_keyword(title, exclude_keywords)
