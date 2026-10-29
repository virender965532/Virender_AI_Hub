"""LLM match analyses by job text + profile + matcher version (MongoDB ``job_analysis_cache``).

The score is computed in code from an analysis (matching/fit_score.py), so the
same analysis always gives the same score. This store makes a job that is seen
again — even after it was removed from the saved jobs — reuse its analysis
instead of asking the LLM again, whose labels can differ slightly between calls.

The key is a fingerprint of the job's ``raw_text_hash``, the candidate profile
and the matcher version (prompts, model, output schema), so a change to any of
them gets a fresh analysis. Documents expire ``ANALYSIS_CACHE_MAX_AGE_HOURS``
(default 720 = 30 days) after they were last written, via a TTL index.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from pymongo import ReplaceOne

from ..matching.fit_score import fingerprint
from .mongo import ANALYSIS_CACHE_COLLECTION, get_collection, to_bson_safe

logger = logging.getLogger(__name__)

_DEFAULT_MAX_AGE_HOURS = 720.0


def _cache():
    return get_collection(ANALYSIS_CACHE_COLLECTION)


def _max_age() -> timedelta:
    try:
        hours = float((os.getenv("ANALYSIS_CACHE_MAX_AGE_HOURS") or "").strip() or _DEFAULT_MAX_AGE_HOURS)
    except ValueError:
        hours = _DEFAULT_MAX_AGE_HOURS
    return timedelta(hours=max(1.0, hours))


def analysis_key(raw_text_hash: str, *, profile_hash: str, matcher_version: str) -> str:
    return fingerprint(raw_text_hash or "", profile_hash, matcher_version)


def cached_analyses(keys: Iterable[str], now: datetime | None = None) -> dict[str, dict[str, Any]]:
    """Stored analyses (``JobMatchAnalysis`` dicts) for these keys."""
    wanted = list({k for k in keys if k})
    if not wanted:
        return {}
    query = {"_id": {"$in": wanted}, "expires_at": {"$gt": now or datetime.now(timezone.utc)}}
    return {doc["_id"]: doc["llm_analysis"] for doc in _cache().find(query, {"llm_analysis": 1})}


def store_analyses(analyses: dict[str, dict[str, Any]], now: datetime | None = None) -> int:
    """Store ``key -> JobMatchAnalysis dict``; returns how many were written."""
    now = now or datetime.now(timezone.utc)
    expires_at = now + _max_age()
    ops = [
        ReplaceOne(
            {"_id": key},
            {"_id": key, "llm_analysis": to_bson_safe(analysis), "stored_at": now.isoformat(), "expires_at": expires_at},
            upsert=True,
        )
        for key, analysis in analyses.items()
        if key
    ]
    if ops:
        _cache().bulk_write(ops, ordered=False)
    return len(ops)
