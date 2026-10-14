"""RankJobs node (section 48): deterministic ranking of non-rejected matches.

Sort order: match_score desc -> fewer missing mandatory
skills -> more recent posting (unknown dates sort last) -> stable original
order (acts as a source-priority tiebreak, since sources are processed in
``config/job_sources.yaml`` order upstream).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _sort_key(indexed_job: tuple[int, dict[str, Any]]) -> tuple:
    idx, job = indexed_job
    posted = (job.get("job") or {}).get("posted_date")
    return (
        -float(job.get("match_score", 0.0)),
        len(job.get("missing_required_skills") or []),
        0 if posted else 1,
        idx,
    )


async def ranking_node(state: dict[str, Any]) -> dict[str, Any]:
    request = state.get("request") or {}
    matched_jobs = state.get("matched_jobs") or []

    # ``rejected`` already reflects the effective (request-overridden) minimum
    # match score — computed once in matching.py -> build_matched_job.
    eligible = [j for j in matched_jobs if not j.get("rejected")]
    indexed = list(enumerate(eligible))
    indexed.sort(key=_sort_key)
    ranked = [job for _idx, job in indexed]

    max_results = request.get("max_results")
    if isinstance(max_results, int) and max_results > 0:
        ranked = ranked[:max_results]

    metrics = dict(state.get("metrics") or {})
    metrics["jobs_recommended"] = len(ranked)
    metrics["match_distribution"] = {}
    for job in matched_jobs:
        rec = job.get("recommendation", "unknown")
        metrics["match_distribution"][rec] = metrics["match_distribution"].get(rec, 0) + 1

    logger.info("ranking_node: %s recommended job(s) after ranking/cap", len(ranked))
    return {"ranked_jobs": ranked, "metrics": metrics}
