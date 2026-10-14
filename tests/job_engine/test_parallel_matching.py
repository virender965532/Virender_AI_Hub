"""LLM matching runs jobs concurrently (JOB_LLM_CONCURRENCY) without mixing up
results or losing metrics."""

from __future__ import annotations

import asyncio
import threading
import time

from jobEngine.config.settings import get_settings
from jobEngine.graph.nodes import matching
from jobEngine.models.job import Job, JobMatchAnalysis


def _job(i: int) -> dict:
    return Job(
        job_id=f"j{i}", source="test", source_job_id=str(i), title=f"Senior Node.js Engineer {i}", company="Acme",
        description="Build Node.js and React services on AWS with TypeScript.",
        application_url=f"https://example.com/apply/{i}", source_url=f"https://example.com/job/{i}",
        raw_text_hash=f"h{i}",
    ).model_dump(mode="json")


def test_llm_calls_overlap_and_results_stay_with_their_job(monkeypatch, candidate_profile):
    monkeypatch.setenv("JOB_LLM_CONCURRENCY", "4")
    monkeypatch.setenv("JOB_LLM_TOP_K", "-1")
    get_settings.cache_clear()

    active, peak, lock = 0, 0, threading.Lock()

    def fake_analyze(self, *, profile, job, metrics):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.2)
        with lock:
            active -= 1
        metrics.calls += 1
        metrics.total_tokens += 100
        score = float(job.source_job_id) * 10  # ties each result to its job
        return JobMatchAnalysis(
            score=score,
            recommendation="good_match", explanation=f"job {job.source_job_id}",
        )

    monkeypatch.setattr(matching.LLMMatcher, "available", lambda self: True)
    monkeypatch.setattr(matching.LLMMatcher, "_get_structured_llm", lambda self: None)
    monkeypatch.setattr(matching.LLMMatcher, "analyze", fake_analyze)

    state = {"validated_jobs": [_job(i) for i in range(1, 9)], "candidate_profile": candidate_profile.model_dump(mode="json"),
             "request": {"minimum_match_score": 0}}
    t0 = time.perf_counter()
    out = asyncio.run(matching.matching_node(state))
    elapsed = time.perf_counter() - t0
    get_settings.cache_clear()

    assert peak == 4
    assert elapsed < 8 * 0.2 * 0.75  # well under the 1.6s a sequential run takes
    for m in out["matched_jobs"]:
        assert m["llm_analysis"]["explanation"] == f"job {m['job']['source_job_id']}"
    assert out["metrics"]["llm_calls"] == 8
    assert out["metrics"]["llm_total_tokens"] == 800
