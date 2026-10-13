"""End-to-end graph tests: fan-out/fan-in, source failure isolation,
rule rejection, and final result shape — all without any network access
(fake in-memory adapters) or LLM calls (disabled via conftest).
"""

from __future__ import annotations

from jobEngine.config.sources import JobSourceConfig
from jobEngine.graph.builder import build_graph
from jobEngine.graph.state import initial_state
from jobEngine.models.errors import SourceError, SourceErrorStatus
from jobEngine.models.job import RawJob
from jobEngine.sources.base import JobSourceAdapter, SourceDiscoveryResult, SourceExecutionContext
from jobEngine.sources.registry import register_adapter

_GOOD_JOB = RawJob(
    source="good_source",
    title="Senior Full Stack AI Engineer",
    company="Acme AI",
    location="Remote",
    description=(
        "Build agentic systems using LangGraph, RAG, and Next.js. "
        "Must have Python and LangGraph."
    ),
    requirements=["Must have Python and LangGraph"],
    skills=["Python", "LangGraph", "RAG", "Next.js"],
    application_url="https://good.example.com/jobs/1",
    source_url="https://good.example.com/careers",
)


@register_adapter("_test_good_adapter")
class _GoodAdapter(JobSourceAdapter):
    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        return SourceDiscoveryResult(
            source_id=context.config.id, status="success", jobs=[_GOOD_JOB], pages_processed=1, duration_ms=1
        )


@register_adapter("_test_failing_adapter")
class _FailingAdapter(JobSourceAdapter):
    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        return SourceDiscoveryResult(
            source_id=context.config.id,
            status="failed",
            errors=[
                SourceError(
                    source_id=context.config.id,
                    status=SourceErrorStatus.ANTI_BOT_BLOCKED,
                    message="simulated anti-bot block",
                    retryable=False,
                )
            ],
        )


def _source_cfg(id_: str, adapter: str) -> JobSourceConfig:
    return JobSourceConfig(
        id=id_,
        name=id_,
        base_url=f"https://{id_}.example.com",
        search_urls=[f"https://{id_}.example.com/careers"],
        adapter=adapter,
        max_jobs=10,
        max_pages=1,
    )


async def _run(configs, *, request=None):
    graph = build_graph(configs)
    state = initial_state(run_id="test-run", request=request or {"max_results": 10, "minimum_match_score": 10})
    state["browser_manager"] = None
    return await graph.ainvoke(state)


async def test_one_failed_source_does_not_fail_the_whole_run():
    configs = [
        _source_cfg("good_source", "_test_good_adapter"),
        _source_cfg("bad_source", "_test_failing_adapter"),
    ]
    final_state = await _run(configs)
    result = final_state["final_result"]

    assert result["status"] in ("completed", "partial")
    assert result["summary"]["sources_processed"] == 2
    assert result["source_errors"]  # the failing source's error is recorded...
    assert any(e["source_id"] == "bad_source" for e in result["source_errors"])
    # ...but the good source's job still made it all the way to the recommendations
    assert any(j["job"]["source"] == "good_source" for j in result["jobs"])


async def test_final_result_has_expected_shape():
    configs = [_source_cfg("good_source", "_test_good_adapter")]
    final_state = await _run(configs)
    result = final_state["final_result"]

    assert set(result.keys()) >= {"run_id", "status", "summary", "jobs", "source_errors", "metrics"}
    summary = result["summary"]
    assert set(summary.keys()) >= {
        "sources_processed", "jobs_discovered", "jobs_validated",
        "duplicates_removed", "jobs_matched", "jobs_recommended",
    }
    for job in result["jobs"]:
        assert 0 <= job["match_score"] <= 100
        assert job["job"]["application_url"]
        assert job["job"]["title"]


async def test_no_enabled_sources_still_completes_gracefully():
    final_state = await _run([])
    result = final_state["final_result"]
    assert result["summary"]["sources_processed"] == 0
    assert result["jobs"] == []


@register_adapter("_test_many_adapter")
class _ManyJobsAdapter(JobSourceAdapter):
    """Returns up to ``config.max_jobs`` distinct copies of the good job."""

    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        jobs = [
            _GOOD_JOB.model_copy(
                update={
                    "source": context.config.id,
                    "title": f"Senior Full Stack AI Engineer {i}",
                    "application_url": f"https://{context.config.id}.example.com/jobs/{i}",
                    "source_url": f"https://{context.config.id}.example.com/careers",
                }
            )
            for i in range(context.config.max_jobs)
        ]
        return SourceDiscoveryResult(
            source_id=context.config.id, status="success", jobs=jobs, pages_processed=1, duration_ms=1
        )


async def test_max_jobs_is_per_search_url_and_combined():
    configs = [_source_cfg("many_a", "_test_many_adapter"), _source_cfg("many_b", "_test_many_adapter")]
    result = (await _run(configs, request={"max_jobs": 3, "max_results": 500, "minimum_match_score": 0}))["final_result"]

    # _source_cfg has one search URL per source -> 3 + 3, all combined and processed.
    assert result["summary"]["jobs_discovered"] == 6
    assert result["metrics"]["jobs_processed"] == 6


async def test_no_max_jobs_processes_everything():
    configs = [_source_cfg("many_a", "_test_many_adapter")]
    result = (await _run(configs, request={"max_results": 500, "minimum_match_score": 0}))["final_result"]

    assert result["metrics"]["jobs_processed"] == 10  # source's own max_jobs


@register_adapter("_test_url_recording_adapter")
class _UrlRecordingAdapter(JobSourceAdapter):
    seen_urls: list[str] = []

    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        _UrlRecordingAdapter.seen_urls = list(context.config.search_urls)
        return SourceDiscoveryResult(
            source_id=context.config.id, status="success", jobs=[_GOOD_JOB], pages_processed=1, duration_ms=1
        )


async def test_search_keywords_only_rewrite_the_search_url():
    cfg = JobSourceConfig(
        id="good_source",
        name="good_source",
        base_url="https://good_source.example.com",
        search_urls=["https://good_source.example.com/jobs?keywords=React&start=0"],
        adapter="_test_url_recording_adapter",
        max_jobs=10,
        max_pages=1,
    )
    # Keywords that the job text doesn't contain: the job must still pass (no keyword filter).
    result = (await _run([cfg], request={"search_keywords": "Rust AND Go", "minimum_match_score": 0}))["final_result"]

    assert _UrlRecordingAdapter.seen_urls == ["https://good_source.example.com/jobs?keywords=Rust%20AND%20Go&start=0"]
    assert result["summary"]["jobs_validated"] == 1


async def test_remote_only_filters_server_side():
    configs = [_source_cfg("good_source", "_test_good_adapter")]
    result = (await _run(configs, request={"remote_only": True, "minimum_match_score": 0}))["final_result"]

    assert result["summary"]["jobs_validated"] == 1  # _GOOD_JOB is "Remote"


@register_adapter("_test_per_url_adapter")
class _PerUrlAdapter(JobSourceAdapter):
    """Each search URL offers 20 jobs; honours ``max_jobs_per_url`` like the real adapters."""

    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        cfg = context.config
        jobs = []
        for u, _url in enumerate(cfg.search_urls):
            per_url = min(20, cfg.max_jobs_per_url or 20)
            jobs += [
                _GOOD_JOB.model_copy(
                    update={
                        "source": cfg.id,
                        "title": f"Senior Full Stack AI Engineer {u}-{i}",
                        "application_url": f"https://{cfg.id}.example.com/jobs/{u}-{i}",
                        "source_url": f"https://{cfg.id}.example.com/careers",
                    }
                )
                for i in range(per_url)
            ]
        return SourceDiscoveryResult(
            source_id=cfg.id, status="success", jobs=jobs[: cfg.max_jobs], pages_processed=1, duration_ms=1
        )


async def test_max_jobs_from_each_of_two_search_urls():
    cfg = JobSourceConfig(
        id="two_urls",
        name="two_urls",
        base_url="https://two_urls.example.com",
        search_urls=["https://two_urls.example.com/jobs?keywords=a", "https://two_urls.example.com/jobs?keywords=b&geoId=1"],
        adapter="_test_per_url_adapter",
        max_jobs=400,
        max_pages=1,
    )
    result = (await _run([cfg], request={"max_jobs": 5, "max_results": 500, "minimum_match_score": 0}))["final_result"]

    assert result["summary"]["jobs_discovered"] == 10  # 5 + 5
    assert result["metrics"]["jobs_processed"] == 10
    assert len(result["jobs"]) == 10  # all shown and saved (every fake job scores well)


async def test_saved_jobs_reuse_stored_analysis_without_rescoring(monkeypatch):
    from jobEngine.graph.nodes import matching
    from jobEngine.persistence.saved_jobs import saved_job_key

    configs = [_source_cfg("many_a", "_test_many_adapter")]
    first = (await _run(configs, request={"max_jobs": 2, "max_results": 500, "minimum_match_score": 0}))["final_result"]
    cached = dict(first["jobs"][0])
    cached.update(
        saved_key=saved_job_key(cached),
        match_score=88.0,
        recommendation="strong_match",
        explanation="stored analysis",
        llm_analysis={
            "score": 88,
            "recommendation": "strong_match", "explanation": "stored analysis",
        },
    )
    monkeypatch.setattr(matching, "list_saved_jobs", lambda: [cached])

    rule_calls = []
    real_evaluate = matching.evaluate_rules
    monkeypatch.setattr(matching, "evaluate_rules", lambda job, *a, **kw: rule_calls.append(job.title) or real_evaluate(job, *a, **kw))

    second = (await _run(configs, request={"max_jobs": 2, "max_results": 500, "minimum_match_score": 0}))["final_result"]
    reused = next(j for j in second["jobs"] if saved_job_key(j) == cached["saved_key"])

    assert reused["match_score"] == 88.0 and reused["explanation"] == "stored analysis"
    assert second["metrics"]["jobs_reused_from_saved"] == 1
    assert rule_calls == [j["job"]["title"] for j in second["jobs"] if saved_job_key(j) != cached["saved_key"]]

    # The current request's minimum still applies to a reused job.
    high_bar = (await _run(configs, request={"max_jobs": 2, "max_results": 500, "minimum_match_score": 95}))["final_result"]
    assert all(saved_job_key(j) != cached["saved_key"] for j in high_bar["jobs"])


@register_adapter("_test_known_recording_adapter")
class _KnownRecordingAdapter(JobSourceAdapter):
    seen_known: dict[str, str] = {}

    async def discover_jobs(self, context: SourceExecutionContext) -> SourceDiscoveryResult:
        _KnownRecordingAdapter.seen_known = dict(context.known_descriptions)
        return SourceDiscoveryResult(source_id=context.config.id, status="success", jobs=[], pages_processed=1, duration_ms=1)


async def test_saved_llm_analyzed_jobs_are_passed_to_the_adapter(monkeypatch):
    from jobEngine.graph.nodes import source_discovery

    saved = [
        {"job": {"source": "known_src", "source_job_id": "111", "description": "saved text"}, "llm_analysis": {"score": 80}},
        {"job": {"source": "known_src", "source_job_id": "222", "description": "rule-only"}, "llm_analysis": None},
        {"job": {"source": "other_src", "source_job_id": "333", "description": "other"}, "llm_analysis": {"score": 70}},
    ]
    monkeypatch.setattr(source_discovery, "list_saved_jobs", lambda: saved)

    await _run([_source_cfg("known_src", "_test_known_recording_adapter")])

    # Only this source's saved jobs that have an LLM analysis skip being opened again.
    assert _KnownRecordingAdapter.seen_known == {"111": "saved text"}
