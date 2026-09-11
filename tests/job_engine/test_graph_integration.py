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
