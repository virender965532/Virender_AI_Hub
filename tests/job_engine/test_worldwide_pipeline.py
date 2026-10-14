"""Worldwide-remote feature wiring: flag gating in normalization, cross-URL
dedup in the LinkedIn adapter, LLM grounding, and saved-jobs merging."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from jobEngine.config.settings import get_settings
from jobEngine.graph.nodes.normalization import normalization_node
from jobEngine.matching.worldwide_llm import WorldwideReview, apply_review, verified_quotes
from jobEngine.matching.worldwide_remote import classify_worldwide
from jobEngine.models.job import RawJob, WorldwideCategory
from jobEngine.persistence import saved_jobs
from jobEngine.sources.linkedin.base import SeenJobIds, merge_search_urls, parse_job_insights
from jobEngine.sources.linkedin.search_results_adapter import LinkedInSearchResultsAdapter

URL_A = "https://www.linkedin.com/jobs/search-results/?f_WT=2&geoId=92000000"
URL_B = "https://www.linkedin.com/jobs/search-results/?f_TPR=r43200&geoId=106204383"
_DESCRIPTION = (
    "We are a fully remote company and hire independent contractors worldwide. "
    "This is a contract role; you can work from any country. "
    "You will build Node.js, React and TypeScript services on AWS with a small product team."
)


@pytest.fixture()
def worldwide_on(monkeypatch):
    monkeypatch.setenv("JOB_WORLDWIDE_REMOTE_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _raw(job_id="111", **extra) -> dict:
    return RawJob(
        source="linkedin",
        source_job_id=job_id,
        title="Senior Node.js Engineer",
        company="Acme",
        location="United States (Remote)",
        description=_DESCRIPTION,
        application_url=f"https://www.linkedin.com/jobs/view/{job_id}/",
        source_url=URL_A,
        extra=extra,
    ).model_dump(mode="json")


def _state(*raws):
    return {"run_id": "r1", "discovered_jobs": list(raws), "source_configs": [{"id": "linkedin", "allowed_domains": ["linkedin.com"]}]}


# --- flag gating ---------------------------------------------------------


def test_flag_off_leaves_jobs_unclassified(monkeypatch):
    monkeypatch.setenv("JOB_WORLDWIDE_REMOTE_ENABLED", "false")  # .env may turn it on
    get_settings.cache_clear()
    out = asyncio.run(normalization_node(_state(_raw())))
    get_settings.cache_clear()
    assert out["normalized_jobs"][0]["worldwide"] is None


def test_flag_on_classifies_with_evidence(worldwide_on):
    out = asyncio.run(normalization_node(_state(_raw(workplace_type="Remote"))))
    worldwide = out["normalized_jobs"][0]["worldwide"]
    assert worldwide["category"] == "worldwide_confirmed"
    assert worldwide["india_eligible"] is True
    assert any(e["signal"] == "positive" and e["found_in"] == "description" for e in worldwide["evidence"])


# --- dedup across search URLs ---------------------------------------------


class _FakePage:
    def __init__(self, cards):
        self._cards = cards

    async def wait_for_selector(self, *_a, **_k):
        return None

    async def eval_on_selector_all(self, *_a, **_k):
        return self._cards


def _card(job_id, title="Senior Node.js Engineer"):
    return {"job_id": job_id, "title": title, "company": "Acme", "location": "Worldwide", "posted": "1 hour ago"}


def test_duplicate_job_ids_from_different_search_urls_are_scraped_once():
    adapter = LinkedInSearchResultsAdapter()
    cfg = SimpleNamespace(id="linkedin", title_keywords=[], title_exclude_keywords=[], base_url="https://www.linkedin.com")
    known = {i: _DESCRIPTION for i in ("111", "222", "333")}  # already saved: the detail pane is not opened
    seen = SeenJobIds()
    jobs = []
    for url, cards in ((URL_A, [_card("111"), _card("222")]), (URL_B, [_card("111"), _card("333")])):
        seen.search_url_for[url] = url  # what LinkedInBaseAdapter._walk_search_url registers
        found, _ = asyncio.run(
            adapter._extract_page(_FakePage(cards), cfg=cfg, source_url=url, seen_ids=seen, limit=25, known=known)
        )
        for job in found:  # what LinkedInBaseAdapter.discover_jobs does per search URL
            job.extra.setdefault("search_urls", [url])
        jobs.extend(found)
    merge_search_urls(jobs, seen.relisted)

    assert [j.source_job_id for j in jobs] == ["111", "222", "333"]
    by_id = {j.source_job_id: j for j in jobs}
    assert by_id["111"].extra["search_urls"] == [URL_A, URL_B]
    assert by_id["222"].extra["search_urls"] == [URL_A]
    assert by_id["333"].extra["search_urls"] == [URL_B]


def test_search_urls_reach_the_normalized_job(worldwide_on):
    out = asyncio.run(normalization_node(_state(_raw(search_urls=[URL_A, URL_B]))))
    assert out["normalized_jobs"][0]["search_urls"] == [URL_A, URL_B]


def test_saved_store_keeps_every_search_url():
    def matched(urls):
        return {"job": {"job_id": "h", "source": "linkedin", "source_job_id": "111", "title": "T",
                        "company": "C", "description": "d", "search_urls": urls}, "match_score": 80}

    saved_jobs.save_jobs_from_run({"run_id": "r1", "jobs": [matched([URL_A])]})
    saved_jobs.save_jobs_from_run({"run_id": "r2", "jobs": [matched([URL_B])]})
    (job,) = saved_jobs.list_saved_jobs()
    assert job["job"]["search_urls"] == [URL_B, URL_A]


# --- LinkedIn chips ----------------------------------------------------------


def test_parse_job_insights_reads_whole_lines_only():
    assert parse_job_insights("Remote\nMatches your job preferences\nFull-time\n$120K/yr") == ("Remote", "Full-time")
    assert parse_job_insights("Senior Remote Contract Manager\nHybrid\nContract") == ("Hybrid", "Contract")
    assert parse_job_insights("") == (None, None)


# --- LLM grounding -----------------------------------------------------------


def _needs_verification():
    return classify_worldwide(
        title="Senior Engineer", location="United States (Remote)",
        description="This is a remote position. " + "We build developer tools with TypeScript and AWS. " * 5,
    )


def test_llm_upgrade_requires_a_verbatim_quote():
    current = _needs_verification()
    assert current.category == WorldwideCategory.NEEDS_VERIFICATION
    text = "We welcome applicants from India and Brazil. " + "Other details. " * 20

    invented = WorldwideReview(category="india_eligible", evidence_quotes=["India-based people are welcome"], reasoning="r")
    assert apply_review(current, invented, text).category == WorldwideCategory.NEEDS_VERIFICATION

    grounded = WorldwideReview(category="india_eligible", evidence_quotes=["we welcome  applicants from India"], reasoning="r")
    result = apply_review(current, grounded, text)
    assert result.category == WorldwideCategory.INDIA_ELIGIBLE
    assert result.method == "llm" and result.india_eligible
    assert result.evidence[0].rule == "llm_quote"


def test_llm_unclear_verdict_keeps_rules_result():
    current = _needs_verification()
    review = WorldwideReview(category="needs_verification", evidence_quotes=[], reasoning="no policy stated")
    result = apply_review(current, review, "text")
    assert result.category == WorldwideCategory.NEEDS_VERIFICATION
    assert result.method == "rules"
    assert "no policy stated" in result.reason


def test_verified_quotes_ignores_short_fragments():
    assert verified_quotes(["India", "Remote anywhere in the world"], "Remote anywhere in the world, India too") == [
        "Remote anywhere in the world"
    ]
