from __future__ import annotations

from pathlib import Path

from jobEngine.config.sources import JobSourceConfig
from jobEngine.sources.generic.adapter import extract_jobs_from_current_page

_FIXTURE = Path(__file__).parent / "fixtures" / "generic" / "listing.html"


def _cfg() -> JobSourceConfig:
    return JobSourceConfig(
        id="generic_test",
        name="Generic Test",
        base_url="https://example.com",
        search_urls=["https://example.com/careers"],
        adapter="generic",
        selectors={
            "job_card": ".job-card",
            "title": ".job-title",
            "company": ".company",
            "location": ".location",
            "url": "a",
            "description": ".description",
        },
    )


async def test_extracts_valid_cards_and_skips_titleless_card(browser_page):
    await browser_page.set_content(_FIXTURE.read_text(encoding="utf-8"))
    jobs = await extract_jobs_from_current_page(browser_page, _cfg(), page_url="https://example.com/careers")

    assert len(jobs) == 2  # the third card has no title and must be skipped, not crash
    titles = {j.title for j in jobs}
    assert titles == {"Senior Full Stack AI Engineer", "Backend Engineer"}


async def test_resolves_relative_application_url(browser_page):
    await browser_page.set_content(_FIXTURE.read_text(encoding="utf-8"))
    jobs = await extract_jobs_from_current_page(browser_page, _cfg(), page_url="https://example.com/careers")

    ai_job = next(j for j in jobs if "AI Engineer" in j.title)
    assert ai_job.application_url == "https://example.com/careers/senior-full-stack-ai-engineer"
    assert ai_job.company == "Example Co"
    assert ai_job.location == "Remote"
    assert "LangGraph" in ai_job.description


async def test_respects_limit(browser_page):
    await browser_page.set_content(_FIXTURE.read_text(encoding="utf-8"))
    jobs = await extract_jobs_from_current_page(browser_page, _cfg(), page_url="https://example.com/careers", limit=1)
    assert len(jobs) == 1
