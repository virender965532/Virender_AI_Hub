"""LinkedIn steps 4-5: once every results page is read, too-old jobs are rejected
(and stored, so the next run skips them) and the rest are fetched all together."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from jobEngine.models.job import RawJob
from jobEngine.sources.linkedin import base as linkedin_base
from jobEngine.sources.linkedin import guest_posting
from jobEngine.sources.linkedin.guest_posting import fetch_guest_postings, parse_guest_posting
from jobEngine.sources.linkedin.search_results_adapter import LinkedInSearchResultsAdapter, _workplace_from_location

URL = "https://www.linkedin.com/jobs/search-results/?geoId=102713980"
LONG = "Build LLM services in Python and TypeScript. " * 5
CFG = SimpleNamespace(id="linkedin", max_posted_age_hours=8, base_url="https://www.linkedin.com")


def _job(job_id, posted="1 hour ago", description=""):
    return RawJob(source="linkedin", source_job_id=job_id, title="AI Engineer", company="Acme",
                  description=description, posted_date_raw=posted,
                  application_url=f"https://www.linkedin.com/jobs/view/{job_id}/", source_url=URL)


class _Page:
    def __init__(self, evaluated=None):
        self.evaluated = evaluated

    async def evaluate(self, script, *_a):
        return "UA" if script == "navigator.userAgent" else self.evaluated


@pytest.fixture()
def stores(monkeypatch):
    """Records what goes to the rejected-jobs store and which ids are fetched."""
    out = SimpleNamespace(rejected=[], fetched=None, details={})

    def fake_save(matched, *, run_id, profile_hash):
        out.rejected.extend(m["job"]["source_job_id"] for m in matched)
        assert all(m["rejected"] and "8h" in m["rejection_reason"] for m in matched)
        return len(matched)

    async def fake_fetch(ids, *, user_agent):
        out.fetched = list(ids)
        return {i: out.details[i] for i in ids if i in out.details}

    monkeypatch.setattr(linkedin_base, "save_rejected_jobs", fake_save)
    monkeypatch.setattr(linkedin_base, "fetch_guest_postings", fake_fetch)
    return out


def test_too_old_jobs_are_rejected_and_stored(stores):
    adapter = LinkedInSearchResultsAdapter()
    jobs = [_job("1", "30 minutes ago"), _job("2", "8 hours ago"), _job("3", "9 hours ago"),
            _job("4", "Reposted 1 day ago"), _job("5", None)]
    kept = asyncio.run(adapter._reject_too_old(jobs, cfg=CFG, run_id="r1"))
    assert [j.source_job_id for j in kept] == ["1", "2", "5"]  # no age ("Viewed") is kept
    assert stores.rejected == ["3", "4"]


def test_details_are_fetched_together_and_a_viewed_card_uses_the_fetched_age(stores):
    adapter = LinkedInSearchResultsAdapter()
    stores.details = {
        "1": {"description": LONG, "posted": "1 hour ago", "employment_type": "Full-time"},
        "2": {"error": "HTTP 429"},
        "3": {"description": LONG, "posted": "3 hours ago"},
        "4": {"description": LONG, "posted": "2 days ago"},
    }
    jobs = [_job("1"), _job("2"), _job("3", None), _job("4", None), _job("5", description="already read")]
    kept = asyncio.run(adapter._fetch_all_details(_Page(), jobs, cfg=CFG, run_id="r1"))

    assert stores.fetched == ["1", "2", "3", "4"]  # one call for every job of every page
    by_id = {j.source_job_id: j for j in kept}
    assert set(by_id) == {"1", "2", "3", "5"}
    assert by_id["1"].description.startswith("Build LLM")
    assert by_id["2"].description == ""  # fetch failed: validation drops it, next run retries
    assert by_id["3"].posted_date_raw == "3 hours ago"
    assert stores.rejected == ["4"]  # "Viewed" card turned out 2 days old


@pytest.mark.parametrize("evaluated, expected", [
    ({"pages": 14, "results": None}, 14),
    ({"pages": None, "results": 330}, 14),       # exact "330 results" -> ceil(330 / 25)
    ({"pages": 10, "results": 330}, 14),         # truncated pagination: the larger wins
    ({"pages": None, "results": None}, None),    # "99+ results", no pagination: unknown
])
def test_page_count(evaluated, expected):
    assert asyncio.run(LinkedInSearchResultsAdapter()._page_count(_Page(evaluated))) == expected


def test_workplace_from_card_location():
    assert _workplace_from_location("Bengaluru, Karnataka, India (Hybrid)") == "Hybrid"
    assert _workplace_from_location("India (Remote)") == "Remote"
    assert _workplace_from_location("Noida, India") is None


_GUEST_HTML = """
<section class="top-card-layout"><span class="posted-time-ago__text topcard__flavor--metadata">
  3 hours ago </span></section>
<div class="description__text description__text--rich"><section class="show-more-less-html">
<div class="show-more-less-html__markup relative overflow-hidden">
  <strong>About us</strong><br>We build AI &amp; data products.<br><br>
  <ul><li>Python</li><li>TypeScript</li></ul><p>Remote in India</p>
</div></section></div>
<ul class="description__job-criteria-list">
<li class="description__job-criteria-item"><h3 class="description__job-criteria-subheader">Seniority level</h3>
<span class="description__job-criteria-text description__job-criteria-text--criteria">Mid-Senior level</span></li>
<li class="description__job-criteria-item"><h3 class="description__job-criteria-subheader">Employment type</h3>
<span class="description__job-criteria-text description__job-criteria-text--criteria">Full-time</span></li>
</ul>
"""


def test_parse_guest_posting():
    parsed = parse_guest_posting(_GUEST_HTML)
    assert parsed["posted"] == "3 hours ago"
    assert parsed["employment_type"] == "Full-time"
    assert parsed["description"].splitlines()[:3] == ["About us", "We build AI & data products.", ""]
    assert "Python" in parsed["description"] and "Remote in India" in parsed["description"]


def test_fetch_backs_off_and_retries_after_429(monkeypatch):
    monkeypatch.setattr(guest_posting, "_BACKOFF_SECONDS", (0.01, 0.01))
    monkeypatch.setattr(guest_posting, "_MIN_INTERVAL_SECONDS", 0)
    calls: dict[str, int] = {}

    def fake_get(url, _ua):
        job_id = url.rsplit("/", 1)[-1]
        calls[job_id] = calls.get(job_id, 0) + 1
        if job_id == "11111" and calls[job_id] < 3:
            return 429, ""
        if job_id == "33333":
            return 429, ""
        return 200, _GUEST_HTML

    monkeypatch.setattr(guest_posting, "_get", fake_get)
    out = asyncio.run(fetch_guest_postings(["11111", "22222", "33333"], user_agent="UA"))
    assert out["11111"]["posted"] == "3 hours ago" and calls["11111"] == 3
    assert out["22222"]["employment_type"] == "Full-time"
    assert out["33333"] == {"error": "HTTP 429"} and calls["33333"] == 3  # 1 try + 2 backoffs
