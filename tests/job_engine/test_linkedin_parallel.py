"""LinkedInBaseAdapter scrapes results pages in a shared pool of tabs
(LINKEDIN_MAX_TABS): several pages of one search URL at once, and several search
URLs, while keeping cross-URL dedup and the shared signed-in session."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from jobEngine.config.settings import get_settings
from jobEngine.config.sources import JobSourceConfig
from jobEngine.models.job import RawJob
from jobEngine.sources.base import SourceExecutionContext
from jobEngine.sources.linkedin.base import LinkedInBaseAdapter, note_relisted


class _FakePage:
    def __init__(self, ctx):
        self.url = "about:blank"
        self.closed = False
        ctx.pages.append(self)

    async def goto(self, url, **_):
        self.url = url

    async def close(self):
        self.closed = True


class _FakeContext:
    def __init__(self):
        self.pages = []

    async def cookies(self, *_):
        return [{"name": "li_at", "value": "x"}]

    async def new_page(self):
        return _FakePage(self)

    async def storage_state(self):
        return {"cookies": [{"name": "li_at", "value": "x", "domain": ".linkedin.com"}], "origins": []}


class _FakeBrowserManager:
    def __init__(self):
        self.context = _FakeContext()

    @asynccontextmanager
    async def open_page(self, **_):
        yield SimpleNamespace(page=_FakePage(self.context), context=self.context)


class _Adapter(LinkedInBaseAdapter):
    """``listings[search_url][page_index]`` = job ids on that results page."""

    def __init__(self, listings):
        self.listings = listings
        self.active = self.peak = 0
        self.pages_seen = []

    async def _extract_page(self, page, *, cfg, source_url, seen_ids, limit, known=None):
        search_url = seen_ids.search_url_for[source_url]
        page_index = int(parse_qs(urlparse(source_url).query)["start"][0]) // 25
        self.pages_seen.append((search_url, page_index))
        self.active += 1
        self.peak = max(self.peak, self.active)
        jobs = []
        for job_id in self.listings[search_url].get(page_index, []):
            if job_id in seen_ids:
                note_relisted(seen_ids, job_id, source_url)
                continue
            seen_ids.add(job_id)
            await asyncio.sleep(0.02)  # the detail-pane click
            jobs.append(RawJob(source=cfg.id, source_job_id=job_id, title="Engineer", company="Acme",
                               application_url=f"https://www.linkedin.com/jobs/view/{job_id}/", source_url=source_url))
        await asyncio.sleep(0.05)
        self.active -= 1
        return jobs, 0


def _run(monkeypatch, listings, tabs):
    monkeypatch.setenv("LINKEDIN_MAX_TABS", str(tabs))
    cfg = JobSourceConfig(id="linkedin", name="LinkedIn", base_url="https://www.linkedin.com",
                          search_urls=list(listings), adapter="linkedin", rate_limit_delay_seconds=0,
                          allowed_domains=["linkedin.com"], max_jobs=1000, max_pages=10)
    browser = _FakeBrowserManager()
    adapter = _Adapter(listings)
    ctx = SourceExecutionContext(run_id="r", config=cfg, browser_manager=browser, settings=get_settings())
    return asyncio.run(adapter.discover_jobs(ctx)), adapter, browser


def _full_page(prefix):
    return [f"{prefix}{n}" for n in range(25)]


def test_one_search_url_scrapes_two_pages_at_once(monkeypatch):
    url = "https://www.linkedin.com/jobs/search/?geoId=92000000"
    # Pages 0-2 are full, page 3 is the short last page.
    listings = {url: {0: _full_page("p0-"), 1: _full_page("p1-"), 2: _full_page("p2-"), 3: ["last"]}}
    result, adapter, browser = _run(monkeypatch, listings, tabs=2)

    assert not result.errors
    assert adapter.peak == 2
    assert len(result.jobs) == 25 * 3 + 1
    assert sorted(i for _, i in adapter.pages_seen) == [0, 1, 2, 3]  # stopped after the short page
    assert len(browser.context.pages) == 2 and browser.context.pages[1].closed


def test_tabs_are_shared_across_search_urls_and_dedup_holds(monkeypatch):
    urls = [f"https://www.linkedin.com/jobs/search/?geoId={g}" for g in (1, 2, 3)]
    listings = {urls[0]: {0: ["a1", "shared"]}, urls[1]: {0: ["shared", "b1"]}, urls[2]: {0: ["c1", "shared"]}}
    result, adapter, browser = _run(monkeypatch, listings, tabs=2)

    assert not result.errors
    assert adapter.peak <= 2
    assert len(browser.context.pages) == 2  # never more than LINKEDIN_MAX_TABS
    assert all(p.closed for p in browser.context.pages[1:])
    assert sorted(j.source_job_id for j in result.jobs) == ["a1", "b1", "c1", "shared"]  # "shared" scraped once
    shared = next(j for j in result.jobs if j.source_job_id == "shared")
    assert sorted(shared.extra["search_urls"]) == sorted(urls)  # but every URL that listed it is kept


def test_one_tab_scrapes_one_page_at_a_time(monkeypatch):
    url = "https://www.linkedin.com/jobs/search/?geoId=92000000"
    listings = {url: {0: _full_page("p0-"), 1: ["last"]}}
    result, adapter, browser = _run(monkeypatch, listings, tabs=1)
    assert adapter.peak == 1
    assert len(browser.context.pages) == 1
    assert len(result.jobs) == 26
