"""``sort_newest_first``: sortBy=DD on the search URL and the early pagination stop."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from jobEngine.models.job import RawJob
from jobEngine.sources.base import TimedDiscovery
from jobEngine.sources.linkedin.base import SeenJobIds, page_has_nothing_new, with_newest_first
from jobEngine.sources.linkedin.search_results_adapter import LinkedInSearchResultsAdapter

SEARCH = "https://www.linkedin.com/jobs/search-results/?keywords=AI%20Engineer&geoId=102713980&f_TPR=r86400"


def _job(job_id: str, posted: str | None = "1 hour ago") -> RawJob:
    return RawJob(
        source="linkedin", source_job_id=job_id, title="Engineer", company="Acme", description="d",
        posted_date_raw=posted,
        application_url=f"https://www.linkedin.com/jobs/view/{job_id}/", source_url=SEARCH,
    )


def test_with_newest_first_adds_or_replaces_sort():
    q = parse_qs(urlparse(with_newest_first(SEARCH)).query)
    assert q["sortBy"] == ["DD"] and q["f_TPR"] == ["r86400"] and q["keywords"] == ["AI Engineer"]
    assert parse_qs(urlparse(with_newest_first(SEARCH + "&sortBy=R")).query)["sortBy"] == ["DD"]


def test_page_has_nothing_new():
    assert page_has_nothing_new([], 8)  # every card saved / rejected / excluded / read already
    assert page_has_nothing_new([_job("1", "9 hours ago"), _job("2", "2 days ago")], 8)
    assert not page_has_nothing_new([_job("1", "9 hours ago"), _job("2", "3 hours ago")], 8)
    assert not page_has_nothing_new([_job("1", None)], 8)  # "Viewed" card: age unknown


class _Pages(LinkedInSearchResultsAdapter):
    """Pages served from a list instead of a browser: (found, filtered) per page index."""

    def __init__(self, pages, total_pages=None):
        super().__init__()
        self.pages = pages
        self.total_pages = total_pages
        self.loaded: list[int] = []
        self.batches: list[list[int]] = []
        self._batch: list[int] = []

    async def _scrape_results_page(self, tabs, browser_context, search_url, page_index, **_kw):
        self.loaded.append(page_index)
        if page_index <= 1 or len(self._batch) == tabs.size:
            self._batch = []
            self.batches.append(self._batch)
        self._batch.append(page_index)
        found, filtered = self.pages[page_index] if page_index < len(self.pages) else ([], 0)
        return found, filtered, f"{search_url}&start={page_index * 25}", self.total_pages if page_index == 0 else None


def _walk(pages, *, newest_first: bool, seen_before: set[str] = frozenset(), tabs: int = 1, total_pages=None,
          adapter=None):
    adapter = adapter or _Pages(pages, total_pages)
    cfg = SimpleNamespace(
        id="linkedin", max_pages=20, max_jobs=400, max_jobs_per_url=None,
        rate_limit_delay_seconds=0, sort_newest_first=newest_first, max_posted_age_hours=8,
    )
    context = SimpleNamespace(seen_before_ids=set(seen_before), known_descriptions={})
    timer = TimedDiscovery("linkedin")
    asyncio.run(adapter._walk_search_url(
        SimpleNamespace(size=tabs), None, SEARCH, cfg=cfg, context=context,
        timer=timer, seen_ids=SeenJobIds(), session=None,
    ))
    return adapter.loaded, [j.source_job_id for j in timer.jobs]


def _page(ids, filtered=0):
    return [_job(i) for i in ids], filtered


PAGES = [
    _page(["n1", "n2"], filtered=21),        # 2 new; 21 saved / rejected / excluded
    _page([], filtered=25),                  # nothing new
    _page(["n3"], filtered=24),              # never reached with newest-first
]


def test_newest_first_stops_after_a_page_with_nothing_new():
    loaded, ids = _walk(PAGES, newest_first=True)
    assert loaded == [0, 1]
    assert ids == ["n1", "n2"]


def test_without_newest_first_every_page_is_walked():
    loaded, ids = _walk(PAGES, newest_first=False)
    assert loaded == [0, 1, 2, 3]  # page 3 is empty: past the last results page
    assert ids[-1] == "n3"


def test_a_page_of_only_too_old_cards_stops_too():
    old = [_job(f"o{i}", "9 hours ago") for i in range(25)]
    loaded, ids = _walk([_page(["n1"], filtered=24), (old, 0), _page(["n2"], 24)], newest_first=True)
    assert loaded == [0, 1]
    assert "o0" in ids  # still collected: step 4 rejects them once every page is read


def _full(prefix):
    return _page([f"{prefix}{n}" for n in range(25)])


def test_page_one_alone_then_batches_of_three_up_to_the_page_count():
    pages = [_full(f"p{i}-") for i in range(6)] + [_page(["last"]), _full("never-")]
    adapter = _Pages(pages, total_pages=7)
    loaded, ids = _walk(pages, newest_first=False, tabs=3, adapter=adapter)
    # Page 1 says there are 7 pages: then 2-4 and 5-7 in parallel, and nothing past 7.
    assert adapter.batches == [[0], [1, 2, 3], [4, 5, 6]]
    assert len(ids) == 6 * 25 + 1


def test_a_full_last_counted_page_keeps_reading():
    """Pagination may show "1 2 3 … 4" of a longer search: a full last page goes on."""
    pages = [_full(f"p{i}-") for i in range(4)] + [_page(["last"])]
    adapter = _Pages(pages, total_pages=4)
    loaded, ids = _walk(pages, newest_first=False, tabs=3, adapter=adapter)
    assert adapter.batches == [[0], [1, 2, 3], [4, 5, 6]]
    assert ids[-1] == "last"


def test_unknown_page_count_reads_until_a_short_page():
    pages = [_full("a"), _full("b"), _full("c"), _page(["last"])]
    adapter = _Pages(pages, total_pages=None)
    loaded, ids = _walk(pages, newest_first=False, tabs=3, adapter=adapter)
    assert adapter.batches == [[0], [1, 2, 3]]
    assert ids[-1] == "last"
