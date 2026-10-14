from __future__ import annotations

import asyncio

import pytest

from jobEngine.sources.linkedin.adapter import LinkedInAdapter, _is_search_page


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.linkedin.com/jobs/search/?f_AL=true&start=25", True),
        ("https://www.linkedin.com/jobs/search?keywords=python", True),
        ("https://www.linkedin.com/jobs/search-results/?keywords=python", False),
        ("https://www.linkedin.com/jobs/collections/recommended/", False),
    ],
)
def test_is_search_page(url, expected):
    assert _is_search_page(url) is expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.linkedin.com/jobs/search/?f_AL=true", "search"),
        ("https://www.linkedin.com/jobs/search-results/?keywords=x", "search_results"),
    ],
)
def test_router_delegates_by_url(url, expected):
    adapter = LinkedInAdapter()
    calls = []

    def _fake(name):
        async def _extract_page(page, **kwargs):
            calls.append((name, kwargs["source_url"]))
            return []
        return _extract_page

    adapter._search._extract_page = _fake("search")
    adapter._search_results._extract_page = _fake("search_results")

    asyncio.run(adapter._extract_page(None, cfg=None, source_url=url, seen_ids=set(), limit=10))
    assert calls == [(expected, url)]
