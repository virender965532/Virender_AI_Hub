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


def test_router_follows_redirect_to_new_page():
    """LinkedIn redirects the retired /jobs/search/ to /jobs/search-results/: the
    extractor is chosen from the page actually loaded, not the configured URL."""
    from types import SimpleNamespace

    adapter = LinkedInAdapter()
    calls = []

    async def _fake(page, **kwargs):
        calls.append("search_results")
        return []

    async def _classic(page, **kwargs):
        calls.append("search")
        return []

    adapter._search._extract_page = _classic
    adapter._search_results._extract_page = _fake
    page = SimpleNamespace(url="https://www.linkedin.com/jobs/search-results/?keywords=x&start=0")
    asyncio.run(adapter._extract_page(
        page, cfg=None, source_url="https://www.linkedin.com/jobs/search/?keywords=x", seen_ids=set(), limit=10,
    ))
    assert calls == ["search_results"]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://www.linkedin.com/jobs/search/?f_WT=2&geoId=92000000&f_TPR=r3600",
            "https://www.linkedin.com/jobs/search-results/?f_WT=2&geoId=92000000&f_TPR=r3600",
        ),
        ("https://www.linkedin.com/jobs/search?keywords=py", "https://www.linkedin.com/jobs/search-results/?keywords=py"),
        (
            "https://www.linkedin.com/jobs/search-results/?keywords=jobs&f_SAL=f_SA_id_225001%3A272001",
            "https://www.linkedin.com/jobs/search-results/?keywords=jobs&f_SAL=f_SA_id_225001%3A272001",
        ),
    ],
)
def test_classic_urls_rewritten_to_new_page(url, expected):
    from jobEngine.sources.linkedin.base import to_search_results_url

    assert to_search_results_url(url) == expected
    assert LinkedInAdapter()._target_search_url(url) == expected


def test_explicit_classic_adapter_keeps_url():
    from jobEngine.sources.linkedin.search_adapter import LinkedInSearchAdapter

    url = "https://www.linkedin.com/jobs/search/?f_WT=2"
    assert LinkedInSearchAdapter()._target_search_url(url) == url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://www.linkedin.com/jobs/search-results/?f_WT=2", True),
        ("https://www.linkedin.com/jobs/search-results/?f_WT=1%2C2", True),
        ("https://www.linkedin.com/jobs/search-results/?f_WT=1", False),
        ("https://www.linkedin.com/jobs/search-results/?keywords=jobs", False),
    ],
)
def test_is_remote_filter(url, expected):
    from jobEngine.sources.linkedin.base import is_remote_filter

    assert is_remote_filter(url) is expected
