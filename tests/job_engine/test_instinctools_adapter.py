from __future__ import annotations

from pathlib import Path

from jobEngine.config.sources import JobSourceConfig
from jobEngine.sources.instinctools.adapter import InstinctoolsAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "instinctools" / "listing.html"


def _cfg() -> JobSourceConfig:
    return JobSourceConfig(
        id="instinctools",
        name="Instinctools",
        base_url="https://www.instinctools.eu",
        search_urls=["https://www.instinctools.eu/vacancies/"],
        adapter="instinctools",
        max_jobs=10,
    )


async def test_structured_tier_finds_nothing_on_plain_link_fixture(browser_page):
    await browser_page.set_content(_FIXTURE.read_text(encoding="utf-8"))
    adapter = InstinctoolsAdapter()
    jobs = await adapter._extract_structured(browser_page, cfg=_cfg())
    assert jobs == []  # fixture has no .vacancy-card-style structure — must degrade quietly


async def test_link_scan_fallback_finds_vacancy_links_and_excludes_listing_page(browser_page):
    await browser_page.set_content(_FIXTURE.read_text(encoding="utf-8"))
    adapter = InstinctoolsAdapter()
    cfg = _cfg()
    jobs = await adapter._extract_via_link_scan(browser_page, cfg=cfg, listing_url=cfg.search_urls[0])

    # extract_link_cards canonicalizes URLs (strips trailing "/") for dedup purposes
    urls = {j.application_url for j in jobs}
    assert "https://www.instinctools.eu/vacancies/senior-python-developer" in urls
    assert "https://www.instinctools.eu/vacancies/react-frontend-engineer" in urls
    # the "All vacancies" link back to the listing page itself must be excluded
    assert cfg.search_urls[0].rstrip("/") not in {u.rstrip("/") for u in urls}
    # a non-vacancy link (About us) must not be picked up
    assert not any("about-us" in u for u in urls)
