from __future__ import annotations

from pathlib import Path

from jobEngine.config.sources import JobSourceConfig
from jobEngine.sources.instinctools.adapter import InstinctoolsAdapter

_FIXTURE = Path(__file__).parent / "fixtures" / "instinctools" / "listing.html"
_CAREER_ITEM_FIXTURE = Path(__file__).parent / "fixtures" / "instinctools" / "listing_career_items.html"


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


async def test_structured_tier_extracts_career_item_cards(browser_page):
    """Real ``.career-item`` markup (verified against the live vacancies page):
    location, description, and "#Office"/"#Remote"/skill tags must all be
    parsed, with remote_type split out of the skills list.
    """
    await browser_page.set_content(_CAREER_ITEM_FIXTURE.read_text(encoding="utf-8"))
    adapter = InstinctoolsAdapter()
    jobs = await adapter._extract_structured(browser_page, cfg=_cfg())

    assert len(jobs) == 2
    by_title = {j.title: j for j in jobs}

    remote_job = by_title["Senior Fullstack Engineer (Python+React)"]
    assert remote_job.location == "Poland"
    assert "Senior FullStack" in remote_job.description
    assert remote_job.skills == ["AI", "AWS", "Cloud", "Full Stack", "LLM", "Python", "React"]
    # both #Office and #Remote are present on this card — remote must win
    assert remote_job.remote_type == "remote"

    onsite_job = by_title["Onsite DevOps Engineer"]
    assert onsite_job.location == "Berlin, Germany"
    assert onsite_job.skills == ["AWS", "Docker"]
    # only #Office is present — no remote tag, so onsite
    assert onsite_job.remote_type == "onsite"


def test_split_tags_prefers_remote_over_office():
    remote_type, skills = InstinctoolsAdapter._split_tags(
        ["#Office", "#Remote", "#AI", "#AWS", "#Cloud", "#Full Stack", "#LLM", "#Python", "#React"]
    )
    assert remote_type == "remote"
    assert skills == ["AI", "AWS", "Cloud", "Full Stack", "LLM", "Python", "React"]


def test_split_tags_office_only_is_onsite():
    remote_type, skills = InstinctoolsAdapter._split_tags(["#Office", "#AWS", "#Docker"])
    assert remote_type == "onsite"
    assert skills == ["AWS", "Docker"]


def test_split_tags_no_work_mode_tag_returns_none():
    remote_type, skills = InstinctoolsAdapter._split_tags(["#AI", "#Python"])
    assert remote_type is None
    assert skills == ["AI", "Python"]
