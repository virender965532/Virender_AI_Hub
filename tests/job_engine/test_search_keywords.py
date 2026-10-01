from jobEngine.sources.search_keywords import url_keywords, with_keywords

_LINKEDIN = (
    "https://www.linkedin.com/jobs/search-results/?keywords=%28JavaScript%20OR%20TypeScript%29%20AND%20AWS"
    "%20NOT%20%22C%23%22&origin=JOB_SEARCH_PAGE_JOB_FILTER&start=0&f_SAL=f_SA_id_225001%3A272001"
)


def test_reads_decoded_keywords():
    assert url_keywords(_LINKEDIN) == '(JavaScript OR TypeScript) AND AWS NOT "C#"'
    assert url_keywords("https://example.com/careers") is None


def test_replaces_only_the_keywords_param():
    new = with_keywords(_LINKEDIN, '(React OR "Next.js") AND AWS NOT "C#"')
    assert url_keywords(new) == '(React OR "Next.js") AND AWS NOT "C#"'
    assert new.startswith("https://www.linkedin.com/jobs/search-results/?keywords=%28React%20OR%20%22Next.js%22%29")
    assert new.endswith("&origin=JOB_SEARCH_PAGE_JOB_FILTER&start=0&f_SAL=f_SA_id_225001%3A272001")


def test_url_without_keywords_is_untouched():
    assert with_keywords("https://example.com/careers?page=1", "React") == "https://example.com/careers?page=1"
