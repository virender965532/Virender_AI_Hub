from __future__ import annotations

import pytest

from jobEngine.guardrails.url_guardrails import (
    SSRFViolation,
    check_url,
    is_domain_allowed,
    validate_navigation_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "ftp://example.com/",
    ],
)
def test_blocks_ssrf_and_disallowed_schemes(url: str):
    result = check_url(url, resolve_dns=False)
    assert not result.ok


def test_allows_https_within_allowlist():
    result = check_url("https://www.linkedin.com/jobs/search/", allowed_domains=["www.linkedin.com"], resolve_dns=False)
    assert result.ok


def test_blocks_host_outside_allowlist():
    result = check_url("https://evil.com/phish", allowed_domains=["www.linkedin.com"], resolve_dns=False)
    assert not result.ok
    assert "not in allowlist" in result.reason


def test_subdomain_of_allowed_domain_is_allowed():
    assert is_domain_allowed("https://sub.example.com/x", ["example.com"])


def test_validate_navigation_url_raises_on_violation():
    with pytest.raises(SSRFViolation):
        validate_navigation_url("http://localhost/admin")


def test_validate_navigation_url_returns_url_on_success():
    url = "https://example.com/jobs"
    assert validate_navigation_url(url) == url
