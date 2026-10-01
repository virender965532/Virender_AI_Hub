"""URL / SSRF guardrails — Guardrails 2 and 3 (input guardrails, section 14).

Every URL the engine ever navigates to (browser) or otherwise fetches must
pass ``validate_navigation_url``. This blocks:
  - non-http(s) schemes (``file://``, ``javascript:``, ``data:``, ...)
  - localhost / loopback / link-local / private RFC1918 ranges
  - cloud metadata endpoints (169.254.169.254 and friends)
  - hosts outside the configured source's allowlist
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

_ALLOWED_SCHEMES = {"http", "https"}

_BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
}

_BLOCKED_EXACT_IPS = {
    "169.254.169.254",  # AWS/GCP/Azure metadata endpoint
    "0.0.0.0",
}


class SSRFViolation(ValueError):
    """Raised when a URL fails SSRF/domain-allowlist validation."""


@dataclass(frozen=True)
class UrlCheckResult:
    ok: bool
    reason: str | None = None


def _is_blocked_ip(ip_str: str) -> bool:
    if ip_str in _BLOCKED_EXACT_IPS:
        return True
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _resolve_host_safe(hostname: str) -> list[str]:
    """Best-effort DNS resolution; empty list on failure (caller decides policy)."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return []
    return sorted({info[4][0] for info in infos})


def check_url(
    url: str,
    *,
    allowed_domains: list[str] | None = None,
    resolve_dns: bool = True,
) -> UrlCheckResult:
    """Validate scheme, host, and (optionally) resolved IPs against SSRF rules."""

    if not url or not url.strip():
        return UrlCheckResult(False, "empty URL")

    parsed = urlparse(url.strip())
    if parsed.scheme not in _ALLOWED_SCHEMES:
        return UrlCheckResult(False, f"disallowed scheme {parsed.scheme!r}")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        return UrlCheckResult(False, "missing host")

    if hostname in _BLOCKED_HOSTNAMES:
        return UrlCheckResult(False, f"blocked hostname {hostname!r}")

    if allowed_domains:
        allowed = {d.lower() for d in allowed_domains}
        if not any(hostname == d or hostname.endswith(f".{d}") for d in allowed):
            return UrlCheckResult(False, f"host {hostname!r} not in allowlist {sorted(allowed)}")

    try:
        ipaddress.ip_address(hostname)
        if _is_blocked_ip(hostname):
            return UrlCheckResult(False, f"blocked literal IP {hostname!r}")
    except ValueError:
        if resolve_dns:
            for ip_str in _resolve_host_safe(hostname):
                if _is_blocked_ip(ip_str):
                    return UrlCheckResult(False, f"host {hostname!r} resolves to blocked IP {ip_str}")

    return UrlCheckResult(True)


def validate_navigation_url(url: str, *, allowed_domains: list[str] | None = None) -> str:
    """Raise ``SSRFViolation`` if ``url`` is unsafe to navigate to; else return it unchanged."""
    result = check_url(url, allowed_domains=allowed_domains)
    if not result.ok:
        raise SSRFViolation(f"Refusing to navigate to {url!r}: {result.reason}")
    return url


def is_domain_allowed(url: str, allowed_domains: list[str]) -> bool:
    return check_url(url, allowed_domains=allowed_domains, resolve_dns=False).ok
