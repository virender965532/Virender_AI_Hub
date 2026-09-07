"""Small Playwright DOM helpers shared by adapters (kept dependency-free of
any one site so both the generic and dedicated adapters can reuse them).
"""

from __future__ import annotations

from urllib.parse import urljoin

from playwright.async_api import ElementHandle, Page


async def text_or_none(card: ElementHandle, selector: str) -> str | None:
    if not selector:
        return None
    try:
        el = await card.query_selector(selector)
        if el is None:
            return None
        text = await el.inner_text()
        text = text.strip()
        return text or None
    except Exception:  # noqa: BLE001
        return None


async def href_or_none(card: ElementHandle, selector: str, *, base_url: str) -> str | None:
    if not selector:
        return None
    try:
        el = await card.query_selector(selector)
        if el is None:
            return None
        href = await el.get_attribute("href")
        if not href:
            return None
        return urljoin(base_url, href)
    except Exception:  # noqa: BLE001
        return None


async def extract_link_cards(
    page: Page,
    *,
    href_contains: tuple[str, ...],
    base_url: str,
    exclude_urls: tuple[str, ...] = (),
    min_text_len: int = 3,
    max_text_len: int = 150,
    max_items: int = 200,
) -> list[dict[str, str]]:
    """Fallback discovery strategy: anchors whose href matches a detail-page
    pattern, deduped by resolved URL. Used when dedicated CSS selectors are
    unavailable/stale — the same "fall back to link scanning" shape already
    used by ``jobSearch/utils/fetch_jobs_node.py``'s legacy extraction path.
    """
    anchors = await page.query_selector_all("a[href]")
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    excluded = {u.rstrip("/") for u in exclude_urls}

    for a in anchors:
        try:
            href = await a.get_attribute("href")
        except Exception:  # noqa: BLE001
            continue
        if not href:
            continue
        if not any(marker in href for marker in href_contains):
            continue
        resolved = urljoin(base_url, href).split("?")[0].rstrip("/")
        if not resolved or resolved in seen or resolved in excluded:
            continue

        try:
            text = (await a.inner_text() or "").strip()
        except Exception:  # noqa: BLE001
            text = ""
        if not (min_text_len <= len(text) <= max_text_len):
            continue

        seen.add(resolved)
        out.append({"title": text, "url": resolved})
        if len(out) >= max_items:
            break

    return out
