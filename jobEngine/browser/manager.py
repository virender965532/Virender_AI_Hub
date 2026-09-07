"""Async Playwright lifecycle for the job engine.

Generalizes the single-session pattern in ``jobSearch/utils/
playwright_manager.py`` (this repo's only existing Playwright usage) into a
shared, concurrency-limited manager: one ``Browser`` process, one
``BrowserContext`` per source run (isolation), a semaphore capping how many
contexts are open at once, heavy-asset blocking, a per-context domain
allowlist enforced at the request-routing level (defense in depth alongside
``guardrails.url_guardrails``), and screenshot/HTML capture on failure.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from ..config.settings import JOB_ENGINE_DEBUG_DIR, EngineSettings, get_settings
from ..guardrails.url_guardrails import is_domain_allowed

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)


@dataclass
class ManagedPage:
    context: BrowserContext
    page: Page
    source_id: str


class BrowserManager:
    """Owns one Chromium ``Browser`` for the lifetime of a run."""

    def __init__(self, settings: EngineSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._start_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._settings.browser_max_concurrency)

    async def _ensure_browser(self) -> Browser:
        if self._browser is not None:
            return self._browser
        async with self._start_lock:
            if self._browser is None:
                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=self._settings.browser_headless,
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
                )
                logger.info(
                    "BrowserManager launched Chromium (headless=%s, max_concurrency=%s)",
                    self._settings.browser_headless,
                    self._settings.browser_max_concurrency,
                )
        return self._browser

    @asynccontextmanager
    async def open_page(
        self,
        *,
        source_id: str,
        allowed_domains: list[str] | None = None,
        storage_state_path: str | None = None,
    ) -> AsyncIterator[ManagedPage]:
        """Yield an isolated ``ManagedPage``; the context is always closed on exit."""
        browser = await self._ensure_browser()
        async with self._semaphore:
            context_kwargs: dict[str, object] = {
                "viewport": {"width": 1440, "height": 900},
                "user_agent": _USER_AGENT,
                "locale": "en-US",
            }
            if storage_state_path and Path(storage_state_path).exists():
                context_kwargs["storage_state"] = storage_state_path

            context = await browser.new_context(**context_kwargs)

            async def _route_guard(route) -> None:  # noqa: ANN001
                request = route.request
                if request.resource_type in {"image", "media", "font"}:
                    await route.abort()
                    return
                if allowed_domains and not is_domain_allowed(request.url, allowed_domains):
                    logger.warning(
                        "Blocked navigation to disallowed domain for source=%s url=%s",
                        source_id,
                        request.url,
                    )
                    await route.abort()
                    return
                await route.continue_()

            await context.route("**/*", _route_guard)
            page = await context.new_page()
            page.set_default_timeout(self._settings.browser_timeout_ms)
            page.set_default_navigation_timeout(self._settings.browser_navigation_timeout_ms)

            try:
                yield ManagedPage(context=context, page=page, source_id=source_id)
            finally:
                try:
                    await context.close()
                except Exception:  # noqa: BLE001
                    logger.exception("Error closing BrowserContext for source=%s", source_id)

    async def capture_failure_artifacts(
        self, managed_page: ManagedPage, *, label: str, run_id: str
    ) -> dict[str, str]:
        """Best-effort screenshot (always) + HTML snapshot (debug mode only)."""
        artifacts: dict[str, str] = {}
        debug_dir = JOB_ENGINE_DEBUG_DIR / run_id
        try:
            debug_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = debug_dir / f"{managed_page.source_id}_{label}.png"
            await managed_page.page.screenshot(path=str(screenshot_path))
            artifacts["screenshot"] = str(screenshot_path)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to capture screenshot for %s/%s", managed_page.source_id, label)

        if self._settings.debug:
            try:
                html = await managed_page.page.content()
                html_path = debug_dir / f"{managed_page.source_id}_{label}.html"
                html_path.write_text(html, encoding="utf-8", errors="replace")
                artifacts["html_snapshot"] = str(html_path)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to capture HTML snapshot for %s/%s", managed_page.source_id, label)

        return artifacts

    async def close(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error closing Browser")
            self._browser = None
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:  # noqa: BLE001
                logger.exception("Error stopping Playwright")
            self._playwright = None
