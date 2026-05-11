"""Async Playwright lifecycle: Chromium, headless=False by default."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass
class PlaywrightSession:
    playwright: Playwright
    browser: Browser
    context: BrowserContext
    page: Page


def _require_credentials() -> tuple[str, str]:
    email = (os.getenv("NAUKRI_EMAIL") or "").strip()
    password = (os.getenv("NAUKRI_PASSWORD") or "").strip()
    if not email or not password:
        raise ValueError(
            "NAUKRI_EMAIL and NAUKRI_PASSWORD must be set in the project root .env file."
        )
    return email, password


async def launch_session(*, headless: bool = False) -> PlaywrightSession:
    """
    Launch Chromium with a single context and page.
    headless=False for visible automation (project default).
    """
    _require_credentials()
    pw = await async_playwright().start()
    launch_kwargs: dict[str, Any] = {
        "headless": headless,
        "args": [
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
        ],
    }
    _ch = (os.getenv("NAUKRI_PLAYWRIGHT_CHANNEL") or "").strip()
    if _ch:
        launch_kwargs["channel"] = _ch
    browser = await pw.chromium.launch(**launch_kwargs)
    context = await browser.new_context(
        viewport={"width": 1440, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
        locale="en-IN",
    )
    page = await context.new_page()
    page.set_default_timeout(25_000)
    page.set_default_navigation_timeout(45_000)
    logger.info("Playwright Chromium launched (headless=%s)", headless)
    return PlaywrightSession(playwright=pw, browser=browser, context=context, page=page)


async def close_session(session: PlaywrightSession | None) -> None:
    if session is None:
        return
    try:
        await session.context.close()
    except Exception:  # noqa: BLE001
        logger.exception("Error closing BrowserContext")
    try:
        await session.browser.close()
    except Exception:  # noqa: BLE001
        logger.exception("Error closing Browser")
    try:
        await session.playwright.stop()
    except Exception:  # noqa: BLE001
        logger.exception("Error stopping Playwright")
