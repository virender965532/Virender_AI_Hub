"""Interactive LinkedIn sign-in that saves the session to MongoDB.

    .venv\\Scripts\\python.exe -m jobEngine.sources.linkedin.login [--timeout 300]

Opens a visible Chromium window at the LinkedIn login page and waits for the
candidate to sign in by hand (2FA, CAPTCHA and "verify it's you" steps
included). No credentials are read or typed by this script. Once LinkedIn sets
the ``li_at`` auth cookie and leaves the login/checkpoint pages, the context's
``storage_state`` is written to the ``browser_sessions`` collection, where the
adapter (including headless runs on the server, which share the database)
picks it up on its next run.

Run it on a machine with a display; the server runs headless and has none.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from playwright.async_api import async_playwright

from ...browser.manager import _USER_AGENT
from ...persistence.sessions import save_storage_state
from .adapter import _DEFAULT_SIGNIN_URL, _is_checkpoint_url, _is_login_url

_PROJECT_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger("jobEngine.linkedin_login")

SOURCE_ID = "linkedin"


LOGIN_URL = _DEFAULT_SIGNIN_URL


async def is_signed_in(context, page) -> bool:
    """LinkedIn has set the ``li_at`` auth cookie and the page left login/checkpoint."""
    cookies = await context.cookies("https://www.linkedin.com")
    has_cookie = any(c.get("name") == "li_at" and c.get("value") for c in cookies)
    return has_cookie and not _is_login_url(page.url) and not _is_checkpoint_url(page.url)


async def interactive_login(*, timeout_seconds: int = 300, source_id: str = SOURCE_ID) -> bool:
    """Return True once a signed-in session has been saved; False on timeout or a closed window."""
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=False, args=["--disable-blink-features=AutomationControlled"]
        )
        try:
            # Same user agent/locale as BrowserManager so the saved cookies match the
            # browser the job engine replays them in. No asset blocking here: the
            # CAPTCHA and verification screens need their images.
            context = await browser.new_context(
                viewport={"width": 1280, "height": 900}, user_agent=_USER_AGENT, locale="en-US"
            )
            page = await context.new_page()
            await page.goto(LOGIN_URL, wait_until="domcontentloaded")
            logger.info("Sign in to LinkedIn in the browser window (waiting up to %ss)...", timeout_seconds)

            deadline = time.monotonic() + timeout_seconds
            while time.monotonic() < deadline:
                if page.is_closed():
                    logger.error("Browser window was closed before sign-in finished")
                    return False
                try:
                    signed_in = await is_signed_in(context, page)
                except Exception:  # noqa: BLE001
                    if page.is_closed():
                        logger.error("Browser window was closed before sign-in finished")
                        return False
                    raise
                if signed_in:
                    # Let LinkedIn finish setting the remaining session cookies.
                    await page.wait_for_load_state("domcontentloaded")
                    await asyncio.sleep(2)
                    state = await context.storage_state()
                    await asyncio.to_thread(save_storage_state, source_id, state)
                    logger.info(
                        "Signed in; saved %s cookies to MongoDB (browser_sessions/%s)",
                        len(state.get("cookies") or []), source_id,
                    )
                    return True
                await asyncio.sleep(1)

            logger.error("Timed out after %ss waiting for LinkedIn sign-in", timeout_seconds)
            return False
        finally:
            await browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sign in to LinkedIn by hand and save the session to MongoDB.")
    parser.add_argument("--timeout", type=int, default=300, help="seconds to wait for sign-in (default 300)")
    args = parser.parse_args(argv)
    load_dotenv(_PROJECT_ROOT / ".env")  # MONGODB_URI is read when the session is saved
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return 0 if asyncio.run(interactive_login(timeout_seconds=max(30, args.timeout))) else 1


if __name__ == "__main__":
    sys.exit(main())
