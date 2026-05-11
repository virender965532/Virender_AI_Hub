"""Login to Naukri via Playwright (modal flow, credential env vars)."""

from __future__ import annotations

import logging
import os
from typing import Any

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from ..state import WorkflowState
from ..utils.playwright_manager import PlaywrightSession, launch_session

logger = logging.getLogger(__name__)

NAUKRI_HOME = "https://www.naukri.com"

_LOGIN_CLICK_SELECTORS: list[str] = [
    "a.loginLayer",
    'a[title="Login"]',
    "div.headGNB a.loginLayer",
    "div.headGNB a[href*='login']",
]

_USER_FIELD_SELECTORS: list[str] = [
    "input#username",
    "input[name='username']",
    "input[type='text'][placeholder*='Email']",
    "input[placeholder*='mail']",
]

_PASSWORD_FIELD_SELECTORS: list[str] = [
    "input#password",
    "input[name='password']",
    "input[type='password']",
]

_SUBMIT_SELECTORS: list[str] = [
    "button[type='submit']",
    "button.loginButton",
    ".btn-primary.loginButton",
]


async def _click_login_entry(page: Page) -> bool:
    for css in _LOGIN_CLICK_SELECTORS:
        loc = page.locator(css).first
        try:
            await loc.wait_for(state="visible", timeout=12_000)
            await loc.click(timeout=12_000)
            logger.info("Opened login layer via selector: %s", css)
            return True
        except PlaywrightTimeoutError:
            continue
        except Exception:  # noqa: BLE001
            logger.debug("Login click failed for %s", css, exc_info=True)
            continue
    try:
        await page.get_by_role("link", name="Login").first.click(timeout=12_000)
        logger.info("Opened login layer via role=link Login")
        return True
    except Exception:  # noqa: BLE001
        logger.debug("Login role link failed", exc_info=True)
    return False


async def _fill_first_visible(page: Page, selectors: list[str], value: str) -> bool:
    for css in selectors:
        loc = page.locator(css).first
        try:
            await loc.wait_for(state="visible", timeout=10_000)
            await loc.fill(value, timeout=10_000)
            return True
        except PlaywrightTimeoutError:
            continue
        except Exception:  # noqa: BLE001
            logger.debug("Fill failed for %s", css, exc_info=True)
            continue
    return False


async def _submit_login(page: Page) -> None:
    for css in _SUBMIT_SELECTORS:
        loc = page.locator(css).first
        try:
            if await loc.count() > 0 and await loc.is_enabled():
                await loc.click(timeout=12_000)
                logger.info("Submitted login via selector: %s", css)
                return
        except Exception:  # noqa: BLE001
            continue
    await page.locator("input[type='password']").first.press("Enter")
    logger.info("Submitted login via Enter on password field")


async def _wait_logged_in(page: Page, *, timeout_ms: int = 120_000) -> None:
    await page.wait_for_function(
        """() => {
            const u = (location.href || '').toLowerCase();
            if (u.includes('/mnjuser') || u.includes('mynaukri')) return true;
            return !!document.querySelector(
              "a[href*='logout'], a[href*='mnjuser/profile'], .nI-gnb-avatar, " +
              "img[alt*='profile'], [class*='gnb-user'], [data-gnb-name]"
            );
        }""",
        timeout=timeout_ms,
    )


async def _perform_login(page: Page) -> None:
    email = (os.getenv("NAUKRI_EMAIL") or "").strip()
    password = (os.getenv("NAUKRI_PASSWORD") or "").strip()
    if not email or not password:
        raise ValueError("NAUKRI_EMAIL and NAUKRI_PASSWORD must be set.")

    await page.goto(NAUKRI_HOME, wait_until="domcontentloaded")
    await page.wait_for_selector("body")

    # Optional consent / interstitial: best-effort dismiss
    for sel in ("button:has-text('Got it')", "[aria-label='Close']", ".gdpr-banner button"):
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=1500):
                await btn.click(timeout=3000)
        except PlaywrightTimeoutError:
            pass
        except Exception:  # noqa: BLE001
            logger.debug("Dismiss attempt skipped for %s", sel, exc_info=True)

    if not await _click_login_entry(page):
        raise RuntimeError("Could not open Naukri login modal.")

    await page.wait_for_selector(
        ",".join(_USER_FIELD_SELECTORS[:3]),
        timeout=15_000,
    )

    if not await _fill_first_visible(page, _USER_FIELD_SELECTORS, email):
        raise RuntimeError("Username/email field not found in login modal.")
    if not await _fill_first_visible(page, _PASSWORD_FIELD_SELECTORS, password):
        raise RuntimeError("Password field not found in login modal.")

    await _submit_login(page)
    await _wait_logged_in(page)
    logger.info("Login appears successful (URL/nav signal).")


async def login_node(state: WorkflowState) -> dict[str, Any]:
    """
    Ensure Playwright session exists, perform Naukri login.
    Returns partial state updates only.
    """
    errs = list(state.get("errors") or [])
    session: PlaywrightSession | None = state.get("session")

    try:
        if session is None:
            session = await launch_session(headless=False)
        page = session.page
        await _perform_login(page)
        return {
            "session": session,
            "login_complete": True,
            "errors": errs,
        }
    except Exception as e:  # noqa: BLE001
        logger.exception("login_node failed")
        errs.append(str(e))
        return {
            "session": session,
            "login_complete": False,
            "errors": errs,
        }
