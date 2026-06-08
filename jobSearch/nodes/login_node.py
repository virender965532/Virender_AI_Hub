"""Login to Naukri via Playwright (direct login page preferred, modal fallback)."""

from __future__ import annotations

import logging
import os
from typing import Any

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from ..state import WorkflowState
from ..utils.playwright_manager import PlaywrightSession, launch_session

logger = logging.getLogger(__name__)

NAUKRI_HOME = "https://www.naukri.com"
NAUKRI_LOGIN_URL = (
    (os.getenv("NAUKRI_LOGIN_URL") or "https://www.naukri.com/nlogin/login").strip()
    or "https://www.naukri.com/nlogin/login"
)
NAUKRI_LOGIN_URL_ALT = "https://login.naukri.com/nLogin/Login.php"

_LOGIN_CLICK_SELECTORS: list[str] = [
    "a.loginLayer",
    "#login_Layer",
    'a[title="Login"]',
    "div.headGNB a.loginLayer",
    "div.headGNB a[href*='login']",
]

_USER_FIELD_SELECTORS: list[str] = [
    "input#username",
    "input#usernameField",
    "input#eLoginNew",
    "input[name='username']",
    "input[name='USERNAME']",
    "input[type='text'][placeholder*='Email']",
    "input[placeholder*='mail']",
]

_PASSWORD_FIELD_SELECTORS: list[str] = [
    "input#password",
    "input#passwordField",
    "input#pLogin",
    "input[name='password']",
    "input[name='PASSWORD']",
    "input[type='password']",
]

_SUBMIT_SELECTORS: list[str] = [
    "button[type='submit']",
    "button.loginButton",
    ".btn-primary.loginButton",
    "button.blueBtn",
]

_USER_LOCATOR = ", ".join(_USER_FIELD_SELECTORS)
_PASSWORD_LOCATOR = ", ".join(_PASSWORD_FIELD_SELECTORS)
_LOGIN_ENTRY_LOCATOR = ", ".join(_LOGIN_CLICK_SELECTORS)
_SUBMIT_LOCATOR = ", ".join(_SUBMIT_SELECTORS)

_FIELD_TIMEOUT_MS = 6_000
_NAV_TIMEOUT_MS = 25_000
_LOGIN_WAIT_MS = 60_000


def _combined_locator(page: Page, selectors: str):
    return page.locator(selectors).first


async def _dismiss_banners(page: Page) -> None:
    for sel in ("button:has-text('Got it')", "[aria-label='Close']", ".gdpr-banner button"):
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=400):
                await btn.click(timeout=800)
        except PlaywrightTimeoutError:
            pass
        except Exception:  # noqa: BLE001
            pass


async def _fill_credentials(page: Page, email: str, password: str) -> None:
    user = _combined_locator(page, _USER_LOCATOR)
    await user.wait_for(state="visible", timeout=_FIELD_TIMEOUT_MS)
    await user.fill(email, timeout=_FIELD_TIMEOUT_MS)

    pwd = _combined_locator(page, _PASSWORD_LOCATOR)
    await pwd.wait_for(state="visible", timeout=_FIELD_TIMEOUT_MS)
    await pwd.fill(password, timeout=_FIELD_TIMEOUT_MS)


async def _submit_login(page: Page) -> None:
    btn = _combined_locator(page, _SUBMIT_LOCATOR)
    try:
        if await btn.is_visible(timeout=800):
            await btn.click(timeout=_FIELD_TIMEOUT_MS)
            logger.info("Submitted login via submit button")
            return
    except Exception:  # noqa: BLE001
        pass
    await _combined_locator(page, _PASSWORD_LOCATOR).press("Enter")
    logger.info("Submitted login via Enter on password field")


async def _wait_logged_in(page: Page, *, timeout_ms: int = _LOGIN_WAIT_MS) -> None:
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


async def _login_form_visible(page: Page) -> bool:
    try:
        await _combined_locator(page, _USER_LOCATOR).wait_for(
            state="visible", timeout=_FIELD_TIMEOUT_MS
        )
        return True
    except PlaywrightTimeoutError:
        return False


async def _try_login_at_url(page: Page, url: str, email: str, password: str) -> bool:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
        await _dismiss_banners(page)
        if not await _login_form_visible(page):
            return False
        await _fill_credentials(page, email, password)
        await _submit_login(page)
        await _wait_logged_in(page)
        return True
    except PlaywrightTimeoutError:
        logger.info("Login timed out at %s", url)
        return False
    except Exception as e:  # noqa: BLE001
        logger.info("Login attempt failed at %s: %s", url, e)
        return False


async def _try_direct_login(page: Page, email: str, password: str) -> bool:
    for url in (NAUKRI_LOGIN_URL, NAUKRI_LOGIN_URL_ALT):
        logger.info("Trying direct login: %s", url)
        if await _try_login_at_url(page, url, email, password):
            logger.info("Direct login succeeded via %s", url)
            return True
    return False


async def _try_modal_login(page: Page, email: str, password: str) -> None:
    await page.goto(NAUKRI_HOME, wait_until="domcontentloaded", timeout=_NAV_TIMEOUT_MS)
    await _dismiss_banners(page)

    login_btn = _combined_locator(page, _LOGIN_ENTRY_LOCATOR)
    try:
        await login_btn.wait_for(state="visible", timeout=_FIELD_TIMEOUT_MS)
        await login_btn.click(timeout=_FIELD_TIMEOUT_MS)
        logger.info("Opened login layer from homepage")
    except PlaywrightTimeoutError:
        await page.get_by_role("link", name="Login").first.click(timeout=_FIELD_TIMEOUT_MS)
        logger.info("Opened login layer via role=link Login")

    await _fill_credentials(page, email, password)
    await _submit_login(page)
    await _wait_logged_in(page)


async def _perform_login(page: Page) -> None:
    email = (os.getenv("NAUKRI_EMAIL") or "").strip()
    password = (os.getenv("NAUKRI_PASSWORD") or "").strip()
    if not email or not password:
        raise ValueError("NAUKRI_EMAIL and NAUKRI_PASSWORD must be set.")

    if await _try_direct_login(page, email, password):
        logger.info("Login successful (direct URL).")
        return

    logger.info("Direct login unavailable; falling back to homepage modal.")
    await _try_modal_login(page, email, password)
    logger.info("Login successful (homepage modal).")


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
