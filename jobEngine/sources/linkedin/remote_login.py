"""LinkedIn sign-in driven from the web UI ("Connect LinkedIn" popup).

The server has no display, so instead of a visible window the browser runs
headless here and the popup drives it remotely: it polls ``frame()`` for a JPEG
screenshot and forwards the candidate's clicks, keys and scrolls through
``send_input()``. The candidate signs in by hand (2FA / CAPTCHA included); a
watcher task saves the context's ``storage_state`` to MongoDB as soon as
LinkedIn sets the ``li_at`` cookie, exactly like ``login.py`` does locally.

Flask is synchronous, so Playwright lives on one dedicated thread with its own
event loop and every call is marshalled onto it. Only one login runs at a time;
starting another closes the previous one. A login closes itself after
``LINKEDIN_REMOTE_LOGIN_TIMEOUT_SECONDS`` (default 300).
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, Playwright, async_playwright

from ...browser.manager import _USER_AGENT
from ...guardrails.url_guardrails import is_domain_allowed
from ...persistence.sessions import save_storage_state
from .login import LOGIN_URL, SOURCE_ID, is_signed_in

logger = logging.getLogger(__name__)

VIEWPORT = {"width": 1000, "height": 760}
# Top-level navigation is kept on LinkedIn; sub-frames (the CAPTCHA widget) may load anywhere.
_ALLOWED_TOP_LEVEL = ["linkedin.com", "licdn.com"]
# Keys forwarded with ``keyboard.press``; anything else must arrive as text.
_NAMED_KEYS = {
    "Enter", "Tab", "Backspace", "Delete", "Escape", "ArrowLeft", "ArrowRight",
    "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown", " ",
}


def _timeout_seconds() -> int:
    try:
        return max(60, int((os.getenv("LINKEDIN_REMOTE_LOGIN_TIMEOUT_SECONDS") or "300").strip()))
    except ValueError:
        return 300


class LoginNotFound(LookupError):
    pass


@dataclass
class _Login:
    id: str
    context: BrowserContext
    page: Page
    expires_at: float
    state: str = "waiting"  # waiting | saved | expired | failed | cancelled
    message: str = "Sign in to LinkedIn in the window below."
    watcher: asyncio.Task | None = field(default=None, repr=False)

    def current_page(self) -> Page:
        """The newest open tab; a popup tab LinkedIn closed falls back to an older one."""
        if self.page.is_closed():
            open_pages = [p for p in self.context.pages if not p.is_closed()]
            if open_pages:
                self.page = open_pages[-1]
        return self.page

    def status(self) -> dict[str, Any]:
        return {
            "login_id": self.id,
            "state": self.state,
            "message": self.message,
            "url": "" if self.current_page().is_closed() else self.page.url,
            "seconds_left": max(0, int(self.expires_at - time.monotonic())),
            "viewport": VIEWPORT,
        }


class RemoteLoginManager:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread_lock = threading.Lock()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._login: _Login | None = None
        # Kept after the browser closes so the popup can still read the outcome.
        self._finished: dict[str, dict[str, Any]] = {}

    # --- thread / loop plumbing -------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._thread_lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(
                    target=loop.run_forever, name="linkedin-remote-login", daemon=True
                ).start()
                self._loop = loop
            return self._loop

    def _call(self, coro, timeout: float = 30):
        return asyncio.run_coroutine_threadsafe(coro, self._ensure_loop()).result(timeout=timeout)

    # --- public (sync, called from Flask threads) -------------------------------

    def start(self) -> dict[str, Any]:
        return self._call(self._start(), timeout=60)

    def status(self, login_id: str) -> dict[str, Any]:
        return self._call(self._status(login_id))

    def frame(self, login_id: str) -> bytes:
        return self._call(self._frame(login_id))

    def send_input(self, login_id: str, event: dict[str, Any]) -> None:
        self._call(self._input(login_id, event))

    def cancel(self, login_id: str) -> None:
        self._call(self._cancel(login_id))

    # --- coroutines (run on the manager's loop) ---------------------------------

    def _active(self, login_id: str) -> _Login:
        login = self._login
        if login is None or login.id != login_id:
            raise LoginNotFound(login_id)
        return login

    async def _ensure_browser(self) -> Browser:
        if self._browser is None or not self._browser.is_connected():
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
        return self._browser

    async def _start(self) -> dict[str, Any]:
        if self._login is not None:
            await self._finish(self._login, "cancelled", "Replaced by a new sign-in.")
        browser = await self._ensure_browser()
        # Same user agent/locale as BrowserManager so the cookies match the browser
        # the job engine replays them in. Images are not blocked: CAPTCHAs need them.
        context = await browser.new_context(viewport=VIEWPORT, user_agent=_USER_AGENT, locale="en-US")

        async def _guard(route) -> None:  # noqa: ANN001
            request = route.request
            if (
                request.is_navigation_request()
                and request.frame.parent_frame is None
                and not is_domain_allowed(request.url, _ALLOWED_TOP_LEVEL)
            ):
                logger.warning("Remote LinkedIn login: blocked navigation to %s", request.url)
                await route.abort()
                return
            await route.continue_()

        await context.route("**/*", _guard)
        page = await context.new_page()
        login = _Login(
            id=secrets.token_urlsafe(16),
            context=context,
            page=page,
            expires_at=time.monotonic() + _timeout_seconds(),
        )
        # Links that open a new tab: follow the newest page.
        context.on("page", lambda p: setattr(login, "page", p))
        self._login = login
        try:
            await page.goto(LOGIN_URL, wait_until="domcontentloaded", timeout=45_000)
        except Exception as exc:  # noqa: BLE001
            await self._finish(login, "failed", f"Could not open the LinkedIn login page: {exc}")
            return self._finished[login.id]
        login.watcher = asyncio.create_task(self._watch(login))
        logger.info("Remote LinkedIn login %s started", login.id[:6])
        return login.status()

    async def _watch(self, login: _Login) -> None:
        try:
            while self._login is login:
                if time.monotonic() >= login.expires_at:
                    await self._finish(login, "expired", "Timed out waiting for sign-in. Start again.")
                    return
                page = login.current_page()
                if not page.is_closed() and await is_signed_in(login.context, page):
                    await page.wait_for_load_state("domcontentloaded")
                    await asyncio.sleep(2)  # let LinkedIn finish setting session cookies
                    state = await login.context.storage_state()
                    await asyncio.to_thread(save_storage_state, SOURCE_ID, state)
                    logger.info(
                        "Remote LinkedIn login %s: saved %s cookies to MongoDB",
                        login.id[:6], len(state.get("cookies") or []),
                    )
                    await self._finish(login, "saved", "Signed in. The LinkedIn session is saved.")
                    return
                await asyncio.sleep(1)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Remote LinkedIn login %s failed", login.id[:6])
            await self._finish(login, "failed", f"Sign-in failed: {exc}")

    async def _finish(self, login: _Login, state: str, message: str) -> None:
        login.state, login.message = state, message
        self._finished[login.id] = {**login.status(), "seconds_left": 0}
        if len(self._finished) > 20:
            self._finished.pop(next(iter(self._finished)))
        if self._login is login:
            self._login = None
        if login.watcher is not None and login.watcher is not asyncio.current_task():
            login.watcher.cancel()
        try:
            await login.context.close()
        except Exception:  # noqa: BLE001
            logger.debug("Error closing remote login context", exc_info=True)
        # Free the Chromium process between logins; the next start relaunches it.
        if self._login is None and self._browser is not None:
            try:
                await self._browser.close()
            except Exception:  # noqa: BLE001
                logger.debug("Error closing remote login browser", exc_info=True)
            self._browser = None

    async def _status(self, login_id: str) -> dict[str, Any]:
        if login_id in self._finished:
            return self._finished[login_id]
        return self._active(login_id).status()

    async def _frame(self, login_id: str) -> bytes:
        page = self._active(login_id).current_page()
        return await page.screenshot(type="jpeg", quality=60, timeout=10_000)

    async def _input(self, login_id: str, event: dict[str, Any]) -> None:
        page = self._active(login_id).current_page()
        kind = event.get("type")
        if kind == "click":
            x = min(max(float(event["x"]), 0), VIEWPORT["width"])
            y = min(max(float(event["y"]), 0), VIEWPORT["height"])
            await page.mouse.click(x, y)
        elif kind == "type":
            text = str(event.get("text") or "")[:500]
            if text:
                await page.keyboard.type(text)
        elif kind == "press":
            key = str(event.get("key") or "")
            if key not in _NAMED_KEYS:
                raise ValueError(f"unsupported key {key!r}")
            await page.keyboard.press(key)
        elif kind == "scroll":
            await page.mouse.wheel(0, max(-2000, min(2000, float(event.get("dy") or 0))))
        else:
            raise ValueError(f"unsupported input type {kind!r}")

    async def _cancel(self, login_id: str) -> None:
        login = self._login
        if login is not None and login.id == login_id:
            await self._finish(login, "cancelled", "Sign-in cancelled.")


remote_login_manager = RemoteLoginManager()
