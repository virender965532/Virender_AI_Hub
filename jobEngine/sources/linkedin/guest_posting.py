"""LinkedIn's public job-posting endpoint: ``/jobs-guest/jobs/api/jobPosting/<id>``.

It returns the same HTML a signed-out visitor of ``/jobs/view/<id>`` gets — full
description, age ("1 hour ago") and the criteria list (employment type, seniority) —
so a job's details are read without opening it in the browser. Fetched from Python
(not from inside the signed-in page, whose security policy blocks it), a few at a
time; parsed with the standard library only.
"""

from __future__ import annotations

import asyncio
import logging
import re
import urllib.error
import urllib.request
from html.parser import HTMLParser

logger = logging.getLogger(__name__)

GUEST_POSTING_URL = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
_JOB_ID_RE = re.compile(r"^\d{5,20}$")
_TIMEOUT_SECONDS = 10
# LinkedIn rate-limits this endpoint per IP (HTTP 429), and the results pages are read
# in parallel tabs — so one limiter is shared by every caller in the process: at most
# _MAX_IN_FLIGHT requests at once, starts spaced by _MIN_INTERVAL_SECONDS, and after
# a 429 *everyone* pauses (2s, 4s, 8s, ...) before that job is retried.
_MAX_IN_FLIGHT = 3
_MIN_INTERVAL_SECONDS = 0.3
_BACKOFF_SECONDS = (2.0, 4.0, 8.0)

_DESCRIPTION_CLASSES = ("show-more-less-html__markup", "description__text")
_POSTED_CLASS = "posted-time-ago__text"
_CRITERIA_KEY_CLASS = "description__job-criteria-subheader"
_CRITERIA_VALUE_CLASS = "description__job-criteria-text"
_TARGET_CLASSES = (*_DESCRIPTION_CLASSES, _POSTED_CLASS, _CRITERIA_KEY_CLASS, _CRITERIA_VALUE_CLASS)
_VOID_TAGS = {"br", "img", "hr", "input", "meta", "link", "source", "wbr", "area", "base", "col", "embed", "track"}
_BLOCK_TAGS = {"p", "li", "div", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "section", "tr"}


class _Collector(HTMLParser):
    """Text of every element carrying one of ``_TARGET_CLASSES``, in document order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: list[tuple[str, str]] = []
        self._open: list[list] = []  # [class, depth, parts]

    def handle_starttag(self, tag, attrs):
        if tag in _VOID_TAGS:
            if tag == "br":
                self._text("\n")
            return
        for capture in self._open:
            capture[1] += 1
        if tag in _BLOCK_TAGS:
            self._text("\n")
        classes = (dict(attrs).get("class") or "").split()
        target = next((c for c in _TARGET_CLASSES if c in classes), None)
        if target:
            self._open.append([target, 1, []])

    def handle_startendtag(self, tag, attrs):
        if tag == "br":
            self._text("\n")

    def handle_endtag(self, tag):
        if tag in _VOID_TAGS:
            return
        if tag in _BLOCK_TAGS:
            self._text("\n")
        still_open = []
        for capture in self._open:
            capture[1] -= 1
            if capture[1] <= 0:
                self.found.append((capture[0], _clean("".join(capture[2]))))
            else:
                still_open.append(capture)
        self._open = still_open

    def handle_data(self, data):
        self._text(data)

    def _text(self, text: str) -> None:
        for capture in self._open:
            capture[2].append(text)


def _clean(text: str) -> str:
    lines = [re.sub(r"[ \t\xa0]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def parse_guest_posting(html: str) -> dict:
    """``{description, posted, employment_type}`` from the endpoint's HTML."""
    collector = _Collector()
    collector.feed(html)
    collector.close()
    description = next((t for c, t in collector.found if c in _DESCRIPTION_CLASSES and t), None)
    posted = next((t for c, t in collector.found if c == _POSTED_CLASS and t), None)
    keys = [t for c, t in collector.found if c == _CRITERIA_KEY_CLASS]
    values = [t for c, t in collector.found if c == _CRITERIA_VALUE_CLASS]
    criteria = dict(zip(keys, values))
    return {"description": description, "posted": posted, "employment_type": criteria.get("Employment type")}


def _get(url: str, user_agent: str) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "text/html"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310 — fixed https host
            return response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""


class _RateLimiter:
    """Shared by every page of every run on one event loop (see module notes above)."""

    def __init__(self) -> None:
        self.slots = asyncio.Semaphore(_MAX_IN_FLIGHT)
        self.lock = asyncio.Lock()
        self.next_start = 0.0
        self.cooldown_until = 0.0

    async def wait_turn(self) -> None:
        loop = asyncio.get_running_loop()
        async with self.lock:
            delay = max(self.next_start, self.cooldown_until) - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self.next_start = loop.time() + _MIN_INTERVAL_SECONDS

    def cool_down(self, seconds: float) -> None:
        until = asyncio.get_running_loop().time() + seconds
        self.cooldown_until = max(self.cooldown_until, until)


_limiters: dict[int, _RateLimiter] = {}


def _limiter() -> _RateLimiter:
    loop = asyncio.get_running_loop()
    limiter = _limiters.get(id(loop))
    if limiter is None:
        _limiters.clear()  # a new event loop (e.g. a new run in a new thread): start fresh
        limiter = _limiters[id(loop)] = _RateLimiter()
    return limiter


async def fetch_guest_postings(job_ids: list[str], *, user_agent: str) -> dict[str, dict]:
    """``job_id -> {description, posted, employment_type}`` or ``{error}`` for every id."""
    limiter = _limiter()

    async def one(job_id: str) -> tuple[str, dict]:
        if not _JOB_ID_RE.match(job_id):
            return job_id, {"error": "bad job id"}
        url = GUEST_POSTING_URL.format(job_id=job_id)
        for attempt in range(len(_BACKOFF_SECONDS) + 1):
            async with limiter.slots:
                await limiter.wait_turn()
                try:
                    status, html = await asyncio.to_thread(_get, url, user_agent)
                except Exception as exc:  # noqa: BLE001 — network errors
                    return job_id, {"error": type(exc).__name__}
            if status == 429:
                if attempt < len(_BACKOFF_SECONDS):
                    limiter.cool_down(_BACKOFF_SECONDS[attempt])
                    continue
                return job_id, {"error": "HTTP 429"}
            if status != 200:
                return job_id, {"error": f"HTTP {status}"}
            try:
                return job_id, parse_guest_posting(html)
            except Exception as exc:  # noqa: BLE001
                return job_id, {"error": f"parse: {type(exc).__name__}"}
        return job_id, {"error": "HTTP 429"}

    return dict(await asyncio.gather(*(one(i) for i in job_ids)))
