"""Pushover push notifications (https://pushover.net/api#messages).

Reads ``PUSHOVER_TOKEN`` (application API token) and ``PUSHOVER_USER`` (user
key) from the environment. Never raises: a missing config or a failed request
is logged and reported as ``False`` so a notification can't break a search.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)

_API_URL = "https://api.pushover.net/1/messages.json"
# Limits from the Pushover API docs.
_MAX_TITLE = 250
_MAX_MESSAGE = 1024
_MAX_URL = 512
_MAX_URL_TITLE = 100


def pushover_configured() -> bool:
    return bool(os.getenv("PUSHOVER_TOKEN") and os.getenv("PUSHOVER_USER"))


def send_pushover(
    message: str,
    *,
    title: str | None = None,
    url: str | None = None,
    url_title: str | None = None,
    priority: int = 0,
    html: bool = False,
    timeout: float = 15.0,
) -> bool:
    token = os.getenv("PUSHOVER_TOKEN", "").strip()
    user = os.getenv("PUSHOVER_USER", "").strip()
    if not token or not user:
        logger.warning("Pushover not configured (PUSHOVER_TOKEN / PUSHOVER_USER missing); notification skipped")
        return False

    fields = {"token": token, "user": user, "message": message[:_MAX_MESSAGE], "priority": str(priority)}
    if title:
        fields["title"] = title[:_MAX_TITLE]
    if url and len(url) <= _MAX_URL:
        fields["url"] = url
        if url_title:
            fields["url_title"] = url_title[:_MAX_URL_TITLE]
    if html:
        fields["html"] = "1"

    request = urllib.request.Request(_API_URL, data=urllib.parse.urlencode(fields).encode("utf-8"), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        # 4xx responses carry {"status": 0, "errors": [...]} — log it without the token.
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("errors")
        except Exception:  # noqa: BLE001
            detail = None
        logger.error("Pushover request failed: HTTP %s %s", exc.code, detail or "")
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Pushover request failed: %s", exc)
        return False

    if body.get("status") != 1:
        logger.error("Pushover rejected the message: %s", body.get("errors"))
        return False
    return True
