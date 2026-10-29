"""Plain SMTP email, IMAP drafts and Gmail compose drafts (Gmail by default).

Every mail login uses ``EMAIL_SENDER`` (address and login) and ``EMAIL_PASSWORD``:
sending over SMTP (``SMTP_HOST`` [smtp.gmail.com], ``SMTP_PORT`` [587, STARTTLS;
465 = implicit TLS]) and saving drafts over IMAP into that mailbox (``IMAP_HOST``
[imap.gmail.com]). Gmail accepts only an App Password here
(https://myaccount.google.com/apppasswords); the account password is refused.

``open_gmail_compose`` needs no password: it opens a prefilled Gmail compose
window in this computer's browser (signed in as ``EMAIL_SENDER``), which Gmail
keeps in Drafts — the same way the Autonomous Application Agent page does it.

Never raises: a missing config or a failed send is logged and reported as
``False`` / an error string so a notification can't break a search.
"""

from __future__ import annotations

import imaplib
import logging
import mimetypes
import os
import re
import smtplib
import ssl
import time
import urllib.parse
import webbrowser
from email.message import EmailMessage
from pathlib import Path

logger = logging.getLogger(__name__)

_DRAFTS_FLAG = re.compile(rb"\\Drafts\b", re.IGNORECASE)
_LIST_NAME = re.compile(rb'"([^"]+)"\s*$|(\S+)\s*$')
# Browsers and Gmail cut very long URLs; the body is trimmed to stay under this.
_MAX_COMPOSE_URL = 7500


def _credentials() -> tuple[str, str]:
    return os.getenv("EMAIL_SENDER", "").strip(), os.getenv("EMAIL_PASSWORD", "").strip()


def email_configured() -> bool:
    return all(_credentials())


def send_email(to: str, subject: str, text: str, *, html: str | None = None, timeout: float = 30.0) -> bool:
    sender, password = _credentials()
    if not sender or not password:
        logger.warning("Email not configured (EMAIL_SENDER / EMAIL_PASSWORD missing); email skipped")
        return False
    host = os.getenv("SMTP_HOST", "").strip() or "smtp.gmail.com"
    try:
        port = int(os.getenv("SMTP_PORT", "").strip() or "587")
    except ValueError:
        port = 587

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")

    context = ssl.create_default_context()
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=timeout, context=context) as smtp:
                smtp.login(sender, password)
                smtp.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=timeout) as smtp:
                smtp.starttls(context=context)
                smtp.login(sender, password)
                smtp.send_message(msg)
    except smtplib.SMTPServerDisconnected as exc:
        # Gmail drops the connection at login (instead of a 535) for a password it won't take.
        logger.error(
            "Email to %s failed: %s. If this happens at login, EMAIL_PASSWORD is likely not a "
            "16-letter Gmail App Password for %s", to, exc, sender,
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("Email to %s failed: %s", to, exc)
        return False
    return True


def _drafts_folder(imap: imaplib.IMAP4_SSL) -> str:
    """The folder flagged ``\\Drafts`` (Gmail localizes its name), else Gmail's default."""
    status, lines = imap.list()
    if status == "OK":
        for line in lines or []:
            if isinstance(line, bytes) and _DRAFTS_FLAG.search(line):
                m = _LIST_NAME.search(line)
                if m:
                    return (m.group(1) or m.group(2)).decode("utf-8", "replace")
    return "[Gmail]/Drafts"


def save_draft(
    to: str, subject: str, text: str, *, attachments: list[Path] | None = None, timeout: float = 30.0
) -> str | None:
    """Save a draft addressed to ``to`` in the ``EMAIL_SENDER`` mailbox; returns None or the error."""
    user, password = _credentials()
    if not user or not password:
        logger.warning("Draft mailbox not configured (EMAIL_SENDER / EMAIL_PASSWORD missing)")
        return "draft mailbox not configured"
    host = os.getenv("IMAP_HOST", "").strip() or "imap.gmail.com"

    msg = EmailMessage()
    msg["From"] = user
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    for path in attachments or []:
        if not path.is_file():
            continue
        ctype, _ = mimetypes.guess_type(path.name)
        maintype, subtype = (ctype or "application/octet-stream").split("/", 1)
        msg.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)

    try:
        with imaplib.IMAP4_SSL(host, timeout=timeout, ssl_context=ssl.create_default_context()) as imap:
            imap.login(user, password)
            folder = _drafts_folder(imap)
            status, data = imap.append(
                f'"{folder}"', r"(\Draft)", imaplib.Time2Internaldate(time.time()), msg.as_bytes()
            )
            if status != "OK":
                raise RuntimeError(f"APPEND to {folder} failed: {data}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Saving draft to %s failed: %s", to, exc)
        return str(exc) or exc.__class__.__name__
    return None


def gmail_compose_url(to: str, subject: str, body: str) -> str:
    """Prefilled Gmail compose link, opened in the ``EMAIL_SENDER`` account when signed in to several."""
    params = {"view": "cm", "fs": "1", "tf": "1", "to": to, "su": subject, "body": body}
    sender = os.getenv("EMAIL_SENDER", "").strip()
    if sender:
        params["authuser"] = sender
    url = "https://mail.google.com/mail/?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    while len(url) > _MAX_COMPOSE_URL and len(params["body"]) > 200:
        params["body"] = params["body"][: int(len(params["body"]) * 0.9)].rstrip() + "\n…"
        url = "https://mail.google.com/mail/?" + urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return url


def open_gmail_compose(to: str, subject: str, body: str) -> str | None:
    """Open a prefilled Gmail compose window in this computer's browser; returns None or the error."""
    try:
        if not webbrowser.open(gmail_compose_url(to, subject, body), new=2):
            return "no browser available to open Gmail"
    except Exception as exc:  # noqa: BLE001
        logger.error("Opening Gmail compose for %s failed: %s", to, exc)
        return str(exc) or exc.__class__.__name__
    return None
