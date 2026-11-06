"""Email the jobs of a finished run whose match score is above a threshold.

Called after a run's jobs are saved (manual search and scheduled run alike).
Each job is mailed once: a successful send stamps ``mailed_at`` on its saved-jobs
document, and jobs already stamped are skipped by later runs.

When a mailed job's posting lists an email address, the LLM reads it into an
``HREmail`` (``hr_drafts.extract_hr_email``). If it is an HR / recruiter address the
alert shows, highlighted, the posting's sentence quoting it and a "Create draft
email" link; opening that link runs the Autonomous Application Agent with the job
description and the profile and saves the draft (``/hr-draft/<saved key>``). The
extraction is stored on the saved job (``hr_contact``) so a retried alert reuses it.

Environment (defaults in brackets):
    JOB_MAIL_THRESHOLD_SCORE [unset = off]  mail jobs scoring strictly above this
    JOB_MAIL_TO              [virenderfullstack@gmail.com]
    JOB_CRON_SAVED_JOBS_URL  [https://ai.ritishka.in/saved-jobs] link in the email
    JOB_HR_DRAFT_URL         [https://ai.ritishka.in/hr-draft] "Create draft email" link
plus the SMTP settings read by ``email.send_email``.
"""

from __future__ import annotations

import html
import logging
import os
from typing import Any
from urllib.parse import quote

from ..persistence.saved_jobs import (
    mark_jobs_mailed,
    record_hr_contact,
    saved_job_key,
    stored_hr_contacts,
    unmailed_job_keys,
)
from .email import email_configured, send_email
from .hr_drafts import extract_hr_email, find_hr_email

logger = logging.getLogger(__name__)

_DEFAULT_TO = "virenderfullstack@gmail.com"
_DEFAULT_SAVED_JOBS_URL = "https://ai.ritishka.in/saved-jobs"
_DEFAULT_HR_DRAFT_URL = "https://ai.ritishka.in/hr-draft"
_HIGHLIGHT = "background:#fff3a3;padding:2px 6px;border-radius:3px;font-weight:bold"


def mail_threshold() -> float | None:
    raw = (os.getenv("JOB_MAIL_THRESHOLD_SCORE") or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("JOB_MAIL_THRESHOLD_SCORE=%r is not a number; job emails are off", raw)
        return None


def _score(matched: dict[str, Any]) -> float:
    try:
        return float(matched.get("match_score") or 0)
    except (TypeError, ValueError):
        return 0.0


def _line(matched: dict[str, Any]) -> tuple[str, str, str]:
    """(heading, details, link) for one job."""
    job = matched.get("job") or {}
    heading = f"{round(_score(matched))}% · {job.get('title') or 'Untitled'} at {job.get('company') or 'Unknown company'}"
    details = " · ".join(p for p in (job.get("location"), job.get("remote_type"), job.get("source")) if p)
    b = matched.get("score_breakdown") or {}
    if b:
        details += (" — " if details else "") + (
            f"skills {round(b.get('skills') or 0)}, experience {round(b.get('experience') or 0)}, "
            f"keywords {round(b.get('keywords') or 0)}"
        )
    return heading, details, str(job.get("application_url") or job.get("source_url") or "")


def hr_draft_link(key: str) -> str:
    base = os.getenv("JOB_HR_DRAFT_URL", "").strip() or _DEFAULT_HR_DRAFT_URL
    return f"{base.rstrip('/')}/{quote(key, safe=':')}"


def _hr_note(key: str, contact: dict[str, Any] | None) -> tuple[str, str]:
    """(plain text, html) for a job's HR email; empty unless the posting lists an HR address."""
    if not contact or not contact.get("is_hr_mail") or not contact.get("email_id"):
        return "", ""
    email = str(contact["email_id"])
    sentence = str(contact.get("reason") or "") or f"HR email id: {email}"
    link = hr_draft_link(key)
    text = f'HR email in the posting: "{sentence}"\nCreate draft email: {link}'
    # The sentence, highlighted, with the address underlined.
    quoted = html.escape(sentence)
    at = sentence.lower().find(email.lower())
    if at >= 0:
        quoted = (
            html.escape(sentence[:at]) + f"<u>{html.escape(sentence[at:at + len(email)])}</u>"
            + html.escape(sentence[at + len(email):])
        )
    note_html = (
        f'<div style="margin-top:4px"><span style="{_HIGHLIGHT}">&ldquo;{quoted}&rdquo;</span>'
        f' &nbsp;<a href="{html.escape(link, quote=True)}">Create draft email</a></div>'
    )
    return text, note_html


def _compose(
    jobs: list[dict[str, Any]], threshold: float, contacts: dict[str, dict[str, Any]]
) -> tuple[str, str, str]:
    saved_url = os.getenv("JOB_CRON_SAVED_JOBS_URL", "").strip() or _DEFAULT_SAVED_JOBS_URL
    if len(jobs) == 1:
        job = jobs[0].get("job") or {}
        subject = f"{round(_score(jobs[0]))}% match: {job.get('title') or 'New job'} at {job.get('company') or ''}".strip()
    else:
        subject = f"{len(jobs)} new jobs above {threshold:g}% match"
    with_hr = sum(1 for k in (saved_job_key(j) for j in jobs) if _hr_note(k or "", contacts.get(k or ""))[0])
    if with_hr:
        subject += f" ({with_hr} with HR email)"

    text_parts, html_items = [], []
    for matched in jobs:
        heading, details, link = _line(matched)
        key = saved_job_key(matched) or ""
        note_text, note_html = _hr_note(key, contacts.get(key))
        text_parts.append("\n".join(p for p in (heading, details, link, note_text) if p))
        title = html.escape(heading)
        if link:
            title = f'<a href="{html.escape(link, quote=True)}">{title}</a>'
        html_items.append(
            f"<li style=\"margin-bottom:10px\"><b>{title}</b>"
            + (f"<br><small>{html.escape(details)}</small>" if details else "")
            + note_html
            + "</li>"
        )

    text = "\n\n".join(text_parts) + f"\n\nAll saved jobs: {saved_url}\n"
    body = (
        f"<p>Jobs scoring above {threshold:g}%:</p><ul>{''.join(html_items)}</ul>"
        f'<p><a href="{html.escape(saved_url, quote=True)}">Open saved jobs</a></p>'
    )
    return subject, text, body


def _hr_contacts(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """``HREmail`` (as a dict) per saved key, for the jobs whose posting contains an email address."""
    with_email = {}
    for matched in jobs:
        key = saved_job_key(matched)
        if key and find_hr_email(matched.get("job") or {}):
            with_email[key] = matched
    if not with_email:
        return {}

    contacts = stored_hr_contacts(list(with_email))  # from an earlier run whose alert failed to send
    for key, matched in with_email.items():
        if key in contacts:
            continue
        contact = extract_hr_email(matched.get("job") or {}).model_dump()
        logger.info("HR email for %s: %s (is_hr_mail=%s)", key, contact["email_id"], contact["is_hr_mail"])
        contacts[key] = contact
        record_hr_contact(key, contact)
    return contacts


def mail_high_score_jobs(result: dict[str, Any]) -> int:
    """Email this run's not-yet-mailed jobs scoring above the threshold; returns how many were mailed."""
    threshold = mail_threshold()
    if threshold is None:
        return 0
    candidates: dict[str, dict[str, Any]] = {}
    for matched in result.get("jobs") or []:
        if not isinstance(matched, dict) or matched.get("rejected") or _score(matched) <= threshold:
            continue
        key = saved_job_key(matched)
        if key:
            candidates[key] = matched
    if not candidates:
        return 0

    pending = unmailed_job_keys(list(candidates))
    jobs = sorted((candidates[k] for k in pending), key=_score, reverse=True)
    if not jobs:
        return 0
    if not email_configured():
        # Checked before the HR email extraction: an alert that can't be sent would retry it every run.
        logger.warning("%s job(s) above %s%% not mailed: EMAIL_SENDER / EMAIL_PASSWORD missing", len(jobs), threshold)
        return 0

    contacts = _hr_contacts(jobs)
    to = os.getenv("JOB_MAIL_TO", "").strip() or _DEFAULT_TO
    subject, text, body = _compose(jobs, threshold, contacts)
    if not send_email(to, subject, text, html=body):
        return 0  # not stamped: the next run retries (stored extractions are reused)
    mark_jobs_mailed([saved_job_key(j) for j in jobs])
    logger.info("Mailed %s job(s) above %s%% to %s (run %s)", len(jobs), threshold, to, result.get("run_id"))
    return len(jobs)
