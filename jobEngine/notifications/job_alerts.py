"""Email the jobs of a finished run whose match score is above a threshold.

Called after a run's jobs are saved (manual search and scheduled run alike).
Each job is mailed once: a successful send stamps ``mailed_at`` on its saved-jobs
document, and jobs already stamped are skipped by later runs.

When a mailed job's posting lists an HR / recruiter email, an application email
is generated for it with the Autonomous Application Agent and saved as a draft
(``hr_drafts.py``); the alert says so, highlighted. The draft outcome is stored on
the saved job (``hr_draft``) so a retried alert reuses it instead of drafting twice.

Environment (defaults in brackets):
    JOB_MAIL_THRESHOLD_SCORE [unset = off]  mail jobs scoring strictly above this
    JOB_MAIL_TO              [virenderfullstack@gmail.com]
    JOB_CRON_SAVED_JOBS_URL  [https://ai.ritishka.in/saved-jobs] link in the email
plus the SMTP settings read by ``email.send_email`` and the draft settings in ``hr_drafts.py``.
"""

from __future__ import annotations

import html
import logging
import os
from typing import Any

from ..persistence.saved_jobs import (
    mark_jobs_mailed,
    record_hr_draft,
    saved_job_key,
    stored_hr_drafts,
    unmailed_job_keys,
)
from .email import email_configured, send_email
from .hr_drafts import create_hr_draft, find_hr_email, hr_drafts_enabled, max_drafts_per_run

logger = logging.getLogger(__name__)

_DEFAULT_TO = "virenderfullstack@gmail.com"
_DEFAULT_SAVED_JOBS_URL = "https://ai.ritishka.in/saved-jobs"
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


def _draft_note(draft: dict[str, Any] | None) -> tuple[str, str]:
    """(plain text, html) for a job's HR draft outcome; empty when no HR email was found."""
    if not draft:
        return "", ""
    email = str(draft.get("hr_email") or "")
    extra = ""
    if draft.get("saved"):
        text = f"Draft email is saved with HR email id as: {email}"
        if draft.get("method") == "gmail_compose":
            extra = "via Gmail compose in the browser; attach the resume before sending"
    else:
        text = f"HR email id found: {email} (draft could not be saved: {draft.get('error') or 'unknown error'})"
    note_html = f'<div style="margin-top:4px"><span style="{_HIGHLIGHT}">{html.escape(text)}</span>'
    if extra:
        note_html += f" <small>({html.escape(extra)})</small>"
        text += f" ({extra})"
    return text, note_html + "</div>"


def _compose(
    jobs: list[dict[str, Any]], threshold: float, drafts: dict[str, dict[str, Any]]
) -> tuple[str, str, str]:
    saved_url = os.getenv("JOB_CRON_SAVED_JOBS_URL", "").strip() or _DEFAULT_SAVED_JOBS_URL
    if len(jobs) == 1:
        job = jobs[0].get("job") or {}
        subject = f"{round(_score(jobs[0]))}% match: {job.get('title') or 'New job'} at {job.get('company') or ''}".strip()
    else:
        subject = f"{len(jobs)} new jobs above {threshold:g}% match"
    saved_drafts = sum(1 for d in drafts.values() if d.get("saved"))
    if saved_drafts:
        subject += f" ({saved_drafts} HR draft{'s' if saved_drafts != 1 else ''} saved)"

    text_parts, html_items = [], []
    for matched in jobs:
        heading, details, link = _line(matched)
        note_text, note_html = _draft_note(drafts.get(saved_job_key(matched) or ""))
        text_parts.append("\n".join(p for p in (heading, details, link, f">>> {note_text} <<<" if note_text else "") if p))
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


def _hr_drafts(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Draft outcome per saved key, for the jobs whose posting lists an HR email."""
    with_email = {}
    for matched in jobs:
        key = saved_job_key(matched)
        hr_email = find_hr_email(matched.get("job") or {})
        if key and hr_email:
            with_email[key] = (matched, hr_email)
    if not with_email:
        return {}

    drafts = stored_hr_drafts(list(with_email))  # from an earlier run whose alert failed to send
    budget = max_drafts_per_run()
    for key, (matched, hr_email) in with_email.items():
        if key in drafts:
            continue
        if not hr_drafts_enabled() or budget <= 0:
            drafts[key] = {"hr_email": hr_email, "saved": False, "error": "drafting skipped (disabled or per-run cap reached)"}
            continue
        budget -= 1
        outcome = create_hr_draft(matched.get("job") or {}, hr_email)
        logger.info("HR draft for %s (%s): %s", key, hr_email, "saved" if outcome["saved"] else outcome["error"])
        drafts[key] = outcome
        record_hr_draft(key, outcome)
    return drafts


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
        # Checked before drafting: an alert that can't be sent would retry (and draft) every run.
        logger.warning("%s job(s) above %s%% not mailed: EMAIL_SENDER / EMAIL_PASSWORD missing", len(jobs), threshold)
        return 0

    drafts = _hr_drafts(jobs)
    to = os.getenv("JOB_MAIL_TO", "").strip() or _DEFAULT_TO
    subject, text, body = _compose(jobs, threshold, drafts)
    if not send_email(to, subject, text, html=body):
        return 0  # not stamped: the next run retries (stored drafts are reused)
    mark_jobs_mailed([saved_job_key(j) for j in jobs])
    logger.info("Mailed %s job(s) above %s%% to %s (run %s)", len(jobs), threshold, to, result.get("run_id"))
    return len(jobs)
