"""Draft an application email for a job whose posting lists an HR / recruiter address.

Runs the Autonomous Application Agent (``jobApplyPro.run_recruitment_pipeline``)
with the job description and the address found in it, the same as pasting both
into its page, then saves the generated subject + body as a draft addressed to
that HR email in the ``EMAIL_SENDER`` mailbox (see ``create_hr_draft``).

Environment (defaults in brackets):
    JOB_HR_DRAFTS_ENABLED       [true]  false = never run the agent
    JOB_HR_DRAFTS_MAX_PER_RUN   [5]     cap on agent runs per search (each is several LLM calls)
plus the IMAP settings read by ``email.save_draft``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re
from typing import Any

from .email import open_gmail_compose, save_draft

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# Addresses that are never a person to apply to.
_IGNORED = re.compile(
    r"(no-?reply|do-?not-?reply|noreply|notifications?|privacy|abuse|support|unsubscribe|example\.(com|org))",
    re.IGNORECASE,
)
# Preferred when a posting lists several addresses.
_HR_HINT = re.compile(r"(hr|career|job|recruit|talent|hiring|people|resume|cv|apply)", re.IGNORECASE)
# File names like logo@2x.png match the address pattern.
_FILE_SUFFIX = re.compile(r"\.(png|jpe?g|gif|svg|webp)$", re.IGNORECASE)


def hr_drafts_enabled() -> bool:
    return os.getenv("JOB_HR_DRAFTS_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


def max_drafts_per_run() -> int:
    try:
        return max(0, int(os.getenv("JOB_HR_DRAFTS_MAX_PER_RUN", "5").strip() or "5"))
    except ValueError:
        return 5


def _job_text(job: dict[str, Any]) -> str:
    return "\n".join(
        [str(job.get("description") or "")]
        + [str(r) for r in job.get("requirements") or []]
        + [str(r) for r in job.get("responsibilities") or []]
    )


def find_hr_email(job: dict[str, Any]) -> str | None:
    """The HR / recruiter address in the posting, if any (an HR-looking one wins)."""
    found: list[str] = []
    for match in _EMAIL_RE.findall(_job_text(job)):
        email = match.strip(".").lower()
        if _IGNORED.search(email) or _FILE_SUFFIX.search(email) or email in found:
            continue
        found.append(email)
    if not found:
        return None
    return next((e for e in found if _HR_HINT.search(e.split("@")[0])), found[0])


def job_description_text(job: dict[str, Any]) -> str:
    """What gets pasted into the agent's "Job description" field."""
    head = [
        f"Job title: {job.get('title') or ''}",
        f"Company: {job.get('company') or ''}",
        f"Location: {job.get('location') or ''}",
    ]
    if job.get("experience_required"):
        head.append(f"Experience: {job['experience_required']}")
    parts = ["\n".join(head), str(job.get("description") or "")]
    if job.get("requirements"):
        parts.append("Requirements:\n" + "\n".join(f"- {r}" for r in job["requirements"]))
    if job.get("responsibilities"):
        parts.append("Responsibilities:\n" + "\n".join(f"- {r}" for r in job["responsibilities"]))
    return "\n\n".join(p for p in parts if p.strip())


def _run_coroutine(coro):
    """``asyncio.run`` from sync code, also when this thread already runs a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def create_hr_draft(job: dict[str, Any], hr_email: str) -> dict[str, Any]:
    """Generate the email with the Autonomous Application Agent and save it as a draft.

    The draft is saved over IMAP into the ``EMAIL_SENDER`` mailbox (resume attached).
    If that fails (e.g. Gmail refuses the password), it falls back to opening a prefilled
    Gmail compose window in this computer's browser, which Gmail keeps in Drafts (no
    attachment there). Returns ``{"hr_email", "saved", "method", "subject", "error"}``;
    never raises.
    """
    outcome: dict[str, Any] = {"hr_email": hr_email, "saved": False, "method": "", "subject": "", "error": None}
    try:
        # Imported here: jobApplyPro loads the OpenAI Agents SDK and .env on import.
        from jobApplyPro import run_recruitment_pipeline
        from jobApplyPro.config import PROFILE_DIR, RESUME_FILENAME

        result = _run_coroutine(
            run_recruitment_pipeline(
                job_requirement_text=job_description_text(job),
                hiring_manager_email=hr_email,
                open_email_client=False,
            )
        )
        if not result:
            raise RuntimeError("the application agent returned no email")
        outcome["subject"] = result["subject"]
        imap_error = save_draft(hr_email, result["subject"], result["body"], attachments=[PROFILE_DIR / RESUME_FILENAME])
        if imap_error is None:
            outcome.update(saved=True, method="imap")
            return outcome
        compose_error = open_gmail_compose(hr_email, result["subject"], result["body"])
        if compose_error is None:
            outcome.update(saved=True, method="gmail_compose")
        else:
            outcome["error"] = f"IMAP: {imap_error}; Gmail compose: {compose_error}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("HR draft for %s failed", hr_email)
        outcome["error"] = str(exc) or exc.__class__.__name__
    return outcome
