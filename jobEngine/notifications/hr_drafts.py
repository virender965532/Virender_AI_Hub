"""HR / recruiter address of a job posting, and the application draft sent to it.

``extract_hr_email`` reads the posting with the job engine's LLM into ``HREmail``
(the address, whether it is meant for applications, and the sentence quoting it);
without an LLM it falls back to the address pattern below. The job alert email
shows that sentence with a "Create draft email" link (``/hr-draft/<saved key>``).

Clicking the link runs ``create_hr_draft``: the Autonomous Application Agent
(``jobApplyPro.run_recruitment_pipeline``) with the job description, the address
and the candidate profile, the same as pasting both into its page; the generated
subject + body are saved as a draft addressed to that HR email in the
``EMAIL_SENDER`` mailbox (IMAP settings read by ``email.save_draft``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .email import gmail_compose_url, save_draft

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


class HREmail(BaseModel):
    """The application address found in a job posting."""

    model_config = ConfigDict(extra="forbid")

    email_id: str = Field(default="", description="The HR / recruiter email address, copied exactly. Empty if none.")
    is_hr_mail: bool = Field(
        default=False, description="True if candidates are asked to email this address to apply, else false."
    )
    reason: str = Field(
        default="", description="The sentence from the posting containing the email address, copied verbatim."
    )


def _job_text(job: dict[str, Any]) -> str:
    return "\n".join(
        [str(job.get("description") or "")]
        + [str(r) for r in job.get("requirements") or []]
        + [str(r) for r in job.get("responsibilities") or []]
    )


def _candidate_emails(text: str) -> list[str]:
    found: list[str] = []
    for match in _EMAIL_RE.findall(text):
        email = match.strip(".").lower()
        if _IGNORED.search(email) or _FILE_SUFFIX.search(email) or email in found:
            continue
        found.append(email)
    return found


def find_hr_email(job: dict[str, Any]) -> str | None:
    """The HR / recruiter address in the posting, if any (an HR-looking one wins)."""
    found = _candidate_emails(_job_text(job))
    if not found:
        return None
    return next((e for e in found if _HR_HINT.search(e.split("@")[0])), found[0])


_SENTENCE_ENDS = (". ", "! ", "? ")


def sentence_with(text: str, email: str) -> str:
    """The sentence (or line) of ``text`` that contains ``email``; empty if it isn't there."""
    if not email:
        return ""
    for line in text.splitlines():
        at = line.lower().find(email.lower())
        if at < 0:
            continue
        # Split at ". " and the like — never at the dots inside the address.
        start = max(line.rfind(sep, 0, at) for sep in _SENTENCE_ENDS)
        start = 0 if start < 0 else start + 2
        ends = [i for i in (line.find(sep, at + len(email)) for sep in _SENTENCE_ENDS) if i >= 0]
        end = min(ends) + 1 if ends else len(line)
        return " ".join(line[start:end].split())[:500]
    return ""


def _llm_extract(job: dict[str, Any], text: str) -> HREmail | None:
    """``HREmail`` from the job engine's LLM; None when no LLM is configured or the call fails."""
    from llm_provider import get_chat_model, llm_configured

    from ..config.settings import get_settings
    from ..guardrails.prompt_injection import wrap_untrusted_content
    from ..matching.llm_matcher import _load_prompt

    settings = get_settings()
    if not (settings.llm_enabled and llm_configured(settings.openai_api_key)):
        return None
    user = "\n".join(
        [f"Title: {job.get('title') or ''}", f"Company: {job.get('company') or ''}", "Job text:", wrap_untrusted_content(text)]
    )
    try:
        llm = get_chat_model(
            settings.openai_model,
            openai_api_key=settings.openai_api_key,
            timeout=settings.openai_timeout,
            temperature=0,
        ).with_structured_output(HREmail)
        result = llm.invoke(
            [{"role": "system", "content": _load_prompt("hr_email_system.txt")}, {"role": "user", "content": user}]
        )
    except Exception:  # noqa: BLE001
        logger.exception("HR email extraction failed for %r", job.get("title"))
        return None
    return result if isinstance(result, HREmail) else None


def extract_hr_email(job: dict[str, Any]) -> HREmail:
    """The posting's application address as ``HREmail``.

    The LLM is asked only when the text contains an address at all, and its answer is
    kept only if that address really occurs there; ``reason`` is then the posting's own
    sentence quoting it. Without an LLM answer the address pattern decides.
    """
    text = _job_text(job)
    candidates = _candidate_emails(text)
    if not candidates:
        return HREmail(reason="No email address in the job description.")
    review = _llm_extract(job, text)
    if review is not None:
        email = review.email_id.strip().strip(".").lower()
        if email in candidates:
            return HREmail(email_id=email, is_hr_mail=review.is_hr_mail, reason=sentence_with(text, email) or review.reason)
        if not email:
            return HREmail(reason=review.reason or "No application email in the job description.")
    email = find_hr_email(job) or ""
    return HREmail(email_id=email, is_hr_mail=bool(email), reason=sentence_with(text, email))


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
    If that fails (e.g. Gmail refuses the password), ``compose_url`` is a prefilled Gmail
    compose link for the browser that asked, which Gmail keeps in Drafts once opened (no
    attachment there). Returns ``{"hr_email", "saved", "method", "subject", "body",
    "compose_url", "error"}``; never raises.
    """
    outcome: dict[str, Any] = {
        "hr_email": hr_email, "saved": False, "method": "", "subject": "", "body": "", "compose_url": "", "error": None,
    }
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
        outcome.update(subject=result["subject"], body=result["body"])
        imap_error = save_draft(hr_email, result["subject"], result["body"], attachments=[PROFILE_DIR / RESUME_FILENAME])
        if imap_error is None:
            outcome.update(saved=True, method="imap")
            return outcome
        outcome.update(
            method="gmail_compose",
            compose_url=gmail_compose_url(hr_email, result["subject"], result["body"]),
            error=f"IMAP: {imap_error}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("HR draft for %s failed", hr_email)
        outcome["error"] = str(exc) or exc.__class__.__name__
    return outcome
