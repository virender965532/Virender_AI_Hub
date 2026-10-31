"""High-score job email: threshold, once-per-job, retry after a failed send, HR email + draft link."""

from __future__ import annotations

import pytest

from jobEngine.notifications import hr_drafts, job_alerts


def _matched(job_id: str, score: float, **extra):
    return {
        "match_score": score,
        "job": {"source": "linkedin", "source_job_id": job_id, "title": f"Job {job_id}", "company": "Acme",
                "application_url": f"https://example.com/{job_id}"},
        **extra,
    }


@pytest.fixture
def store(monkeypatch):
    """In-memory stand-in for the saved-jobs ``mailed_at`` / ``hr_contact`` fields, plus a captured outbox."""
    mailed: set[str] = set()
    outbox: list[dict] = []
    state = {"send_ok": True, "llm": None, "llm_calls": [], "recorded": {}}

    monkeypatch.setattr(job_alerts, "unmailed_job_keys", lambda keys: {k for k in keys if k not in mailed})
    monkeypatch.setattr(job_alerts, "mark_jobs_mailed", lambda keys: mailed.update(keys))
    monkeypatch.setattr(job_alerts, "email_configured", lambda: True)
    monkeypatch.setattr(
        job_alerts, "stored_hr_contacts", lambda keys: {k: v for k, v in state["recorded"].items() if k in keys}
    )
    monkeypatch.setattr(job_alerts, "record_hr_contact", lambda key, c: state["recorded"].__setitem__(key, c))

    def fake_llm(job, text):
        state["llm_calls"].append(text)
        return state["llm"]

    monkeypatch.setattr(hr_drafts, "_llm_extract", fake_llm)

    def fake_send(to, subject, text, *, html=None):
        outbox.append({"to": to, "subject": subject, "text": text, "html": html})
        return state["send_ok"]

    monkeypatch.setattr(job_alerts, "send_email", fake_send)
    monkeypatch.setenv("JOB_MAIL_THRESHOLD_SCORE", "80")
    monkeypatch.delenv("JOB_MAIL_TO", raising=False)
    monkeypatch.delenv("JOB_HR_DRAFT_URL", raising=False)
    return mailed, outbox, state


def test_only_jobs_strictly_above_threshold_are_mailed(store):
    mailed, outbox, _ = store
    result = {"jobs": [_matched("a", 86), _matched("b", 80), _matched("c", 95), _matched("d", 99, rejected=True)]}
    assert job_alerts.mail_high_score_jobs(result) == 2
    assert outbox[0]["to"] == "virenderfullstack@gmail.com"
    assert mailed == {"linkedin:a", "linkedin:c"}
    # Highest score first.
    assert outbox[0]["text"].index("Job c") < outbox[0]["text"].index("Job a")


def test_a_job_is_mailed_once(store):
    _, outbox, _ = store
    result = {"jobs": [_matched("a", 90)]}
    assert job_alerts.mail_high_score_jobs(result) == 1
    assert job_alerts.mail_high_score_jobs(result) == 0
    assert len(outbox) == 1


def test_failed_send_is_retried_next_run(store):
    mailed, outbox, state = store
    state["send_ok"] = False
    result = {"jobs": [_matched("a", 90)]}
    assert job_alerts.mail_high_score_jobs(result) == 0
    assert not mailed
    state["send_ok"] = True
    assert job_alerts.mail_high_score_jobs(result) == 1
    assert len(outbox) == 2


def test_no_threshold_means_no_email(store, monkeypatch):
    _, outbox, _ = store
    monkeypatch.delenv("JOB_MAIL_THRESHOLD_SCORE")
    assert job_alerts.mail_high_score_jobs({"jobs": [_matched("a", 100)]}) == 0
    assert not outbox


def _with_hr(job_id: str, score: float, description: str):
    m = _matched(job_id, score)
    m["job"]["description"] = description
    return m


def test_hr_sentence_and_draft_link_are_shown(store):
    _, outbox, state = store
    result = {"jobs": [_with_hr("a", 90, "We are hiring. Send your CV to careers@acme.io or call us. Thanks."), _matched("b", 85)]}
    assert job_alerts.mail_high_score_jobs(result) == 2
    mail = outbox[0]
    assert 'HR email in the posting: "Send your CV to careers@acme.io or call us."' in mail["text"]
    assert "Create draft email: https://ai.ritishka.in/hr-draft/linkedin:a" in mail["text"]
    assert "Draft email is saved" not in mail["text"]
    assert "background:#fff3a3" in mail["html"] and "<u>careers@acme.io</u>" in mail["html"]
    assert 'href="https://ai.ritishka.in/hr-draft/linkedin:a">Create draft email</a>' in mail["html"]
    assert "1 with HR email" in mail["subject"]
    assert state["recorded"]["linkedin:a"] == {
        "email_id": "careers@acme.io", "is_hr_mail": True, "reason": "Send your CV to careers@acme.io or call us.",
    }


def test_llm_decides_whether_the_address_is_for_applications(store):
    _, outbox, state = store
    state["llm"] = hr_drafts.HREmail(email_id="legal@acme.io", is_hr_mail=False, reason="x")
    job = _with_hr("a", 90, "To report fraud write to legal@acme.io.")
    assert job_alerts.mail_high_score_jobs({"jobs": [job]}) == 1
    assert len(state["llm_calls"]) == 1
    assert "Create draft email" not in outbox[0]["text"]
    assert state["recorded"]["linkedin:a"]["reason"] == "To report fraud write to legal@acme.io."


def test_llm_address_not_in_posting_is_ignored(store):
    _, outbox, state = store
    state["llm"] = hr_drafts.HREmail(email_id="made.up@acme.io", is_hr_mail=True, reason="made up")
    assert job_alerts.mail_high_score_jobs({"jobs": [_with_hr("a", 90, "Apply: jobs@acme.io")]}) == 1
    assert state["recorded"]["linkedin:a"]["email_id"] == "jobs@acme.io"


def test_retried_alert_does_not_extract_twice(store):
    _, outbox, state = store
    state["send_ok"] = False
    result = {"jobs": [_with_hr("a", 90, "Apply: hr@acme.io")]}
    assert job_alerts.mail_high_score_jobs(result) == 0
    state["send_ok"] = True
    assert job_alerts.mail_high_score_jobs(result) == 1
    assert len(state["llm_calls"]) == 1
    assert "hr-draft/linkedin:a" in outbox[-1]["text"]


def test_no_llm_call_without_an_address(store):
    _, _, state = store
    assert job_alerts.mail_high_score_jobs({"jobs": [_with_hr("a", 90, "No contact here.")]}) == 1
    assert not state["llm_calls"] and not state["recorded"]


def test_no_extraction_when_alert_cannot_be_sent(store, monkeypatch):
    _, outbox, state = store
    monkeypatch.setattr(job_alerts, "email_configured", lambda: False)
    assert job_alerts.mail_high_score_jobs({"jobs": [_with_hr("a", 90, "hr@acme.io")]}) == 0
    assert not state["llm_calls"] and not outbox


@pytest.mark.parametrize(
    ("text", "email", "expected"),
    [
        ("Intro. Mail jobs@acme.io today! Bye.", "jobs@acme.io", "Mail jobs@acme.io today!"),
        ("line one\nContact: hr@acme.io\nline three", "hr@acme.io", "Contact: hr@acme.io"),
        ("Send to hr@acme.io.", "hr@acme.io", "Send to hr@acme.io."),
        ("nothing", "hr@acme.io", ""),
    ],
)
def test_sentence_with(text, email, expected):
    assert hr_drafts.sentence_with(text, email) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Contact noreply@acme.io or jobs@acme.io", "jobs@acme.io"),
        ("Questions: jane.doe@acme.io; HR: talent@acme.io.", "talent@acme.io"),
        ("Reach jane.doe@acme.io.", "jane.doe@acme.io"),
        ("<img src='logo@2x.png'> no address here", None),
        ("Nothing to see", None),
    ],
)
def test_find_hr_email(text, expected):
    assert hr_drafts.find_hr_email({"description": text}) == expected


def test_hr_draft_falls_back_to_gmail_compose(monkeypatch):
    import sys
    import types
    from pathlib import Path

    from jobEngine.notifications import hr_drafts as hd

    async def fake_pipeline(**kwargs):
        assert kwargs["open_email_client"] is False
        return {"subject": "Hi", "body": "Body"}

    # Stub modules: importing the real jobApplyPro loads .env (override=True) into os.environ.
    monkeypatch.setitem(sys.modules, "jobApplyPro", types.SimpleNamespace(run_recruitment_pipeline=fake_pipeline))
    monkeypatch.setitem(
        sys.modules, "jobApplyPro.config", types.SimpleNamespace(PROFILE_DIR=Path("."), RESUME_FILENAME="none.pdf")
    )
    monkeypatch.setattr(hd, "save_draft", lambda *a, **k: "login refused")
    out = hd.create_hr_draft({"title": "SE", "description": "x"}, "hr@acme.io")
    assert not out["saved"] and out["method"] == "gmail_compose" and out["error"] == "IMAP: login refused"
    assert out["subject"] == "Hi" and out["body"] == "Body"
    assert "to=hr%40acme.io" in out["compose_url"]


def test_compose_url_targets_sender_account(monkeypatch):
    from jobEngine.notifications.email import gmail_compose_url

    monkeypatch.setenv("EMAIL_SENDER", "me@gmail.com")
    url = gmail_compose_url("hr@acme.io", "Sub & ject", "x" * 20000)
    assert "authuser=me%40gmail.com" in url and "to=hr%40acme.io" in url and "Sub%20%26%20ject" in url
    assert len(url) <= 7500
