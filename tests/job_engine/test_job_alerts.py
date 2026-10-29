"""High-score job email: threshold, once-per-job, retry after a failed send, HR drafts."""

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
    """In-memory stand-in for the saved-jobs ``mailed_at`` / ``hr_draft`` fields, plus a captured outbox."""
    mailed: set[str] = set()
    outbox: list[dict] = []
    state = {"send_ok": True, "draft_ok": True, "agent_calls": [], "recorded": {}}

    monkeypatch.setattr(job_alerts, "unmailed_job_keys", lambda keys: {k for k in keys if k not in mailed})
    monkeypatch.setattr(job_alerts, "mark_jobs_mailed", lambda keys: mailed.update(keys))
    monkeypatch.setattr(job_alerts, "email_configured", lambda: True)
    monkeypatch.setattr(
        job_alerts, "stored_hr_drafts",
        lambda keys: {k: v for k, v in state["recorded"].items() if k in keys and v["saved"]},
    )
    monkeypatch.setattr(job_alerts, "record_hr_draft", lambda key, outcome: state["recorded"].__setitem__(key, outcome))

    def fake_draft(job, hr_email):
        state["agent_calls"].append((job.get("description"), hr_email))
        return {"hr_email": hr_email, "saved": state["draft_ok"], "subject": "S", "error": None if state["draft_ok"] else "imap down"}

    monkeypatch.setattr(job_alerts, "create_hr_draft", fake_draft)

    def fake_send(to, subject, text, *, html=None):
        outbox.append({"to": to, "subject": subject, "text": text, "html": html})
        return state["send_ok"]

    monkeypatch.setattr(job_alerts, "send_email", fake_send)
    monkeypatch.setenv("JOB_MAIL_THRESHOLD_SCORE", "80")
    monkeypatch.delenv("JOB_MAIL_TO", raising=False)
    monkeypatch.delenv("JOB_HR_DRAFTS_ENABLED", raising=False)
    monkeypatch.delenv("JOB_HR_DRAFTS_MAX_PER_RUN", raising=False)
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


def test_hr_email_in_posting_creates_a_draft_and_is_highlighted(store):
    _, outbox, state = store
    result = {"jobs": [_with_hr("a", 90, "Send your CV to careers@acme.io or call us."), _matched("b", 85)]}
    assert job_alerts.mail_high_score_jobs(result) == 2
    assert state["agent_calls"] == [("Send your CV to careers@acme.io or call us.", "careers@acme.io")]
    mail = outbox[0]
    assert "Draft email is saved with HR email id as: careers@acme.io" in mail["text"]
    assert "background:#fff3a3" in mail["html"] and "careers@acme.io" in mail["html"]
    assert "1 HR draft saved" in mail["subject"]


def test_retried_alert_does_not_draft_twice(store):
    _, outbox, state = store
    state["send_ok"] = False
    result = {"jobs": [_with_hr("a", 90, "Apply: hr@acme.io")]}
    assert job_alerts.mail_high_score_jobs(result) == 0
    state["send_ok"] = True
    assert job_alerts.mail_high_score_jobs(result) == 1
    assert len(state["agent_calls"]) == 1
    assert "Draft email is saved with HR email id as: hr@acme.io" in outbox[-1]["text"]


def test_failed_draft_is_reported_and_retried(store):
    _, outbox, state = store
    state["draft_ok"] = False
    assert job_alerts.mail_high_score_jobs({"jobs": [_with_hr("a", 90, "hr@acme.io")]}) == 1
    assert "draft could not be saved: imap down" in outbox[0]["text"]


def test_no_drafts_when_alert_cannot_be_sent(store, monkeypatch):
    _, outbox, state = store
    monkeypatch.setattr(job_alerts, "email_configured", lambda: False)
    assert job_alerts.mail_high_score_jobs({"jobs": [_with_hr("a", 90, "hr@acme.io")]}) == 0
    assert not state["agent_calls"] and not outbox


def test_draft_cap_per_run(store, monkeypatch):
    _, outbox, state = store
    monkeypatch.setenv("JOB_HR_DRAFTS_MAX_PER_RUN", "1")
    jobs = [_with_hr("a", 95, "hr@a.io"), _with_hr("b", 90, "hr@b.io")]
    assert job_alerts.mail_high_score_jobs({"jobs": jobs}) == 2
    assert [c[1] for c in state["agent_calls"]] == ["hr@a.io"]  # highest score drafted first
    assert "drafting skipped" in outbox[0]["text"]


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


def test_gmail_compose_draft_is_noted(store):
    _, outbox, state = store
    original = job_alerts.create_hr_draft

    def compose_draft(job, hr_email):
        out = original(job, hr_email)
        return {**out, "method": "gmail_compose"}

    job_alerts.create_hr_draft = compose_draft
    try:
        assert job_alerts.mail_high_score_jobs({"jobs": [_with_hr("a", 90, "hr@acme.io")]}) == 1
    finally:
        job_alerts.create_hr_draft = original
    assert "Draft email is saved with HR email id as: hr@acme.io" in outbox[0]["text"]
    assert "attach the resume" in outbox[0]["text"]


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
    opened = []
    monkeypatch.setattr(hd, "open_gmail_compose", lambda to, s, b: opened.append((to, s, b)))
    out = hd.create_hr_draft({"title": "SE", "description": "x"}, "hr@acme.io")
    assert out["saved"] and out["method"] == "gmail_compose"
    assert opened == [("hr@acme.io", "Hi", "Body")]


def test_compose_url_targets_sender_account(monkeypatch):
    from jobEngine.notifications.email import gmail_compose_url

    monkeypatch.setenv("EMAIL_SENDER", "me@gmail.com")
    url = gmail_compose_url("hr@acme.io", "Sub & ject", "x" * 20000)
    assert "authuser=me%40gmail.com" in url and "to=hr%40acme.io" in url and "Sub%20%26%20ject" in url
    assert len(url) <= 7500
