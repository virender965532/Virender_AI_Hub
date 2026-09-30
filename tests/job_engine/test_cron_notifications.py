import io
import json
import urllib.parse

import pytest

from jobEngine import cron
from jobEngine.notifications import pushover


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(dict(urllib.parse.parse_qsl(request.data.decode("utf-8"))))
        return _Response(json.dumps({"status": 1, "request": "abc"}).encode())

    monkeypatch.setenv("PUSHOVER_TOKEN", "tok")
    monkeypatch.setenv("PUSHOVER_USER", "usr")
    monkeypatch.setattr(pushover.urllib.request, "urlopen", fake_urlopen)
    return calls


def test_send_pushover_posts_token_user_and_message(captured):
    assert pushover.send_pushover("hello", title="Hi", url="http://x/saved-jobs", url_title="Open", html=True)
    sent = captured[0]
    assert sent["token"] == "tok" and sent["user"] == "usr"
    assert sent["message"] == "hello" and sent["title"] == "Hi"
    assert sent["url"] == "http://x/saved-jobs" and sent["html"] == "1"


def test_send_pushover_without_config_is_a_noop(monkeypatch):
    monkeypatch.delenv("PUSHOVER_TOKEN", raising=False)
    monkeypatch.delenv("PUSHOVER_USER", raising=False)
    assert pushover.send_pushover("hello") is False


def _matched(n):
    return {
        "match_score": 80 + n,
        "job": {
            "source": "linkedin",
            "source_job_id": str(n),
            "title": f"Full Stack <Engineer> {n}",
            "company": "Acme",
            "application_url": f"https://x/{n}",
        },
    }


def test_message_is_only_the_run_summary():
    text = cron._format_summary({"jobs_discovered": 5, "jobs_recommended": 4})
    assert text == "<i>5 scanned · 4 matched</i>"


def test_cron_request_defaults(monkeypatch):
    for name in ("JOB_CRON_MIN_MATCH_SCORE", "JOB_CRON_MAX_RESULTS", "JOB_CRON_REMOTE_ONLY", "JOB_CRON_SOURCES"):
        monkeypatch.delenv(name, raising=False)
    request = cron._build_request()
    assert request["minimum_match_score"] == 30
    assert request["max_jobs"] is None  # -1 = all jobs from every search URL
    assert request["remote_only"] is True
    assert "instinctools" not in (request["sources"] or [])


async def test_only_jobs_not_saved_before_are_pushed(monkeypatch, captured):
    old, new = _matched(1), _matched(2)
    monkeypatch.setattr(cron, "list_saved_jobs", lambda: [{"saved_key": cron.saved_job_key(old)}])

    async def fake_search(request):
        return {"run_id": "r1", "status": "completed", "summary": {"jobs_discovered": 2, "jobs_recommended": 2}, "jobs": [old, new]}

    monkeypatch.setattr(cron, "run_job_search", fake_search)
    assert await cron.run_once() == 0
    assert captured[0]["title"] == "1 new remote job"
    assert captured[0]["message"] == "<i>2 scanned · 2 matched</i>"
