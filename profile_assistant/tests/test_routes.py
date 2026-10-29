"""Flask endpoints via the test client (UI mounting disabled by conftest)."""

from __future__ import annotations

import pytest
from flask import Flask

from profile_assistant import register_profile_assistant

from .conftest import answer, verdict


@pytest.fixture
def client():
    app = Flask(__name__)
    register_profile_assistant(app)
    return app.test_client()


def test_health(client):
    response = client.get("/profile-assistant/api/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "answer_model": "gpt-4o-mini", "critic_model": "gpt-4o"}


def test_ask(client, make_pipeline):
    make_pipeline([answer("He knows LangGraph.", evidence=["LangGraph"])], [verdict(True)])
    response = client.post("/profile-assistant/api/ask", json={"question": "Does he know LangGraph?", "history": []})

    body = response.get_json()
    assert response.status_code == 200
    assert body["answer"] == "He knows LangGraph."
    assert body["status"] == "passed"
    assert set(body) == {"answer", "status", "revisions", "evidence", "trace_id", "latency_ms"}


def test_ask_rejects_bad_body(client):
    assert client.post("/profile-assistant/api/ask", json={"history": []}).status_code == 400
    assert client.post("/profile-assistant/api/ask", json={"question": "hi", "history": "x"}).status_code == 400
    assert client.post("/profile-assistant/api/ask", data="not json").status_code == 400


def test_reload_profile_requires_admin_token(client):
    assert client.post("/profile-assistant/api/reload-profile").status_code == 401
    response = client.post("/profile-assistant/api/reload-profile", headers={"X-Admin-Token": "test-admin-token"})
    assert response.status_code == 200
    assert response.get_json()["name"] == "Virender Kumar"
