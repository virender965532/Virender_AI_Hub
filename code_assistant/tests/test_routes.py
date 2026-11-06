"""Flask endpoints via the test client (UI mounting disabled by conftest), including the SSE stream."""

from __future__ import annotations

import json

import pytest
from flask import Flask

from code_assistant import pipeline as pipeline_module
from code_assistant import register_code_assistant
from code_assistant.config import get_settings
from code_assistant.pipeline import CodeAssistantPipeline

from .conftest import CODER, REVIEWER, ROUTER, reply, tool_call
from .test_pipeline import FIX, review, route

PREFIX = "/code-assistant/api"


@pytest.fixture
def setup(make_llm, workspace_dir):
    def _make(scripts=None, chat_scripts=None):
        llm, fake = make_llm(scripts, chat_scripts)
        pipeline_module.set_pipeline(CodeAssistantPipeline(get_settings(), llm))
        app = Flask(__name__, template_folder="../../templates")
        _hub_stubs(app)
        register_code_assistant(app)
        client = app.test_client()
        ws = client.post(f"{PREFIX}/workspaces", json={"path": str(workspace_dir)}).get_json()["workspace"]["id"]
        return client, ws, fake

    yield _make
    pipeline_module.set_pipeline(None)


HUB_ENDPOINTS = {
    "main": ["home", "basic_rag_search", "dynamic_rag_search", "enterprise_rag_search", "page_index_rag_search",
             "interview_ui", "job_search", "autonomous_application_agent"],
    "remote_job_finder": ["remote_job_finder_ui", "saved_jobs_ui", "applied_jobs_ui"],
    "resume_intel": ["resume_intelligence_ui"],
    "profile_assistant": ["career_consultant_ui"],
    "stock_finder": ["stock_finder_ui"],
}


def _hub_stubs(app: Flask) -> None:
    """``base.html`` links to the other hub apps; stub their endpoints so the page renders standalone."""
    from flask import Blueprint

    for bp_name, endpoints in HUB_ENDPOINTS.items():
        bp = Blueprint(bp_name, __name__, url_prefix=f"/_stub/{bp_name}")
        for endpoint in endpoints:
            bp.add_url_rule(f"/{endpoint}", endpoint, lambda: "")
        app.register_blueprint(bp)


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events = []
    for block in body.strip().split("\n\n"):
        lines = [line for line in block.splitlines() if not line.startswith(":")]
        if not lines:
            continue
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        events.append((event, data))
    return events


def test_health(setup):
    client, _, _ = setup()
    body = client.get(f"{PREFIX}/health").get_json()
    assert body["status"] == "ok" and body["models"]["reviewer"] == "o3-mini" and body["circuit_open"] is False


def test_workspace_must_be_in_allowlist(setup, tmp_path):
    client, ws, _ = setup()
    outside = tmp_path.parent
    response = client.post(f"{PREFIX}/workspaces", json={"path": str(outside)})
    assert response.status_code == 400 and "CA_WORKSPACE_ROOTS" in response.get_json()["error"]
    assert client.post(f"{PREFIX}/workspaces", json={}).status_code == 400
    info = client.get(f"{PREFIX}/workspaces/{ws}").get_json()
    assert "calc.py" in info["file_tree"] and ".env" not in info["file_tree"]
    assert client.get(f"{PREFIX}/workspaces/nope").status_code == 404


def test_index_endpoint(setup):
    client, ws, _ = setup()
    response = client.post(f"{PREFIX}/workspaces/{ws}/index")
    assert response.status_code == 202 and response.get_json()["ok"]


def test_chat_streams_sse_events(setup, workspace_dir):
    client, ws, _ = setup({ROUTER: [route("fix_bug")], REVIEWER: [review(True)]}, {CODER: [
        reply("", [tool_call("read_file", {"path": "pkg/calc.py"})]),
        reply("", [tool_call("propose_edit", {"path": "pkg/calc.py", "unified_diff": FIX, "explanation": "fix"})]),
        reply("Fixed add."),
    ]})
    response = client.post(f"{PREFIX}/chat", json={"workspace_id": ws, "message": "fix test_add"})
    assert response.status_code == 200 and response.mimetype == "text/event-stream"
    events = parse_sse(response.get_data(as_text=True))
    names = [e for e, _ in events]
    assert names[0] == "session" and names[-1] == "final"
    for expected in ("route", "tool_call", "tool_result", "diff", "test_result", "review"):
        assert expected in names, expected
    final = events[-1][1]
    assert final["status"] == "changes_pending_approval" and final["changes"][0]["path"] == "pkg/calc.py"

    session_id = events[0][1]["session_id"]
    changes = client.get(f"{PREFIX}/changes/{session_id}").get_json()["changes"]
    assert [c["status"] for c in changes] == ["pending"]
    original = (workspace_dir / "pkg" / "calc.py").read_bytes()
    assert b"a - b" in original

    change_id = changes[0]["change_id"]
    assert client.post(f"{PREFIX}/changes/{change_id}/approve").get_json()["ok"]
    assert b"a + b" in (workspace_dir / "pkg" / "calc.py").read_bytes()
    assert client.post(f"{PREFIX}/changes/{change_id}/approve").status_code == 409
    assert client.post(f"{PREFIX}/rollback/{session_id}").get_json()["restored"] == ["pkg/calc.py"]
    assert (workspace_dir / "pkg" / "calc.py").read_bytes() == original


def test_chat_json_mode_and_reject(setup):
    client, ws, _ = setup({ROUTER: [route("fix_bug")], REVIEWER: [review(True)]}, {CODER: [
        reply("", [tool_call("propose_edit", {"path": "pkg/calc.py", "unified_diff": FIX, "explanation": "fix"})]),
        reply("Fixed add."),
    ]})
    body = client.post(f"{PREFIX}/chat", json={"workspace_id": ws, "message": "fix", "stream": False}).get_json()
    assert body["ok"] and body["status"] == "changes_pending_approval"
    change_id = body["changes"][0]["change_id"]
    assert client.post(f"{PREFIX}/changes/{change_id}/reject").get_json()["change"]["status"] == "rejected"


def test_chat_validation(setup):
    client, ws, _ = setup()
    assert client.post(f"{PREFIX}/chat", data="nope").status_code == 400
    assert client.post(f"{PREFIX}/chat", json={"workspace_id": ws, "message": ""}).status_code == 400
    assert client.post(f"{PREFIX}/chat", json={"workspace_id": "zzz", "message": "hi"}).status_code == 404
    assert client.get(f"{PREFIX}/changes/unknown").status_code == 404
    assert client.post(f"{PREFIX}/changes/unknown/approve").status_code == 404
    assert client.post(f"{PREFIX}/rollback/unknown").status_code == 404
    assert client.post(f"{PREFIX}/plans/unknown/approve").status_code == 404


def test_hub_page_renders(setup):
    client, _, _ = setup()
    response = client.get("/code-assistant/")
    assert response.status_code == 200 and b"/code-assistant/ui/" in response.data


def test_registration_is_lazy(monkeypatch):
    """Registering builds neither the pipeline nor the UI; the UI is built on its first request."""
    import code_assistant.ui as ui_module

    monkeypatch.setenv("CA_UI_ENABLED", "true")
    get_settings.cache_clear()
    pipeline_module.set_pipeline(None)
    built = []

    def fake_build(settings):
        built.append(settings)

        def wsgi(environ, start_response):
            start_response("200 OK", [("Content-Type", "text/plain")])
            return [b"gradio"]

        return wsgi

    monkeypatch.setattr(ui_module, "build_wsgi_app", fake_build)
    app = Flask(__name__)
    register_code_assistant(app)
    assert pipeline_module._pipeline is None and built == []

    client = app.test_client()
    assert client.get("/code-assistant/ui/").data == b"gradio"
    assert client.get("/code-assistant/ui/").data == b"gradio"
    assert len(built) == 1  # built once, then reused


def test_ui_failure_returns_503_and_retries(monkeypatch):
    import code_assistant.ui as ui_module

    monkeypatch.setenv("CA_UI_ENABLED", "true")
    get_settings.cache_clear()
    calls = []
    monkeypatch.setattr(ui_module, "build_wsgi_app", lambda s: calls.append(1))
    app = Flask(__name__)
    register_code_assistant(app)
    client = app.test_client()
    assert client.get("/code-assistant/ui/").status_code == 503
    assert client.get("/code-assistant/ui/").status_code == 503
    assert len(calls) == 2
