"""End-to-end pipeline with a scripted fake OpenAI: routing, verify loop, review loop, approval, rollback, budget."""

from __future__ import annotations

import httpx
import openai
import pytest

from code_assistant import pipeline as pipeline_module
from code_assistant.config import get_settings
from code_assistant.pipeline import CodeAssistantPipeline, SessionBusyError
from code_assistant.schemas import Plan, PlanStep, ReviewVerdict, RouteDecision

from .conftest import CODER, PLANNER, REVIEWER, ROUTER, reply, tool_call

FIX = "@@ -4,2 +4,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
WRONG_FIX = "@@ -4,2 +4,2 @@\n def add(a, b):\n-    return a - b\n+    return a * b\n"


def route(intent: str, difficulty: str = "easy", confidence: float = 0.9) -> RouteDecision:
    return RouteDecision(reasoning="r", intent=intent, difficulty=difficulty, confidence=confidence)


def review(ok: bool, feedback: str = "") -> ReviewVerdict:
    score = 5 if ok else 2
    return ReviewVerdict(reasoning="r", correctness=score, safety=5, style=5, breaks_existing_behavior=False,
                         verdict="PASS" if ok else "FAIL", feedback=feedback)


def propose(diff: str = FIX) -> dict:
    return reply("", [tool_call("propose_edit", {"path": "pkg/calc.py", "unified_diff": diff, "explanation": "fix add"})])


@pytest.fixture
def make_pipeline(make_llm, monkeypatch):
    monkeypatch.setenv("CA_MAX_FIX_ATTEMPTS", "2")
    get_settings.cache_clear()

    def _make(scripts=None, chat_scripts=None):
        llm, fake = make_llm(scripts, chat_scripts)
        pipe = CodeAssistantPipeline(get_settings(), llm)
        pipeline_module.set_pipeline(pipe)
        return pipe, fake

    yield _make
    pipeline_module.set_pipeline(None)


@pytest.fixture
def calc(workspace_dir):
    path = workspace_dir / "pkg" / "calc.py"
    return path, path.read_bytes()


def _ws(pipe, workspace_dir) -> str:
    return pipe.register_workspace(str(workspace_dir))["id"]


def test_explain_is_read_only(make_pipeline, workspace_dir, calc):
    pipe, fake = make_pipeline({ROUTER: [route("explain")]}, {CODER: [
        reply("", [tool_call("read_file", {"path": "pkg/calc.py"})]),
        reply("`add` subtracts its arguments (pkg/calc.py:5)."),
    ]})
    events = []
    result = pipe.chat(_ws(pipe, workspace_dir), "What does add do?", listener=lambda e, d: events.append(e))
    assert result.status == "answered" and "subtracts" in result.summary and not result.changes
    assert calc[0].read_bytes() == calc[1]
    assert {"route", "tool_call", "tool_result", "final"} <= set(events)
    offered = {t["function"]["name"] for t in fake.calls_for(CODER, "chat")[0]["tools"]}
    assert "propose_edit" not in offered
    assert result.metrics["llm_calls"] == 3 and result.metrics["status"] == "answered"


def test_fix_bug_verify_review_approve_rollback(make_pipeline, workspace_dir, calc):
    pipe, fake = make_pipeline({ROUTER: [route("fix_bug")], REVIEWER: [review(True)]},
                               {CODER: [propose(), reply("Fixed `add` to use +.")]})
    result = pipe.chat(_ws(pipe, workspace_dir), "test_add fails, fix it")
    assert result.status == "changes_pending_approval" and not result.needs_human_review
    assert result.verification.passed and result.verification.attempts == 0
    assert result.review.verdict == "PASS"
    [change] = result.changes
    assert change.path == "pkg/calc.py" and "+    return a + b" in change.unified_diff
    assert calc[0].read_bytes() == calc[1]  # nothing written without approval
    assert "new_content" not in result.model_dump()["changes"][0]

    session_id = result.metrics["session_id"]
    assert [c["status"] for c in pipe.list_changes(session_id)] == ["pending"]
    applied = pipe.approve_change(change.change_id)
    assert applied["ok"] and "return a + b" in calc[0].read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="applied"):
        pipe.approve_change(change.change_id)

    rolled = pipe.rollback(session_id)
    assert rolled["restored"] == ["pkg/calc.py"]
    assert calc[0].read_bytes() == calc[1]
    assert pipe.list_changes(session_id)[0]["status"] == "rolled_back"


def test_reject_change(make_pipeline, workspace_dir, calc):
    pipe, _ = make_pipeline({ROUTER: [route("fix_bug")], REVIEWER: [review(True)]}, {CODER: [propose(), reply("done")]})
    result = pipe.chat(_ws(pipe, workspace_dir), "fix add")
    assert pipe.reject_change(result.changes[0].change_id)["change"]["status"] == "rejected"
    assert calc[0].read_bytes() == calc[1]


def test_verify_loop_retries_then_stops_at_max_attempts(make_pipeline, workspace_dir):
    coder = [propose(WRONG_FIX), reply("Changed it."), reply("Tried again."), reply("Still trying.")]
    pipe, fake = make_pipeline({ROUTER: [route("fix_bug")], REVIEWER: [review(True)]}, {CODER: coder})
    result = pipe.chat(_ws(pipe, workspace_dir), "fix add")
    assert result.verification.attempts == 2 and not result.verification.passed
    assert result.metrics["fix_attempts"] == 2
    feedback = fake.calls_for(CODER, "chat")[2]["messages"][-1]["content"]
    assert "verification FAILED" in feedback and "(attempt 1 of 2)" in feedback and "test_add" in feedback
    assert result.needs_human_review and result.status == "changes_pending_approval"


def test_verify_loop_succeeds_after_feedback(make_pipeline, workspace_dir):
    coder = [propose(WRONG_FIX), reply("Changed it."),
             reply("", [tool_call("propose_edit", {"path": "pkg/calc.py", "explanation": "use +",
                                                   "unified_diff": "@@ -4,2 +4,2 @@\n def add(a, b):\n-    return a * b\n+    return a + b\n"})]),
             reply("Fixed properly.")]
    pipe, _ = make_pipeline({ROUTER: [route("fix_bug")], REVIEWER: [review(True)]}, {CODER: coder})
    result = pipe.chat(_ws(pipe, workspace_dir), "fix add")
    assert result.verification.passed and result.verification.attempts == 1 and not result.needs_human_review
    assert "+    return a + b" in result.changes[0].unified_diff and "a * b" not in result.changes[0].unified_diff


def test_reviewer_fail_triggers_one_revision_then_needs_human_review(make_pipeline, workspace_dir):
    pipe, fake = make_pipeline(
        {ROUTER: [route("fix_bug")], REVIEWER: [review(False, "add a docstring"), review(False, "still missing")]},
        {CODER: [propose(), reply("done"), reply("I disagree; docstrings are not needed here.")]},
    )
    result = pipe.chat(_ws(pipe, workspace_dir), "fix add")
    assert len(fake.calls_for(REVIEWER)) == 2
    assert "add a docstring" in fake.calls_for(CODER, "chat")[2]["messages"][-1]["content"]
    assert result.needs_human_review and "still missing" in result.changes[0].review_concerns
    assert result.changes[0].needs_human_review and result.status == "changes_pending_approval"


def test_reviewer_fail_then_pass(make_pipeline, workspace_dir):
    pipe, _ = make_pipeline({ROUTER: [route("fix_bug")], REVIEWER: [review(False, "x"), review(True)]},
                            {CODER: [propose(), reply("done"), reply("Addressed.")]})
    result = pipe.chat(_ws(pipe, workspace_dir), "fix add")
    assert not result.needs_human_review and result.review.verdict == "PASS"


def test_budget_exceeded_returns_clean_partial_result(make_pipeline, workspace_dir, monkeypatch):
    monkeypatch.setenv("CA_MAX_LLM_CALLS", "2")
    get_settings.cache_clear()
    pipe, _ = make_pipeline({ROUTER: [route("fix_bug")]},
                            {CODER: [reply("Looking at calc.py first.", [tool_call("read_file", {"path": "pkg/calc.py"})]), reply("never")]})
    result = pipe.chat(_ws(pipe, workspace_dir), "fix add")
    assert result.budget_exceeded and result.status in ("answered", "failed")
    assert "Looking at calc.py" in result.summary and result.remaining
    assert result.metrics["llm_calls"] == 2


def test_budget_hit_after_edit_keeps_changes_for_approval(make_pipeline, workspace_dir, monkeypatch):
    monkeypatch.setenv("CA_MAX_LLM_CALLS", "2")
    get_settings.cache_clear()
    pipe, _ = make_pipeline({ROUTER: [route("fix_bug")]}, {CODER: [propose(), reply("never")]})
    result = pipe.chat(_ws(pipe, workspace_dir), "fix add")
    assert result.budget_exceeded and result.status == "changes_pending_approval"
    assert result.needs_human_review and len(result.changes) == 1


def test_plan_requires_approval_then_executes(make_pipeline, workspace_dir, calc):
    plan = Plan(goal="Add multiply", steps=[
        PlanStep(id="s1", description="Read calc.py", action="investigate", files_involved=["pkg/calc.py"], depends_on=[]),
        PlanStep(id="s2", description="Read the tests", action="investigate", files_involved=["tests/test_calc.py"], depends_on=[]),
        PlanStep(id="s3", description="Add mul()", action="edit", files_involved=["pkg/calc.py"], depends_on=["s1", "s2"]),
    ])
    add_mul = "@@ -4,2 +4,6 @@\n def add(a, b):\n     return a - b\n+\n+\n+def mul(a, b):\n+    return a * b\n"
    pipe, fake = make_pipeline(
        {ROUTER: [route("implement_feature", "hard")], PLANNER: [plan], REVIEWER: [review(True)]},
        {CODER: [reply("calc.py has add."), reply("tests cover add."),
                 reply("", [tool_call("propose_edit", {"path": "pkg/calc.py", "unified_diff": add_mul, "explanation": "add mul"})]),
                 reply("Added mul.")]},
    )
    ws = _ws(pipe, workspace_dir)
    first = pipe.chat(ws, "add a multiply function")
    assert first.status == "plan_pending_approval" and first.plan.steps[2].depends_on == ["s1", "s2"]
    assert not fake.calls_for(CODER, "chat")

    second = pipe.approve_plan(first.metrics["session_id"])
    assert second.status == "changes_pending_approval" and second.plan is not None
    assert "def mul" in second.changes[0].unified_diff and not second.remaining
    assert calc[0].read_bytes() == calc[1]
    with pytest.raises(ValueError):
        pipe.approve_plan(first.metrics["session_id"])


def test_llm_down_returns_fallback_status(make_pipeline, workspace_dir):
    err = openai.APIConnectionError(request=httpx.Request("POST", "https://api.openai.com"))
    pipe, _ = make_pipeline({ROUTER: [err] * 10}, {CODER: [err] * 10})
    result = pipe.chat(_ws(pipe, workspace_dir), "explain add")
    assert result.status == "fallback" and "unavailable" in result.summary and result.trace_id in result.summary


def test_unhandled_error_is_friendly(make_pipeline, workspace_dir, monkeypatch, caplog):
    pipe, _ = make_pipeline({ROUTER: [route("explain")]})
    monkeypatch.setattr(pipe.coder, "run", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("kaboom")))
    result = pipe.chat(_ws(pipe, workspace_dir), "explain add")
    assert result.status == "failed" and "kaboom" not in result.summary and result.trace_id in result.summary
    assert "kaboom" in caplog.text


def test_guard_rejects_bad_input(make_pipeline, workspace_dir):
    pipe, _ = make_pipeline()
    ws = _ws(pipe, workspace_dir)
    with pytest.raises(ValueError):
        pipe.chat(ws, "   ")
    with pytest.raises(ValueError):
        pipe.chat(ws, "x" * 30000)
    with pytest.raises(KeyError):
        pipe.chat("nope", "hi")


def test_session_busy(make_pipeline, workspace_dir):
    pipe, _ = make_pipeline()
    ws = _ws(pipe, workspace_dir)
    session = pipe.sessions.get_or_create("s-busy", ws)
    session.busy.acquire()
    try:
        with pytest.raises(SessionBusyError):
            pipe.chat(ws, "hi", "s-busy")
    finally:
        session.busy.release()
