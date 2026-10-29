"""Router mapping, ReAct loop bounds, planner DAG handling, reviewer rubric."""

from __future__ import annotations

import pytest

from code_assistant.agents.coder_agent import CoderAgent
from code_assistant.agents.planner import Planner, layers, normalize_plan
from code_assistant.agents.reviewer_agent import Reviewer, passes
from code_assistant.agents.router import Router, select_executor
from code_assistant.config import get_settings
from code_assistant.schemas import Plan, PlanStep, ReviewVerdict, RouteDecision
from code_assistant.tools.registry import build_default_registry

from .conftest import CODER, PLANNER, REVIEWER, ROUTER, reply, tool_call


def decision(intent: str, difficulty: str = "easy", confidence: float = 0.9) -> RouteDecision:
    return RouteDecision(reasoning="r", intent=intent, difficulty=difficulty, confidence=confidence)


def verdict(passed: bool = True, **scores) -> ReviewVerdict:
    base = {"correctness": 5, "safety": 5, "style": 5}
    base.update(scores)
    return ReviewVerdict(reasoning="r", breaks_existing_behavior=False, verdict="PASS" if passed else "FAIL", feedback="", **base)


@pytest.mark.parametrize("intent,difficulty,expected", [
    ("explain", "easy", "react_readonly"),
    ("answer_question", "hard", "react_readonly"),
    ("general_chat", "easy", "react_readonly"),
    ("fix_bug", "easy", "react_edit"),
    ("refactor", "easy", "react_edit"),
    ("write_tests", "easy", "react_edit"),
    ("fix_bug", "hard", "plan_execute"),
    ("implement_feature", "easy", "plan_execute"),
])
def test_router_maps_intents(intent, difficulty, expected):
    assert select_executor(decision(intent, difficulty))[0] == expected


def test_low_confidence_goes_to_default():
    assert select_executor(decision("explain", confidence=0.3))[0] == "react_edit"


def test_router_uses_cheap_model_at_temperature_zero(make_llm, trace):
    llm, fake = make_llm({ROUTER: [decision("explain")]})
    result = Router(llm, get_settings()).route("what does add do?", "", trace)
    assert result.executor == "react_readonly"
    assert fake.calls[0]["model"] == ROUTER and fake.calls[0]["temperature"] == 0.0


def test_router_failure_falls_back_to_default(make_llm, trace):
    import httpx
    import openai

    err = openai.AuthenticationError("bad", response=httpx.Response(401, request=httpx.Request("POST", "http://x")), body=None)
    llm, _ = make_llm({ROUTER: [err]})  # router model == fallback model, so no second attempt
    assert Router(llm, get_settings()).route("hi", "", trace).executor == "react_edit"


def test_react_loop_stops_at_max_steps(make_llm, tool_ctx, monkeypatch):
    monkeypatch.setenv("CA_MAX_AGENT_STEPS", "3")
    get_settings.cache_clear()
    settings = get_settings()
    looping = [reply("", [tool_call("list_dir", {"path": "."})]) for _ in range(10)]
    llm, fake = make_llm(chat_scripts={CODER: looping})
    agent = CoderAgent(llm, build_default_registry(), settings)
    run = agent.run(agent.start_messages("read_only", "task"), tool_ctx, "read_only")
    assert run.stopped == "max_steps" and run.steps == 3 and len(fake.calls) == 3
    assert "step limit (3 steps)" in run.final_text


def test_unknown_tool_error_goes_back_to_agent(make_llm, tool_ctx):
    llm, fake = make_llm(chat_scripts={CODER: [reply("", [tool_call("delete_everything", {})]), reply("Done: I could not do that.")]})
    agent = CoderAgent(llm, build_default_registry(), get_settings())
    run = agent.run(agent.start_messages("edit", "task"), tool_ctx, "edit")
    assert run.stopped == "done" and run.final_text == "Done: I could not do that."
    tool_msg = fake.calls[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "Error: unknown tool 'delete_everything'" in tool_msg["content"]
    assert "not instructions" in tool_msg["content"]


def test_read_only_mode_hides_and_blocks_write_tools(make_llm, tool_ctx, workspace_dir):
    diff = "@@ -4,2 +4,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
    llm, fake = make_llm(chat_scripts={CODER: [
        reply("", [tool_call("propose_edit", {"path": "pkg/calc.py", "unified_diff": diff, "explanation": "x"})]),
        reply("ok"),
    ]})
    agent = CoderAgent(llm, build_default_registry(), get_settings())
    agent.run(agent.start_messages("read_only", "task"), tool_ctx, "read_only")
    offered = {t["function"]["name"] for t in fake.calls[0]["tools"]}
    assert "propose_edit" not in offered and "run_command" not in offered
    assert "not available in this mode" in fake.calls[1]["messages"][-1]["content"]
    assert tool_ctx.working_set.is_empty()


def test_agent_returns_cleanly_on_budget(make_llm, tool_ctx):
    from code_assistant.observability import Trace

    tool_ctx.trace = Trace(max_llm_calls=1, max_tokens=10**6)
    llm, _ = make_llm(chat_scripts={CODER: [reply("looking", [tool_call("list_dir", {})]), reply("never")]})
    agent = CoderAgent(llm, build_default_registry(), get_settings())
    run = agent.run(agent.start_messages("edit", "task"), tool_ctx, "edit")
    assert run.stopped == "budget" and run.final_text == "looking"


def test_plan_normalisation_and_layers():
    plan = Plan(goal="g", steps=[
        PlanStep(id="a", description="read x", action="investigate", files_involved=[], depends_on=["c"]),  # forward dep dropped
        PlanStep(id="b", description="read y", action="investigate", files_involved=[], depends_on=[]),
        PlanStep(id="c", description="edit", action="edit", files_involved=[], depends_on=["a", "b", "zzz"]),
        PlanStep(id="c", description="edit again", action="edit", files_involved=[], depends_on=["c"]),
    ])
    norm = normalize_plan(plan)
    assert [s.id for s in norm.steps] == ["a", "b", "c", "c_4"]
    assert norm.steps[0].depends_on == [] and norm.steps[2].depends_on == ["a", "b"]
    assert [[s.id for s in wave] for wave in layers(norm)] == [["a", "b"], ["c"], ["c_4"]]
    assert [[s.id for s in wave] for wave in layers(norm, {"a", "b"})] == [["c"], ["c_4"]]


def test_planner_emits_plan(make_llm, trace):
    plan = Plan(goal="g", steps=[PlanStep(id="s1", description="d", action="edit", files_involved=["a.py"], depends_on=[])])
    llm, _ = make_llm({PLANNER: [plan]})
    events = []
    trace.listener = lambda e, d: events.append(e)
    assert Planner(llm, get_settings()).make_plan("task", trace).steps[0].id == "s1"
    assert "plan" in events


def test_reviewer_rubric_overrides_inconsistent_pass(make_llm, trace):
    llm, fake = make_llm({REVIEWER: [verdict(True, safety=3)]})
    result = Reviewer(llm, get_settings()).review("task", "diff", "tests passed", trace)
    assert result.verdict == "FAIL" and not passes(result)
    assert fake.calls[0]["model"] == REVIEWER != CODER
    assert passes(verdict(True))
