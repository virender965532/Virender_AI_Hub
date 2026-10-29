"""Planner (Plan-and-Execute): produces a dependency DAG of steps, and replans on request.

The plan is normalised before use: duplicate ids are renamed, dependencies on
unknown or later-defined steps are dropped (so the graph is always acyclic),
and the plan is capped at ``MAX_STEPS``. ``layers`` groups steps into waves
that can run once all their dependencies are done.
"""

from __future__ import annotations

from ..config import CodeAssistantSettings
from ..llm_client import LLMClient
from ..observability import Trace
from ..prompts import PLANNER_SYSTEM, REPLAN_TEMPLATE
from ..schemas import Plan, PlanStep

MAX_STEPS = 8


def normalize_plan(plan: Plan) -> Plan:
    """Unique ids, only backward dependencies (acyclic), at most ``MAX_STEPS`` steps."""
    steps: list[PlanStep] = []
    seen: set[str] = set()
    for i, step in enumerate(plan.steps[:MAX_STEPS], start=1):
        step_id = (step.id or f"s{i}").strip() or f"s{i}"
        if step_id in seen:
            step_id = f"{step_id}_{i}"
        deps = [d for d in dict.fromkeys(step.depends_on) if d in seen]
        steps.append(step.model_copy(update={"id": step_id, "depends_on": deps}))
        seen.add(step_id)
    return Plan(goal=plan.goal, steps=steps)


def layers(plan: Plan, done: set[str] | None = None) -> list[list[PlanStep]]:
    """Topological waves of the not-yet-done steps."""
    done = set(done or ())
    remaining = [s for s in plan.steps if s.id not in done]
    waves: list[list[PlanStep]] = []
    while remaining:
        ready = [s for s in remaining if all(d in done for d in s.depends_on)]
        if not ready:  # unreachable after normalize_plan; guard anyway
            ready = [remaining[0]]
        waves.append(ready)
        done |= {s.id for s in ready}
        remaining = [s for s in remaining if s.id not in done]
    return waves


def render_plan(plan: Plan) -> str:
    rows = [f"Goal: {plan.goal}"]
    for step in plan.steps:
        deps = f" (after {', '.join(step.depends_on)})" if step.depends_on else ""
        files = f" [{', '.join(step.files_involved)}]" if step.files_involved else ""
        rows.append(f"{step.id}. [{step.action}] {step.description}{files}{deps}")
    return "\n".join(rows)


class Planner:
    def __init__(self, llm: LLMClient, settings: CodeAssistantSettings) -> None:
        self.llm = llm
        self.settings = settings

    def make_plan(self, task_context: str, trace: Trace) -> Plan:
        result = self.llm.parse(
            model=self.settings.planner_model,
            messages=[{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": task_context}],
            schema=Plan, trace=trace, purpose="plan",
        )
        plan = normalize_plan(result.output)  # type: ignore[arg-type]
        if not plan.steps:
            plan = Plan(goal=plan.goal, steps=[PlanStep(id="s1", description=plan.goal, action="edit", files_involved=[], depends_on=[])])
        trace.emit("plan", {"plan": plan.model_dump()})
        return plan

    def replan(self, plan: Plan, results: dict[str, str], trace: Trace) -> Plan:
        """Updated plan containing only the remaining steps."""
        rendered = "\n".join(f"{sid}: {text[:1500]}" for sid, text in results.items())
        result = self.llm.parse(
            model=self.settings.planner_model,
            messages=[{"role": "system", "content": PLANNER_SYSTEM},
                      {"role": "user", "content": REPLAN_TEMPLATE.format(plan=render_plan(plan), results=rendered)}],
            schema=Plan, trace=trace, purpose="replan",
        )
        new_plan: Plan = result.output  # type: ignore[assignment]
        # Dependencies on completed steps are satisfied; drop them before normalising.
        cleaned = Plan(goal=new_plan.goal, steps=[
            s.model_copy(update={"id": s.id if s.id not in results else f"{s.id}_r", "depends_on": [d for d in s.depends_on if d not in results]})
            for s in new_plan.steps
        ])
        plan_out = normalize_plan(cleaned)
        trace.emit("plan", {"plan": plan_out.model_dump(), "replanned": True})
        return plan_out
