"""Router: classifies intent + difficulty (cheap model, temperature 0) and picks the executor.

Executors:
- ``react_readonly``: ReAct coder with READ tools only (explain, answer_question, general_chat)
- ``react_edit``:     ReAct coder with edit + verify tools (fix_bug, refactor, write_tests; the default)
- ``plan_execute``:   Planner first, then the coder per step (implement_feature, or any hard edit task)

Low confidence (< 0.5) or a router failure falls back to ``react_edit``: it can
answer questions too, and nothing it proposes is written without approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..config import CodeAssistantSettings
from ..llm_client import LLMClient, LLMUnavailableError
from ..observability import BudgetExceededError, Trace
from ..prompts import ROUTER_SYSTEM
from ..schemas import RouteDecision

Executor = Literal["react_readonly", "react_edit", "plan_execute"]

DEFAULT_EXECUTOR: Executor = "react_edit"
LOW_CONFIDENCE = 0.5


@dataclass(frozen=True)
class RouteResult:
    executor: Executor
    decision: RouteDecision | None
    reason: str


def select_executor(decision: RouteDecision) -> tuple[Executor, str]:
    if decision.confidence < LOW_CONFIDENCE:
        return DEFAULT_EXECUTOR, f"low confidence ({decision.confidence:.2f}); using the default coder agent"
    if decision.intent in ("explain", "answer_question", "general_chat"):
        return "react_readonly", f"{decision.intent}: read-only agent"
    if decision.intent == "implement_feature" or decision.difficulty == "hard":
        return "plan_execute", f"{decision.intent} ({decision.difficulty}): plan first"
    return "react_edit", f"{decision.intent} ({decision.difficulty}): coder agent with edit + verify tools"


class Router:
    def __init__(self, llm: LLMClient, settings: CodeAssistantSettings) -> None:
        self.llm = llm
        self.settings = settings

    def route(self, message: str, history: str, trace: Trace) -> RouteResult:
        user = (f"Conversation so far:\n{history[-3000:]}\n\n" if history else "") + f"Request:\n{message[:6000]}"
        try:
            result = self.llm.parse(
                model=self.settings.router_model,
                messages=[{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": user}],
                schema=RouteDecision, trace=trace, purpose="route", temperature=0.0,
            )
        except (LLMUnavailableError, BudgetExceededError) as exc:
            route = RouteResult(DEFAULT_EXECUTOR, None, f"router unavailable ({exc}); using the default coder agent")
            trace.emit("route", {"executor": route.executor, "reason": route.reason})
            return route
        decision: RouteDecision = result.output  # type: ignore[assignment]
        executor, reason = select_executor(decision)
        trace.emit("route", {
            "executor": executor, "reason": reason, "intent": decision.intent, "difficulty": decision.difficulty,
            "confidence": decision.confidence, "reasoning": decision.reasoning,
        })
        return RouteResult(executor, decision, reason)
