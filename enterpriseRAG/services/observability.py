from __future__ import annotations

import json
import logging
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_trace_id_ctx: ContextVar[str | None] = ContextVar("enterprise_rag_trace_id", default=None)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_trace_id() -> str:
    return str(uuid.uuid4())


def set_trace_id(trace_id: str) -> None:
    _trace_id_ctx.set(trace_id)


def get_trace_id() -> str | None:
    return _trace_id_ctx.get()


@dataclass
class AgentTrace:
    agent: str
    started_at: str
    ended_at: str = ""
    elapsed_ms: float = 0.0
    status: str = "success"
    details: dict[str, Any] = field(default_factory=dict)

    def finish(self, status: str = "success", **details: Any) -> "AgentTrace":
        self.ended_at = utc_now_iso()
        self.status = status
        self.details.update(details)
        return self


class ObservabilityService:
    """Enterprise-grade observability for agent workflows."""

    def __init__(self) -> None:
        self.trace_id: str = ""
        self.agent_traces: list[AgentTrace] = []
        self.tool_usage: list[dict[str, Any]] = []
        self.token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.api_logs: list[dict[str, Any]] = []
        self.guardrail_events: list[dict[str, Any]] = []
        self._workflow_start: float = 0.0

    def start_workflow(self, trace_id: str) -> None:
        self.trace_id = trace_id
        set_trace_id(trace_id)
        self._workflow_start = time.perf_counter()
        self.agent_traces.clear()
        self.tool_usage.clear()
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.api_logs.clear()
        self.guardrail_events.clear()

    def start_agent(self, agent_name: str) -> tuple[AgentTrace, float]:
        trace = AgentTrace(agent=agent_name, started_at=utc_now_iso())
        return trace, time.perf_counter()

    def finish_agent(
        self,
        trace: AgentTrace,
        t0: float,
        status: str = "success",
        **details: Any,
    ) -> AgentTrace:
        trace.elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        trace.finish(status=status, **details)
        self.agent_traces.append(trace)
        logger.info(
            "Agent %s completed in %.2fms [%s]",
            trace.agent,
            trace.elapsed_ms,
            status,
        )
        return trace

    def record_tool(self, tool_name: str, elapsed_ms: float, **details: Any) -> None:
        entry = {
            "tool": tool_name,
            "elapsed_ms": elapsed_ms,
            "timestamp": utc_now_iso(),
            **details,
        }
        self.tool_usage.append(entry)

    def record_tokens(self, usage: dict[str, int]) -> None:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.token_usage[key] += usage.get(key, 0)

    def record_api_log(self, log: dict[str, Any]) -> None:
        log.setdefault("trace_id", self.trace_id)
        self.api_logs.append(log)
        if "prompt_tokens" in log or "total_tokens" in log:
            self.record_tokens(
                {
                    "prompt_tokens": log.get("prompt_tokens", 0),
                    "completion_tokens": log.get("completion_tokens", 0),
                    "total_tokens": log.get("total_tokens", 0),
                }
            )

    def record_guardrail(self, stage: str, check: str, passed: bool, **details: Any) -> None:
        self.guardrail_events.append(
            {
                "stage": stage,
                "check": check,
                "passed": passed,
                "timestamp": utc_now_iso(),
                **details,
            }
        )

    def workflow_elapsed_ms(self) -> float:
        return round((time.perf_counter() - self._workflow_start) * 1000, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "agent_traces": [
                {
                    "agent": t.agent,
                    "elapsed_ms": t.elapsed_ms,
                    "status": t.status,
                    "started_at": t.started_at,
                    "ended_at": t.ended_at,
                    "details": t.details,
                }
                for t in self.agent_traces
            ],
            "tool_usage": self.tool_usage,
            "token_usage": self.token_usage,
            "guardrail_events": self.guardrail_events,
            "workflow_elapsed_ms": self.workflow_elapsed_ms(),
        }

    def log_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)
