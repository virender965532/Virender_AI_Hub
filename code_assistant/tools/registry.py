"""Tool registry: JSON schemas, risk levels, argument validation and dispatch.

Risk levels:
- READ       auto-allowed (list_dir, read_file, search_code, find_symbol, get_file_outline, git_status, git_diff)
- VERIFY     auto-allowed (run_tests, run_linter), always against the private verify copy
- WRITE      approval required to touch disk; the agent only *proposes* (propose_edit, create_file);
             apply_patch refuses unless the user approved the change
- DANGEROUS  approval required (run_command is queued for the user, never run by the agent)

``dispatch`` never raises: unknown tools, bad JSON, invalid arguments and
handler errors all come back as an ``Error: ...`` string the agent can read.
Every call is logged with arguments, duration and result size.
"""

from __future__ import annotations

import enum
import json
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from ..workspace.sandbox import SandboxError

if TYPE_CHECKING:
    from ..config import CodeAssistantSettings
    from ..memory.session import Session
    from ..observability import Trace
    from ..workspace.indexer import CodeIndex
    from ..workspace.retriever import Retriever
    from ..workspace.sandbox import Sandbox, Workspace
    from .edit_tools import WorkingSet
    from .exec_tools import CommandRunner, VerifyEnvironment

TRUNCATION_MARKER = "\n...[truncated]"


class RiskLevel(str, enum.Enum):
    READ = "read"
    VERIFY = "verify"
    WRITE = "write"
    DANGEROUS = "dangerous"


class ToolError(Exception):
    """A tool failed in a way the agent should be told about (message is shown to the agent)."""


@dataclass
class ToolContext:
    """Everything a tool handler may touch during one request."""

    settings: CodeAssistantSettings
    workspace: Workspace
    sandbox: Sandbox
    trace: Trace
    working_set: WorkingSet
    runner: CommandRunner
    verify_env: VerifyEnvironment
    session: Session | None = None
    index: CodeIndex | None = None
    retriever: Retriever | None = None
    read_cache: dict[tuple[str, float], str] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock)


Handler = Callable[..., str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    risk: RiskLevel
    handler: Handler
    agent_visible: bool = True

    def openai_schema(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}


def truncate(text: str, limit: int) -> str:
    """Cap tool output, keeping the start, with an explicit marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + TRUNCATION_MARKER + f" ({len(text) - limit} more characters; narrow the request)"


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,), "array": (list,), "object": (dict,),
}


def validate_args(schema: dict[str, Any], args: Any) -> dict[str, Any]:
    """Minimal JSON-schema check: object, required keys, known keys, primitive types, enums."""
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    props: dict[str, Any] = schema.get("properties", {})
    missing = [k for k in schema.get("required", []) if k not in args or args[k] is None]
    if missing:
        raise ToolError(f"missing required argument(s): {', '.join(missing)}")
    unknown = [k for k in args if k not in props]
    if unknown:
        raise ToolError(f"unknown argument(s): {', '.join(unknown)}; allowed: {', '.join(props) or 'none'}")
    clean: dict[str, Any] = {}
    for key, value in args.items():
        if value is None:
            continue
        spec = props[key]
        expected = _JSON_TYPES.get(spec.get("type", ""), ())
        if expected and (not isinstance(value, expected) or (isinstance(value, bool) and bool not in expected)):
            raise ToolError(f"argument {key!r} must be of type {spec['type']}")
        if "enum" in spec and value not in spec["enum"]:
            raise ToolError(f"argument {key!r} must be one of {spec['enum']}")
        clean[key] = value
    return clean


class ToolRegistry:
    """Holds tools and dispatches calls from the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def openai_tools(self, risks: set[RiskLevel]) -> list[dict[str, Any]]:
        """Schemas of the agent-visible tools at the given risk levels."""
        return [t.openai_schema() for t in self._tools.values() if t.agent_visible and t.risk in risks]

    def dispatch(self, name: str, raw_args: str | dict[str, Any], ctx: ToolContext, *, allowed: set[RiskLevel] | None = None) -> str:
        """Run a tool; always returns text (errors included) and never raises."""
        started = time.monotonic()
        tool = self._tools.get(name)
        ok = False
        args: Any = raw_args
        try:
            if tool is None or not tool.agent_visible:
                raise ToolError(f"unknown tool {name!r}. Available tools: {', '.join(sorted(n for n, t in self._tools.items() if t.agent_visible))}")
            if allowed is not None and tool.risk not in allowed:
                raise ToolError(f"tool {name!r} is not available in this mode")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args or "{}")
                except json.JSONDecodeError as exc:
                    raise ToolError(f"arguments are not valid JSON: {exc}") from exc
            args = validate_args(tool.parameters, args)
            result = tool.handler(ctx, **args)
            ok = True
        except (ToolError, SandboxError, ValueError, KeyError) as exc:
            result = f"Error: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface to the agent, keep the request alive
            ctx.trace.log("tool_exception", tool=name, error=f"{type(exc).__name__}: {exc}")
            result = f"Error: tool {name!r} failed unexpectedly ({type(exc).__name__}: {exc})"
        result = truncate(result, ctx.settings.max_tool_output_chars)
        ctx.trace.record_tool(name)
        ctx.trace.log(
            "tool_call", tool=name, args=args if isinstance(args, dict) else str(args)[:500],
            ok=ok, duration_ms=int((time.monotonic() - started) * 1000), result_chars=len(result),
            risk=tool.risk.value if tool else None,
        )
        return result


def build_default_registry() -> ToolRegistry:
    """Registry with every built-in tool."""
    from . import edit_tools, exec_tools, file_tools, git_tools

    registry = ToolRegistry()
    for module in (file_tools, git_tools, exec_tools, edit_tools):
        for tool in module.TOOLS:
            registry.register(tool)
    return registry
