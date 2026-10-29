"""Orchestrator: guard -> route -> execute (ReAct or Plan-and-Execute) -> verify -> review -> approve.

- guard:   validate input, workspace and session; one request per session at a time
- route:   ``Router`` picks read-only ReAct, edit ReAct, or plan-and-execute
- execute: the coder proposes edits into a per-request ``WorkingSet`` (never disk)
- verify:  lint + tests run on a private copy with the proposals; *new* failures
           (compared with the untouched workspace) go back to the coder, at most
           ``CA_MAX_FIX_ATTEMPTS`` times
- review:  an independent reviewer model scores the diff; FAIL -> one revision,
           still FAIL -> the changes are returned marked "needs human review"
- approve: every change waits for explicit approval; applying creates a
           checkpoint first, and ``rollback`` restores it

Every loop is bounded, the request budget is enforced by the LLM client, and
any unexpected error becomes a friendly message plus a full log under the trace_id.
"""

from __future__ import annotations

import logging
import math
import shlex
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .agents.coder_agent import AgentRun, CoderAgent, Mode
from .agents.planner import Planner, layers, render_plan
from .agents.reviewer_agent import Reviewer, passes
from .agents.router import Router
from .config import CodeAssistantSettings, get_settings
from .errors import SessionBusyError
from .llm_client import LLMClient, LLMUnavailableError, get_llm_client
from .memory.session import Session, SessionStore
from .observability import BudgetExceededError, Listener, Trace
from .prompts import FIX_FEEDBACK_TEMPLATE, REVIEW_FEEDBACK_TEMPLATE, SUMMARIZE_SYSTEM, TASK_CONTEXT_TEMPLATE
from .schemas import CheckResult, FinalResponse, Plan, ReviewVerdict, Verification
from .tools.edit_tools import ConflictError, WorkingSet, apply_change_to_disk
from .tools.exec_tools import CommandRunner, VerifyEnvironment, failure_signatures, lint_command, run_check, test_command
from .tools.git_tools import CheckpointManager, RollbackConflictError
from .tools.registry import ToolContext, ToolRegistry, build_default_registry
from .workspace.indexer import Indexer, file_tree
from .workspace.retriever import Retriever
from .workspace.sandbox import SandboxError, WorkspaceRegistry

logger = logging.getLogger(__name__)

FRIENDLY_ERROR = "Something went wrong while handling your request. Please try again. (trace id: {trace_id})"
FRIENDLY_LLM_DOWN = ("The AI model is unavailable right now (it may be rate-limited or down). "
                     "Please try again in a minute. (trace id: {trace_id})")
_PLAN_CHANGE = "PLAN CHANGE:"
_MAX_SNIPPET_CHARS = 12_000


@dataclass
class _Request:
    """Per-request state shared by the stages."""

    ctx: ToolContext
    session: Session
    task: str
    intent: str | None = None
    existing_change_ids: set[str] = field(default_factory=set)
    baseline: VerifyEnvironment | None = None
    baseline_results: dict[str, CheckResult] = field(default_factory=dict)


class CodeAssistantPipeline:
    """Owns the agents, stores and tools; all public methods are thread-safe."""

    def __init__(
        self,
        settings: CodeAssistantSettings,
        llm: LLMClient,
        *,
        registry: ToolRegistry | None = None,
        sessions: SessionStore | None = None,
        workspaces: WorkspaceRegistry | None = None,
        indexer: Indexer | None = None,
        checkpoints: CheckpointManager | None = None,
        runner: CommandRunner | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.registry = registry or build_default_registry()
        self.sessions = sessions or SessionStore()
        self.workspaces = workspaces or WorkspaceRegistry(settings.workspace_roots, max_file_bytes=settings.max_file_bytes)
        self.indexer = indexer or Indexer(
            settings.index_dir, embed_fn=self._embed if llm.available else None, max_file_bytes=settings.max_file_bytes,
        )
        self.checkpoints = checkpoints or CheckpointManager(settings.checkpoint_dir)
        self.runner = runner or CommandRunner(settings)
        self.router = Router(llm, settings)
        self.planner = Planner(llm, settings)
        self.coder = CoderAgent(llm, self.registry, settings)
        self.reviewer = Reviewer(llm, settings)

    # --- workspaces ---------------------------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self.llm.embed(texts)

    def register_workspace(self, path: str) -> dict[str, Any]:
        workspace = self.workspaces.register(path)
        if self.settings.auto_index:
            self.indexer.build_async(workspace)
        return self.workspace_info(workspace.id)

    def index_workspace(self, workspace_id: str) -> bool:
        return self.indexer.build_async(self.workspaces.get(workspace_id))

    def workspace_info(self, workspace_id: str) -> dict[str, Any]:
        workspace = self.workspaces.get(workspace_id)
        return {"id": workspace.id, "name": workspace.name, "root": str(workspace.root),
                "index": self.indexer.get(workspace).summary()}

    def file_tree(self, workspace_id: str, max_entries: int = 400) -> str:
        return file_tree(self.workspaces.sandbox(workspace_id), max_entries=max_entries)

    # --- request plumbing ---------------------------------------------------------------------

    def validate_chat(self, workspace_id: str, message: str, session_id: str | None) -> None:
        """Input guard; raises ``ValueError`` / ``KeyError`` before any work starts."""
        if not isinstance(message, str) or not message.strip():
            raise ValueError("'message' must be a non-empty string")
        if len(message) > self.settings.max_message_chars:
            raise ValueError(f"'message' is too long (max {self.settings.max_message_chars} characters)")
        if session_id is not None and (not isinstance(session_id, str) or len(session_id) > 64):
            raise ValueError("'session_id' must be a short string")
        self.workspaces.get(workspace_id)

    def _context(self, session: Session, trace: Trace) -> ToolContext:
        workspace = self.workspaces.get(session.workspace_id)
        sandbox = self.workspaces.sandbox(workspace.id)
        index = self.indexer.get(workspace)
        embed_query = None
        if index.vectors and self.llm.available:
            embed_query = lambda q: self.llm.embed([q], trace=trace)[0]  # noqa: E731
        return ToolContext(
            settings=self.settings, workspace=workspace, sandbox=sandbox, trace=trace,
            working_set=WorkingSet(sandbox), runner=self.runner, verify_env=VerifyEnvironment(sandbox),
            session=session, index=index, retriever=Retriever(sandbox, index, embed_query),
        )

    def _new_trace(self, listener: Listener | None) -> Trace:
        return Trace(max_llm_calls=self.settings.max_llm_calls, max_tokens=self.settings.max_tokens_request, listener=listener)

    def _run_request(
        self, session: Session, task: str, listener: Listener | None, body: Callable[[_Request], FinalResponse],
        *, record_turn: bool = True,
    ) -> FinalResponse:
        """Common wrapper: busy lock, trace, error handling, cleanup, metrics and the final event."""
        if not session.busy.acquire(blocking=False):
            raise SessionBusyError("another request is still running for this session")
        trace = self._new_trace(listener)
        trace.log("request_start", session_id=session.id, workspace_id=session.workspace_id, message_chars=len(task))
        request: _Request | None = None
        try:
            request = _Request(self._context(session, trace), session, task, existing_change_ids=set(session.changes))
            response = body(request)
        except LLMUnavailableError as exc:
            logger.warning("trace_id=%s LLM unavailable: %s", trace.trace_id, exc)
            response = FinalResponse(summary=FRIENDLY_LLM_DOWN.format(trace_id=trace.trace_id), status="fallback", trace_id=trace.trace_id)
        except BudgetExceededError as exc:
            response = FinalResponse(summary=f"I stopped because the request budget ran out ({exc}) before I could start.",
                                     status="failed", budget_exceeded=True, trace_id=trace.trace_id)
        except Exception:  # noqa: BLE001 - never leak internals to the user
            logger.exception("trace_id=%s unhandled error in Code Assistant pipeline", trace.trace_id)
            response = FinalResponse(summary=FRIENDLY_ERROR.format(trace_id=trace.trace_id), status="failed", trace_id=trace.trace_id)
        finally:
            if request is not None:
                request.ctx.verify_env.cleanup()
                if request.baseline is not None:
                    request.baseline.cleanup()
            session.busy.release()
        response.trace_id = trace.trace_id
        response.degraded = response.degraded or trace.degraded
        if record_turn:
            session.add_turn("user", task)
            session.add_turn("assistant", response.summary)
        response.metrics = trace.finish(response.status)
        trace.emit("final", response.model_dump(mode="json"), log=False)
        return response

    # --- chat ---------------------------------------------------------------------------------

    def chat(self, workspace_id: str, message: str, session_id: str | None = None, *, listener: Listener | None = None) -> FinalResponse:
        """Handle one user message. Raises ``ValueError``/``KeyError`` for bad input, ``SessionBusyError`` if busy."""
        self.validate_chat(workspace_id, message, session_id)
        session = self.sessions.get_or_create(session_id, workspace_id)
        message = message.strip()

        def body(req: _Request) -> FinalResponse:
            session.compact(self.settings.history_turns, lambda text: self._summarize(text, req.ctx.trace))
            route = self.router.route(message, session.history_text(), req.ctx.trace)
            intent = route.decision.intent if route.decision else None
            req.intent = intent
            task_context = self._task_context(req, message)
            if route.executor == "plan_execute":
                plan = self.planner.make_plan(task_context, req.ctx.trace)
                if self.settings.require_plan_approval:
                    with session.lock:
                        session.pending_plan, session.pending_task = plan, message
                    return FinalResponse(
                        summary=f"This is a larger task, so here is my plan. Approve it to start, or reject it and refine the request.\n\n{render_plan(plan)}",
                        status="plan_pending_approval", plan=plan, intent=intent, trace_id=req.ctx.trace.trace_id,
                    )
                return self._execute_plan(req, plan, task_context, intent)
            mode: Mode = "read_only" if route.executor == "react_readonly" else "edit"
            run = self.coder.run(self.coder.start_messages(mode, task_context), req.ctx, mode)
            return self._finish(req, run, intent)

        response = self._run_request(session, message, listener, body)
        response.metrics["session_id"] = session.id
        return response

    def approve_plan(self, session_id: str, *, listener: Listener | None = None) -> FinalResponse:
        session = self.sessions.get(session_id)
        with session.lock:
            plan, task = session.pending_plan, session.pending_task
            session.pending_plan = session.pending_task = None
        if plan is None or task is None:
            raise ValueError("there is no plan waiting for approval in this session")
        session.add_turn("user", "(approved the plan)")

        def body(req: _Request) -> FinalResponse:
            req.ctx.trace.log("plan_approved", session_id=session_id, steps=len(plan.steps))
            return self._execute_plan(req, plan, self._task_context(req, task), "implement_feature")

        response = self._run_request(session, task, listener, body, record_turn=False)
        session.add_turn("assistant", response.summary)
        return response

    def reject_plan(self, session_id: str) -> dict[str, Any]:
        session = self.sessions.get(session_id)
        with session.lock:
            had_plan = session.pending_plan is not None
            session.pending_plan = session.pending_task = None
        if had_plan:
            session.add_turn("user", "(rejected the plan)")
        return {"ok": had_plan}

    def _summarize(self, text: str, trace: Trace) -> str:
        result = self.llm.chat(
            model=self.settings.router_model,
            messages=[{"role": "system", "content": SUMMARIZE_SYSTEM}, {"role": "user", "content": text[:20000]}],
            trace=trace, purpose="summarize",
        )
        return result.content

    def _task_context(self, req: _Request, task: str) -> str:
        sandbox = req.ctx.sandbox
        try:
            hits = req.ctx.retriever.search(task, k=6) if req.ctx.retriever else []
        except Exception:  # noqa: BLE001 - retrieval is an optimisation
            logger.warning("trace_id=%s initial retrieval failed", req.ctx.trace.trace_id, exc_info=True)
            hits = []
        snippets = "\n\n".join(h.render() for h in hits)[:_MAX_SNIPPET_CHARS] or "(no matches; use the search tools)"
        history = req.session.history_text()
        return TASK_CONTEXT_TEMPLATE.format(
            workspace_name=req.ctx.workspace.name,
            file_tree=file_tree(sandbox, max_entries=250),
            snippets=snippets,
            history=f"Conversation context:\n{history}\n\n" if history else "",
            task=task,
        )

    # --- plan-and-execute ---------------------------------------------------------------------

    def _execute_plan(self, req: _Request, plan: Plan, task_context: str, intent: str | None) -> FinalResponse:
        trace = req.ctx.trace
        results: dict[str, str] = {}
        stopped: str | None = None
        replans = 0
        step_budget = max(3, math.ceil(self.settings.max_agent_steps / 2))
        max_waves = len(plan.steps) + self.settings.max_replans * 8 + 1
        for _ in range(max_waves):
            waves = layers(plan, set(results))
            if not waves or stopped:
                break
            wave = waves[0]

            def run_step(step, mode: Mode) -> tuple[str, AgentRun]:
                deps = "\n".join(f"- {d}: {results.get(d, '')[:1500]}" for d in step.depends_on)
                context = (f"{task_context}\n\nOverall plan:\n{render_plan(plan)}\n\n"
                           + (f"Results of earlier steps:\n{deps}\n\n" if deps else "")
                           + f"YOUR STEP ({step.id}, {step.action}): {step.description}")
                trace.emit("plan_step", {"step": step.id, "status": "started", "description": step.description})
                run = self.coder.run(self.coder.start_messages(mode, context, plan_step=True), req.ctx, mode,
                                     max_steps=step_budget, label=f"step:{step.id}")
                trace.emit("plan_step", {"step": step.id, "status": run.stopped, "result": run.final_text[:1500]})
                return step.id, run

            investigate = [s for s in wave if s.action == "investigate"]
            edits = [s for s in wave if s.action != "investigate"]
            runs: list[tuple[str, AgentRun]] = []
            if investigate:  # independent read-only steps run in parallel
                with ThreadPoolExecutor(max_workers=min(4, len(investigate)), thread_name_prefix="ca-step") as pool:
                    runs += list(pool.map(lambda s: run_step(s, "read_only"), investigate))
            for step in edits:
                runs.append(run_step(step, "edit"))
                if runs[-1][1].stopped in ("budget", "llm_unavailable"):
                    break
            for step_id, run in runs:
                results[step_id] = run.final_text
                if run.stopped in ("budget", "llm_unavailable"):
                    stopped = run.stopped
            if stopped:
                break
            if any(_PLAN_CHANGE in run.final_text for _, run in runs) and replans < self.settings.max_replans:
                replans += 1
                try:
                    plan = self.planner.replan(plan, results, trace)
                except (LLMUnavailableError, BudgetExceededError) as exc:
                    trace.log("replan_failed", error=str(exc))

        remaining = [s for s in plan.steps if s.id not in results]
        summary_lines = [f"Plan: {plan.goal}"] + [f"- {sid}: {text.strip()[:600]}" for sid, text in results.items()]
        summary = "\n".join(summary_lines)
        messages = self.coder.start_messages("edit", f"{task_context}\n\nThe plan has been executed by step agents:\n{summary}")
        messages.append({"role": "assistant", "content": summary})
        run = AgentRun(summary, messages, 0, "budget" if stopped == "budget" else ("llm_unavailable" if stopped else "done"))
        response = self._finish(req, run, intent)
        response.plan = plan
        if remaining:
            response.remaining = "Steps not completed: " + "; ".join(f"{s.id} {s.description}" for s in remaining)
        return response

    # --- verify + review ----------------------------------------------------------------------

    def _finish(self, req: _Request, run: AgentRun, intent: str | None) -> FinalResponse:
        ctx = req.ctx
        trace = ctx.trace
        working_set = ctx.working_set
        if run.stopped == "llm_unavailable" and working_set.is_empty() and not run.final_text:
            raise LLMUnavailableError(run.error or "LLM unavailable")
        budget_hit = run.stopped == "budget"
        queued_commands = [c for c in req.session.list_changes("pending")
                           if c.kind == "command" and c.change_id not in req.existing_change_ids]

        if working_set.is_empty():
            summary = run.final_text or "I could not produce an answer."
            if budget_hit:
                summary += f"\n\n(The request budget ran out: {run.error}.)"
            return FinalResponse(
                summary=summary, status="changes_pending_approval" if queued_commands else ("failed" if budget_hit and not run.final_text else "answered"),
                changes=queued_commands, intent=intent, budget_exceeded=budget_hit, trace_id=trace.trace_id,
                remaining="The task was not finished before the budget ran out; ask me to continue." if budget_hit else None,
            )

        verification, run = self._verify_loop(req, run)
        review: ReviewVerdict | None = None
        needs_human = False
        concerns = ""
        if run.stopped not in ("budget", "llm_unavailable"):
            review, run, verification, needs_human, concerns = self._review_loop(req, run, verification)
        else:
            needs_human, concerns = True, f"Not reviewed: {run.error or run.stopped}."
            budget_hit = budget_hit or run.stopped == "budget"
        if not verification.passed:
            needs_human = True
            concerns = (concerns + " Verification failed after the allowed fix attempts.").strip()

        changes = working_set.to_changes()
        req.session.supersede_pending({c.path for c in changes})
        for change in changes:
            change.needs_human_review = needs_human
            change.review_concerns = concerns
            req.session.add_change(change)
            trace.emit("diff", {"change_id": change.change_id, "path": change.path, "unified_diff": change.unified_diff,
                                "explanation": change.explanation, "risk": change.risk, "kind": change.kind,
                                "needs_human_review": needs_human, "proposed": False})
        summary = run.final_text or "I proposed the changes below."
        if needs_human:
            summary += "\n\nNeeds human review: " + concerns
        if budget_hit:
            summary += "\n\nThe request budget ran out, so this is partial progress."
        return FinalResponse(
            summary=summary, changes=changes + queued_commands, verification=verification, status="changes_pending_approval",
            intent=intent, review=review, needs_human_review=needs_human, budget_exceeded=budget_hit, trace_id=trace.trace_id,
            remaining="The task may be incomplete; review the changes and ask me to continue." if budget_hit else None,
        )

    def _verify(self, req: _Request) -> Verification:
        ctx = req.ctx
        cwd = ctx.verify_env.sync(ctx.working_set)
        changed = ctx.working_set.changed_paths()
        checks: list[CheckResult] = []
        for name, argv in (("lint", lint_command(self.runner, ctx.sandbox.root, changed)), ("tests", test_command(self.runner, cwd))):
            check = run_check(ctx, name, argv, cwd=cwd)
            if not check.passed and not check.skipped and argv is not None:
                check = self._compare_with_baseline(req, check, argv)
            checks.append(check)
        return Verification(passed=all(c.passed for c in checks), checks=checks)

    def _compare_with_baseline(self, req: _Request, check: CheckResult, argv: list[str]) -> CheckResult:
        """A failure that already happens without our changes is pre-existing, not ours.

        Not applied to fix_bug requests: there the failing test is usually the bug itself.
        """
        if check.timed_out or not check.new_failures or req.intent == "fix_bug":
            return check
        key = shlex.join(argv)
        baseline = req.baseline_results.get(key)
        if baseline is None:
            if req.baseline is None:
                req.baseline = VerifyEnvironment(req.ctx.sandbox)
            result = self.runner.run(argv, cwd=req.baseline.prepare(), workspace_root=req.ctx.sandbox.root)
            baseline = CheckResult(name=check.name, command=argv, exit_code=result.exit_code, passed=result.exit_code == 0,
                                   output="", new_failures=failure_signatures(result.output))
            req.baseline_results[key] = baseline
        before = set(baseline.new_failures)
        new = [f for f in check.new_failures if f not in before]
        pre = [f for f in check.new_failures if f in before]
        return check.model_copy(update={"passed": not new, "new_failures": new, "preexisting_failures": pre})

    def _verify_loop(self, req: _Request, run: AgentRun) -> tuple[Verification, AgentRun]:
        trace = req.ctx.trace
        attempts = 0
        verification = self._verify(req)
        while not verification.passed and attempts < self.settings.max_fix_attempts and run.stopped not in ("budget", "llm_unavailable"):
            attempts += 1
            trace.fix_attempts = attempts
            trace.log("fix_attempt", attempt=attempts, failures=[f for c in verification.checks for f in c.new_failures][:20])
            feedback = FIX_FEEDBACK_TEMPLATE.format(attempt=attempts, max_attempts=self.settings.max_fix_attempts,
                                                    failures=render_verification(verification, failures_only=True))
            run = self.coder.run([*run.messages, {"role": "user", "content": feedback}], req.ctx, "edit", label="coder-fix")
            verification = self._verify(req)
        verification.attempts = attempts
        return verification, run

    def _review_loop(self, req: _Request, run: AgentRun, verification: Verification):
        trace = req.ctx.trace
        try:
            verdict = self.reviewer.review(req.task, req.ctx.working_set.combined_diff(), render_verification(verification), trace)
        except (LLMUnavailableError, BudgetExceededError) as exc:
            return None, run, verification, True, f"The reviewer was unavailable ({exc})."
        if passes(verdict):
            return verdict, run, verification, False, ""
        before = req.ctx.working_set.combined_diff()
        run = self.coder.run([*run.messages, {"role": "user", "content": REVIEW_FEEDBACK_TEMPLATE.format(feedback=verdict.feedback)}],
                             req.ctx, "edit", label="coder-revise")
        if req.ctx.working_set.is_empty():
            return verdict, run, verification, True, verdict.feedback
        if req.ctx.working_set.combined_diff() != before:
            verification = self._verify(req)
        try:
            second = self.reviewer.review(req.task, req.ctx.working_set.combined_diff(), render_verification(verification), trace)
        except (LLMUnavailableError, BudgetExceededError) as exc:
            return verdict, run, verification, True, f"{verdict.feedback} (Second review unavailable: {exc}.)"
        if passes(second):
            return second, run, verification, False, ""
        return second, run, verification, True, second.feedback or verdict.feedback

    # --- approvals ----------------------------------------------------------------------------

    def list_changes(self, session_id: str) -> list[dict[str, Any]]:
        session = self.sessions.get(session_id)
        return [c.model_dump(mode="json") for c in session.list_changes()]

    def approve_change(self, change_id: str) -> dict[str, Any]:
        """Apply ONE change after explicit approval (checkpoint first). Raises on conflicts."""
        session, change = self.sessions.find_change(change_id)
        trace = self._new_trace(None)
        with session.lock:
            if change.status != "pending":
                raise ValueError(f"change {change_id} is {change.status}, not pending")
            sandbox = self.workspaces.sandbox(session.workspace_id)
            if change.kind == "command":
                result = self.runner.run(change.command or [], cwd=sandbox.root, workspace_root=sandbox.root)
                change.status = "applied" if result.exit_code == 0 else "failed"
                trace.log("approval", change_id=change_id, kind="command", command=change.command, exit_code=result.exit_code)
                return {"ok": True, "change": change.model_dump(mode="json"), "output": result.render()}
            checkpoint = self.checkpoints.create(sandbox, [change.path])
            try:
                apply_change_to_disk(sandbox, change)
            except ConflictError:
                change.status = "failed"
                trace.log("approval_conflict", change_id=change_id, path=change.path)
                raise
            self.checkpoints.mark_applied(checkpoint, sandbox)
            session.checkpoints.append(checkpoint.id)
            session.change_checkpoints[change_id] = checkpoint.id
            change.status = "applied"
        trace.log("approval", change_id=change_id, path=change.path, kind=change.kind, checkpoint=checkpoint.id,
                  git_branch=checkpoint.git_branch, needs_human_review=change.needs_human_review)
        if self.settings.auto_index:
            self.indexer.build_async(self.workspaces.get(session.workspace_id))
        return {"ok": True, "change": change.model_dump(mode="json"), "checkpoint_id": checkpoint.id, "git_branch": checkpoint.git_branch}

    def reject_change(self, change_id: str) -> dict[str, Any]:
        session, change = self.sessions.find_change(change_id)
        with session.lock:
            if change.status != "pending":
                raise ValueError(f"change {change_id} is {change.status}, not pending")
            change.status = "rejected"
        self._new_trace(None).log("rejection", change_id=change_id, path=change.path)
        return {"ok": True, "change": change.model_dump(mode="json")}

    def rollback(self, session_id: str, *, force: bool = False) -> dict[str, Any]:
        """Undo every applied change of the session, newest first."""
        session = self.sessions.get(session_id)
        sandbox = self.workspaces.sandbox(session.workspace_id)
        restored: list[str] = []
        with session.lock:
            for checkpoint_id in reversed(session.checkpoints):
                checkpoint = self.checkpoints.load(checkpoint_id)
                restored += self.checkpoints.rollback(checkpoint, sandbox, force=force)
            for change_id, checkpoint_id in session.change_checkpoints.items():
                change = session.changes.get(change_id)
                if change is not None and change.status == "applied":
                    change.status = "rolled_back"
            session.checkpoints.clear()
            session.change_checkpoints.clear()
        self._new_trace(None).log("rollback", session_id=session_id, restored=restored, force=force)
        if restored and self.settings.auto_index:
            self.indexer.build_async(self.workspaces.get(session.workspace_id))
        return {"ok": True, "restored": sorted(set(restored))}


def render_verification(verification: Verification, *, failures_only: bool = False) -> str:
    rows = []
    for check in verification.checks:
        if failures_only and (check.passed or check.skipped):
            continue
        state = "skipped" if check.skipped else ("passed" if check.passed else ("timed out" if check.timed_out else "FAILED"))
        rows.append(f"## {check.name}: {state}\n$ {shlex.join(check.command) if check.command else '(none)'}")
        if check.new_failures:
            rows.append("New failures:\n" + "\n".join(f"- {f}" for f in check.new_failures[:30]))
        if check.preexisting_failures:
            rows.append(f"({len(check.preexisting_failures)} failure(s) also happen without the changes; ignored)")
        if not check.passed or not failures_only:
            rows.append(check.output[-6000:] if not check.passed else check.output[-800:])
    return "\n".join(rows) or "No verification commands ran."


_pipeline: CodeAssistantPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> CodeAssistantPipeline:
    """Process-wide pipeline (built on first use)."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = CodeAssistantPipeline(get_settings(), get_llm_client())
    return _pipeline


def set_pipeline(pipeline: CodeAssistantPipeline | None) -> None:
    """Replace the shared pipeline (tests)."""
    global _pipeline
    with _pipeline_lock:
        _pipeline = pipeline


__all__ = ["CodeAssistantPipeline", "SessionBusyError", "RollbackConflictError", "ConflictError", "SandboxError",
           "get_pipeline", "set_pipeline", "render_verification"]
