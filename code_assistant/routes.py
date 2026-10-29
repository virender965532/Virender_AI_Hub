"""Flask blueprint for the Code Assistant (JSON + Server-Sent Events).

The app has no authentication layer, so these routes are open to anyone who can
reach the server (see README "Security model"); keep ``FLASK_HOST=127.0.0.1``.

Streaming endpoints send ``text/event-stream`` events: session, route, plan,
plan_step, thinking, tool_call, tool_result, diff, test_result, review, final
(and error). Send ``"stream": false`` to get one JSON response instead.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from typing import TYPE_CHECKING, Any, Callable, Iterator

from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context

from .config import CodeAssistantSettings
from .errors import ConflictError, RollbackConflictError, SandboxError, SessionBusyError

if TYPE_CHECKING:
    from .schemas import FinalResponse

logger = logging.getLogger(__name__)

_HEARTBEAT_SECONDS = 15.0
_DONE = object()


def _error(message: str, status: int):
    return jsonify({"ok": False, "error": message}), status


def _json_body() -> dict[str, Any] | None:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else None


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


def _stream(run: Callable[[Callable[[str, dict], None]], FinalResponse], first: dict[str, Any]) -> Response:
    """Run ``run(listener)`` in a worker thread and relay its events as SSE."""
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            run(lambda event, data: events.put((event, data)))
        except SessionBusyError as exc:
            events.put(("error", {"error": str(exc), "status": 409}))
        except Exception:  # noqa: BLE001 - the pipeline already converts expected errors
            logger.exception("Code Assistant stream worker failed")
            events.put(("error", {"error": "internal error", "status": 500}))
        finally:
            events.put(_DONE)

    threading.Thread(target=worker, name="ca-stream", daemon=True).start()

    def generate() -> Iterator[str]:
        yield sse("session", first)
        while True:
            try:
                item = events.get(timeout=_HEARTBEAT_SECONDS)
            except queue.Empty:
                yield ": keep-alive\n\n"
                continue
            if item is _DONE:
                return
            event, data = item
            yield sse(event, data)

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def build_blueprint(settings: CodeAssistantSettings) -> Blueprint:
    """Create the ``code_assistant`` blueprint mounted at ``settings.url_prefix``."""
    bp = Blueprint("code_assistant", __name__, url_prefix=settings.url_prefix, template_folder="templates")

    def pipe():
        """The shared pipeline, built (and its heavy imports loaded) on the first API call."""
        from .pipeline import get_pipeline

        return get_pipeline()

    @bp.route("/", methods=["GET"])
    def code_assistant_ui():
        """Hub-styled page (extends ``base.html``) that embeds the Gradio UI."""
        return render_template(
            "code_assistant/code_assistant.html",
            title="Code Assistant",
            ui_url=f"{settings.url_prefix}/ui/?__theme=dark",
            api_url=f"{settings.url_prefix}/api",
            roots=[str(r) for r in settings.workspace_roots],
        )

    # --- workspaces ---------------------------------------------------------------------------

    @bp.route("/api/workspaces", methods=["GET"])
    def api_list_workspaces():
        return jsonify({"ok": True, "roots": [str(r) for r in settings.workspace_roots],
                        "workspaces": [pipe().workspace_info(w.id) for w in pipe().workspaces.list()]})

    @bp.route("/api/workspaces", methods=["POST"])
    def api_register_workspace():
        payload = _json_body()
        if payload is None or not isinstance(payload.get("path"), str):
            return _error("body must be a JSON object with a 'path' string", 400)
        try:
            info = pipe().register_workspace(payload["path"])
        except SandboxError as exc:
            return _error(str(exc), 400)
        return jsonify({"ok": True, "workspace": info}), 201

    @bp.route("/api/workspaces/<workspace_id>", methods=["GET"])
    def api_workspace(workspace_id: str):
        try:
            info = pipe().workspace_info(workspace_id)
        except KeyError:
            return _error("unknown workspace", 404)
        return jsonify({"ok": True, "workspace": info, "file_tree": pipe().file_tree(workspace_id)})

    @bp.route("/api/workspaces/<workspace_id>/index", methods=["POST"])
    def api_index(workspace_id: str):
        try:
            started = pipe().index_workspace(workspace_id)
        except KeyError:
            return _error("unknown workspace", 404)
        return jsonify({"ok": True, "started": started, "workspace": pipe().workspace_info(workspace_id)}), 202

    # --- chat ---------------------------------------------------------------------------------

    @bp.route("/api/chat", methods=["POST"])
    def api_chat():
        payload = _json_body()
        if payload is None:
            return _error("request body must be a JSON object", 400)
        workspace_id, message = payload.get("workspace_id"), payload.get("message")
        session_id = payload.get("session_id") or uuid.uuid4().hex[:16]
        if not isinstance(workspace_id, str):
            return _error("'workspace_id' must be a string", 400)
        try:
            pipe().validate_chat(workspace_id, message, session_id)
        except KeyError:
            return _error("unknown workspace", 404)
        except ValueError as exc:
            return _error(str(exc), 400)

        if payload.get("stream", True) is False:
            try:
                result = pipe().chat(workspace_id, message, session_id)
            except SessionBusyError as exc:
                return _error(str(exc), 409)
            except ValueError as exc:
                return _error(str(exc), 400)
            return jsonify({"ok": True, "session_id": session_id, **result.model_dump(mode="json")})
        return _stream(lambda listener: pipe().chat(workspace_id, message, session_id, listener=listener),
                       {"session_id": session_id, "workspace_id": workspace_id})

    @bp.route("/api/plans/<session_id>/approve", methods=["POST"])
    def api_approve_plan(session_id: str):
        payload = _json_body() or {}
        try:
            session = pipe().sessions.get(session_id)
        except KeyError:
            return _error("unknown session", 404)
        if session.pending_plan is None:
            return _error("there is no plan waiting for approval in this session", 409)
        if payload.get("stream", True) is False:
            try:
                result = pipe().approve_plan(session_id)
            except SessionBusyError as exc:
                return _error(str(exc), 409)
            except ValueError as exc:
                return _error(str(exc), 409)
            return jsonify({"ok": True, "session_id": session_id, **result.model_dump(mode="json")})
        return _stream(lambda listener: pipe().approve_plan(session_id, listener=listener),
                       {"session_id": session_id, "workspace_id": session.workspace_id})

    @bp.route("/api/plans/<session_id>/reject", methods=["POST"])
    def api_reject_plan(session_id: str):
        try:
            return jsonify(pipe().reject_plan(session_id))
        except KeyError:
            return _error("unknown session", 404)

    # --- changes ------------------------------------------------------------------------------

    @bp.route("/api/changes/<session_id>", methods=["GET"])
    def api_changes(session_id: str):
        try:
            return jsonify({"ok": True, "changes": pipe().list_changes(session_id)})
        except KeyError:
            return _error("unknown session", 404)

    @bp.route("/api/changes/<change_id>/approve", methods=["POST"])
    def api_approve(change_id: str):
        try:
            return jsonify(pipe().approve_change(change_id))
        except KeyError:
            return _error("unknown change", 404)
        except (ValueError, ConflictError) as exc:
            return _error(str(exc), 409)
        except SandboxError as exc:
            return _error(str(exc), 400)

    @bp.route("/api/changes/<change_id>/reject", methods=["POST"])
    def api_reject(change_id: str):
        try:
            return jsonify(pipe().reject_change(change_id))
        except KeyError:
            return _error("unknown change", 404)
        except ValueError as exc:
            return _error(str(exc), 409)

    @bp.route("/api/rollback/<session_id>", methods=["POST"])
    def api_rollback(session_id: str):
        payload = _json_body() or {}
        try:
            return jsonify(pipe().rollback(session_id, force=payload.get("force") is True))
        except KeyError:
            return _error("unknown session", 404)
        except RollbackConflictError as exc:
            return _error(str(exc), 409)

    @bp.route("/api/health", methods=["GET"])
    def api_health():
        llm = pipe().llm
        return jsonify({
            "status": "ok",
            "models": {"router": settings.router_model, "coder": settings.coder_model, "planner": settings.planner_model,
                       "reviewer": settings.reviewer_model, "fallback": settings.fallback_model,
                       "embedding": settings.embedding_model},
            "openai_configured": llm.available,
            "circuit_open": llm.breaker.is_open,
            "workspace_roots": len(settings.workspace_roots),
            "docker": settings.use_docker,
        }), 200

    return bp
