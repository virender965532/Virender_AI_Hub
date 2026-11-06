"""Gradio UI, mounted into the Flask app under ``{CA_URL_PREFIX}/ui``.

Layout: left = workspace selector + file tree; centre = chat with streamed,
collapsible agent steps; right = pending diffs with Approve / Reject / Rollback
and the verification results. Large tasks show the plan with Approve / Reject.

Mounting (option A from the spec): Gradio is ASGI and Flask is WSGI, so the
Gradio app is wrapped with ``a2wsgi`` and attached with werkzeug's
``DispatcherMiddleware``; only paths under the UI prefix are intercepted. This
is the same approach as ``profile_assistant/ui.py``, except that the Gradio app
is only built (``build_wsgi_app``) on the first request to the UI. The UI calls
the pipeline in-process (no HTTP hop).
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import queue
import threading
from typing import Any, Iterator

from . import pipeline as pipeline_module
from .config import CodeAssistantSettings, get_settings
from .pipeline import ConflictError, RollbackConflictError, SandboxError, SessionBusyError
from .schemas import FinalResponse

logger = logging.getLogger(__name__)

_running_lifespans: list[Any] = []
_DONE = object()

_HUB = {
    "bg_deep": "#05060d", "bg_panel": "#080a16", "bg_input": "rgba(8, 10, 22, 0.85)",
    "border": "rgba(120, 160, 255, 0.22)", "border_strong": "rgba(120, 160, 255, 0.35)",
    "neon_blue": "#38b6ff", "neon_purple": "#b967ff", "neon_cyan": "#5cf0e8",
    "text": "#e8ecff", "text_muted": "#98a2c9",
}

BADGES = {
    "answered": ("Answered", "#5cf0e8"),
    "changes_pending_approval": ("Pending approval", "#ffc857"),
    "plan_pending_approval": ("Plan awaiting approval", "#ffc857"),
    "applied": ("Applied", "#4ade80"),
    "failed": ("Failed", "#ff6b81"),
    "fallback": ("Model unavailable", "#ff6b81"),
    "degraded": ("Degraded (fallback model)", "#f59e0b"),
    "needs_human_review": ("Needs human review", "#ff9f43"),
    "budget": ("Budget reached", "#f59e0b"),
    "working": ("Working…", "#38b6ff"),
}

CSS = f"""
.gradio-container {{ font-family: "Rajdhani", system-ui, sans-serif !important; max-width: 100% !important; }}
.ca-badges span {{ display: inline-block; margin: 0 6px 6px 0; padding: 2px 10px; border-radius: 999px;
    font-size: 0.85rem; font-weight: 600; border: 1px solid currentColor; }}
.ca-tree textarea, .ca-tree .cm-content {{ font-size: 0.8rem !important; }}
.gradio-container .message.user {{
    background: linear-gradient(120deg, rgba(56, 182, 255, 0.16), rgba(185, 103, 255, 0.16)) !important;
}}
:root {{ color-scheme: dark; }}
footer {{ display: none !important; }}
"""


def build_theme() -> Any:
    import gradio as gr

    values = {
        "body_background_fill": _HUB["bg_panel"], "body_text_color": _HUB["text"],
        "body_text_color_subdued": _HUB["text_muted"], "background_fill_primary": _HUB["bg_panel"],
        "background_fill_secondary": "rgba(12, 14, 28, 0.85)", "block_background_fill": "rgba(12, 14, 28, 0.55)",
        "block_border_color": _HUB["border"], "block_label_text_color": _HUB["text_muted"],
        "block_title_text_color": _HUB["text"], "border_color_primary": _HUB["border"],
        "input_background_fill": _HUB["bg_input"], "input_border_color": _HUB["border_strong"],
        "button_primary_background_fill": f"linear-gradient(120deg, {_HUB['neon_blue']}, {_HUB['neon_purple']})",
        "button_primary_text_color": _HUB["bg_deep"],
        "button_secondary_background_fill": "rgba(56, 182, 255, 0.1)", "button_secondary_text_color": _HUB["neon_cyan"],
        "color_accent": _HUB["neon_purple"], "code_background_fill": "rgba(8, 10, 22, 0.85)",
    }
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple, secondary_hue=gr.themes.colors.sky, neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Rajdhani"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )
    dark = {f"{key}_dark": value for key, value in values.items() if hasattr(theme, f"{key}_dark")}
    return theme.set(**values, **dark)


# --- rendering helpers -------------------------------------------------------------------------

def badges_html(statuses: list[str]) -> str:
    spans = []
    for status in statuses:
        label, color = BADGES.get(status, (status, _HUB["text_muted"]))
        spans.append(f'<span style="color:{color}">{html.escape(label)}</span>')
    return f'<div class="ca-badges">{"".join(spans)}</div>'


def response_badges(result: FinalResponse) -> list[str]:
    statuses = [result.status]
    if result.needs_human_review:
        statuses.append("needs_human_review")
    if result.degraded:
        statuses.append("degraded")
    if result.budget_exceeded:
        statuses.append("budget")
    return statuses


def fence(text: str, lang: str = "") -> str:
    text = text.replace("```", "ˋˋˋ")
    return f"```{lang}\n{text}\n```"


def step(title: str, content: str, status: str = "done") -> dict[str, Any]:
    return {"role": "assistant", "content": content, "metadata": {"title": title, "status": status}}


def apply_event(history: list[dict[str, Any]], event: str, data: dict[str, Any]) -> None:
    """Translate one pipeline event into chat messages (collapsible agent steps)."""
    if event == "route":
        history.append(step(f"🧭 Route: {data.get('executor')}", data.get("reason", "")))
    elif event == "plan":
        plan = data.get("plan", {})
        rows = [f"**Goal:** {plan.get('goal', '')}"] + [
            f"{i}. `{s['id']}` [{s['action']}] {s['description']}" + (f" _(after {', '.join(s['depends_on'])})_" if s["depends_on"] else "")
            for i, s in enumerate(plan.get("steps", []), start=1)
        ]
        history.append(step("🗺️ Plan" + (" (revised)" if data.get("replanned") else ""), "\n".join(rows)))
    elif event == "plan_step":
        history.append(step(f"▶ Step {data.get('step')}: {data.get('status')}", data.get("result") or data.get("description", "")))
    elif event == "thinking":
        history.append(step("💭 Thinking", data.get("text", "")))
    elif event == "tool_call":
        args = json.dumps(data.get("args"), indent=1, ensure_ascii=False)[:1500]
        history.append({"role": "assistant", "content": fence(args, "json"),
                        "metadata": {"title": f"🔧 {data.get('tool')}", "status": "pending", "id": data.get("id")}})
    elif event == "tool_result":
        for message in reversed(history):
            meta = message.get("metadata") or {}
            if meta.get("id") == data.get("id"):
                icon = "✅" if data.get("ok") else "⚠️"
                message["content"] += "\n" + fence(str(data.get("output", ""))[:1500])
                meta["status"] = "done"
                meta["title"] = f"{meta['title']} {icon}"
                break
    elif event == "diff" and data.get("proposed"):
        title = f"📝 Proposed {'command' if data.get('command') else 'edit'}: {data.get('path')}"
        body = fence(" ".join(data["command"]), "bash") if data.get("command") else fence(data.get("unified_diff", "")[:4000], "diff")
        history.append(step(title, f"{data.get('explanation', '')}\n{body}"))
    elif event == "test_result":
        state = "passed" if data.get("passed") else ("timed out" if data.get("timed_out") else "FAILED")
        history.append(step(f"🧪 {data.get('check')}: {state}", f"`{data.get('command', '')}`\n" + fence(str(data.get("output", ""))[-2000:])))
    elif event == "review":
        scores = f"correctness {data.get('correctness')}/5 · safety {data.get('safety')}/5 · style {data.get('style')}/5"
        history.append(step(f"🔍 Review: {data.get('verdict')}", f"{scores}\n\n{data.get('feedback') or data.get('reasoning', '')}"))


def verification_md(result: FinalResponse | None) -> str:
    if result is None or result.verification is None:
        return ""
    rows = [f"**Verification:** {'passed' if result.verification.passed else 'FAILED'} · fix attempts: {result.verification.attempts}"]
    for check in result.verification.checks:
        state = "skipped" if check.skipped else ("passed" if check.passed else "FAILED")
        rows.append(f"- **{check.name}** {state}: `{' '.join(check.command) or '-'}`")
        for failure in check.new_failures[:10]:
            rows.append(f"  - {failure}")
    if result.review:
        rows.append(f"\n**Review:** {result.review.verdict} — {result.review.feedback or result.review.reasoning}"[:1500])
    return "\n".join(rows)


def change_choices(session_id: str | None) -> list[tuple[str, str]]:
    if not session_id:
        return []
    try:
        changes = pipeline_module.get_pipeline().list_changes(session_id)
    except KeyError:
        return []
    out = []
    for c in changes:
        flag = " ⚠ review" if c["needs_human_review"] and c["status"] == "pending" else ""
        out.append((f"[{c['status']}] {c['path'] if c['kind'] != 'command' else ' '.join(c['command'] or [])} ({c['risk']}){flag}", c["change_id"]))
    return out


def change_detail(session_id: str | None, change_id: str | None) -> str:
    if not session_id or not change_id:
        return "_No change selected._"
    try:
        changes = {c["change_id"]: c for c in pipeline_module.get_pipeline().list_changes(session_id)}
    except KeyError:
        return "_Unknown session._"
    c = changes.get(change_id)
    if c is None:
        return "_Unknown change._"
    header = f"**{c['path']}** · {c['kind']} · risk **{c['risk']}** · status **{c['status']}**\n\n{c['explanation']}\n"
    if c["needs_human_review"]:
        header += f"\n> ⚠ **Needs human review:** {c['review_concerns']}\n"
    body = fence(" ".join(c["command"] or []), "bash") if c["kind"] == "command" else fence(c["unified_diff"], "diff")
    return header + "\n" + body


# --- callbacks ---------------------------------------------------------------------------------

def open_workspace(path: str):
    import gradio as gr

    try:
        info = pipeline_module.get_pipeline().register_workspace(path or "")
    except SandboxError as exc:
        return None, f"⚠️ {exc}", "", None
    pipe = pipeline_module.get_pipeline()
    index = info["index"]
    status = f"**{info['name']}** · `{info['root']}` · index: {index['status']} ({index['files']} files)"
    return info["id"], status, pipe.file_tree(info["id"]), gr.update(value=None)


def reindex(workspace_id: str | None) -> str:
    if not workspace_id:
        return "⚠️ Open a workspace first."
    pipe = pipeline_module.get_pipeline()
    started = pipe.index_workspace(workspace_id)
    index = pipe.workspace_info(workspace_id)["index"]
    return f"Re-indexing {'started' if started else 'already running'} · {index['files']} files indexed so far"


def _run_streaming(target, history: list[dict[str, Any]]) -> Iterator[tuple[list[dict[str, Any]], FinalResponse | None, str | None]]:
    """Run ``target(listener)`` in a thread; yield the growing history, then the final response."""
    events: queue.Queue = queue.Queue()
    holder: dict[str, Any] = {}

    def worker() -> None:
        try:
            holder["result"] = target(lambda e, d: events.put((e, d)))
        except (SessionBusyError, ValueError, KeyError) as exc:
            holder["error"] = str(exc)
        except Exception:  # noqa: BLE001
            logger.exception("Code Assistant UI request failed")
            holder["error"] = "internal error (see server logs)"
        finally:
            events.put(_DONE)

    threading.Thread(target=worker, name="ca-ui", daemon=True).start()
    while True:
        item = events.get()
        if item is _DONE:
            break
        event, data = item
        if event != "final":
            apply_event(history, event, data)
            yield history, None, None
    yield history, holder.get("result"), holder.get("error")


def _outputs(history, result: FinalResponse | None, session_id, error: str | None):
    import gradio as gr

    if error:
        history.append({"role": "assistant", "content": f"⚠️ {error}"})
        statuses = ["failed"]
    elif result is not None:
        history.append({"role": "assistant", "content": result.summary + (f"\n\n_{result.remaining}_" if result.remaining else "")})
        statuses = response_badges(result)
    else:
        statuses = ["working"]
    choices = change_choices(session_id)
    pending = [value for label, value in choices if label.startswith("[pending]")]
    selected = pending[0] if pending else (choices[-1][1] if choices else None)
    plan_visible = bool(result and result.status == "plan_pending_approval")
    return (history, session_id, badges_html(statuses), gr.update(choices=choices, value=selected),
            change_detail(session_id, selected), verification_md(result), gr.update(visible=plan_visible))


def send_message(message: str, history: list, workspace_id: str | None, session_id: str | None):
    import uuid

    history = list(history or [])
    if not workspace_id:
        yield _outputs(history, None, session_id, "Open a workspace first (left panel).")
        return
    if not (message or "").strip():
        return
    session_id = session_id or uuid.uuid4().hex[:16]
    history.append({"role": "user", "content": message})
    pipe = pipeline_module.get_pipeline()
    for hist, result, error in _run_streaming(lambda listener: pipe.chat(workspace_id, message, session_id, listener=listener), history):
        yield _outputs(hist, result, session_id, error)


def approve_plan(history: list, session_id: str | None):
    history = list(history or [])
    if not session_id:
        return
    history.append({"role": "user", "content": "✅ Approve the plan"})
    pipe = pipeline_module.get_pipeline()
    for hist, result, error in _run_streaming(lambda listener: pipe.approve_plan(session_id, listener=listener), history):
        yield _outputs(hist, result, session_id, error)


def reject_plan(history: list, session_id: str | None):
    import gradio as gr

    history = list(history or [])
    if session_id:
        pipeline_module.get_pipeline().reject_plan(session_id)
        history.append({"role": "assistant", "content": "Plan rejected. Refine the request and try again."})
    return history, gr.update(visible=False)


def _change_action(action: str, session_id: str | None, change_id: str | None):
    import gradio as gr

    pipe = pipeline_module.get_pipeline()
    try:
        if action == "rollback":
            if not session_id:
                raise ValueError("no session")
            result = pipe.rollback(session_id)
            note = f"↩️ Rolled back: {', '.join(result['restored']) or 'nothing to roll back'}"
            statuses = ["answered"]
        else:
            if not change_id:
                raise ValueError("select a change first")
            if action == "approve":
                result = pipe.approve_change(change_id)
                extra = f" (checkpoint `{result.get('checkpoint_id')}`)" if result.get("checkpoint_id") else ""
                note = f"✅ Applied {result['change']['path']}{extra}" + (f"\n{fence(result['output'])}" if result.get("output") else "")
                statuses = ["applied"]
            else:
                result = pipe.reject_change(change_id)
                note = f"🗑️ Rejected {result['change']['path']}"
                statuses = ["answered"]
    except (KeyError, ValueError, ConflictError, RollbackConflictError, SandboxError) as exc:
        note, statuses = f"⚠️ {exc}", ["failed"]
    choices = change_choices(session_id)
    pending = [value for label, value in choices if label.startswith("[pending]")]
    selected = pending[0] if pending else change_id
    return gr.update(choices=choices, value=selected), change_detail(session_id, selected), note, badges_html(statuses)


def build_ui(settings: CodeAssistantSettings | None = None) -> Any:
    """Build the Gradio Blocks app."""
    import gradio as gr

    settings = settings or get_settings()
    default_root = str(settings.workspace_roots[0]) if settings.workspace_roots else ""
    with gr.Blocks(title="Code Assistant", analytics_enabled=False, fill_height=True) as demo:
        workspace_id = gr.State(None)
        session_id = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=240):
                gr.Markdown("### Workspace")
                path = gr.Textbox(label="Project folder", value=default_root,
                                  placeholder="A folder inside CA_WORKSPACE_ROOTS", lines=1)
                with gr.Row():
                    open_btn = gr.Button("Open", variant="primary", size="sm")
                    index_btn = gr.Button("Re-index", size="sm")
                ws_status = gr.Markdown("_No workspace open._")
                tree = gr.Code(label="Files", language=None, lines=24, max_lines=24, interactive=False, elem_classes="ca-tree")
            with gr.Column(scale=6, min_width=360):
                badge = gr.HTML(badges_html([]))
                chat = gr.Chatbot(label="Chat", height=600, buttons=["copy"], placeholder="Ask about your code: explain, find bugs, fix, add features, refactor, write tests.")
                with gr.Row(visible=False) as plan_row:
                    approve_plan_btn = gr.Button("✅ Approve plan", variant="primary")
                    reject_plan_btn = gr.Button("✖ Reject plan")
                message = gr.Textbox(label="Message", placeholder="e.g. Why does test_add fail? Fix it.", lines=2, autofocus=False)
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    new_btn = gr.Button("New session")
            with gr.Column(scale=5, min_width=320):
                gr.Markdown("### Changes")
                change_dd = gr.Dropdown(label="Changes in this session", choices=[], interactive=True)
                diff = gr.Markdown("_No change selected._")
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve", variant="primary", size="sm")
                    reject_btn = gr.Button("🗑️ Reject", size="sm")
                    rollback_btn = gr.Button("↩️ Rollback session", size="sm")
                action_note = gr.Markdown()
                verification = gr.Markdown()

        open_btn.click(open_workspace, [path], [workspace_id, ws_status, tree, session_id])
        index_btn.click(reindex, [workspace_id], [ws_status])
        chat_outputs = [chat, session_id, badge, change_dd, diff, verification, plan_row]
        send_event = dict(fn=send_message, inputs=[message, chat, workspace_id, session_id], outputs=chat_outputs)
        send_btn.click(**send_event).then(lambda: "", None, message)
        message.submit(**send_event).then(lambda: "", None, message)
        new_btn.click(lambda: ([], None, badges_html([]), None, "_No change selected._", "", gr.update(visible=False)),
                      None, chat_outputs)
        approve_plan_btn.click(approve_plan, [chat, session_id], chat_outputs)
        reject_plan_btn.click(reject_plan, [chat, session_id], [chat, plan_row])
        change_dd.change(change_detail, [session_id, change_dd], [diff])
        approve_btn.click(lambda s, c: _change_action("approve", s, c), [session_id, change_dd], [change_dd, diff, action_note, badge])
        reject_btn.click(lambda s, c: _change_action("reject", s, c), [session_id, change_dd], [change_dd, diff, action_note, badge])
        rollback_btn.click(lambda s, c: _change_action("rollback", s, c), [session_id, change_dd], [change_dd, diff, action_note, badge])
    return demo


def _start_lifespan(asgi_app: Any, timeout: float = 30.0) -> asyncio.AbstractEventLoop:
    """Run the ASGI lifespan startup on a background loop (a2wsgi never sends lifespan events).

    Without it Gradio's request queue never starts and streamed requests hang.
    The same loop is handed to ``ASGIMiddleware`` and the lifespan stays open.
    """
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="code-assistant-ui", daemon=True).start()
    ready = threading.Event()

    async def _hold_lifespan() -> None:
        async with asgi_app.router.lifespan_context(asgi_app):
            ready.set()
            await asyncio.Event().wait()

    future = asyncio.run_coroutine_threadsafe(_hold_lifespan(), loop)
    if not ready.wait(timeout):
        future.cancel()
        raise RuntimeError("Gradio startup did not complete")
    if future.done():
        future.result()
    _running_lifespans.append(future)  # keep a strong reference for the process lifetime
    return loop


def build_wsgi_app(settings: CodeAssistantSettings) -> Any | None:
    """Build the Gradio app, start its lifespan and wrap it as WSGI. Returns None (and logs) on failure.

    Called on the first request to ``{prefix}/ui`` (see ``_LazyUI`` in ``__init__.py``).
    """
    try:
        import gradio as gr
        from a2wsgi import ASGIMiddleware
        from fastapi import FastAPI
    except ImportError as exc:
        logger.warning("Code Assistant UI unavailable (missing dependency: %s); the API still works.", exc)
        return None

    mount_path = f"{settings.url_prefix}/ui"
    try:
        asgi_app = gr.mount_gradio_app(
            FastAPI(), build_ui(settings), path="/", root_path=mount_path, theme=build_theme(), css=CSS,
            head='<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&display=swap" rel="stylesheet">',
        )
        loop = _start_lifespan(asgi_app)
    except Exception:  # noqa: BLE001 - a UI failure must never take down the other hub apps
        logger.exception("Code Assistant UI failed to start; the API still works.")
        return None
    logger.info("Code Assistant UI started at %s/", mount_path)
    return ASGIMiddleware(asgi_app, loop=loop)
