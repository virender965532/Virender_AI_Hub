"""Gradio chat UI, mounted into the Flask app under ``{PA_URL_PREFIX}/ui``.

Gradio is ASGI; Flask is WSGI. The Gradio app is wrapped with ``a2wsgi`` and
attached with werkzeug's ``DispatcherMiddleware``, which only intercepts
paths under the UI prefix and passes everything else to the original Flask
``wsgi_app`` unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from flask import Flask

from . import pipeline
from .config import ProfileAssistantSettings
from .profile_loader import get_profile

logger = logging.getLogger(__name__)

_running_lifespans: list[Any] = []

# Palette and fonts of the hub (static/css/style.css), so the chat matches the rest of the app.
_HUB = {
    "bg_deep": "#05060d",
    "bg_panel": "#080a16",
    "bg_input": "rgba(8, 10, 22, 0.85)",
    "border": "rgba(120, 160, 255, 0.22)",
    "border_strong": "rgba(120, 160, 255, 0.35)",
    "neon_blue": "#38b6ff",
    "neon_purple": "#b967ff",
    "neon_cyan": "#5cf0e8",
    "text": "#e8ecff",
    "text_muted": "#98a2c9",
}

HUB_CSS = f"""
.gradio-container {{ font-family: "Rajdhani", system-ui, sans-serif !important; font-size: 1.05rem; }}
.gradio-container h1 {{
    font-family: "Orbitron", system-ui, sans-serif !important;
    letter-spacing: 0.04em;
    background: linear-gradient(120deg, {_HUB["neon_blue"]}, {_HUB["neon_purple"]});
    -webkit-background-clip: text; background-clip: text; color: transparent !important;
}}
.gradio-container .message.user {{
    background: linear-gradient(120deg, rgba(56, 182, 255, 0.16), rgba(185, 103, 255, 0.16)) !important;
    border-color: rgba(185, 103, 255, 0.45) !important;
}}
.gradio-container .message.bot {{ background: rgba(12, 14, 28, 0.85) !important; border-color: {_HUB["border"]} !important; }}
.gradio-container button.primary, .gradio-container .submit-button {{
    box-shadow: 0 0 24px rgba(185, 103, 255, 0.25);
}}
:root {{ color-scheme: dark; }}
* {{ scrollbar-color: {_HUB["border_strong"]} transparent; scrollbar-width: thin; }}
footer {{ display: none !important; }}
"""


def build_theme() -> Any:
    """Gradio theme using the hub's dark glass + neon palette (same in light and dark mode)."""
    import gradio as gr

    values = {
        "body_background_fill": _HUB["bg_panel"],
        "body_text_color": _HUB["text"],
        "body_text_color_subdued": _HUB["text_muted"],
        "background_fill_primary": _HUB["bg_panel"],
        "background_fill_secondary": "rgba(12, 14, 28, 0.85)",
        "block_background_fill": "rgba(12, 14, 28, 0.55)",
        "block_border_color": _HUB["border"],
        "block_label_background_fill": "rgba(12, 14, 28, 0.85)",
        "block_label_text_color": _HUB["text_muted"],
        "block_title_text_color": _HUB["text"],
        "panel_background_fill": "rgba(12, 14, 28, 0.55)",
        "panel_border_color": _HUB["border"],
        "border_color_primary": _HUB["border"],
        "border_color_accent": _HUB["neon_purple"],
        "input_background_fill": _HUB["bg_input"],
        "input_border_color": _HUB["border_strong"],
        "input_border_color_focus": _HUB["neon_purple"],
        "input_placeholder_color": _HUB["text_muted"],
        "button_primary_background_fill": f"linear-gradient(120deg, {_HUB['neon_blue']}, {_HUB['neon_purple']})",
        "button_primary_background_fill_hover": f"linear-gradient(120deg, {_HUB['neon_cyan']}, {_HUB['neon_purple']})",
        "button_primary_text_color": _HUB["bg_deep"],
        "button_secondary_background_fill": "rgba(56, 182, 255, 0.1)",
        "button_secondary_background_fill_hover": "rgba(56, 182, 255, 0.2)",
        "button_secondary_text_color": _HUB["neon_cyan"],
        "button_secondary_border_color": "rgba(92, 240, 232, 0.35)",
        "color_accent": _HUB["neon_purple"],
        "color_accent_soft": "rgba(185, 103, 255, 0.16)",
        "link_text_color": _HUB["neon_cyan"],
        "code_background_fill": "rgba(8, 10, 22, 0.85)",
        "loader_color": _HUB["neon_purple"],
        "accordion_text_color": _HUB["text"],
        "table_border_color": _HUB["border"],
        "shadow_drop": "0 20px 50px rgba(0, 0, 0, 0.45)",
    }
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.purple,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Rajdhani"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )
    dark = {f"{key}_dark": value for key, value in values.items() if hasattr(theme, f"{key}_dark")}
    return theme.set(**values, **dark)


def format_review_details(result: dict[str, Any]) -> str:
    """Markdown for the collapsible "Review details" panel."""
    lines = [
        f"**Status:** {result['status']}  ",
        f"**Revisions:** {result['revisions']}  ",
        f"**Trace ID:** `{result['trace_id']}`  ",
        f"**Latency:** {result['latency_ms']} ms",
        "",
        "**Evidence:**",
        "",
    ]
    snippets = [" ".join(snippet.split()).lstrip("-*• ").strip() for snippet in result["evidence"]]
    lines += [f"- {snippet}" for snippet in snippets if snippet] or ["- (none)"]
    return "\n".join(lines)


def respond(message: str, history: list) -> tuple[str, str]:
    """ChatInterface callback: run the pipeline directly (no HTTP hop)."""
    result = pipeline.answer_question(message, history)
    return result["answer"], format_review_details(result)


def build_ui() -> Any:
    """Build the Gradio Blocks app for the current profile."""
    import gradio as gr

    name = get_profile().name
    first_name = name.split()[0]
    title = f"Ask {first_name} - Career Consultant"
    with gr.Blocks(title=title, analytics_enabled=False, fill_height=True) as demo:
        details = gr.Markdown("Ask a question to see how the answer was reviewed.", render=False)
        gr.ChatInterface(
            fn=respond,
            examples=[[f"What are {first_name}'s main technical skills?"], ["What is the capital of France?"]],
            cache_examples=False,
            additional_outputs=[details],
            fill_height=True,
            autofocus=False,  # focusing inside the iframe would scroll the hub page past its heading
            analytics_enabled=False,
        )
        with gr.Accordion("Review details", open=False):
            details.render()
    return demo


def _start_lifespan(asgi_app: Any, timeout: float = 30.0) -> asyncio.AbstractEventLoop:
    """Run the ASGI app's lifespan startup on a dedicated background event loop.

    a2wsgi never sends ASGI lifespan events, but Gradio starts its request
    queue in its lifespan handler; without this, chat requests wait forever.
    The same loop is handed to ``ASGIMiddleware`` so requests and the queue
    worker share it. The lifespan context stays open for the process lifetime.
    """
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, name="profile-assistant-ui", daemon=True).start()
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
        future.result()  # re-raise a startup failure
    # Strong reference: otherwise the pending lifespan task is garbage-collected
    # (and Gradio's queue shut down) as soon as this function returns.
    _running_lifespans.append(future)
    return loop


def mount_ui(app: Flask, settings: ProfileAssistantSettings) -> bool:
    """Mount the Gradio UI at ``{prefix}/ui``. Returns False (and logs) if unavailable."""
    try:
        import gradio as gr
        from a2wsgi import ASGIMiddleware
        from fastapi import FastAPI
        from werkzeug.middleware.dispatcher import DispatcherMiddleware
    except ImportError as exc:
        logger.warning("Profile Assistant UI not mounted (missing dependency: %s); the JSON API still works.", exc)
        return False

    mount_path = f"{settings.url_prefix}/ui"
    try:
        asgi_app = gr.mount_gradio_app(
            FastAPI(), build_ui(), path="/", root_path=mount_path,
            theme=build_theme(), css=HUB_CSS,
            head='<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&display=swap" rel="stylesheet">',
        )
        loop = _start_lifespan(asgi_app)
    except Exception:  # noqa: BLE001 - a UI failure must never take down the other hub apps
        logger.exception("Profile Assistant UI not mounted; the JSON API still works.")
        return False
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {mount_path: ASGIMiddleware(asgi_app, loop=loop)})  # type: ignore[method-assign]
    logger.info("Profile Assistant UI mounted at %s/", mount_path)
    return True
