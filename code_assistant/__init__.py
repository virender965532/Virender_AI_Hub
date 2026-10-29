"""Code Assistant: an AI coding agent over a sandboxed project folder.

Self-contained module; the host app only calls ``register_code_assistant(app)``.
See ``code_assistant/README.md`` for setup, the security model and architecture.

Nothing heavy runs at app startup: registration only validates settings and
adds the URL routes. The pipeline (OpenAI client, index, tools) is built on
the first API call, and the Gradio UI is built and started on the first
request to ``{prefix}/ui`` (i.e. when someone opens the Code Assistant page).
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Iterable

from flask import Flask

logger = logging.getLogger(__name__)

__all__ = ["register_code_assistant"]


class _LazyUI:
    """WSGI app that builds the Gradio UI on its first request, then delegates to it."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self._app: Callable | None = None
        self._lock = threading.Lock()

    def _load(self) -> Callable | None:
        if self._app is None:
            with self._lock:
                if self._app is None:
                    from .ui import build_wsgi_app

                    self._app = build_wsgi_app(self.settings)  # None if Gradio failed; retried next request
        return self._app

    def __call__(self, environ: dict, start_response: Callable) -> Iterable[bytes]:
        app = self._load()
        if app is None:
            start_response("503 Service Unavailable", [("Content-Type", "text/plain; charset=utf-8")])
            return [b"The Code Assistant UI is unavailable; see the server logs. The API still works."]
        return app(environ, start_response)


def register_code_assistant(app: Flask) -> None:
    """Register the JSON/SSE API blueprint and a lazily-built Gradio UI on ``app``.

    Raises ``CodeAssistantConfigError`` if the reviewer and coder models are the same.
    """
    from werkzeug.middleware.dispatcher import DispatcherMiddleware

    from .config import get_settings
    from .routes import build_blueprint

    settings = get_settings()
    app.register_blueprint(build_blueprint(settings))
    if settings.ui_enabled:
        mount_path = f"{settings.url_prefix}/ui"
        app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {mount_path: _LazyUI(settings)})  # type: ignore[method-assign]
    logger.info("Code Assistant registered at %s (loads on first use)", settings.url_prefix)
