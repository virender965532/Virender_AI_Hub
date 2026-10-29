"""Flask blueprint for the Profile Assistant JSON API.

The repo has no existing admin-auth mechanism, so ``reload-profile`` is
protected by a shared secret: the ``X-Admin-Token`` header must match
``PA_ADMIN_TOKEN``. If ``PA_ADMIN_TOKEN`` is unset the route is disabled.
"""

from __future__ import annotations

import hmac
import logging
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, jsonify, render_template, request

from . import pipeline
from .config import ProfileAssistantSettings
from .profile_loader import ProfileLoadError, get_profile, reload_profile

logger = logging.getLogger(__name__)


def _admin_required(settings: ProfileAssistantSettings) -> Callable:
    def decorator(view: Callable) -> Callable:
        @wraps(view)
        def wrapped(*args: Any, **kwargs: Any):
            if not settings.admin_token:
                return jsonify({"ok": False, "error": "admin routes are disabled (PA_ADMIN_TOKEN not set)"}), 403
            supplied = request.headers.get("X-Admin-Token", "")
            if not hmac.compare_digest(supplied.encode(), settings.admin_token.encode()):
                return jsonify({"ok": False, "error": "unauthorized"}), 401
            return view(*args, **kwargs)

        return wrapped

    return decorator


def build_blueprint(settings: ProfileAssistantSettings) -> Blueprint:
    """Create the ``profile_assistant`` blueprint mounted at ``settings.url_prefix``."""
    bp = Blueprint("profile_assistant", __name__, url_prefix=settings.url_prefix, template_folder="templates")

    @bp.route("/", methods=["GET"])
    def career_consultant_ui():
        """Hub-styled page (extends ``base.html``) that embeds the Gradio chat."""
        name = get_profile().name
        first_name = name.split()[0]
        return render_template(
            "profile_assistant/career_consultant.html",
            name=name,
            first_name=first_name,
            title=f"Ask {first_name} - Career Consultant",
            ui_url=f"{settings.url_prefix}/ui/?__theme=dark",
            api_url=f"{settings.url_prefix}/api/ask",
        )

    @bp.route("/api/ask", methods=["POST"])
    def api_ask():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"ok": False, "error": "request body must be a JSON object"}), 400
        question = payload.get("question")
        history = payload.get("history", [])
        if not isinstance(question, str):
            return jsonify({"ok": False, "error": "'question' must be a string"}), 400
        if history is None:
            history = []
        if not isinstance(history, list):
            return jsonify({"ok": False, "error": "'history' must be a list"}), 400
        return jsonify(pipeline.answer_question(question, history)), 200

    @bp.route("/api/health", methods=["GET"])
    def api_health():
        return jsonify(
            {"status": "ok", "answer_model": settings.answer_model, "critic_model": settings.critic_model}
        ), 200

    @bp.route("/api/reload-profile", methods=["POST"])
    @_admin_required(settings)
    def api_reload_profile():
        try:
            profile = reload_profile()
        except ProfileLoadError as exc:
            logger.error("Profile reload failed: %s", exc)
            return jsonify({"ok": False, "error": str(exc)}), 422
        logger.info("Profile reloaded from %s", profile.source_path)
        return jsonify({"ok": True, "name": profile.name, "chars": len(profile.text)}), 200

    return bp
