"""Profile Assistant: a profile-grounded chatbot with a critic reflection loop.

Self-contained module; the host app only calls ``register_profile_assistant(app)``.
See ``profile_assistant/README.md`` for setup and architecture.
"""

from __future__ import annotations

import logging

from flask import Flask

logger = logging.getLogger(__name__)

__all__ = ["register_profile_assistant"]


def register_profile_assistant(app: Flask) -> None:
    """Register the JSON API blueprint and mount the Gradio UI on ``app``.

    Raises ``ProfileAssistantConfigError`` if the answer and critic models are
    the same, and ``ProfileLoadError`` if the profile is missing or invalid.
    """
    from .config import get_settings
    from .profile_loader import get_profile
    from .routes import build_blueprint
    from .ui import mount_ui

    settings = get_settings()
    profile = get_profile()
    app.register_blueprint(build_blueprint(settings))
    if settings.ui_enabled:
        mount_ui(app, settings)
    logger.info(
        "Profile Assistant registered at %s for %r (answer=%s, critic=%s)",
        settings.url_prefix, profile.name, settings.answer_model, settings.critic_model,
    )
