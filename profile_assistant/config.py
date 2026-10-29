"""Environment-driven settings for the Profile Assistant.

Follows the ``resumeIntel/config/settings.py`` pattern already used in this
repo: a frozen dataclass populated from env vars via small typed helpers,
cached with ``get_settings()``. The API key is read here but never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ProfileAssistantConfigError(ValueError):
    """Raised at startup when the Profile Assistant configuration is invalid."""


def _env_str(name: str, default: str) -> str:
    raw = (os.getenv(name) or "").strip()
    return raw if raw else default


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise ProfileAssistantConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = (os.getenv(name) or "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError as exc:
        raise ProfileAssistantConfigError(f"{name} must be a number, got {raw!r}") from exc
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_paths(name: str, default: str) -> tuple[Path, ...]:
    """Comma-separated paths; an explicitly empty variable means "none"."""
    raw = os.getenv(name)
    raw = default if raw is None else raw
    return tuple(_resolve_path(part.strip()) for part in raw.split(",") if part.strip())


def _normalize_prefix(raw: str) -> str:
    prefix = "/" + raw.strip().strip("/")
    if prefix == "/":
        raise ProfileAssistantConfigError("PA_URL_PREFIX must not be '/' (it would shadow the existing apps)")
    return prefix


@dataclass(frozen=True)
class ProfileAssistantSettings:
    """All tunables for the Profile Assistant module."""

    openai_api_key: str = field(repr=False)
    answer_model: str
    critic_model: str
    profile_path: Path
    extra_profile_paths: tuple[Path, ...]
    redact_contact_details: bool
    max_revisions: int
    request_timeout: float
    max_retries: int
    url_prefix: str
    admin_token: str = field(repr=False)
    ui_enabled: bool
    history_turns: int = 4
    max_question_chars: int = 2000

    def validate(self) -> None:
        """Fail fast on configuration that would make the critic meaningless."""
        if self.answer_model.strip().lower() == self.critic_model.strip().lower():
            raise ProfileAssistantConfigError(
                "PA_ANSWER_MODEL and PA_CRITIC_MODEL must be different models "
                f"(both are {self.answer_model!r}); the critic must be an independent reviewer."
            )


def load_settings() -> ProfileAssistantSettings:
    """Build and validate settings from the current environment (uncached)."""
    settings = ProfileAssistantSettings(
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
        answer_model=_env_str("PA_ANSWER_MODEL", "gpt-4o-mini"),
        critic_model=_env_str("PA_CRITIC_MODEL", "gpt-4o"),
        profile_path=_resolve_path(_env_str("PA_PROFILE_PATH", "profile_assistant/data/profile.json")),
        extra_profile_paths=_env_paths("PA_EXTRA_PROFILE_PATHS", "Profile/Profile.pdf"),
        redact_contact_details=_env_bool("PA_REDACT_CONTACT_DETAILS", True),
        max_revisions=_env_int("PA_MAX_REVISIONS", 2, minimum=0),
        request_timeout=_env_float("PA_REQUEST_TIMEOUT", 30.0, minimum=1.0),
        max_retries=_env_int("PA_MAX_RETRIES", 3, minimum=0),
        url_prefix=_normalize_prefix(_env_str("PA_URL_PREFIX", "/profile-assistant")),
        admin_token=(os.getenv("PA_ADMIN_TOKEN") or "").strip(),
        ui_enabled=_env_bool("PA_UI_ENABLED", True),
    )
    settings.validate()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> ProfileAssistantSettings:
    """Cached settings; call ``get_settings.cache_clear()`` after changing env vars."""
    return load_settings()
