"""Environment-driven settings for Resume Intelligence.

Follows the ``jobEngine/config/settings.py`` pattern already used in this
repo: a frozen dataclass populated from env vars via small typed helpers,
cached with ``get_settings()``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROFILE_DIR = PROJECT_ROOT / "Profile"


def _env_str(name: str, default: str) -> str:
    raw = (os.getenv(name) or "").strip()
    return raw if raw else default


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        value = default
    else:
        try:
            value = float(raw)
        except ValueError:
            value = default
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ResumeIntelSettings:
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(
        default_factory=lambda: _env_str("RESUME_INTEL_OPENAI_MODEL", "gpt-4o-mini")
    )
    openai_timeout: float = field(
        default_factory=lambda: _env_float("OPENAI_TIMEOUT", 60.0, minimum=1.0)
    )
    openai_max_retries: int = field(
        default_factory=lambda: _env_int("RESUME_INTEL_LLM_MAX_RETRIES", 1, minimum=0)
    )
    llm_enabled: bool = field(
        default_factory=lambda: _env_bool("RESUME_INTEL_LLM_ENABLED", True)
    )
    default_resume_filename: str = field(
        default_factory=lambda: _env_str(
            "RESUME_INTEL_DEFAULT_RESUME_FILE", "SeniorFullStackAIEngineer.pdf"
        )
    )
    max_upload_bytes: int = field(
        default_factory=lambda: _env_int(
            "RESUME_INTEL_MAX_UPLOAD_BYTES", 5_000_000, minimum=10_000
        )
    )
    max_resume_chars: int = field(
        default_factory=lambda: _env_int(
            "RESUME_INTEL_MAX_RESUME_CHARS", 20_000, minimum=500
        )
    )
    max_jd_chars: int = field(
        default_factory=lambda: _env_int("RESUME_INTEL_MAX_JD_CHARS", 20_000, minimum=500)
    )

    @property
    def default_resume_path(self) -> Path:
        return PROFILE_DIR / self.default_resume_filename


@lru_cache(maxsize=1)
def get_settings() -> ResumeIntelSettings:
    return ResumeIntelSettings()
