"""Environment-driven settings for the Code Assistant.

Follows the pattern used by ``profile_assistant/config.py``: a frozen dataclass
populated from env vars via small typed helpers, validated at startup and
cached with ``get_settings()``. The API key is read here but never logged.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from llm_provider import warn_if_shared_model

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = Path(__file__).resolve().parent

DEFAULT_ALLOWLIST = "pytest,python -m pytest,ruff,flake8,mypy,npm test"


class CodeAssistantConfigError(ValueError):
    """Raised at startup when the Code Assistant configuration is invalid."""


def _env_str(name: str, default: str) -> str:
    raw = (os.getenv(name) or "").strip()
    return raw if raw else default


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise CodeAssistantConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = (os.getenv(name) or "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError as exc:
        raise CodeAssistantConfigError(f"{name} must be a number, got {raw!r}") from exc
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _env_paths(name: str) -> tuple[Path, ...]:
    """Comma-separated folders, resolved with ``realpath`` so later prefix checks are exact."""
    raw = os.getenv(name) or ""
    return tuple(Path(os.path.realpath(_resolve_path(part.strip()))) for part in raw.split(",") if part.strip())


def _env_commands(name: str, default: str) -> tuple[tuple[str, ...], ...]:
    """Comma-separated commands, each tokenized (``"python -m pytest"`` -> ``("python", "-m", "pytest")``)."""
    raw = os.getenv(name)
    raw = default if raw is None or not raw.strip() else raw
    commands = []
    for part in raw.split(","):
        tokens = tuple(shlex.split(part.strip()))
        if tokens:
            commands.append(tokens)
    return tuple(commands)


def _normalize_prefix(raw: str) -> str:
    prefix = "/" + raw.strip().strip("/")
    if prefix == "/":
        raise CodeAssistantConfigError("CA_URL_PREFIX must not be '/' (it would shadow the existing apps)")
    return prefix


@dataclass(frozen=True)
class CodeAssistantSettings:
    """All tunables for the Code Assistant module."""

    openai_api_key: str = field(repr=False)
    router_model: str
    coder_model: str
    planner_model: str
    reviewer_model: str
    fallback_model: str
    embedding_model: str
    workspace_roots: tuple[Path, ...]
    max_agent_steps: int
    max_fix_attempts: int
    max_llm_calls: int
    max_tokens_request: int
    command_timeout: float
    command_allowlist: tuple[tuple[str, ...], ...]
    url_prefix: str
    use_docker: bool
    docker_image: str
    request_timeout: float
    max_retries: int
    context_tokens: int
    ui_enabled: bool
    auto_index: bool
    require_plan_approval: bool
    data_dir: Path
    max_tool_output_chars: int = 12000
    max_read_lines: int = 400
    max_file_bytes: int = 1_000_000
    max_message_chars: int = 20000
    output_tail_lines: int = 200
    circuit_failure_threshold: int = 5
    circuit_cooldown_seconds: float = 30.0
    max_replans: int = 2
    history_turns: int = 6

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def checkpoint_dir(self) -> Path:
        return self.data_dir / "checkpoints"

    def validate(self) -> None:
        """Fail fast on configuration that would make the critic meaningless."""
        if self.reviewer_model.strip().lower() == self.coder_model.strip().lower():
            raise CodeAssistantConfigError(
                "CA_REVIEWER_MODEL and CA_CODER_MODEL must be different models "
                f"(both are {self.coder_model!r}); the reviewer must be an independent critic."
            )
        if not self.command_allowlist:
            raise CodeAssistantConfigError("CA_COMMAND_ALLOWLIST must contain at least one command")

    def warn_if_critic_shared(self) -> None:
        """On Ollama, warn if coder and reviewer map to one local model (called on first use, not at startup)."""
        warn_if_shared_model("Code Assistant", {"CA_CODER_MODEL": self.coder_model, "CA_REVIEWER_MODEL": self.reviewer_model})


def load_settings() -> CodeAssistantSettings:
    """Build and validate settings from the current environment (uncached)."""
    settings = CodeAssistantSettings(
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
        router_model=_env_str("CA_ROUTER_MODEL", "gpt-4o-mini"),
        coder_model=_env_str("CA_CODER_MODEL", "gpt-4o"),
        planner_model=_env_str("CA_PLANNER_MODEL", "gpt-4o"),
        reviewer_model=_env_str("CA_REVIEWER_MODEL", "o3-mini"),
        fallback_model=_env_str("CA_FALLBACK_MODEL", "gpt-4o-mini"),
        embedding_model=_env_str("CA_EMBEDDING_MODEL", "text-embedding-3-small"),
        workspace_roots=_env_paths("CA_WORKSPACE_ROOTS"),
        max_agent_steps=_env_int("CA_MAX_AGENT_STEPS", 15, minimum=1),
        max_fix_attempts=_env_int("CA_MAX_FIX_ATTEMPTS", 3, minimum=0),
        max_llm_calls=_env_int("CA_MAX_LLM_CALLS", 40, minimum=1),
        max_tokens_request=_env_int("CA_MAX_TOKENS_REQUEST", 200_000, minimum=1000),
        command_timeout=_env_float("CA_COMMAND_TIMEOUT", 120.0, minimum=1.0),
        command_allowlist=_env_commands("CA_COMMAND_ALLOWLIST", DEFAULT_ALLOWLIST),
        url_prefix=_normalize_prefix(_env_str("CA_URL_PREFIX", "/code-assistant")),
        use_docker=_env_bool("CA_USE_DOCKER", False),
        docker_image=_env_str("CA_DOCKER_IMAGE", "python:3.12-slim"),
        request_timeout=_env_float("CA_REQUEST_TIMEOUT", 90.0, minimum=1.0),
        max_retries=_env_int("CA_MAX_RETRIES", 3, minimum=0),
        context_tokens=_env_int("CA_CONTEXT_TOKENS", 60_000, minimum=4000),
        ui_enabled=_env_bool("CA_UI_ENABLED", True),
        auto_index=_env_bool("CA_AUTO_INDEX", True),
        require_plan_approval=_env_bool("CA_REQUIRE_PLAN_APPROVAL", True),
        data_dir=_resolve_path(_env_str("CA_DATA_DIR", "code_assistant/data")),
    )
    settings.validate()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> CodeAssistantSettings:
    """Cached settings; call ``get_settings.cache_clear()`` after changing env vars."""
    return load_settings()
