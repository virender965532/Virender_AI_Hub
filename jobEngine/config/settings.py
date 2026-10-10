"""Environment-driven settings for the job engine.

Follows the ``enterpriseRAG/config/settings.py`` pattern already used in this
repo: a frozen dataclass populated from env vars via small typed helpers, and
a single module-level cache (``get_settings``). Nothing here is hard-coded
per-source; source-specific values live in ``config/job_sources.yaml``
(see ``jobEngine/config/sources.py``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
JOB_SOURCES_CONFIG_PATH = PROJECT_ROOT / "config" / "job_sources.yaml"
PROFILE_DIR = PROJECT_ROOT / "Profile"
PROFILE_SUMMARY_PATH = PROFILE_DIR / "Summary.txt"
PROFILE_RESUME_PDF_PATH = PROFILE_DIR / "VirenderFullstackResumeUpdated.pdf"

JOB_ENGINE_OUTPUT_DIR = PROJECT_ROOT / "output" / "job_engine"
JOB_ENGINE_DEBUG_DIR = PROJECT_ROOT / "scraper_debug" / "job_engine"
JOB_ENGINE_HISTORY_DIR = PROJECT_ROOT / "history" / "job_engine"


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
class MatchingWeights:
    """Weights must sum to ~1.0; validated in ``EngineSettings.__post_init__``."""

    technical_skills: float = field(
        default_factory=lambda: _env_float("JOB_MATCH_WEIGHT_TECHNICAL", 0.30)
    )
    role_alignment: float = field(
        default_factory=lambda: _env_float("JOB_MATCH_WEIGHT_ROLE", 0.20)
    )
    experience: float = field(
        default_factory=lambda: _env_float("JOB_MATCH_WEIGHT_EXPERIENCE", 0.15)
    )
    ai_alignment: float = field(
        default_factory=lambda: _env_float("JOB_MATCH_WEIGHT_AI", 0.20)
    )
    architecture_cloud: float = field(
        default_factory=lambda: _env_float("JOB_MATCH_WEIGHT_ARCHITECTURE", 0.10)
    )
    domain: float = field(
        default_factory=lambda: _env_float("JOB_MATCH_WEIGHT_DOMAIN", 0.05)
    )

    def as_dict(self) -> dict[str, float]:
        return {
            "technical_skills": self.technical_skills,
            "role_alignment": self.role_alignment,
            "experience": self.experience,
            "ai_alignment": self.ai_alignment,
            "architecture_cloud": self.architecture_cloud,
            "domain": self.domain,
        }

    def total(self) -> float:
        return round(sum(self.as_dict().values()), 4)


@dataclass(frozen=True)
class EngineSettings:
    # --- LLM ---
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(
        default_factory=lambda: _env_str("JOB_ENGINE_OPENAI_MODEL", "gpt-4o-mini")
    )
    openai_timeout: float = field(
        default_factory=lambda: _env_float("OPENAI_TIMEOUT", 60.0, minimum=1.0)
    )
    openai_max_retries: int = field(
        default_factory=lambda: _env_int("JOB_LLM_MAX_RETRIES", 2, minimum=0)
    )
    llm_enabled: bool = field(
        default_factory=lambda: _env_bool("JOB_ENGINE_LLM_ENABLED", True)
    )
    prompt_version: str = field(
        default_factory=lambda: _env_str("JOB_MATCH_PROMPT_VERSION", "v1")
    )
    # Approximate, operator-configurable pricing for cost estimation/observability only
    # (section 27/40) — not used for any billing decision. Defaults are for gpt-4o-mini.
    llm_price_per_1k_prompt_usd: float = field(
        default_factory=lambda: _env_float("JOB_LLM_PRICE_PER_1K_PROMPT_USD", 0.00015)
    )
    llm_price_per_1k_completion_usd: float = field(
        default_factory=lambda: _env_float("JOB_LLM_PRICE_PER_1K_COMPLETION_USD", 0.0006)
    )

    # --- LangSmith / tracing (reuses standard LangChain env vars; opt-in) ---
    langchain_tracing_v2: bool = field(
        default_factory=lambda: _env_bool("LANGCHAIN_TRACING_V2", False)
    )
    langchain_project: str = field(
        default_factory=lambda: _env_str("LANGCHAIN_PROJECT", "job-engine")
    )

    # --- Browser ---
    browser_headless: bool = field(
        default_factory=lambda: _env_bool("JOB_BROWSER_HEADLESS", True)
    )
    browser_timeout_ms: int = field(
        default_factory=lambda: _env_int("JOB_BROWSER_TIMEOUT", 60_000, minimum=1_000)
    )
    browser_navigation_timeout_ms: int = field(
        default_factory=lambda: _env_int(
            "JOB_BROWSER_NAV_TIMEOUT", 45_000, minimum=1_000
        )
    )
    browser_max_retries: int = field(
        default_factory=lambda: _env_int("JOB_BROWSER_MAX_RETRIES", 3, minimum=0)
    )
    browser_max_concurrency: int = field(
        default_factory=lambda: _env_int("JOB_BROWSER_MAX_CONCURRENCY", 3, minimum=1)
    )
    browser_retry_backoff_seconds: float = field(
        default_factory=lambda: _env_float(
            "JOB_BROWSER_RETRY_BACKOFF_SECONDS", 2.0, minimum=0.1
        )
    )

    # --- Circuit breaker ---
    circuit_breaker_failure_threshold: int = field(
        default_factory=lambda: _env_int("JOB_CIRCUIT_BREAKER_THRESHOLD", 3, minimum=1)
    )
    circuit_breaker_cooldown_seconds: float = field(
        default_factory=lambda: _env_float(
            "JOB_CIRCUIT_BREAKER_COOLDOWN_SECONDS", 900.0, minimum=1.0
        )
    )

    # --- Source / run limits ---
    max_sources_per_run: int = field(
        default_factory=lambda: _env_int("JOB_MAX_SOURCES", 10, minimum=1)
    )
    max_jobs_per_source: int = field(
        default_factory=lambda: _env_int("JOB_MAX_JOBS_PER_SOURCE", 100, minimum=1)
    )
    max_pagination_depth: int = field(
        default_factory=lambda: _env_int("JOB_MAX_PAGINATION_DEPTH", 20, minimum=1)
    )
    max_description_chars: int = field(
        default_factory=lambda: _env_int("JOB_MAX_DESCRIPTION_CHARS", 20_000, minimum=500)
    )
    max_html_bytes: int = field(
        default_factory=lambda: _env_int("JOB_MAX_HTML_BYTES", 5_000_000, minimum=10_000)
    )

    # --- Matching ---
    weights: MatchingWeights = field(default_factory=MatchingWeights)
    minimum_match_score: float = field(
        default_factory=lambda: _env_float("JOB_MIN_MATCH_SCORE", 70.0, minimum=0.0)
    )
    # <= 0 (e.g. -1) sends every validated job to the LLM, uncapped.
    llm_matching_top_k: int = field(
        default_factory=lambda: _env_int("JOB_LLM_TOP_K", 50)
    )
    rule_prefilter_min_score: float = field(
        default_factory=lambda: _env_float(
            "JOB_RULE_PREFILTER_MIN_SCORE", 15.0, minimum=0.0
        )
    )
    candidate_total_experience_years: float = field(
        default_factory=lambda: _env_float(
            "JOB_CANDIDATE_EXPERIENCE_YEARS", 7.0, minimum=0.0
        )
    )
    max_negative_experience_gap_years: float = field(
        default_factory=lambda: _env_float(
            "JOB_MAX_EXPERIENCE_GAP_YEARS", 3.0, minimum=0.0
        )
    )

    # --- Operational modes ---
    dry_run: bool = field(default_factory=lambda: _env_bool("JOB_ENGINE_DRY_RUN", False))
    debug: bool = field(default_factory=lambda: _env_bool("JOB_ENGINE_DEBUG", False))
    persist_results: bool = field(
        default_factory=lambda: _env_bool("JOB_ENGINE_PERSIST_RESULTS", True)
    )
    degrade_on_llm_failure: bool = field(
        default_factory=lambda: _env_bool("JOB_ENGINE_DEGRADE_ON_LLM_FAILURE", True)
    )

    def __post_init__(self) -> None:
        total = self.weights.total()
        if not (0.98 <= total <= 1.02):
            raise ValueError(
                f"Matching weights must sum to ~1.0, got {total} ({self.weights.as_dict()})"
            )


@lru_cache(maxsize=1)
def get_settings() -> EngineSettings:
    return EngineSettings()
