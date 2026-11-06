"""Environment-driven settings for Stock Finder.

Follows the pattern used by ``code_assistant/config.py``: a frozen dataclass
populated from env vars via small typed helpers, validated at startup and
cached with ``get_settings()``. The API keys are read here but never logged.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from llm_provider import resolve_embedding_model, resolve_model, warn_if_shared_model

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = Path(__file__).resolve().parent

MARKETS = ("IN", "US")
RISK_PROFILES = ("conservative", "moderate", "aggressive")
SCORE_COMPONENTS = ("fundamentals", "valuation", "momentum", "news_sentiment", "risk")

# Default composite-score weights per risk profile (each row sums to 1.0).
DEFAULT_SCORE_WEIGHTS: dict[str, dict[str, float]] = {
    "conservative": {"fundamentals": 0.30, "valuation": 0.25, "momentum": 0.10, "news_sentiment": 0.10, "risk": 0.25},
    "moderate": {"fundamentals": 0.30, "valuation": 0.20, "momentum": 0.20, "news_sentiment": 0.15, "risk": 0.15},
    "aggressive": {"fundamentals": 0.25, "valuation": 0.15, "momentum": 0.30, "news_sentiment": 0.20, "risk": 0.10},
}

DISCLAIMERS = {
    "IN": (
        "This is AI-generated research for informational purposes only and is not financial advice. "
        "Please do your own research or consult a SEBI-registered investment adviser before investing."
    ),
    "US": (
        "This is AI-generated research for informational purposes only and is not financial advice. "
        "Please do your own research or consult an SEC-registered investment adviser or FINRA-registered "
        "broker-dealer before investing."
    ),
}


class StockFinderConfigError(ValueError):
    """Raised at startup when the Stock Finder configuration is invalid."""


def _env_str(name: str, default: str) -> str:
    raw = (os.getenv(name) or "").strip()
    return raw if raw else default


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError as exc:
        raise StockFinderConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    raw = (os.getenv(name) or "").strip()
    try:
        value = float(raw) if raw else default
    except ValueError as exc:
        raise StockFinderConfigError(f"{name} must be a number, got {raw!r}") from exc
    if minimum is not None:
        value = max(minimum, value)
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _env_list(name: str, default: str) -> tuple[str, ...]:
    raw = _env_str(name, default)
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _resolve_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_prefix(raw: str) -> str:
    prefix = "/" + raw.strip().strip("/")
    if prefix == "/":
        raise StockFinderConfigError("SF_URL_PREFIX must not be '/' (it would shadow the existing apps)")
    return prefix


def _score_weights() -> dict[str, dict[str, float]]:
    """``SF_SCORE_WEIGHTS`` (JSON, optional) overrides the defaults per profile; rows are re-normalized."""
    weights = {profile: dict(row) for profile, row in DEFAULT_SCORE_WEIGHTS.items()}
    raw = (os.getenv("SF_SCORE_WEIGHTS") or "").strip()
    if raw:
        try:
            override = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StockFinderConfigError(f"SF_SCORE_WEIGHTS must be JSON: {exc}") from exc
        if not isinstance(override, dict):
            raise StockFinderConfigError("SF_SCORE_WEIGHTS must be a JSON object keyed by risk profile")
        for profile, row in override.items():
            if profile not in RISK_PROFILES or not isinstance(row, dict):
                raise StockFinderConfigError(f"SF_SCORE_WEIGHTS: unknown risk profile {profile!r}")
            for key, value in row.items():
                if key not in SCORE_COMPONENTS or not isinstance(value, (int, float)) or value < 0:
                    raise StockFinderConfigError(f"SF_SCORE_WEIGHTS[{profile}]: invalid weight {key}={value!r}")
                weights[profile][key] = float(value)
    for profile, row in weights.items():
        total = sum(row.values())
        if total <= 0:
            raise StockFinderConfigError(f"SF_SCORE_WEIGHTS[{profile}] must not be all zero")
        weights[profile] = {k: round(v / total, 4) for k, v in row.items()}
    return weights


@dataclass(frozen=True)
class StockFinderSettings:
    """All tunables for the Stock Finder module."""

    openai_api_key: str = field(repr=False)
    newsapi_key: str = field(repr=False)
    mongodb_uri: str = field(repr=False)
    router_model: str
    analyst_model: str
    worker_model: str
    critic_model: str
    fallback_model: str
    embedding_model: str
    market: str
    exchange_suffix: str
    market_data_providers: tuple[str, ...]
    news_providers: tuple[str, ...]
    max_candidates: int
    max_parallel_workers: int
    news_lookback_days: int
    max_llm_calls: int
    risk_profile_default: str
    url_prefix: str
    mongodb_db_name: str
    feeds_path: Path
    universe_path: Path
    request_timeout: float
    data_timeout: float
    max_retries: int
    ui_enabled: bool
    max_stock_pct: float
    max_sector_pct: float
    max_peers: int
    screener_max_steps: int
    penny_price_in: float
    penny_price_us: float
    score_weights: dict[str, dict[str, float]]
    job_workers: int = 2
    circuit_failure_threshold: int = 5
    circuit_cooldown_seconds: float = 60.0
    max_news_for_analysis: int = 60
    news_batch_size: int = 15
    max_company_articles: int = 8

    @property
    def disclaimer(self) -> str:
        return DISCLAIMERS[self.market]

    @property
    def penny_price(self) -> float:
        return self.penny_price_in if self.market == "IN" else self.penny_price_us

    @property
    def currency(self) -> str:
        return "INR" if self.market == "IN" else "USD"

    def models(self) -> dict[str, str]:
        """Model ids in use (saved with every report for auditability) — the ones that actually
        serve the calls, so on LLM_PROVIDER=ollama these are the mapped local models."""
        return {
            "router": resolve_model(self.router_model), "analyst": resolve_model(self.analyst_model),
            "worker": resolve_model(self.worker_model), "critic": resolve_model(self.critic_model),
            "fallback": resolve_model(self.fallback_model), "embedding": resolve_embedding_model(self.embedding_model),
        }

    def validate(self) -> None:
        """Fail fast on configuration that would make the critic meaningless or the market ambiguous."""
        if self.critic_model.strip().lower() == self.analyst_model.strip().lower():
            raise StockFinderConfigError(
                "SF_CRITIC_MODEL and SF_ANALYST_MODEL must be different models "
                f"(both are {self.analyst_model!r}); the critic must be an independent reviewer."
            )
        if self.market not in MARKETS:
            raise StockFinderConfigError(f"SF_MARKET must be one of {MARKETS}, got {self.market!r}")
        if self.risk_profile_default not in RISK_PROFILES:
            raise StockFinderConfigError(
                f"SF_RISK_PROFILE_DEFAULT must be one of {RISK_PROFILES}, got {self.risk_profile_default!r}"
            )
        if not self.market_data_providers:
            raise StockFinderConfigError("SF_MARKET_DATA_PROVIDER must name at least one provider")

    def warn_if_critic_shared(self) -> None:
        """On Ollama, warn if analyst and critic map to one local model (called on first use, not at startup)."""
        warn_if_shared_model("Stock Finder", {"SF_ANALYST_MODEL": self.analyst_model, "SF_CRITIC_MODEL": self.critic_model})


def load_settings() -> StockFinderSettings:
    """Build and validate settings from the current environment (uncached)."""
    market = _env_str("SF_MARKET", "IN").upper()
    settings = StockFinderSettings(
        openai_api_key=(os.getenv("OPENAI_API_KEY") or "").strip(),
        newsapi_key=(os.getenv("NEWSAPI_KEY") or "").strip(),
        # Own connection settings; falls back to the hub-wide MongoDB used by jobEngine.
        mongodb_uri=(os.getenv("SF_MONGODB_URI") or os.getenv("MONGODB_URI") or "").strip(),
        router_model=_env_str("SF_ROUTER_MODEL", "gpt-4o-mini"),
        analyst_model=_env_str("SF_ANALYST_MODEL", "gpt-4o"),
        worker_model=_env_str("SF_WORKER_MODEL", "gpt-4o-mini"),
        # The spec's o3-mini is deprecated by OpenAI (404); gpt-4.1 is a different model family from the analyst.
        critic_model=_env_str("SF_CRITIC_MODEL", "gpt-4.1"),
        fallback_model=_env_str("SF_FALLBACK_MODEL", "gpt-4o-mini"),
        embedding_model=_env_str("SF_EMBEDDING_MODEL", "text-embedding-3-small"),
        market=market,
        exchange_suffix=_env_str("SF_EXCHANGE_SUFFIX", ".NS" if market == "IN" else ""),
        market_data_providers=_env_list("SF_MARKET_DATA_PROVIDER", "yfinance"),
        news_providers=_env_list("SF_NEWS_PROVIDERS", "rss,newsapi"),
        max_candidates=_env_int("SF_MAX_CANDIDATES", 8, minimum=1),
        max_parallel_workers=_env_int("SF_MAX_PARALLEL_WORKERS", 4, minimum=1),
        news_lookback_days=_env_int("SF_NEWS_LOOKBACK_DAYS", 7, minimum=1),
        max_llm_calls=_env_int("SF_MAX_LLM_CALLS", 60, minimum=1),
        risk_profile_default=_env_str("SF_RISK_PROFILE_DEFAULT", "moderate").lower(),
        url_prefix=_normalize_prefix(_env_str("SF_URL_PREFIX", "/stock-finder")),
        mongodb_db_name=_env_str("SF_MONGODB_DB_NAME", "stock_research"),
        feeds_path=_resolve_path(_env_str("SF_FEEDS_FILE", "config/stock_finder_feeds.yaml")),
        universe_path=_resolve_path(_env_str("SF_UNIVERSE_FILE", "config/stock_finder_universe.yaml")),
        request_timeout=_env_float("SF_REQUEST_TIMEOUT", 90.0, minimum=1.0),
        data_timeout=_env_float("SF_DATA_TIMEOUT", 15.0, minimum=1.0),
        max_retries=_env_int("SF_MAX_RETRIES", 3, minimum=0),
        ui_enabled=_env_bool("SF_UI_ENABLED", True),
        max_stock_pct=_env_float("SF_MAX_STOCK_PCT", 25.0, minimum=1.0),
        max_sector_pct=_env_float("SF_MAX_SECTOR_PCT", 40.0, minimum=1.0),
        max_peers=_env_int("SF_MAX_PEERS", 5, minimum=0),
        screener_max_steps=_env_int("SF_SCREENER_MAX_STEPS", 8, minimum=1),
        penny_price_in=_env_float("SF_PENNY_PRICE_IN", 20.0, minimum=0.0),
        penny_price_us=_env_float("SF_PENNY_PRICE_US", 5.0, minimum=0.0),
        score_weights=_score_weights(),
    )
    settings.validate()
    return settings


@lru_cache(maxsize=1)
def get_settings() -> StockFinderSettings:
    """Cached settings; call ``get_settings.cache_clear()`` after changing env vars."""
    return load_settings()
