"""Data freshness and sanity checks; every company gets a ``DataQuality`` flag.

- Prices older than 1 trading day are stale (weekends are skipped; exchange
  holidays are not modelled, so the day after a holiday may be flagged).
- Obviously invalid values (non-positive prices, absurd ratios) are dropped,
  never passed to the LLM, and recorded as issues.
- Missing key fundamentals lower the quality level.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..schemas import DataQuality

KEY_FUNDAMENTALS = ("pe", "pb", "roe", "debt_to_equity", "revenue_growth", "profit_margin")

# (field, min, max) — values outside are treated as provider errors and dropped.
_SANE_RANGES: dict[str, tuple[float, float]] = {
    "pe": (-1000.0, 1000.0),
    "pb": (0.0, 500.0),
    "roe": (-5.0, 5.0),
    "debt_to_equity": (0.0, 50.0),
    "revenue_growth": (-1.0, 20.0),
    "earnings_growth": (-50.0, 50.0),
    "profit_margin": (-10.0, 1.0),
    "operating_margin": (-10.0, 1.0),
    "insider_holding": (0.0, 1.0),
    "beta": (-5.0, 10.0),
    "dividend_yield": (0.0, 1.0),
}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def trading_days_between(earlier: date, later: date) -> int:
    """Business days strictly after ``earlier`` up to and including ``later`` (0 if same day)."""
    if later <= earlier:
        return 0
    return int(np.busday_count(earlier, later))


def check_price(price: Any, as_of: datetime | None, *, now: datetime | None = None) -> tuple[list[str], bool]:
    """Issues for a quote and whether it is stale (older than 1 trading day)."""
    issues: list[str] = []
    if not is_number(price) or price <= 0:
        issues.append("price missing or invalid")
    stale = False
    if as_of is None:
        issues.append("price timestamp missing")
        stale = True
    else:
        now = now or datetime.now(timezone.utc)
        as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=timezone.utc)
        age = trading_days_between(as_of_utc.date(), now.date())
        if age > 1:
            stale = True
            issues.append(f"price is stale (last update {as_of_utc.date().isoformat()}, {age} trading days old)")
    return issues, stale


def sanitize_fundamentals(values: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Drop invalid ratios; report missing key fields. Returns ``(clean_values, issues)``."""
    clean = dict(values)
    issues: list[str] = []
    for name, (low, high) in _SANE_RANGES.items():
        value = clean.get(name)
        if value is None:
            continue
        if not is_number(value) or not (low <= value <= high):
            issues.append(f"{name} value {value!r} rejected as invalid")
            clean[name] = None
    for name in ("market_cap", "free_cash_flow", "total_revenue", "ebit", "total_assets", "current_liabilities"):
        if clean.get(name) is not None and not is_number(clean[name]):
            clean[name] = None
    missing = [name for name in KEY_FUNDAMENTALS if clean.get(name) is None]
    if missing:
        issues.append(f"missing fundamentals: {', '.join(missing)}")
    return clean, issues


def check_history(history: pd.DataFrame | None) -> list[str]:
    if history is None or history.empty:
        return ["price history unavailable"]
    if len(history) < 200:
        return [f"only {len(history)} days of price history (200-day indicators unavailable)"]
    return []


def build_quality(
    *,
    price_issues: Iterable[str] = (),
    stale: bool = False,
    fundamentals_issues: Iterable[str] = (),
    history_issues: Iterable[str] = (),
    other_issues: Iterable[str] = (),
    as_of: datetime | None = None,
    has_price: bool = True,
    has_fundamentals: bool = True,
) -> DataQuality:
    """Combine checks into one flag: poor if no price / fundamentals or stale, partial if anything is missing."""
    issues = [*price_issues, *fundamentals_issues, *history_issues, *other_issues]
    missing_many = any(i.startswith("missing fundamentals") and i.count(",") >= 3 for i in fundamentals_issues)
    if not has_price or not has_fundamentals or stale or missing_many:
        level = "poor"
    elif issues:
        level = "partial"
    else:
        level = "good"
    return DataQuality(level=level, stale=stale, issues=issues, as_of=as_of.isoformat() if as_of else None)
