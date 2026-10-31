"""Transparent, deterministic composite score.

Five sub-scores (0-100, higher is better): fundamentals, valuation, momentum,
news sentiment and risk (high = *lower* risk). Each is the mean of simple,
documented component rules; a component whose input is missing is skipped, and
a sub-score with no inputs is ``None`` and drops out of the composite (the
remaining weights are re-normalized). Weights come from the user's risk profile
(``SF_SCORE_WEIGHTS``). Every rule that fired is written to ``notes`` so the
analyst and the UI can explain the score.
"""

from __future__ import annotations

import re
from typing import Iterable

from ..data.news_data import Article
from ..schemas import QualityLevel, ScoreBreakdown
from .fundamentals import FundamentalMetrics
from .indicators import TechnicalIndicators

POSITIVE_WORDS = (
    "beat", "beats", "surge", "surges", "soar", "rally", "rallies", "upgrade", "upgraded", "record", "growth",
    "rise", "rises", "jump", "jumps", "gain", "gains", "wins", "win", "order", "expansion", "approval", "approved",
    "profit up", "strong", "outperform", "buyback", "dividend", "raises guidance", "bullish", "boost",
)
NEGATIVE_WORDS = (
    "miss", "misses", "plunge", "plunges", "slump", "downgrade", "downgraded", "loss", "losses", "probe", "fraud",
    "default", "penalty", "decline", "declines", "fall", "falls", "drop", "drops", "resign", "resigns",
    "lawsuit", "ban", "raid", "weak", "underperform", "bearish", "warning", "pledge",
)
STRONG_BUY_MIN_SCORE = 60.0


def linear(value: float, worst: float, best: float) -> float:
    """Map ``value`` to 0-100 linearly between ``worst`` (0) and ``best`` (100); works when worst > best."""
    if best == worst:
        return 50.0
    return max(0.0, min(100.0, (value - worst) / (best - worst) * 100))


def _mean(parts: list[float]) -> float | None:
    return round(sum(parts) / len(parts), 1) if parts else None


def fundamentals_score(m: FundamentalMetrics, notes: list[str]) -> float | None:
    parts: list[float] = []
    if m.roe_pct is not None:
        parts.append(linear(m.roe_pct, 0, 25))
        notes.append(f"fundamentals: ROE {m.roe_pct}% (0% -> 0, 25%+ -> 100)")
    if m.roce_pct is not None:
        parts.append(linear(m.roce_pct, 0, 25))
    if m.revenue_growth_pct is not None:
        parts.append(linear(m.revenue_growth_pct, -5, 25))
        notes.append(f"fundamentals: revenue growth {m.revenue_growth_pct}%")
    if m.profit_growth_pct is not None:
        parts.append(linear(m.profit_growth_pct, -10, 30))
    if m.profit_margin_pct is not None:
        parts.append(linear(m.profit_margin_pct, 0, 25))
    if m.debt_to_equity is not None:
        parts.append(linear(m.debt_to_equity, 2.0, 0.0))
        notes.append(f"fundamentals: debt/equity {m.debt_to_equity} (2.0+ -> 0, 0 -> 100)")
    if m.fcf_positive is not None:
        parts.append(100.0 if m.fcf_positive else 20.0)
    return _mean(parts)


def valuation_score(m: FundamentalMetrics, notes: list[str]) -> float | None:
    parts: list[float] = []
    if m.pe is not None:
        if m.pe <= 0:
            parts.append(10.0)
            notes.append("valuation: negative earnings (P/E not meaningful)")
        else:
            parts.append(linear(m.pe, 60, 10))
    if m.pe_vs_sector_pct is not None:
        parts.append(linear(m.pe_vs_sector_pct, 50, -30))
        notes.append(f"valuation: P/E {m.pe} vs sector median {m.sector_median_pe} ({m.pe_vs_sector_pct}% premium/discount)")
    if m.pb_vs_sector_pct is not None:
        parts.append(linear(m.pb_vs_sector_pct, 50, -30))
    return _mean(parts)


def _rsi_points(value: float) -> float:
    if 45 <= value <= 65:
        return 100.0
    if 30 <= value < 45:
        return linear(value, 30, 45) * 0.4 + 60
    if 65 < value <= 75:
        return linear(value, 75, 65) * 0.5 + 50
    return 30.0 if value > 75 else 40.0


def momentum_score(t: TechnicalIndicators, notes: list[str]) -> float | None:
    parts: list[float] = []
    if t.above_sma200 is not None:
        parts.append(100.0 if t.above_sma200 else 0.0)
        notes.append(f"momentum: price {'above' if t.above_sma200 else 'below'} 200-day average ({t.price_vs_sma200_pct}%)")
    if t.above_sma50 is not None:
        parts.append(100.0 if t.above_sma50 else 0.0)
    if t.rsi_14 is not None:
        parts.append(_rsi_points(t.rsi_14))
        notes.append(f"momentum: RSI(14) {t.rsi_14} (45-65 ideal; above 75 overbought)")
    if t.macd_hist is not None:
        parts.append(100.0 if t.macd_hist > 0 else 30.0)
    if t.rs_vs_index_3m_pct is not None:
        parts.append(linear(t.rs_vs_index_3m_pct, -10, 10))
        notes.append(f"momentum: 3-month relative strength vs index {t.rs_vs_index_3m_pct} pts")
    if t.return_3m_pct is not None:
        parts.append(linear(t.return_3m_pct, -15, 20))
    return _mean(parts)


def news_sentiment(articles: Iterable[Article]) -> tuple[float, int, int]:
    """Lexicon score 0-100 (50 = neutral) shrunk toward neutral when there are few hits."""
    pos = neg = 0
    for article in articles:
        text = f"{article.title} {article.summary}".lower()
        pos += sum(1 for w in POSITIVE_WORDS if re.search(rf"\b{re.escape(w)}\b", text))
        neg += sum(1 for w in NEGATIVE_WORDS if re.search(rf"\b{re.escape(w)}\b", text))
    return round(50 + 50 * (pos - neg) / (pos + neg + 2), 1), pos, neg


def news_score(articles: list[Article], notes: list[str]) -> float:
    if not articles:
        notes.append("news: no recent company news (neutral 50)")
        return 50.0
    score, pos, neg = news_sentiment(articles)
    notes.append(f"news: {len(articles)} articles, {pos} positive / {neg} negative keyword hits")
    return score


def risk_score(t: TechnicalIndicators, m: FundamentalMetrics, notes: list[str]) -> float | None:
    parts: list[float] = []
    if t.volatility_ann_pct is not None:
        parts.append(linear(t.volatility_ann_pct, 60, 15))
        notes.append(f"risk: annualized volatility {t.volatility_ann_pct}% (15% -> 100, 60%+ -> 0)")
    if t.max_drawdown_pct is not None:
        parts.append(linear(t.max_drawdown_pct, -50, -10))
        notes.append(f"risk: 1-year max drawdown {t.max_drawdown_pct}%")
    if m.debt_to_equity is not None:
        parts.append(linear(m.debt_to_equity, 2.5, 0.2))
    if m.beta is not None:
        parts.append(linear(m.beta, 2.0, 0.6))
    if m.pledged_pct is not None and m.pledged_pct > 0:
        parts.append(linear(m.pledged_pct, 30, 0))
    return _mean(parts)


def score_company(
    metrics: FundamentalMetrics,
    technicals: TechnicalIndicators,
    articles: list[Article],
    *,
    risk_profile: str,
    weights: dict[str, dict[str, float]],
) -> ScoreBreakdown:
    """Weighted composite of the five sub-scores for ``risk_profile``."""
    notes: list[str] = []
    subs = {
        "fundamentals": fundamentals_score(metrics, notes),
        "valuation": valuation_score(metrics, notes),
        "momentum": momentum_score(technicals, notes),
        "news_sentiment": news_score(articles, notes),
        "risk": risk_score(technicals, metrics, notes),
    }
    profile_weights = weights[risk_profile]
    available = {k: v for k, v in subs.items() if v is not None and profile_weights.get(k, 0) > 0}
    total_weight = sum(profile_weights[k] for k in available)
    composite = round(sum(profile_weights[k] * v for k, v in available.items()) / total_weight, 1) if total_weight else None
    missing = [k for k, v in subs.items() if v is None]
    if missing:
        notes.append(f"not scored (missing data): {', '.join(missing)}; weights re-normalized")
    return ScoreBreakdown(**subs, composite=composite, risk_profile=risk_profile, weights=profile_weights, notes=notes)


def verdict_consistent(verdict: str, composite: float | None, quality: QualityLevel) -> tuple[bool, str]:
    """Deterministic guardrail: is the verdict compatible with the score and data quality?"""
    if verdict == "STRONG_BUY_CANDIDATE":
        if composite is None or composite < STRONG_BUY_MIN_SCORE:
            return False, f"STRONG_BUY_CANDIDATE needs a composite score of at least {STRONG_BUY_MIN_SCORE:.0f} (got {composite})"
        if quality == "poor":
            return False, "STRONG_BUY_CANDIDATE is not allowed when data quality is poor"
    if verdict == "AVOID" and composite is not None and composite >= 75:
        return False, f"AVOID contradicts a high composite score ({composite}) without a stated overriding risk"
    return True, ""
