"""Technical indicators computed in Python from price history (never by the LLM).

All outputs are rounded to 2 decimals so the exact values the LLM sees are the
values the critic checks its numbers against. Anything that needs more history
than is available is ``None`` rather than a guess.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel

TRADING_DAYS = 252


class TechnicalIndicators(BaseModel):
    last_close: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    price_vs_sma50_pct: Optional[float] = None
    price_vs_sma200_pct: Optional[float] = None
    above_sma50: Optional[bool] = None
    above_sma200: Optional[bool] = None
    golden_cross: Optional[bool] = None
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    pct_from_52w_high: Optional[float] = None
    pct_from_52w_low: Optional[float] = None
    volatility_ann_pct: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    return_1m_pct: Optional[float] = None
    return_3m_pct: Optional[float] = None
    return_6m_pct: Optional[float] = None
    return_1y_pct: Optional[float] = None
    rs_vs_index_3m_pct: Optional[float] = None
    rs_vs_index_6m_pct: Optional[float] = None
    history_days: int = 0


def _r(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def sma(close: pd.Series, window: int) -> float | None:
    if len(close) < window:
        return None
    return float(close.iloc[-window:].mean())


def rsi(close: pd.Series, period: int = 14) -> float | None:
    """Wilder's RSI on closing prices."""
    if len(close) <= period:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean().iloc[-1]
    if loss == 0:
        return 100.0 if gain > 0 else 50.0
    return float(100 - 100 / (1 + gain / loss))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float, float, float] | None:
    """(MACD line, signal line, histogram) for the last bar."""
    if len(close) < slow + signal:
        return None
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return float(line.iloc[-1]), float(sig.iloc[-1]), float(line.iloc[-1] - sig.iloc[-1])


def period_return(close: pd.Series, days: int) -> float | None:
    """Percent return over the last ``days`` trading days."""
    if len(close) <= days:
        return None
    return float((close.iloc[-1] / close.iloc[-1 - days] - 1) * 100)


def annualized_volatility(close: pd.Series, window: int = TRADING_DAYS) -> float | None:
    returns = np.log(close.iloc[-(window + 1):]).diff().dropna()
    if len(returns) < 20:
        return None
    return float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS) * 100)


def max_drawdown(close: pd.Series, window: int = TRADING_DAYS) -> float | None:
    """Largest peak-to-trough fall (negative percent) over ``window`` days."""
    recent = close.iloc[-window:]
    if len(recent) < 2:
        return None
    return float((recent / recent.cummax() - 1).min() * 100)


def relative_strength(close: pd.Series, index_close: pd.Series | None, days: int) -> float | None:
    """Stock return minus index return over the same dates (percentage points)."""
    if index_close is None or index_close.empty:
        return None
    left = close.copy()
    right = index_close.copy()
    left.index = pd.to_datetime(left.index).normalize()
    right.index = pd.to_datetime(right.index).normalize()
    joined = pd.concat([left.rename("s"), right.rename("i")], axis=1, join="inner").dropna()
    stock, index = period_return(joined["s"], days), period_return(joined["i"], days)
    if stock is None or index is None:
        return None
    return stock - index


def compute_indicators(history: pd.DataFrame | None, index_history: pd.DataFrame | None = None) -> TechnicalIndicators:
    """All indicators for one stock; missing history yields ``None`` fields."""
    if history is None or history.empty or "close" not in history:
        return TechnicalIndicators()
    close = history["close"].astype(float).dropna()
    if close.empty:
        return TechnicalIndicators()
    last = float(close.iloc[-1])
    year = history.iloc[-TRADING_DAYS:]
    high = float((year["high"] if "high" in year else year["close"]).max())
    low = float((year["low"] if "low" in year else year["close"]).min())
    s50, s200 = sma(close, 50), sma(close, 200)
    m = macd(close)
    index_close = index_history["close"].astype(float) if index_history is not None and "close" in index_history else None
    return TechnicalIndicators(
        last_close=_r(last),
        sma_50=_r(s50),
        sma_200=_r(s200),
        price_vs_sma50_pct=_r((last / s50 - 1) * 100) if s50 else None,
        price_vs_sma200_pct=_r((last / s200 - 1) * 100) if s200 else None,
        above_sma50=last > s50 if s50 else None,
        above_sma200=last > s200 if s200 else None,
        golden_cross=s50 > s200 if s50 and s200 else None,
        rsi_14=_r(rsi(close)),
        macd=_r(m[0]) if m else None,
        macd_signal=_r(m[1]) if m else None,
        macd_hist=_r(m[2]) if m else None,
        high_52w=_r(high),
        low_52w=_r(low),
        pct_from_52w_high=_r((last / high - 1) * 100) if high else None,
        pct_from_52w_low=_r((last / low - 1) * 100) if low else None,
        volatility_ann_pct=_r(annualized_volatility(close)),
        max_drawdown_pct=_r(max_drawdown(close)),
        return_1m_pct=_r(period_return(close, 21)),
        return_3m_pct=_r(period_return(close, 63)),
        return_6m_pct=_r(period_return(close, 126)),
        return_1y_pct=_r(period_return(close, TRADING_DAYS - 1)),
        rs_vs_index_3m_pct=_r(relative_strength(close, index_close, 63)),
        rs_vs_index_6m_pct=_r(relative_strength(close, index_close, 126)),
        history_days=len(close),
    )
