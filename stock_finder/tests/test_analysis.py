"""Indicators, fundamentals and scoring on fixed sample data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_finder.analysis.fundamentals import compute_fundamentals, sector_median
from stock_finder.analysis.indicators import compute_indicators, macd, max_drawdown, rsi
from stock_finder.analysis.scoring import linear, news_sentiment, score_company, verdict_consistent
from stock_finder.data.market_data import Fundamentals
from stock_finder.data.news_data import make_article


def _frame(close) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range("2025-01-01", periods=len(close))
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000}, index=idx)


# --- indicators -----------------------------------------------------------------------------------

def test_moving_averages_and_returns_on_linear_series():
    t = compute_indicators(_frame(range(1, 301)))
    assert t.last_close == 300
    assert t.sma_50 == 275.5  # mean of 251..300
    assert t.sma_200 == 200.5  # mean of 101..300
    assert t.above_sma50 and t.above_sma200 and t.golden_cross
    assert t.rsi_14 == 100.0  # only gains
    assert t.max_drawdown_pct == 0.0
    assert t.return_1m_pct == pytest.approx((300 / 279 - 1) * 100, abs=0.01)
    assert t.high_52w == 300 and t.low_52w == 49  # last 252 bars: 49..300
    assert t.pct_from_52w_high == 0.0
    assert t.history_days == 300


def test_rsi_alternating_is_neutral():
    close = pd.Series([100 + (i % 2) for i in range(200)], dtype=float)
    assert rsi(close) == pytest.approx(50, abs=3)


def test_rsi_falling_is_zero():
    assert rsi(pd.Series(np.arange(100, 50, -1), dtype=float)) == 0.0


def test_macd_positive_in_uptrend_and_drawdown():
    up = pd.Series(np.linspace(100, 200, 120))
    line, signal, hist = macd(up)
    assert line > 0 and hist >= 0
    assert max_drawdown(pd.Series([100, 120, 90, 110], dtype=float)) == pytest.approx(-25.0)


def test_relative_strength_vs_index():
    stock = _frame(np.linspace(100, 130, 200))
    index = _frame(np.linspace(100, 110, 200))
    t = compute_indicators(stock, index)
    expected = (stock.close.iloc[-1] / stock.close.iloc[-64] - 1) * 100 - (index.close.iloc[-1] / index.close.iloc[-64] - 1) * 100
    assert t.rs_vs_index_3m_pct == pytest.approx(expected, abs=0.01)
    assert t.sma_200 == pytest.approx(115.0, abs=0.01)


def test_short_history_gives_none():
    t = compute_indicators(_frame(range(1, 31)))  # MACD needs 26 + 9 bars
    assert t.sma_50 is None and t.sma_200 is None and t.macd is None and t.rsi_14 == 100.0
    assert compute_indicators(None).last_close is None


# --- fundamentals ---------------------------------------------------------------------------------

def _fund(symbol="X.NS", **kw) -> Fundamentals:
    base = dict(symbol=symbol, name=symbol, pe=20.0, pb=4.0, roe=0.18, debt_to_equity=0.5, revenue_growth=0.12,
                earnings_growth=0.2, profit_margin=0.15, operating_margin=0.2, free_cash_flow=5e10, total_revenue=5e11,
                ebit=1e11, total_assets=8e11, current_liabilities=3e11, market_cap=2e12, insider_holding=0.55, beta=1.1)
    base.update(kw)
    return Fundamentals(**base)


def test_fundamental_ratios_and_sector_comparison():
    peers = [_fund("A.NS", pe=10.0, pb=2.0), _fund("B.NS", pe=30.0, pb=6.0), _fund("C.NS", pe=-5.0, pb=None)]
    m = compute_fundamentals(_fund(pe=25.0, pb=5.0), peers, market="IN")
    assert m.roe_pct == 18.0 and m.revenue_growth_pct == 12.0 and m.profit_growth_pct == 20.0
    assert m.roce_pct == 20.0  # 1e11 / (8e11 - 3e11)
    assert m.sector_median_pe == 20.0  # negative P/E excluded
    assert m.pe_vs_sector_pct == 25.0
    assert m.sector_median_pb == 4.0 and m.pb_vs_sector_pct == 25.0
    assert m.free_cash_flow == 5000.0 and m.money_unit == "INR crore"  # 5e10 / 1e7
    assert m.market_cap == 200000.0
    assert m.fcf_margin_pct == 10.0 and m.fcf_positive
    assert m.insider_or_promoter_holding_pct == 55.0
    assert "pledged_pct" in m.unavailable


def test_sector_median_needs_two_peers():
    assert sector_median([15.0]) is None
    assert sector_median([10.0, 20.0, None]) == 15.0


def test_us_money_unit():
    assert compute_fundamentals(_fund(market_cap=3e12), market="US").market_cap == 3000.0


# --- scoring --------------------------------------------------------------------------------------

def test_linear_mapping():
    assert linear(25, 0, 25) == 100 and linear(-1, 0, 25) == 0 and linear(1.0, 2.0, 0.0) == 50


def test_score_breakdown_is_transparent(settings):
    m = compute_fundamentals(_fund(), [_fund("A.NS", pe=25.0), _fund("B.NS", pe=15.0)])
    t = compute_indicators(_frame(np.linspace(100, 150, 300)))
    score = score_company(m, t, [], risk_profile="moderate", weights=settings.score_weights)
    for sub in (score.fundamentals, score.valuation, score.momentum, score.news_sentiment, score.risk, score.composite):
        assert sub is not None and 0 <= sub <= 100
    assert score.news_sentiment == 50.0 and any("no recent company news" in n for n in score.notes)
    w = settings.score_weights["moderate"]
    expected = sum(w[k] * getattr(score, k) for k in w) / sum(w.values())
    assert score.composite == pytest.approx(expected, abs=0.1)


def test_risk_profile_changes_composite(settings):
    m = compute_fundamentals(_fund(), [])
    t = compute_indicators(_frame(np.linspace(100, 150, 300)))  # strong momentum
    aggressive = score_company(m, t, [], risk_profile="aggressive", weights=settings.score_weights).composite
    conservative = score_company(m, t, [], risk_profile="conservative", weights=settings.score_weights).composite
    assert aggressive != conservative


def test_missing_data_renormalizes(settings):
    score = score_company(compute_fundamentals(None), compute_indicators(None), [], risk_profile="moderate",
                          weights=settings.score_weights)
    assert score.fundamentals is None and score.momentum is None and score.composite == 50.0  # only news
    assert any("missing data" in n for n in score.notes)


def test_news_sentiment_lexicon():
    pos = make_article({"source": "s", "title": "Infosys shares surge after record deal win", "url": "https://x.com/1"}, "t")
    neg = make_article({"source": "s", "title": "Company faces fraud probe, shares plunge", "url": "https://x.com/2"}, "t")
    assert news_sentiment([pos])[0] > 50 > news_sentiment([neg])[0]


def test_verdict_consistency():
    assert verdict_consistent("STRONG_BUY_CANDIDATE", 70, "good")[0]
    assert not verdict_consistent("STRONG_BUY_CANDIDATE", 50, "good")[0]
    assert not verdict_consistent("STRONG_BUY_CANDIDATE", 70, "poor")[0]
    assert not verdict_consistent("AVOID", 80, "good")[0]
    assert verdict_consistent("WATCHLIST", 10, "poor")[0]
