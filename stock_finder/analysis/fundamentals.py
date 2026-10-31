"""Fundamental ratios and sector comparison computed in Python (never by the LLM).

Raw provider fractions (0.12) become display percentages (12.0); money values
are scaled to one unit per market (INR crore / USD billion) so the model never
has to convert units. P/E and P/B are compared against the median of the
company's sector peers.
"""

from __future__ import annotations

import statistics
from typing import Iterable, Optional

from pydantic import BaseModel, Field

from ..data.market_data import Fundamentals

MONEY_UNITS = {"IN": ("INR crore", 1e7), "US": ("USD billion", 1e9)}


class FundamentalMetrics(BaseModel):
    revenue_growth_pct: Optional[float] = None
    profit_growth_pct: Optional[float] = None
    profit_margin_pct: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    roe_pct: Optional[float] = None
    roce_pct: Optional[float] = None
    debt_to_equity: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    sector_median_pe: Optional[float] = None
    sector_median_pb: Optional[float] = None
    pe_vs_sector_pct: Optional[float] = None
    pb_vs_sector_pct: Optional[float] = None
    free_cash_flow: Optional[float] = None
    fcf_margin_pct: Optional[float] = None
    fcf_positive: Optional[bool] = None
    market_cap: Optional[float] = None
    money_unit: str = ""
    insider_or_promoter_holding_pct: Optional[float] = None
    pledged_pct: Optional[float] = None
    dividend_yield_pct: Optional[float] = None
    beta: Optional[float] = None
    peers_compared: list[str] = Field(default_factory=list)
    unavailable: list[str] = Field(default_factory=list)


def pct(value: float | None) -> float | None:
    return round(value * 100, 2) if value is not None else None


def _round(value: float | None, digits: int = 2) -> float | None:
    return round(value, digits) if value is not None else None


def sector_median(values: Iterable[float | None], *, positive_only: bool = True) -> float | None:
    """Median of available peer values (needs at least 2); negative P/E (losses) excluded."""
    usable = [v for v in values if v is not None and (v > 0 or not positive_only)]
    return round(statistics.median(usable), 2) if len(usable) >= 2 else None


def compute_fundamentals(fund: Fundamentals | None, peers: Iterable[Fundamentals] = (), *, market: str = "IN") -> FundamentalMetrics:
    """Display-ready fundamentals plus P/E and P/B versus the sector median."""
    unit, divisor = MONEY_UNITS.get(market, MONEY_UNITS["IN"])
    if fund is None:
        return FundamentalMetrics(money_unit=unit, unavailable=["all fundamentals"])
    peers = [p for p in peers if p.symbol != fund.symbol]
    median_pe = sector_median(p.pe for p in peers)
    median_pb = sector_median(p.pb for p in peers)
    roce = None
    if fund.ebit is not None and fund.total_assets and fund.current_liabilities is not None:
        capital_employed = fund.total_assets - fund.current_liabilities
        if capital_employed > 0:
            roce = fund.ebit / capital_employed
    fcf_margin = fund.free_cash_flow / fund.total_revenue if fund.free_cash_flow is not None and fund.total_revenue else None
    metrics = FundamentalMetrics(
        revenue_growth_pct=pct(fund.revenue_growth),
        profit_growth_pct=pct(fund.earnings_growth),
        profit_margin_pct=pct(fund.profit_margin),
        operating_margin_pct=pct(fund.operating_margin),
        roe_pct=pct(fund.roe),
        roce_pct=pct(roce),
        debt_to_equity=_round(fund.debt_to_equity),
        pe=_round(fund.pe),
        pb=_round(fund.pb),
        sector_median_pe=median_pe,
        sector_median_pb=median_pb,
        pe_vs_sector_pct=_round((fund.pe / median_pe - 1) * 100) if fund.pe and fund.pe > 0 and median_pe else None,
        pb_vs_sector_pct=_round((fund.pb / median_pb - 1) * 100) if fund.pb and median_pb else None,
        free_cash_flow=_round(fund.free_cash_flow / divisor) if fund.free_cash_flow is not None else None,
        fcf_margin_pct=pct(fcf_margin),
        fcf_positive=fund.free_cash_flow > 0 if fund.free_cash_flow is not None else None,
        market_cap=_round(fund.market_cap / divisor) if fund.market_cap is not None else None,
        money_unit=unit,
        insider_or_promoter_holding_pct=pct(fund.insider_holding),
        pledged_pct=_round(fund.pledged_pct),
        dividend_yield_pct=pct(fund.dividend_yield),
        beta=_round(fund.beta),
        peers_compared=[p.symbol for p in peers],
    )
    metrics.unavailable = [
        name for name in ("revenue_growth_pct", "profit_growth_pct", "roe_pct", "roce_pct", "debt_to_equity", "pe",
                          "sector_median_pe", "free_cash_flow", "pledged_pct")
        if getattr(metrics, name) is None
    ]
    return metrics
