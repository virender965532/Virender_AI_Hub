"""Portfolio maths in Python: P&L, allocation, sector concentration and theme overlap.

Allocation uses market value; a holding whose price is unavailable is valued at
cost (and flagged) so totals stay meaningful. Concentration warnings fire when
one stock exceeds ``SF_MAX_STOCK_PCT`` (default 25%) or one sector exceeds
``SF_MAX_SECTOR_PCT`` (default 40%) of the portfolio.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Optional

from ..data.market_data import Quote
from ..schemas import DataQuality, Holding, HoldingPosition, MarketTheme, PortfolioSnapshot


def _r(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None


def compute_snapshot(
    holdings: list[Holding],
    quotes: dict[str, Optional[Quote]],
    *,
    sectors: dict[str, str],
    names: dict[str, str] | None = None,
    themes: list[MarketTheme] | None = None,
    sector_matcher: Callable[[str], Optional[str]] | None = None,
    max_stock_pct: float = 25.0,
    max_sector_pct: float = 40.0,
    currency: str = "",
) -> PortfolioSnapshot:
    """Positions with P&L and allocation, sector allocation and concentration warnings."""
    names = names or {}
    themes = themes or []
    match = sector_matcher or (lambda text: text)
    positions: list[HoldingPosition] = []
    for h in holdings:
        quote = quotes.get(h.symbol)
        price = quote.price if quote is not None else None
        cost = h.quantity * h.avg_cost
        value = h.quantity * price if price else None
        issues = list(quote.issues) if quote is not None else ["price unavailable"]
        if value is None:
            issues.append("valued at cost for allocation (price unavailable)")
        sector = sectors.get(h.symbol) or "Unknown"
        overlap = [t.theme for t in themes if any(match(s) == sector for s in t.sectors)] if sector != "Unknown" else []
        positions.append(HoldingPosition(
            symbol=h.symbol, company_name=names.get(h.symbol, ""), sector=sector, quantity=h.quantity,
            avg_cost=h.avg_cost, current_price=price, cost_value=_r(cost), market_value=_r(value),
            pnl_abs=_r(value - cost) if value is not None else None,
            pnl_pct=_r((price / h.avg_cost - 1) * 100) if price else None,
            theme_overlap=overlap,
            data_quality=DataQuality(level="poor" if price is None or (quote and quote.stale) else
                                     ("partial" if issues else "good"),
                                     stale=bool(quote and quote.stale), issues=issues,
                                     as_of=quote.as_of.isoformat() if quote and quote.as_of else None),
        ))

    total_value = sum(p.market_value if p.market_value is not None else p.cost_value for p in positions)
    total_cost = sum(p.cost_value for p in positions)
    priced = [p for p in positions if p.market_value is not None]
    priced_cost = sum(p.cost_value for p in priced)
    priced_value = sum(p.market_value for p in priced)  # type: ignore[misc]
    sector_alloc: dict[str, float] = {}
    warnings: list[str] = []
    for p in positions:
        basis = p.market_value if p.market_value is not None else p.cost_value
        p.allocation_pct = _r(basis / total_value * 100) if total_value else None
        if p.allocation_pct is not None:
            sector_alloc[p.sector] = sector_alloc.get(p.sector, 0.0) + p.allocation_pct
        if p.allocation_pct is not None and p.allocation_pct > max_stock_pct:
            p.breaches_stock_limit = True
            warnings.append(f"{p.symbol} is {p.allocation_pct}% of the portfolio (limit {max_stock_pct:g}% per stock)")
    sector_alloc = {k: round(v, 2) for k, v in sorted(sector_alloc.items(), key=lambda kv: -kv[1])}
    for sector, share in sector_alloc.items():
        if share > max_sector_pct:
            warnings.append(f"{sector} sector is {share}% of the portfolio (limit {max_sector_pct:g}% per sector)")
            for p in positions:
                if p.sector == sector:
                    p.breaches_sector_limit = True
    return PortfolioSnapshot(
        positions=positions,
        total_cost=round(total_cost, 2),
        total_value=round(total_value, 2),
        total_pnl_abs=round(priced_value - priced_cost, 2),
        total_pnl_pct=_r((priced_value / priced_cost - 1) * 100) if priced_cost else None,
        sector_allocation=sector_alloc,
        concentration_warnings=warnings,
        currency=currency,
        as_of=datetime.now(timezone.utc).isoformat(),
    )
