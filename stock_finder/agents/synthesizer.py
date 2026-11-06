"""Synthesizer: assembles the final report.

Order: market summary -> top opportunities ranked by composite score ->
watchlist -> avoid list -> portfolio actions -> key risks -> disclaimer.
Ranking and assembly are deterministic. Only the executive summary is written
by a model, and it is checked with the same grounding / language rules as the
critic; if it fails (or the budget is spent) a deterministic summary is used.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ..config import StockFinderSettings
from ..llm_client import LLMUnavailableError
from ..observability import BudgetExceededError, Trace
from ..prompts import PROMPT_VERSION, SUMMARY_SYSTEM
from ..schemas import CompanyAnalysis, FinalReport, MarketAnalysis, PortfolioReview, SummaryDraft
from .common import input_message
from .critic import collect_numbers, forbidden_phrases, ungrounded_numbers

logger = logging.getLogger(__name__)


def _score(c: CompanyAnalysis) -> float:
    return c.score_breakdown.composite if c.score_breakdown.composite is not None else -1.0


class Synthesizer:
    def __init__(self, settings: StockFinderSettings, llm: Any) -> None:
        self.settings = settings
        self.llm = llm

    @staticmethod
    def rank(companies: list[CompanyAnalysis]) -> dict[str, list[CompanyAnalysis]]:
        ok = [c for c in companies if c.status == "ok"]
        by_score = sorted(ok, key=_score, reverse=True)
        return {
            "opportunities": [c for c in by_score if c.verdict == "STRONG_BUY_CANDIDATE"],
            "watchlist": [c for c in by_score if c.verdict == "WATCHLIST"],
            "avoid": [c for c in by_score if c.verdict == "AVOID"],
            "unavailable": [c for c in companies if c.status != "ok"],
        }

    @staticmethod
    def key_risks(market: MarketAnalysis | None, ranked: dict[str, list[CompanyAnalysis]],
                  portfolio: PortfolioReview | None) -> list[str]:
        risks: list[str] = list(market.key_risks[:5]) if market else []
        if portfolio is not None:
            risks += portfolio.snapshot.concentration_warnings
        for company in ranked.get("opportunities", [])[:3]:
            if company.risks:
                risks.append(f"{company.symbol}: {company.risks[0]}")
        return list(dict.fromkeys(risks))

    def fallback_summary(self, market: MarketAnalysis | None, ranked: dict[str, list[CompanyAnalysis]],
                         portfolio: PortfolioReview | None) -> str:
        parts: list[str] = []
        if market is not None:
            parts.append(f"Market sentiment is {market.overall_sentiment}"
                         + (f"; leading themes: {', '.join(t.theme for t in market.themes[:3])}." if market.themes else "."))
        if ranked.get("opportunities"):
            parts.append("Top research candidates: " + ", ".join(c.symbol for c in ranked["opportunities"][:5]) + ".")
        elif any(ranked.get(k) for k in ("watchlist", "avoid")):
            parts.append("No company qualified as a strong candidate; see the watchlist.")
        if portfolio is not None and portfolio.reviews:
            actions = ", ".join(f"{r.symbol}: {r.action}" for r in portfolio.reviews)
            parts.append(f"Portfolio actions: {actions}.")
        return " ".join(parts) or "No results."

    def executive_summary(self, *, market: MarketAnalysis | None, ranked: dict[str, list[CompanyAnalysis]],
                          portfolio: PortfolioReview | None, risk_profile: str, trace: Trace) -> str:
        data: dict[str, Any] = {
            "market": {"sentiment": market.overall_sentiment, "summary": market.summary,
                       "themes": [t.theme for t in market.themes]} if market else None,
            "opportunities": [{"symbol": c.symbol, "name": c.company_name, "composite": c.score_breakdown.composite,
                               "thesis": c.thesis[:300]} for c in ranked.get("opportunities", [])[:5]],
            "watchlist": [c.symbol for c in ranked.get("watchlist", [])],
            "avoid": [c.symbol for c in ranked.get("avoid", [])],
            "portfolio_actions": [{"symbol": r.symbol, "action": r.action, "pnl_pct": r.pnl_pct,
                                   "allocation_pct": r.allocation_pct} for r in portfolio.reviews] if portfolio else [],
            "concentration_warnings": portfolio.snapshot.concentration_warnings if portfolio else [],
        }
        fallback = self.fallback_summary(market, ranked, portfolio)
        try:
            result = self.llm.parse(
                model=self.settings.analyst_model, schema=SummaryDraft, trace=trace, purpose="summary", temperature=0.3,
                messages=[{"role": "system", "content": SUMMARY_SYSTEM.format(market=self.settings.market,
                                                                             risk_profile=risk_profile)},
                          input_message(data, task="Write the executive summary.")])
        except (LLMUnavailableError, BudgetExceededError) as exc:
            logger.info("trace_id=%s executive summary fallback: %s", trace.trace_id, exc)
            return fallback
        summary = result.output.executive_summary.strip()  # type: ignore[attr-defined]
        if not summary or ungrounded_numbers([summary], collect_numbers(data)) or forbidden_phrases([summary]):
            trace.log("summary_rejected", reason="ungrounded numbers or forbidden language")
            return fallback
        return summary

    def assemble(
        self,
        *,
        report_id: str,
        kind: str,
        risk_profile: str,
        trace: Trace,
        market: MarketAnalysis | None = None,
        companies: list[CompanyAnalysis] | None = None,
        portfolio: PortfolioReview | None = None,
        summary: str | None = None,
        notes: list[str] | None = None,
        **extra: Any,
    ) -> FinalReport:
        ranked = self.rank(companies or [])
        if summary is None:
            summary = self.executive_summary(market=market, ranked=ranked, portfolio=portfolio,
                                             risk_profile=risk_profile, trace=trace)
        now = datetime.now(timezone.utc).isoformat()
        as_of_values = [c.as_of for c in companies or [] if c.as_of] + ([market.as_of] if market and market.as_of else [])
        unavailable = ranked["unavailable"]
        degraded = trace.degraded or bool(market and market.degraded) or any(c.degraded for c in companies or [])
        partial = trace.budget_exceeded or bool(unavailable) or bool(
            portfolio and any(r.status != "ok" for r in portfolio.reviews))
        return FinalReport(
            report_id=report_id, kind=kind, created_at=now, as_of=min(as_of_values) if as_of_values else now,
            market=self.settings.market, risk_profile=risk_profile, executive_summary=summary,
            market_analysis=market, opportunities=ranked["opportunities"], watchlist=ranked["watchlist"],
            avoid=ranked["avoid"], unavailable=unavailable, portfolio=portfolio,
            key_risks=self.key_risks(market, ranked, portfolio), disclaimer=self.settings.disclaimer,
            degraded=degraded, partial=partial, budget_exceeded=trace.budget_exceeded, notes=notes or [],
            prompt_version=PROMPT_VERSION, models=self.settings.models(), **extra,
        )
