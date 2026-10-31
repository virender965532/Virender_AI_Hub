"""Portfolio analyst: BUY_MORE / HOLD / TRIM / SELL for one holding, with reasons, risks and triggers.

Python has already computed the position (P&L, allocation, concentration flags)
and the company's analysis data. The analyst model decides the action; the
critic checks it, including the hard rule that BUY_MORE is not allowed for a
position (or sector) over its concentration limit. A still-failing review
downgrades the action to HOLD with low confidence and shows the concerns.
"""

from __future__ import annotations

from typing import Any

from ..config import StockFinderSettings
from ..observability import Trace
from ..prompts import COMPANY_REVISION_TEMPLATE, PORTFOLIO_ANALYST_SYSTEM
from ..rag.news_store import NewsStore
from ..schemas import (
    Holding, HoldingPosition, HoldingReview, HoldingReviewDraft, MarketTheme, PortfolioSnapshot, ScoreBreakdown,
)
from .common import input_message
from .company_analyst import CompanyData
from .critic import Critic, ReviewContext, run_with_critic


def action_consistent(action: str, position: HoldingPosition) -> tuple[bool, str]:
    if action == "BUY_MORE" and (position.breaches_stock_limit or position.breaches_sector_limit):
        return False, (f"BUY_MORE is not allowed: {position.symbol} breaches a concentration limit "
                       f"(allocation {position.allocation_pct}%); consider HOLD or TRIM")
    if action == "BUY_MORE" and position.current_price is None:
        return False, "BUY_MORE is not allowed without a current price"
    return True, ""


class PortfolioAnalyst:
    def __init__(self, settings: StockFinderSettings, llm: Any, critic: Critic, store: NewsStore) -> None:
        self.settings = settings
        self.llm = llm
        self.critic = critic
        self.store = store

    def review(self, holding: Holding, position: HoldingPosition, data: CompanyData | None, snapshot: PortfolioSnapshot,
               themes: list[MarketTheme], trace: Trace, risk_profile: str) -> HoldingReview:
        prompt_data: dict[str, Any] = {
            "position": position.model_dump(),
            "original_thesis": holding.thesis or "not saved",
            "buy_date": holding.buy_date.isoformat() if holding.buy_date else None,
            "portfolio": {"total_value": snapshot.total_value, "total_pnl_pct": snapshot.total_pnl_pct,
                          "sector_allocation": snapshot.sector_allocation,
                          "concentration_warnings": snapshot.concentration_warnings, "currency": snapshot.currency,
                          "limits": {"max_stock_pct": self.settings.max_stock_pct,
                                     "max_sector_pct": self.settings.max_sector_pct}},
            "market_themes": [{"theme": t.theme, "sectors": t.sectors, "sentiment": t.sentiment} for t in themes],
            "risk_profile": risk_profile,
        }
        articles = []
        if data is not None:
            company = data.prompt_data(self.settings)
            prompt_data["company_analysis_data"] = {k: company[k] for k in
                                                    ("fundamentals", "technicals", "score_breakdown", "data_quality",
                                                     "risk_flags", "news")}
            articles = data.articles
        else:
            prompt_data["company_analysis_data"] = "unavailable (market data could not be fetched)"
        quality = position.data_quality.level if data is None else min(
            (position.data_quality.level, data.quality.level), key=["poor", "partial", "good"].index)
        system = PORTFOLIO_ANALYST_SYSTEM.format(market=self.settings.market, risk_profile=risk_profile)
        base = [{"role": "system", "content": system},
                input_message(prompt_data, articles, f"Review the holding {holding.symbol}.")]
        state = {"degraded": False, "model": ""}

        def generate(feedback: str | None) -> HoldingReviewDraft:
            messages = list(base)
            if feedback:
                messages.append({"role": "user", "content": COMPANY_REVISION_TEMPLATE.format(feedback=feedback)})
            result = self.llm.parse(model=self.settings.analyst_model, messages=messages, schema=HoldingReviewDraft,
                                    trace=trace, purpose=f"holding:{holding.symbol}" + (":revise" if feedback else ""),
                                    temperature=0.2)
            state["degraded"] = state["degraded"] or result.degraded
            state["model"] = result.model
            return result.output  # type: ignore[return-value]

        def ctx_for(draft: HoldingReviewDraft) -> ReviewContext:
            return ReviewContext(
                kind="holding", label=holding.symbol, input_data=prompt_data,
                allowed_citation_ids={a.id for a in articles},
                news_text=" ".join(a.prompt_line() for a in articles),
                news_fields=("rationale", "reasoning"),
                logic_check=lambda d: action_consistent(d.action, position),
                data_quality=quality, generator_model=state["model"],
            )

        draft, status = run_with_critic(generate, self.critic, ctx_for, trace)
        action, confidence = draft.action, draft.confidence
        if status.downgraded:
            action, confidence = "HOLD", "low"
        citations, _ = self.store.citations(i for i in draft.citation_ids if i in {a.id for a in articles})
        review = HoldingReview(
            symbol=holding.symbol, company_name=position.company_name or (data.name if data else ""),
            quantity=holding.quantity, avg_cost=holding.avg_cost, current_price=position.current_price,
            pnl_pct=position.pnl_pct, allocation_pct=position.allocation_pct, reasoning=draft.reasoning,
            action=action, rationale=draft.rationale, risks=draft.risks, triggers_to_revisit=draft.triggers_to_revisit,
            confidence=confidence, citations=citations,
            score_breakdown=data.score if data else ScoreBreakdown(risk_profile=risk_profile),
            data_quality=data.quality if data else position.data_quality, critic=status, degraded=state["degraded"],
        )
        trace.emit("worker_result", {"symbol": holding.symbol, "action": review.action, "confidence": review.confidence,
                                     "critic_passed": status.passed, "downgraded": status.downgraded})
        return review
