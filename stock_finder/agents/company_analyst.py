"""Company analyst (Planner-Worker): deep analysis of ONE company.

``gather_company_data`` is pure Python: quote, history, fundamentals, sector
peers, indicators, company news and the deterministic score. Each worker sees
only its own company's data. ``CompanyAnalyst.analyze`` asks the worker model
for a ``CompanyAnalysisDraft`` and runs it through the critic (one revision on
FAIL; still FAIL -> verdict WATCHLIST, confidence low, concerns shown).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from ..analysis.fundamentals import FundamentalMetrics, compute_fundamentals
from ..analysis.indicators import TechnicalIndicators, compute_indicators
from ..analysis.scoring import STRONG_BUY_MIN_SCORE, score_company, verdict_consistent
from ..config import StockFinderSettings
from ..data.market_data import DataUnavailableError, Fundamentals, MarketDataService, Quote
from ..data.news_data import Article, NewsService
from ..data.validators import build_quality, check_history
from ..observability import Trace
from ..prompts import COMPANY_ANALYST_SYSTEM, COMPANY_REVISION_TEMPLATE
from ..rag.news_store import NewsStore
from ..schemas import Candidate, CompanyAnalysis, CompanyAnalysisDraft, DataQuality, ScoreBreakdown
from .common import input_message, news_index
from .critic import Critic, ReviewContext, run_with_critic

logger = logging.getLogger(__name__)


@dataclass
class CompanyData:
    """Everything computed in Python for one company (also the audit snapshot)."""

    symbol: str
    name: str
    sector: str
    quote: Optional[Quote]
    fundamentals: Optional[Fundamentals]
    metrics: FundamentalMetrics
    technicals: TechnicalIndicators
    score: ScoreBreakdown
    articles: list[Article]
    quality: DataQuality
    risk_flags: list[str] = field(default_factory=list)

    def prompt_data(self, settings: StockFinderSettings, candidate: Candidate | None = None) -> dict[str, Any]:
        quote = self.quote
        data: dict[str, Any] = {
            "company": {"symbol": self.symbol, "name": self.name, "sector": self.sector, "currency": settings.currency,
                        "price": quote.price if quote else None, "day_change_pct": quote.change_pct if quote else None,
                        "price_as_of": quote.as_of.isoformat() if quote and quote.as_of else None},
            "fundamentals": self.metrics.model_dump(),
            "technicals": self.technicals.model_dump(),
            "score_breakdown": self.score.model_dump(),
            "data_quality": self.quality.model_dump(),
            "risk_flags": self.risk_flags,
            "news": news_index(self.articles),
            "rules": {"strong_buy_min_composite": STRONG_BUY_MIN_SCORE, "risk_profile": self.score.risk_profile},
        }
        if candidate is not None:
            data["screening"] = {"linked_theme": candidate.linked_theme, "why_candidate": candidate.why_candidate}
        return data

    def snapshot(self) -> dict[str, Any]:
        return {"quote": self.quote.model_dump(mode="json") if self.quote else None, "sector": self.sector,
                "metrics": self.metrics.model_dump(), "technicals": self.technicals.model_dump(),
                "score": self.score.model_dump(), "article_ids": [a.id for a in self.articles],
                "data_quality": self.quality.model_dump()}


def gather_company_data(
    symbol: str,
    *,
    settings: StockFinderSettings,
    market: MarketDataService,
    news: NewsService,
    store: NewsStore,
    risk_profile: str,
    index_history: pd.DataFrame | None = None,
    company_name: str = "",
) -> CompanyData:
    """Fetch and compute all data for ``symbol``; raises ``DataUnavailableError`` if nothing at all is available."""
    issues: list[str] = []
    quote = history = fund = None
    try:
        quote = market.get_quote(symbol)
    except DataUnavailableError as exc:
        issues.append("price unavailable")
        logger.info("quote unavailable for %s: %s", symbol, exc)
    try:
        history = market.get_history(symbol, "1y")
    except DataUnavailableError:
        pass
    try:
        fund = market.get_fundamentals(symbol)
    except DataUnavailableError:
        issues.append("fundamentals unavailable")
    if quote is None and history is None and fund is None:
        raise DataUnavailableError(f"no market data available for {symbol}")

    name = (fund.name if fund else "") or company_name or market.universe.name_of(symbol) or symbol
    sector = market.universe.sector_of(symbol) or market.universe.match_sector(fund.sector if fund else "") \
        or (fund.sector if fund and fund.sector else "Unknown")
    peer_symbols = market.get_sector_peers(symbol, fund.sector if fund else None)
    peers = list(market.get_fundamentals_many(peer_symbols).values()) if peer_symbols else []
    if not peers:
        issues.append("no sector peers available for valuation comparison")
    metrics = compute_fundamentals(fund, peers, market=settings.market)
    technicals = compute_indicators(history, index_history)
    if index_history is None:
        issues.append("benchmark index unavailable (relative strength not computed)")

    news_result = news.fetch_company_news(symbol, name)
    store.add(news_result.articles)
    articles = news_result.articles[: settings.max_company_articles]
    if news_result.failed_providers:
        issues.append(f"company news provider failed: {', '.join(news_result.failed_providers)}")

    score = score_company(metrics, technicals, articles, risk_profile=risk_profile, weights=settings.score_weights)
    risk_flags: list[str] = []
    price = quote.price if quote else None
    if price is not None and price < settings.penny_price:
        risk_flags.append(f"penny stock (price below {settings.penny_price:g} {settings.currency}): very high risk")
    if metrics.debt_to_equity is not None and metrics.debt_to_equity > 2:
        risk_flags.append("high leverage (debt/equity above 2)")
    quality = build_quality(
        price_issues=(quote.issues if quote else ["price unavailable"]), stale=bool(quote and quote.stale),
        fundamentals_issues=(fund.issues if fund else ["missing fundamentals: all"]),
        history_issues=check_history(history), other_issues=issues,
        as_of=quote.as_of if quote and quote.as_of else datetime.now(timezone.utc),
        has_price=price is not None, has_fundamentals=fund is not None,
    )
    return CompanyData(symbol=symbol, name=name, sector=sector, quote=quote, fundamentals=fund, metrics=metrics,
                       technicals=technicals, score=score, articles=articles, quality=quality, risk_flags=risk_flags)


class CompanyAnalyst:
    def __init__(self, settings: StockFinderSettings, llm: Any, critic: Critic, store: NewsStore) -> None:
        self.settings = settings
        self.llm = llm
        self.critic = critic
        self.store = store

    def analyze(self, data: CompanyData, trace: Trace, candidate: Candidate | None = None,
                risk_profile: str = "moderate") -> CompanyAnalysis:
        prompt_data = data.prompt_data(self.settings, candidate)
        system = COMPANY_ANALYST_SYSTEM.format(market=self.settings.market, risk_profile=risk_profile)
        base_messages = [{"role": "system", "content": system},
                         input_message(prompt_data, data.articles, f"Analyse {data.symbol}.")]
        state = {"degraded": False, "model": ""}

        def generate(feedback: str | None) -> CompanyAnalysisDraft:
            messages = list(base_messages)
            if feedback:
                messages.append({"role": "user", "content": COMPANY_REVISION_TEMPLATE.format(feedback=feedback)})
            result = self.llm.parse(model=self.settings.worker_model, messages=messages, schema=CompanyAnalysisDraft,
                                    trace=trace, purpose=f"company:{data.symbol}" + (":revise" if feedback else ""),
                                    temperature=0.2)
            state["degraded"] = state["degraded"] or result.degraded
            state["model"] = result.model
            return result.output  # type: ignore[return-value]

        def ctx_for(draft: CompanyAnalysisDraft) -> ReviewContext:
            return ReviewContext(
                kind="company", label=data.symbol, input_data=prompt_data,
                allowed_citation_ids={a.id for a in data.articles},
                news_text=" ".join(a.prompt_line() for a in data.articles),
                logic_check=lambda d: verdict_consistent(d.verdict, data.score.composite, data.quality.level),
                data_quality=data.quality.level, generator_model=state["model"],
            )

        draft, status = run_with_critic(generate, self.critic, ctx_for, trace)
        verdict, confidence = draft.verdict, draft.confidence
        if status.downgraded:
            verdict, confidence = "WATCHLIST", "low"
        if data.quality.level == "poor" and confidence == "high":
            confidence = "medium"
        citations, _ = self.store.citations(i for i in draft.citation_ids if i in {a.id for a in data.articles})
        analysis = CompanyAnalysis(
            symbol=data.symbol, company_name=data.name, linked_theme=candidate.linked_theme if candidate else "",
            reasoning=draft.reasoning, thesis=draft.thesis, why_now=draft.why_now, strengths=draft.strengths,
            risks=[*data.risk_flags, *draft.risks], valuation_view=draft.valuation_view,
            technical_view=draft.technical_view, news_view=draft.news_view, score_breakdown=data.score,
            verdict=verdict, confidence=confidence, what_would_change_view=draft.what_would_change_view,
            citations=citations, data_quality=data.quality, critic=status,
            price=data.quote.price if data.quote else None, currency=self.settings.currency,
            as_of=data.quality.as_of or "", degraded=state["degraded"],
        )
        trace.emit("worker_result", {"symbol": data.symbol, "verdict": analysis.verdict, "confidence": analysis.confidence,
                                     "composite": data.score.composite, "critic_passed": status.passed,
                                     "downgraded": status.downgraded})
        return analysis
