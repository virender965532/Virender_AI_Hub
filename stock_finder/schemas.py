"""Pydantic models for every structured input and output of Stock Finder.

Two kinds of model live here:

- **LLM drafts** (``*Draft`` and ``RouteDecision``/``CriticVerdict``): what a model
  fills via structured output. They have no defaults and no dict fields so they
  stay compatible with OpenAI strict JSON schemas. News evidence is referenced
  by *article id*; code resolves ids to ``Citation`` objects, so a model cannot
  cite an article that is not in the news store.
- **Final models** (``CompanyAnalysis``, ``HoldingReview``, ``FinalReport``...):
  the draft merged with numbers computed in Python (scores, P&L, data quality).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

Intent = Literal["market_overview", "find_opportunities", "analyze_stock", "review_portfolio", "general_question"]
Sentiment = Literal["positive", "negative", "neutral"]
Horizon = Literal["short", "medium", "long"]
Verdict = Literal["STRONG_BUY_CANDIDATE", "WATCHLIST", "AVOID"]
Action = Literal["BUY_MORE", "HOLD", "SELL", "TRIM"]
Confidence = Literal["low", "medium", "high"]
RiskProfile = Literal["conservative", "moderate", "aggressive"]
QualityLevel = Literal["good", "partial", "poor"]


# --- shared ---------------------------------------------------------------------------------------

class Citation(BaseModel):
    """A news article backing a claim; always resolved from the news store, never model-written."""

    id: str = ""
    source: str
    title: str
    published_date: Optional[str] = None
    url: str


class DataQuality(BaseModel):
    """Freshness / completeness flag attached to every company and holding."""

    level: QualityLevel = "good"
    stale: bool = False
    issues: list[str] = Field(default_factory=list)
    as_of: Optional[str] = None


class ScoreBreakdown(BaseModel):
    """Deterministic sub-scores (0-100, higher is better; ``risk`` high = lower risk)."""

    fundamentals: Optional[float] = None
    valuation: Optional[float] = None
    momentum: Optional[float] = None
    news_sentiment: Optional[float] = None
    risk: Optional[float] = None
    composite: Optional[float] = None
    risk_profile: RiskProfile = "moderate"
    weights: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class CriticStatus(BaseModel):
    """What the critic concluded about a recommendation, shown next to it in the UI."""

    passed: bool = False
    revised: bool = False
    downgraded: bool = False
    concerns: list[str] = Field(default_factory=list)
    model: str = ""


# --- router ---------------------------------------------------------------------------------------

class RouteDecision(BaseModel):
    """Router output. ``clarifying_question`` is empty unless confidence is low."""

    reasoning: str
    intent: Intent
    symbols: list[str]
    confidence: float
    clarifying_question: str


# --- news analyst ---------------------------------------------------------------------------------

class ThemeDraft(BaseModel):
    theme: str
    sectors: list[str]
    sentiment: Sentiment
    impact_horizon: Horizon
    evidence_ids: list[str]


class ThemeBatch(BaseModel):
    """Map step: themes found in one batch of articles."""

    themes: list[ThemeDraft]


class MarketAnalysisDraft(BaseModel):
    """Reduce step: merged market view."""

    reasoning: str
    overall_sentiment: Sentiment
    summary: str
    themes: list[ThemeDraft]
    key_risks: list[str]


class MarketTheme(BaseModel):
    theme: str
    sectors: list[str]
    sentiment: Sentiment
    impact_horizon: Horizon
    evidence: list[Citation]


class IndexQuote(BaseModel):
    symbol: str
    name: str
    price: Optional[float] = None
    change_pct: Optional[float] = None
    return_1m_pct: Optional[float] = None
    as_of: Optional[str] = None
    stale: bool = False


class MarketAnalysis(BaseModel):
    overall_sentiment: Sentiment = "neutral"
    summary: str = ""
    themes: list[MarketTheme] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    index_snapshot: list[IndexQuote] = Field(default_factory=list)
    article_count: int = 0
    as_of: str = ""
    data_quality: DataQuality = Field(default_factory=DataQuality)
    degraded: bool = False


# --- screener -------------------------------------------------------------------------------------

class Candidate(BaseModel):
    symbol: str
    company_name: str
    linked_theme: str
    why_candidate: str


# --- company analyst ------------------------------------------------------------------------------

class CompanyAnalysisDraft(BaseModel):
    """What the company-analyst model writes; ``reasoning`` comes first on purpose."""

    reasoning: str
    thesis: str
    why_now: str
    strengths: list[str]
    risks: list[str]
    valuation_view: str
    technical_view: str
    news_view: str
    verdict: Verdict
    confidence: Confidence
    what_would_change_view: str
    citation_ids: list[str]


class CompanyAnalysis(BaseModel):
    symbol: str
    company_name: str = ""
    linked_theme: str = ""
    status: Literal["ok", "unavailable"] = "ok"
    error: Optional[str] = None
    reasoning: str = ""
    thesis: str = ""
    why_now: str = ""
    strengths: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    valuation_view: str = ""
    technical_view: str = ""
    news_view: str = ""
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    verdict: Verdict = "WATCHLIST"
    confidence: Confidence = "low"
    what_would_change_view: str = ""
    citations: list[Citation] = Field(default_factory=list)
    data_quality: DataQuality = Field(default_factory=DataQuality)
    critic: CriticStatus = Field(default_factory=CriticStatus)
    price: Optional[float] = None
    currency: str = ""
    as_of: str = ""
    degraded: bool = False

    @classmethod
    def unavailable(cls, symbol: str, error: str, **extra: Any) -> "CompanyAnalysis":
        return cls(symbol=symbol, status="unavailable", error=error, reasoning="Analysis unavailable.", **extra)


# --- portfolio ------------------------------------------------------------------------------------

class HoldingIn(BaseModel):
    """A holding as entered by the user (API body or CSV row)."""

    model_config = ConfigDict(str_strip_whitespace=True)

    symbol: str = Field(min_length=1, max_length=32)
    quantity: float = Field(gt=0)
    avg_cost: float = Field(gt=0)
    buy_date: Optional[date] = None
    thesis: Optional[str] = Field(default=None, max_length=2000)

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()


class Holding(HoldingIn):
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class HoldingPosition(BaseModel):
    """Python-computed position numbers for one holding."""

    symbol: str
    company_name: str = ""
    sector: str = "Unknown"
    quantity: float
    avg_cost: float
    current_price: Optional[float] = None
    cost_value: float
    market_value: Optional[float] = None
    pnl_abs: Optional[float] = None
    pnl_pct: Optional[float] = None
    allocation_pct: Optional[float] = None
    theme_overlap: list[str] = Field(default_factory=list)
    breaches_stock_limit: bool = False
    breaches_sector_limit: bool = False
    data_quality: DataQuality = Field(default_factory=DataQuality)


class PortfolioSnapshot(BaseModel):
    positions: list[HoldingPosition] = Field(default_factory=list)
    total_cost: float = 0.0
    total_value: float = 0.0
    total_pnl_abs: float = 0.0
    total_pnl_pct: Optional[float] = None
    sector_allocation: dict[str, float] = Field(default_factory=dict)
    concentration_warnings: list[str] = Field(default_factory=list)
    currency: str = ""
    as_of: str = ""


class HoldingReviewDraft(BaseModel):
    reasoning: str
    action: Action
    rationale: str
    risks: list[str]
    triggers_to_revisit: list[str]
    confidence: Confidence
    citation_ids: list[str]


class HoldingReview(BaseModel):
    symbol: str
    company_name: str = ""
    quantity: float
    avg_cost: float
    current_price: Optional[float] = None
    pnl_pct: Optional[float] = None
    allocation_pct: Optional[float] = None
    status: Literal["ok", "unavailable"] = "ok"
    error: Optional[str] = None
    reasoning: str = ""
    action: Action = "HOLD"
    rationale: str = ""
    risks: list[str] = Field(default_factory=list)
    triggers_to_revisit: list[str] = Field(default_factory=list)
    confidence: Confidence = "low"
    citations: list[Citation] = Field(default_factory=list)
    score_breakdown: ScoreBreakdown = Field(default_factory=ScoreBreakdown)
    data_quality: DataQuality = Field(default_factory=DataQuality)
    critic: CriticStatus = Field(default_factory=CriticStatus)
    degraded: bool = False


class PortfolioReview(BaseModel):
    snapshot: PortfolioSnapshot
    reviews: list[HoldingReview] = Field(default_factory=list)


# --- critic ---------------------------------------------------------------------------------------

class CriticVerdict(BaseModel):
    reasoning: str
    numbers_grounded: bool
    claims_cited: bool
    logic_consistent: bool
    risks_balanced: bool
    overconfident: bool
    verdict: Literal["PASS", "FAIL"]
    feedback: str


# --- synthesizer / report -------------------------------------------------------------------------

class SummaryDraft(BaseModel):
    executive_summary: str


class ChatAnswerDraft(BaseModel):
    answer: str
    citation_ids: list[str]


class FinalReport(BaseModel):
    report_id: str
    kind: Literal["market_overview", "opportunities", "analysis", "portfolio_review", "chat"]
    created_at: str
    as_of: str
    market: str
    risk_profile: RiskProfile = "moderate"
    executive_summary: str = ""
    market_analysis: Optional[MarketAnalysis] = None
    opportunities: list[CompanyAnalysis] = Field(default_factory=list)
    watchlist: list[CompanyAnalysis] = Field(default_factory=list)
    avoid: list[CompanyAnalysis] = Field(default_factory=list)
    unavailable: list[CompanyAnalysis] = Field(default_factory=list)
    portfolio: Optional[PortfolioReview] = None
    key_risks: list[str] = Field(default_factory=list)
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    clarifying_question: str = ""
    route: Optional[RouteDecision] = None
    disclaimer: str
    degraded: bool = False
    partial: bool = False
    budget_exceeded: bool = False
    error: str = ""
    notes: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str = ""
    models: dict[str, str] = Field(default_factory=dict)
