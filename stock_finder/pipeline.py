"""Orchestrator for Stock Finder.

Flows:
- market overview: news analyst (RAG + map-reduce) -> report.
- find opportunities: news analyst -> screener (ReAct) -> parallel company
  analysts (asyncio + semaphore, one worker per candidate) each reviewed by the
  critic -> synthesizer.
- analyze: one company analyst worker + critic.
- portfolio review: Python computes positions -> parallel portfolio analysts +
  critic -> synthesizer.
- chat: router -> one of the above, or a grounded answer for general questions.

Every run has a ``Trace`` (trace_id, budget, metrics). A failed worker becomes
"analysis unavailable" without failing the run; a spent budget stops cleanly
with partial results; any unexpected error becomes a friendly message with the
trace_id (details in the logs). Every report is saved for audit. Long runs can
be submitted to ``JobManager`` (a small thread pool) and streamed as events.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, TypeVar

import pandas as pd

from llm_provider import provider_name

from .agents.common import input_message
from .agents.company_analyst import CompanyAnalyst, CompanyData, gather_company_data
from .agents.critic import Critic, collect_numbers, forbidden_phrases, ungrounded_numbers
from .agents.news_analyst import NewsAnalyst
from .agents.portfolio_analyst import PortfolioAnalyst
from .agents.router import route
from .agents.screener import Screener
from .agents.synthesizer import Synthesizer
from .config import RISK_PROFILES, StockFinderSettings, get_settings
from .data.cache import NEWS_TTL, TTLCache
from .data.market_data import DataUnavailableError, MarketDataService, run_parallel
from .data.news_data import Article, NewsService
from .llm_client import LLMUnavailableError, get_llm_client
from .observability import BudgetExceededError, Listener, Trace, current_trace
from .portfolio.models import DEFAULT_PORTFOLIO, HoldingsRepository
from .portfolio.service import compute_snapshot
from .prompts import CHAT_SYSTEM, PROMPT_VERSION
from .rag.news_store import NewsStore
from .reports import ReportStore
from .schemas import (
    Candidate, ChatAnswerDraft, CompanyAnalysis, FinalReport, HoldingReview, MarketAnalysis, PortfolioReview,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

_FRIENDLY_ERROR = "Something went wrong while preparing this report. Please try again later (trace id {trace_id})."


def new_report_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]


def run_workers(items: list[T], fn: Callable[[T], R], limit: int) -> list[R | BaseException]:
    """Run ``fn`` over ``items`` concurrently (asyncio + semaphore, threads for the blocking work).

    Results keep the input order; exceptions are returned, not raised. The trace
    context is copied into each worker by ``asyncio.to_thread``.
    """

    async def gather() -> list[R | BaseException]:
        semaphore = asyncio.Semaphore(max(1, limit))

        async def one(item: T) -> R:
            async with semaphore:
                return await asyncio.to_thread(fn, item)

        return await asyncio.gather(*(one(i) for i in items), return_exceptions=True)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(gather())
    # Called from inside a running event loop (unusual): run the loop in a helper thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        import contextvars

        return pool.submit(contextvars.copy_context().run, asyncio.run, gather()).result()


# --- background jobs ------------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    kind: str
    status: str = "queued"  # queued | running | done | error
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    report: FinalReport | None = None
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cond: threading.Condition = field(default_factory=threading.Condition, repr=False)

    def push(self, event: str, data: dict[str, Any]) -> None:
        with self.cond:
            self.events.append((event, data))
            self.cond.notify_all()

    def info(self) -> dict[str, Any]:
        return {"job_id": self.id, "kind": self.kind, "status": self.status, "created_at": self.created_at,
                "report_id": self.report.report_id if self.report else None, "error": self.error,
                "events": len(self.events)}

    def stream(self, heartbeat: float = 15.0) -> Iterator[tuple[str, dict[str, Any]] | None]:
        """Yield events as they arrive (``None`` = heartbeat) until the job ends."""
        index = 0
        while True:
            with self.cond:
                if index >= len(self.events) and self.status not in ("done", "error"):
                    self.cond.wait(timeout=heartbeat)
                pending = self.events[index:]
                index += len(pending)
                finished = self.status in ("done", "error") and index >= len(self.events)
            if not pending and not finished:
                yield None
            for item in pending:
                yield item
            if finished:
                return


class JobManager:
    """Thread-pool runner for long requests; keeps the most recent jobs in memory."""

    def __init__(self, max_workers: int = 2, keep: int = 50) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sf-job")
        self._jobs: dict[str, Job] = {}
        self._keep = keep
        self._lock = threading.Lock()

    def submit(self, kind: str, run: Callable[[Listener], FinalReport]) -> Job:
        job = Job(id=uuid.uuid4().hex[:16], kind=kind)
        with self._lock:
            self._jobs[job.id] = job
            while len(self._jobs) > self._keep:
                self._jobs.pop(next(iter(self._jobs)))

        def work() -> None:
            job.status = "running"
            job.push("job", {"job_id": job.id, "status": "running"})
            try:
                job.report = run(job.push)
                job.status = "done"
                job.push("final", job.report.model_dump(mode="json"))
            except Exception:  # noqa: BLE001 - the pipeline already converts expected errors
                logger.exception("Stock Finder job %s failed", job.id)
                job.status = "error"
                job.error = "internal error"
                job.push("error", {"error": "internal error", "job_id": job.id})
            with job.cond:
                job.cond.notify_all()

        job.push("job", {"job_id": job.id, "status": "queued"})
        self._pool.submit(work)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


# --- pipeline -------------------------------------------------------------------------------------

class StockFinderPipeline:
    """Wires data services, agents, storage and jobs. All dependencies are injectable for tests."""

    def __init__(
        self,
        settings: StockFinderSettings | None = None,
        *,
        llm: Any | None = None,
        market: MarketDataService | None = None,
        news: NewsService | None = None,
        store: NewsStore | None = None,
        holdings: HoldingsRepository | None = None,
        reports: ReportStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or get_llm_client()
        self.market = market or MarketDataService(self.settings)
        self.news = news or NewsService(self.settings)
        self.store = store or NewsStore(lambda texts: self.llm.embed(texts, trace=current_trace()))
        self.critic = Critic(self.settings, self.llm)
        self.news_analyst = NewsAnalyst(self.settings, self.llm, self.news, self.store, self.market)
        self.screener = Screener(self.settings, self.llm, self.market, self.store)
        self.company_analyst = CompanyAnalyst(self.settings, self.llm, self.critic, self.store)
        self.portfolio_analyst = PortfolioAnalyst(self.settings, self.llm, self.critic, self.store)
        self.synthesizer = Synthesizer(self.settings, self.llm)
        self.holdings = holdings or HoldingsRepository()
        self.reports = reports or ReportStore()
        self.jobs = JobManager(self.settings.job_workers)
        self._market_cache = TTLCache(max_entries=4)

    # --- helpers ------------------------------------------------------------------------------

    def risk_profile(self, value: Any) -> str:
        profile = str(value or self.settings.risk_profile_default).strip().lower()
        if profile not in RISK_PROFILES:
            raise ValueError(f"risk_profile must be one of {', '.join(RISK_PROFILES)}")
        return profile

    def new_trace(self, listener: Listener | None = None) -> Trace:
        return Trace(max_llm_calls=self.settings.max_llm_calls, listener=listener)

    def market_analysis(self, trace: Trace, *, refresh: bool = False) -> tuple[MarketAnalysis, list[Article]]:
        """Market analysis, reused for ``NEWS_TTL`` so follow-up runs don't re-spend LLM calls."""
        cached = None if refresh else self._market_cache.get("market")
        if cached is not None:
            trace.progress("news", "Using the market analysis from the last 15 minutes")
            return cached
        result = self.news_analyst.analyze(trace)
        if result[0].themes and not result[0].degraded:
            self._market_cache.set("market", result, NEWS_TTL)
        return result

    def cached_market_analysis(self) -> MarketAnalysis | None:
        cached = self._market_cache.get("market")
        return cached[0] if cached else None

    def _index_history(self) -> pd.DataFrame | None:
        try:
            return self.market.get_history(self.market.universe.benchmark, "1y")
        except DataUnavailableError:
            return None

    def _finish(self, trace: Trace, report: FinalReport, snapshot: dict[str, Any]) -> FinalReport:
        status = "error" if report.error else "partial" if report.partial else "degraded" if report.degraded else "ok"
        report.metrics = trace.finish(status)
        report.budget_exceeded = report.budget_exceeded or trace.budget_exceeded
        if trace.budget_exceeded and "LLM call budget reached; results are partial." not in report.notes:
            report.notes.append("LLM call budget reached; results are partial.")
            report.partial = True
        self.reports.save(report, snapshot)
        trace.log("report_saved", report_id=report.report_id, kind=report.kind, prompt_version=PROMPT_VERSION)
        return report

    def _guard(self, kind: str, trace: Trace, risk_profile: str, run: Callable[[], FinalReport]) -> FinalReport:
        """Run a flow; turn unexpected errors into a friendly report (details logged with the trace id)."""
        with trace:
            try:
                return run()
            except Exception:  # noqa: BLE001
                logger.exception("trace_id=%s Stock Finder %s failed", trace.trace_id, kind)
                now = datetime.now(timezone.utc).isoformat()
                report = FinalReport(report_id=new_report_id(), kind=kind, created_at=now, as_of=now,
                                     market=self.settings.market, risk_profile=risk_profile,
                                     disclaimer=self.settings.disclaimer, partial=True,
                                     error=_FRIENDLY_ERROR.format(trace_id=trace.trace_id),
                                     prompt_version=PROMPT_VERSION, models=self.settings.models())
                return self._finish(trace, report, {})

    def _analyze_companies(self, items: list[tuple[str, Candidate | None]], *, risk_profile: str, trace: Trace,
                           snapshots: dict[str, Any]) -> list[CompanyAnalysis]:
        index_history = self._index_history()

        def work(item: tuple[str, Candidate | None]) -> CompanyAnalysis:
            symbol, candidate = item
            name = candidate.company_name if candidate else ""
            if trace.remaining_calls < 1:
                return CompanyAnalysis.unavailable(symbol, "analysis unavailable: request budget exhausted",
                                                   company_name=name)
            trace.progress("analysis", f"Analysing {symbol}", symbol=symbol)
            try:
                data = gather_company_data(symbol, settings=self.settings, market=self.market, news=self.news,
                                           store=self.store, risk_profile=risk_profile, index_history=index_history,
                                           company_name=name)
                snapshots[symbol] = data.snapshot()
                return self.company_analyst.analyze(data, trace, candidate, risk_profile)
            except BudgetExceededError:
                return CompanyAnalysis.unavailable(symbol, "analysis unavailable: request budget exhausted",
                                                   company_name=name)
            except DataUnavailableError:
                return CompanyAnalysis.unavailable(symbol, "analysis unavailable: market data could not be fetched",
                                                   company_name=name)
            except LLMUnavailableError:
                return CompanyAnalysis.unavailable(symbol, "analysis unavailable: AI model unavailable",
                                                   company_name=name)
            except Exception:  # noqa: BLE001 - one failed worker must not fail the run
                logger.exception("trace_id=%s worker failed for %s", trace.trace_id, symbol)
                return CompanyAnalysis.unavailable(symbol, "analysis unavailable: unexpected error",
                                                   company_name=name)

        results = run_workers(items, work, self.settings.max_parallel_workers)
        out = [r if isinstance(r, CompanyAnalysis) else CompanyAnalysis.unavailable(items[i][0], "analysis unavailable")
               for i, r in enumerate(results)]
        for analysis in out:
            if analysis.status != "ok":
                trace.emit("worker_result", {"symbol": analysis.symbol, "status": "unavailable", "error": analysis.error})
        return out

    # --- flows --------------------------------------------------------------------------------

    def market_overview(self, *, refresh: bool = False, listener: Listener | None = None) -> FinalReport:
        trace = self.new_trace(listener)
        profile = self.settings.risk_profile_default

        def run() -> FinalReport:
            analysis, articles = self.market_analysis(trace, refresh=refresh)
            report = self.synthesizer.assemble(report_id=new_report_id(), kind="market_overview", risk_profile=profile,
                                               trace=trace, market=analysis, summary=analysis.summary)
            return self._finish(trace, report, {"market": analysis.model_dump(mode="json"),
                                                "article_ids": [a.id for a in articles]})

        return self._guard("market_overview", trace, profile, run)

    def find_opportunities(self, *, risk_profile: str | None = None, sectors: list[str] | None = None,
                           max_candidates: int | None = None, listener: Listener | None = None) -> FinalReport:
        profile = self.risk_profile(risk_profile)
        limit = max(1, min(int(max_candidates or self.settings.max_candidates), self.settings.max_candidates))
        trace = self.new_trace(listener)

        def run() -> FinalReport:
            trace.progress("start", f"Finding opportunities for a {profile} investor", trace_id=trace.trace_id)
            notes: list[str] = []
            try:
                analysis, articles = self.market_analysis(trace)
            except BudgetExceededError:
                analysis, articles = MarketAnalysis(summary="Market analysis not completed (budget)."), []
            trace.progress("screener", "Mapping themes to sectors and companies")
            candidates, used_fallback = self.screener.screen(analysis, risk_profile=profile, sectors=sectors,
                                                             max_candidates=limit, trace=trace)
            if used_fallback:
                notes.append("Candidates came from the deterministic screen (the AI screener did not complete).")
            if not candidates:
                notes.append("No candidates passed screening (data unavailable or filters too strict).")
            snapshots: dict[str, Any] = {}
            trace.progress("analysis", f"Analysing {len(candidates)} companies in parallel",
                           candidates=[c.symbol for c in candidates])
            companies = self._analyze_companies([(c.symbol, c) for c in candidates], risk_profile=profile, trace=trace,
                                                snapshots=snapshots)
            trace.progress("synthesis", "Writing the report")
            report = self.synthesizer.assemble(report_id=new_report_id(), kind="opportunities", risk_profile=profile,
                                               trace=trace, market=analysis, companies=companies, notes=notes)
            return self._finish(trace, report, {"market": analysis.model_dump(mode="json"),
                                                "article_ids": [a.id for a in articles],
                                                "candidates": [c.model_dump() for c in candidates],
                                                "companies": snapshots})

        return self._guard("opportunities", trace, profile, run)

    def analyze(self, symbol: str, *, risk_profile: str | None = None, listener: Listener | None = None) -> FinalReport:
        profile = self.risk_profile(risk_profile)
        normalized = self.market.normalize(symbol)
        if not normalized or len(normalized) > 32:
            raise ValueError("symbol must be a ticker such as TCS, RELIANCE.NS or AAPL")
        trace = self.new_trace(listener)

        def run() -> FinalReport:
            snapshots: dict[str, Any] = {}
            [company] = self._analyze_companies([(normalized, None)], risk_profile=profile, trace=trace,
                                                snapshots=snapshots)
            if company.status == "ok":
                summary = (f"{company.company_name or company.symbol}: {company.verdict.replace('_', ' ').lower()} "
                           f"(confidence {company.confidence}). {company.thesis}")
            else:
                summary = f"{normalized}: {company.error}"
            report = self.synthesizer.assemble(report_id=new_report_id(), kind="analysis", risk_profile=profile,
                                               trace=trace, market=self.cached_market_analysis(), companies=[company],
                                               summary=summary)
            return self._finish(trace, report, {"companies": snapshots})

        return self._guard("analysis", trace, profile, run)

    def review_portfolio(self, *, risk_profile: str | None = None, portfolio_id: str = DEFAULT_PORTFOLIO,
                         listener: Listener | None = None) -> FinalReport:
        profile = self.risk_profile(risk_profile)
        trace = self.new_trace(listener)

        def run() -> FinalReport:
            holdings = self.holdings.list(portfolio_id)
            if not holdings:
                report = self.synthesizer.assemble(
                    report_id=new_report_id(), kind="portfolio_review", risk_profile=profile, trace=trace,
                    summary="No holdings saved yet. Add holdings (or upload a CSV) to get a portfolio review.")
                return self._finish(trace, report, {})
            trace.progress("portfolio", f"Valuing {len(holdings)} holdings")
            notes: list[str] = []
            try:
                analysis, _ = self.market_analysis(trace)
            except BudgetExceededError:
                analysis = None
            if analysis is None or not analysis.themes:
                notes.append("Market themes unavailable; theme overlap not assessed.")
            symbols = [h.symbol for h in holdings]
            quotes = {s: (q if not isinstance(q, Exception) else None)
                      for s, q in run_parallel(self.market.get_quote, symbols, max_workers=4).items()}
            funds = self.market.get_fundamentals_many(symbols)
            sectors = {s: (self.market.universe.sector_of(s) or self.market.universe.match_sector(funds[s].sector if s in funds else "")
                           or (funds[s].sector if s in funds and funds[s].sector else "Unknown")) for s in symbols}
            names = {s: (funds[s].name if s in funds else self.market.universe.name_of(s) or "") for s in symbols}
            snapshot = compute_snapshot(
                holdings, quotes, sectors=sectors, names=names, themes=analysis.themes if analysis else [],
                sector_matcher=self.market.universe.match_sector, max_stock_pct=self.settings.max_stock_pct,
                max_sector_pct=self.settings.max_sector_pct, currency=self.settings.currency)
            trace.emit("portfolio_snapshot", {"total_value": snapshot.total_value, "total_pnl_pct": snapshot.total_pnl_pct,
                                              "warnings": snapshot.concentration_warnings})
            index_history = self._index_history()
            positions = {p.symbol: p for p in snapshot.positions}
            data_snapshots: dict[str, Any] = {}

            def work(holding: Any) -> HoldingReview:
                position = positions[holding.symbol]
                trace.progress("portfolio", f"Reviewing {holding.symbol}", symbol=holding.symbol)
                base = dict(symbol=holding.symbol, company_name=position.company_name, quantity=holding.quantity,
                            avg_cost=holding.avg_cost, current_price=position.current_price, pnl_pct=position.pnl_pct,
                            allocation_pct=position.allocation_pct, data_quality=position.data_quality)
                if trace.remaining_calls < 1:
                    return HoldingReview(**base, status="unavailable", error="review unavailable: request budget exhausted")
                data: CompanyData | None = None
                try:
                    data = gather_company_data(holding.symbol, settings=self.settings, market=self.market, news=self.news,
                                               store=self.store, risk_profile=profile, index_history=index_history,
                                               company_name=position.company_name)
                    data_snapshots[holding.symbol] = data.snapshot()
                except DataUnavailableError:
                    pass
                try:
                    return self.portfolio_analyst.review(holding, position, data, snapshot,
                                                         analysis.themes if analysis else [], trace, profile)
                except BudgetExceededError:
                    return HoldingReview(**base, status="unavailable", error="review unavailable: request budget exhausted")
                except LLMUnavailableError:
                    return HoldingReview(**base, status="unavailable", error="review unavailable: AI model unavailable")
                except Exception:  # noqa: BLE001
                    logger.exception("trace_id=%s holding review failed for %s", trace.trace_id, holding.symbol)
                    return HoldingReview(**base, status="unavailable", error="review unavailable: unexpected error")

            results = run_workers(holdings, work, self.settings.max_parallel_workers)
            reviews = [r if isinstance(r, HoldingReview) else HoldingReview(
                symbol=holdings[i].symbol, quantity=holdings[i].quantity, avg_cost=holdings[i].avg_cost,
                status="unavailable", error="review unavailable") for i, r in enumerate(results)]
            portfolio = PortfolioReview(snapshot=snapshot, reviews=reviews)
            trace.progress("synthesis", "Writing the report")
            report = self.synthesizer.assemble(report_id=new_report_id(), kind="portfolio_review", risk_profile=profile,
                                               trace=trace, market=analysis, portfolio=portfolio, notes=notes)
            return self._finish(trace, report, {"portfolio": snapshot.model_dump(mode="json"),
                                                "holdings": [h.model_dump(mode="json") for h in holdings],
                                                "companies": data_snapshots})

        return self._guard("portfolio_review", trace, profile, run)

    def chat(self, message: str, *, risk_profile: str | None = None, listener: Listener | None = None) -> FinalReport:
        """Free-form question: route, then dispatch (long flows are pointed to their tab / endpoint)."""
        profile = self.risk_profile(risk_profile)
        text = (message or "").strip()
        if not text:
            raise ValueError("message must not be empty")
        trace = self.new_trace(listener)

        def run() -> FinalReport:
            decision = route(self.llm, self.settings, text, trace, self.market.universe)
            assemble = dict(report_id=new_report_id(), kind="chat", risk_profile=profile, trace=trace, route=decision)
            if decision.clarifying_question:
                report = self.synthesizer.assemble(**assemble, summary="", answer=decision.clarifying_question,
                                                   clarifying_question=decision.clarifying_question)
                return self._finish(trace, report, {})
            if decision.intent == "market_overview":
                analysis, _ = self.market_analysis(trace)
                report = self.synthesizer.assemble(**assemble, market=analysis, summary=analysis.summary,
                                                   answer=analysis.summary)
                return self._finish(trace, report, {})
            if decision.intent == "analyze_stock" and decision.symbols:
                snapshots: dict[str, Any] = {}
                companies = self._analyze_companies([(s, None) for s in decision.symbols[:3]], risk_profile=profile,
                                                    trace=trace, snapshots=snapshots)
                answer = "\n".join(
                    f"{c.symbol}: {c.verdict.replace('_', ' ').lower()} (confidence {c.confidence}). {c.thesis}"
                    if c.status == "ok" else f"{c.symbol}: {c.error}" for c in companies)
                report = self.synthesizer.assemble(**assemble, companies=companies, summary=answer, answer=answer)
                return self._finish(trace, report, {"companies": snapshots})
            if decision.intent == "find_opportunities":
                answer = ("Finding opportunities scans market news, screens sectors and analyses several companies, "
                          "which takes a few minutes. Use the Opportunities tab (or POST /api/find-opportunities) to run it "
                          "with live progress.")
                return self._finish(trace, self.synthesizer.assemble(**assemble, summary=answer, answer=answer), {})
            if decision.intent == "review_portfolio":
                answer = ("Use the My Portfolio tab (or POST /api/portfolio/review) to review your saved holdings; "
                          "it values each position and recommends BUY MORE / HOLD / TRIM / SELL with reasons.")
                return self._finish(trace, self.synthesizer.assemble(**assemble, summary=answer, answer=answer), {})
            answer, citations = self._answer_question(text, trace)
            report = self.synthesizer.assemble(**assemble, summary=answer, answer=answer, citations=citations)
            return self._finish(trace, report, {})

        return self._guard("chat", trace, profile, run)

    def _answer_question(self, question: str, trace: Trace) -> tuple[str, list[Any]]:
        articles = self.store.search(question, k=8)
        market = self.cached_market_analysis()
        data = {"question": question,
                "market_context": {"sentiment": market.overall_sentiment, "summary": market.summary,
                                   "themes": [t.theme for t in market.themes]} if market else "not loaded",
                "news": [{"id": a.id, "source": a.source, "title": a.title} for a in articles]}
        messages = [{"role": "system", "content": CHAT_SYSTEM.format(market=self.settings.market)},
                    input_message(data, articles, "Answer the question.")]
        allowed = collect_numbers(data) | collect_numbers(" ".join(a.prompt_line() for a in articles))
        draft: ChatAnswerDraft | None = None
        for attempt in range(2):
            result = self.llm.parse(model=self.settings.analyst_model, messages=messages, schema=ChatAnswerDraft,
                                    trace=trace, purpose="chat" + (":revise" if attempt else ""), temperature=0.3)
            draft = result.output  # type: ignore[assignment]
            problems = []
            bad = ungrounded_numbers([draft.answer], allowed)
            if bad:
                problems.append(f"figures not in the input data: {', '.join(bad[:5])}")
            phrases = forbidden_phrases([draft.answer])
            if phrases:
                problems.append(f"overconfident language: {', '.join(phrases)}")
            if not problems:
                break
            messages = [*messages, {"role": "user", "content": "Revise your answer: " + "; ".join(problems)}]
        else:
            draft = ChatAnswerDraft(answer=draft.answer + "\n\n(Note: some figures or wording above could not be "
                                    "verified against current data; please check them independently.)",
                                    citation_ids=draft.citation_ids)  # type: ignore[union-attr]
        citations, _ = self.store.citations(i for i in draft.citation_ids if i in {a.id for a in articles})
        return draft.answer, citations

    # --- misc ---------------------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        from .mongo import MongoNotConfigured, ping

        try:
            ping()
            mongo = "ok"
        except MongoNotConfigured:
            mongo = "not configured"
        except Exception as exc:  # noqa: BLE001
            mongo = f"unavailable ({type(exc).__name__})"
        return {
            "status": "ok", "market": self.settings.market, "prompt_version": PROMPT_VERSION,
            "models": self.settings.models(), "openai_configured": bool(self.settings.openai_api_key), "llm_provider": provider_name(),
            "openai_circuit": getattr(getattr(self.llm, "breaker", None), "state", "unknown"),
            "market_data_providers": self.market.breaker_states(), "news_providers": self.news.breaker_states(),
            "newsapi_configured": bool(self.settings.newsapi_key), "mongodb": mongo,
            "news_store_articles": len(self.store), "time": datetime.now(timezone.utc).isoformat(),
        }


_pipeline: StockFinderPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> StockFinderPipeline:
    """Process-wide pipeline, built on first use."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                started = time.monotonic()
                get_settings().warn_if_critic_shared()
                _pipeline = StockFinderPipeline()
                logger.info("Stock Finder pipeline ready in %.1fs", time.monotonic() - started)
    return _pipeline


def set_pipeline_for_tests(pipeline: StockFinderPipeline | None) -> None:
    global _pipeline
    _pipeline = pipeline
