"""News analyst (RAG + Map-Reduce): market news -> themes with citations + overall sentiment.

1. Fetch market news (all providers), add it to the news store.
2. Retrieve with a fixed set of market-wide queries (macro, policy, earnings,
   flows, sectors) so the analysis covers the market, not just the latest feed.
3. Map: the worker model extracts themes per batch of articles (evidence ids).
4. Reduce: the analyst model merges them into one ``MarketAnalysis``.
Evidence ids are resolved through the store; themes left without any valid
citation are dropped. If the reduce step cannot run (budget / model down) the
map themes are merged deterministically and the result is marked degraded.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import StockFinderSettings
from ..data.market_data import MarketDataService
from ..data.news_data import Article, NewsService
from ..data.validators import build_quality
from ..llm_client import LLMUnavailableError
from ..observability import BudgetExceededError, Trace
from ..prompts import NEWS_MAP_SYSTEM, NEWS_REDUCE_SYSTEM
from ..rag.news_store import NewsStore
from ..schemas import MarketAnalysis, MarketAnalysisDraft, MarketTheme, ThemeBatch, ThemeDraft
from .common import input_message
from .critic import collect_numbers, ungrounded_numbers

logger = logging.getLogger(__name__)

MARKET_QUERIES = {
    "IN": [
        "Indian stock market outlook Sensex Nifty", "RBI monetary policy interest rates inflation",
        "quarterly earnings results profit growth", "sector rally stocks gain", "government policy budget reform",
        "FII DII foreign investor flows", "global markets crude oil dollar rupee",
    ],
    "US": [
        "US stock market outlook S&P 500 Nasdaq", "Federal Reserve interest rates inflation",
        "quarterly earnings results guidance", "sector rally stocks gain", "government policy tariffs regulation",
        "treasury yields dollar oil prices", "AI technology spending",
    ],
}
_PER_QUERY = 12
_MAX_THEMES = 8


class NewsAnalyst:
    def __init__(self, settings: StockFinderSettings, llm: Any, news: NewsService, store: NewsStore,
                 market: MarketDataService) -> None:
        self.settings = settings
        self.llm = llm
        self.news = news
        self.store = store
        self.market = market

    def _retrieve(self, fetched: list[Article]) -> list[Article]:
        since = datetime.now(timezone.utc) - timedelta(days=self.settings.news_lookback_days)
        picked: dict[str, Article] = {}
        for query in MARKET_QUERIES.get(self.settings.market, MARKET_QUERIES["IN"]):
            for article in self.store.search(query, k=_PER_QUERY, since=since):
                picked.setdefault(article.id, article)
        for article in fetched:  # top up with the newest headlines
            if len(picked) >= self.settings.max_news_for_analysis:
                break
            picked.setdefault(article.id, article)
        articles = list(picked.values())[: self.settings.max_news_for_analysis]
        articles.sort(key=lambda a: a.published.timestamp() if a.published else 0, reverse=True)
        return articles

    def _resolve(self, drafts: list[ThemeDraft], allowed: set[str]) -> list[MarketTheme]:
        themes: list[MarketTheme] = []
        for draft in drafts:
            ids = [i for i in draft.evidence_ids if i in allowed]
            citations, unknown = self.store.citations(ids)
            if not citations:
                logger.info("theme dropped (no valid evidence): %s", draft.theme)
                continue
            themes.append(MarketTheme(theme=draft.theme, sectors=draft.sectors, sentiment=draft.sentiment,
                                      impact_horizon=draft.impact_horizon, evidence=citations))
        return themes[:_MAX_THEMES]

    @staticmethod
    def _merge_without_llm(drafts: list[ThemeDraft]) -> MarketAnalysisDraft:
        merged: dict[str, ThemeDraft] = {}
        for draft in drafts:
            key = draft.theme.strip().lower()
            if key in merged:
                merged[key].evidence_ids.extend(i for i in draft.evidence_ids if i not in merged[key].evidence_ids)
            else:
                merged[key] = draft.model_copy(deep=True)
        votes = Counter(d.sentiment for d in drafts)
        overall = votes.most_common(1)[0][0] if votes else "neutral"
        ranked = sorted(merged.values(), key=lambda d: len(d.evidence_ids), reverse=True)
        return MarketAnalysisDraft(
            reasoning="Merged deterministically (reduce step unavailable).", overall_sentiment=overall,
            summary="Key themes from recent news: " + "; ".join(d.theme for d in ranked[:5]) + ".",
            themes=ranked, key_risks=[],
        )

    def analyze(self, trace: Trace) -> tuple[MarketAnalysis, list[Article]]:
        """Returns the market analysis and the articles it was based on."""
        trace.progress("news", "Fetching market news and index levels")
        fetched = self.news.fetch_market_news()
        self.store.add(fetched.articles)
        snapshot = self.market.get_index_snapshot()
        now = datetime.now(timezone.utc)
        issues = [f"news provider failed: {p}" for p in fetched.failed_providers]
        issues += [f"index {q.name} unavailable or stale" for q in snapshot if q.price is None or q.stale]
        articles = self._retrieve(fetched.articles)
        if not articles:
            quality = build_quality(other_issues=issues + ["no recent market news could be fetched"], has_fundamentals=False)
            trace.progress("news", "No news available; market view limited to index levels")
            return MarketAnalysis(summary="No recent market news could be fetched, so no themes were identified.",
                                  index_snapshot=snapshot, as_of=now.isoformat(), data_quality=quality, degraded=True), []

        allowed = {a.id for a in articles}
        sectors = self.market.universe.sector_names()
        batches = [articles[i:i + self.settings.news_batch_size] for i in range(0, len(articles), self.settings.news_batch_size)]
        trace.progress("news", f"Analysing {len(articles)} articles in {len(batches)} batches")
        map_system = NEWS_MAP_SYSTEM.format(market=self.settings.market)
        drafts: list[ThemeDraft] = []
        degraded = bool(fetched.failed_providers)
        for n, batch in enumerate(batches, 1):
            try:
                result = self.llm.parse(
                    model=self.settings.worker_model, purpose=f"news_map:{n}", trace=trace, temperature=0.2,
                    schema=ThemeBatch, messages=[{"role": "system", "content": map_system},
                                                 input_message({"SECTORS": sectors}, batch, "Extract the market themes.")])
            except BudgetExceededError:
                degraded = True
                break
            except LLMUnavailableError as exc:
                logger.warning("trace_id=%s news map batch %d failed: %s", trace.trace_id, n, exc)
                degraded = True
                continue
            degraded = degraded or result.degraded
            drafts.extend(result.output.themes)  # type: ignore[attr-defined]

        reduce_input = {
            "SECTORS": sectors,
            "index_snapshot": [q.model_dump() for q in snapshot],
            "candidate_themes": [d.model_dump() for d in drafts],
        }
        try:
            if not drafts:
                raise LLMUnavailableError("no themes from the map step")
            result = self.llm.parse(
                model=self.settings.analyst_model, purpose="news_reduce", trace=trace, temperature=0.2,
                schema=MarketAnalysisDraft,
                messages=[{"role": "system", "content": NEWS_REDUCE_SYSTEM.format(market=self.settings.market)},
                          input_message(reduce_input, articles, "Merge into one market analysis.")])
            merged: MarketAnalysisDraft = result.output  # type: ignore[assignment]
            degraded = degraded or result.degraded
        except (BudgetExceededError, LLMUnavailableError) as exc:
            logger.warning("trace_id=%s news reduce skipped: %s", trace.trace_id, exc)
            merged = self._merge_without_llm(drafts)
            degraded = True
            if not drafts:
                merged.summary = "Market news was fetched but themes could not be extracted (analysis model unavailable)."

        themes = self._resolve(merged.themes, allowed)
        # The market summary has no critic round, so apply the number-grounding rule directly.
        allowed_numbers = collect_numbers(reduce_input) | collect_numbers(" ".join(a.prompt_line() for a in articles))
        if ungrounded_numbers([merged.summary, *merged.key_risks], allowed_numbers):
            issues.append("model summary contained unverified figures and was replaced")
            merged.summary = "Key themes from recent news: " + "; ".join(t.theme for t in themes[:5]) + "."
            merged.key_risks = [r for r in merged.key_risks if not ungrounded_numbers([r], allowed_numbers)]
        quality = build_quality(other_issues=issues, has_fundamentals=True)
        analysis = MarketAnalysis(
            overall_sentiment=merged.overall_sentiment, summary=merged.summary, themes=themes,
            key_risks=merged.key_risks, index_snapshot=snapshot, article_count=len(articles),
            as_of=now.isoformat(), data_quality=quality, degraded=degraded,
        )
        trace.progress("news", f"Identified {len(themes)} market themes (sentiment: {analysis.overall_sentiment})",
                       themes=[t.theme for t in themes])
        return analysis, articles
