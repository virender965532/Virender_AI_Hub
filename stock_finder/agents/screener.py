"""Screener (ReAct, tool-calling): market themes -> sectors -> up to SF_MAX_CANDIDATES companies.

Tools: ``search_news``, ``get_sector_peers``, ``get_index_snapshot``,
``screen_by_metrics`` (fundamentals computed in Python) and
``submit_candidates`` (ends the loop). The loop is bounded by
``SF_SCREENER_MAX_STEPS``. Whatever the model submits is validated in code:
symbols are normalized, deduplicated, restricted to the requested sectors and
checked against a penny-stock price floor. If the loop fails or submits
nothing usable, a deterministic screen over the themes' sectors is used.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..analysis.fundamentals import compute_fundamentals
from ..config import StockFinderSettings
from ..data.market_data import DataUnavailableError, MarketDataService
from ..llm_client import LLMUnavailableError
from ..observability import BudgetExceededError, Trace
from ..prompts import SCREENER_SYSTEM
from ..rag.news_store import NewsStore
from ..schemas import Candidate, MarketAnalysis
from .common import input_message, to_json

logger = logging.getLogger(__name__)

_MAX_TOOL_RESULT_CHARS = 6000

TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "search_news", "description": "Search recent news articles in the news store.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_sector_peers", "description": "List universe companies in the same sector as a symbol.",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {
        "name": "get_index_snapshot", "description": "Current index levels, day change and 1-month return.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "screen_by_metrics",
        "description": "Screen a sector's companies on fundamentals computed in code; returns matching companies.",
        "parameters": {"type": "object", "properties": {
            "sector": {"type": "string", "description": "A name from the SECTORS list"},
            "min_roe_pct": {"type": "number"}, "max_pe": {"type": "number"},
            "max_debt_to_equity": {"type": "number"}, "min_revenue_growth_pct": {"type": "number"},
        }, "required": ["sector"]}}},
    {"type": "function", "function": {
        "name": "submit_candidates", "description": "Submit the final candidate list (ends screening).",
        "parameters": {"type": "object", "properties": {"candidates": {"type": "array", "items": {
            "type": "object", "properties": {
                "symbol": {"type": "string"}, "company_name": {"type": "string"},
                "linked_theme": {"type": "string"}, "why_candidate": {"type": "string"}},
            "required": ["symbol", "company_name", "linked_theme", "why_candidate"]}}},
            "required": ["candidates"]}}},
]


class Screener:
    def __init__(self, settings: StockFinderSettings, llm: Any, market: MarketDataService, store: NewsStore) -> None:
        self.settings = settings
        self.llm = llm
        self.market = market
        self.store = store

    # --- tools --------------------------------------------------------------------------------

    def screen_by_metrics(self, sector: str, *, min_roe_pct: float | None = None, max_pe: float | None = None,
                          max_debt_to_equity: float | None = None, min_revenue_growth_pct: float | None = None) -> list[dict[str, Any]]:
        matched = self.market.universe.match_sector(sector)
        if not matched:
            return []
        rows = []
        for symbol, fund in self.market.get_fundamentals_many(self.market.universe.symbols(matched)).items():
            m = compute_fundamentals(fund, market=self.settings.market)
            if min_roe_pct is not None and (m.roe_pct is None or m.roe_pct < min_roe_pct):
                continue
            if max_pe is not None and (m.pe is None or m.pe <= 0 or m.pe > max_pe):
                continue
            if max_debt_to_equity is not None and m.debt_to_equity is not None and m.debt_to_equity > max_debt_to_equity:
                continue
            if min_revenue_growth_pct is not None and (m.revenue_growth_pct is None or m.revenue_growth_pct < min_revenue_growth_pct):
                continue
            rows.append({"symbol": symbol, "name": fund.name, "sector": matched, "pe": m.pe, "roe_pct": m.roe_pct,
                         "debt_to_equity": m.debt_to_equity, "revenue_growth_pct": m.revenue_growth_pct,
                         "market_cap": m.market_cap, "money_unit": m.money_unit})
        rows.sort(key=lambda r: r["roe_pct"] if r["roe_pct"] is not None else -1e9, reverse=True)
        return rows

    def _run_tool(self, name: str, args: dict[str, Any], analysis: MarketAnalysis) -> Any:
        if name == "search_news":
            return [{"id": a.id, "date": a.published.date().isoformat() if a.published else None, "source": a.source,
                     "title": a.title} for a in self.store.search(str(args.get("query", "")), k=6)]
        if name == "get_sector_peers":
            symbol = self.market.normalize(str(args.get("symbol", "")))
            sector = self.market.universe.sector_of(symbol)
            return {"symbol": symbol, "sector": sector,
                    "peers": [{"symbol": s, "name": self.market.universe.name_of(s)}
                              for s in (self.market.universe.symbols(sector) if sector else [])]}
        if name == "get_index_snapshot":
            return [q.model_dump() for q in analysis.index_snapshot]
        if name == "screen_by_metrics":
            kwargs = {k: float(args[k]) for k in ("min_roe_pct", "max_pe", "max_debt_to_equity", "min_revenue_growth_pct")
                      if isinstance(args.get(k), (int, float))}
            return self.screen_by_metrics(str(args.get("sector", "")), **kwargs)
        return {"error": f"unknown tool {name}"}

    # --- validation ---------------------------------------------------------------------------

    def _validate(self, raw: list[dict[str, Any]], *, sectors: list[str] | None, limit: int, trace: Trace) -> list[Candidate]:
        allowed_sectors = {self.market.universe.match_sector(s) for s in sectors or []} - {None}
        out: list[Candidate] = []
        seen: set[str] = set()
        for item in raw:
            if len(out) >= limit:
                break
            try:
                symbol = self.market.normalize(str(item.get("symbol", "")))
            except Exception:  # noqa: BLE001
                continue
            if not symbol or symbol in seen or symbol.startswith("^"):
                continue
            sector = self.market.universe.sector_of(symbol)
            if allowed_sectors and sector not in allowed_sectors:
                trace.log("candidate_rejected", symbol=symbol, reason="outside requested sectors")
                continue
            try:
                quote = self.market.get_quote(symbol)
            except DataUnavailableError:
                trace.log("candidate_rejected", symbol=symbol, reason="no market data")
                continue
            if quote.price is None or quote.price < self.settings.penny_price:
                trace.log("candidate_rejected", symbol=symbol, reason="penny stock or no price", price=quote.price)
                continue
            seen.add(symbol)
            out.append(Candidate(symbol=symbol,
                                 company_name=str(item.get("company_name") or self.market.universe.name_of(symbol) or symbol),
                                 linked_theme=str(item.get("linked_theme") or "")[:200],
                                 why_candidate=str(item.get("why_candidate") or "")[:600]))
        return out

    def fallback(self, analysis: MarketAnalysis, *, sectors: list[str] | None, limit: int, trace: Trace) -> list[Candidate]:
        """Deterministic screen: sectors of positive themes first, best ROE with moderate leverage."""
        ordered = sorted(analysis.themes, key=lambda t: {"positive": 0, "neutral": 1, "negative": 2}[t.sentiment])
        targets: list[tuple[str, str]] = []
        for theme in ordered:
            if theme.sentiment == "negative":
                continue
            for sector in theme.sectors:
                matched = self.market.universe.match_sector(sector)
                if matched and matched not in [t[0] for t in targets]:
                    targets.append((matched, theme.theme))
        for sector in sectors or []:
            matched = self.market.universe.match_sector(sector)
            if matched and matched not in [t[0] for t in targets]:
                targets.insert(0, (matched, "user-selected sector"))
        if sectors:
            wanted = {self.market.universe.match_sector(s) for s in sectors}
            targets = [t for t in targets if t[0] in wanted]
        if not targets:
            targets = [(s, "broad market screen") for s in self.market.universe.sector_names()[:3]]
        raw: list[dict[str, Any]] = []
        per_sector = max(1, -(-limit // len(targets)))
        for sector, theme in targets:
            rows = self.screen_by_metrics(sector, max_debt_to_equity=2.0)
            raw += [{"symbol": r["symbol"], "company_name": r["name"], "linked_theme": theme,
                     "why_candidate": f"Deterministic screen: strong return on equity within {sector}."}
                    for r in rows[:per_sector]]
        trace.log("screener_fallback", sectors=[t[0] for t in targets])
        return self._validate(raw, sectors=sectors, limit=limit, trace=trace)

    # --- ReAct loop ---------------------------------------------------------------------------

    def screen(self, analysis: MarketAnalysis, *, risk_profile: str, sectors: list[str] | None, max_candidates: int,
               trace: Trace) -> tuple[list[Candidate], bool]:
        """Returns ``(candidates, used_fallback)``."""
        limit = max(1, min(max_candidates, self.settings.max_candidates))
        system = SCREENER_SYSTEM.format(market=self.settings.market, max_candidates=limit, risk_profile=risk_profile,
                                        max_steps=self.settings.screener_max_steps)
        data = {
            "SECTORS": self.market.universe.sector_names(),
            "requested_sectors": sectors or "any",
            "overall_sentiment": analysis.overall_sentiment,
            "themes": [{"theme": t.theme, "sectors": t.sectors, "sentiment": t.sentiment, "horizon": t.impact_horizon}
                       for t in analysis.themes],
        }
        messages: list[dict[str, Any]] = [{"role": "system", "content": system},
                                          input_message(data, task="Find candidate companies, then call submit_candidates.")]
        nudged = False
        for step in range(1, self.settings.screener_max_steps + 1):
            try:
                reply = self.llm.chat(model=self.settings.analyst_model, messages=messages, trace=trace,
                                      purpose=f"screener:{step}", tools=TOOLS, temperature=0.2)
            except (LLMUnavailableError, BudgetExceededError) as exc:
                logger.warning("trace_id=%s screener stopped: %s", trace.trace_id, exc)
                break
            if not reply.tool_calls:
                if nudged:
                    break
                nudged = True
                messages += [{"role": "assistant", "content": reply.content or ""},
                             {"role": "user", "content": "Call submit_candidates now with your final list."}]
                continue
            messages.append({"role": "assistant", "content": reply.content or None, "tool_calls": [
                {"id": tc["id"] or f"call_{step}_{i}", "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for i, tc in enumerate(reply.tool_calls)]})
            for i, tc in enumerate(reply.tool_calls):
                call_id = tc["id"] or f"call_{step}_{i}"
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                if tc["name"] == "submit_candidates":
                    found = self._validate(list(args.get("candidates") or []), sectors=sectors, limit=limit, trace=trace)
                    trace.progress("screener", f"Shortlisted {len(found)} candidates",
                                   candidates=[c.symbol for c in found])
                    if found:
                        return found, False
                    return self.fallback(analysis, sectors=sectors, limit=limit, trace=trace), True
                trace.progress("screener", f"Tool {tc['name']}", tool=tc["name"], args=args)
                try:
                    result = self._run_tool(tc["name"], args, analysis)
                except Exception as exc:  # noqa: BLE001 - report the error to the model
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                content = to_json(result)
                messages.append({"role": "tool", "tool_call_id": call_id, "content": content[:_MAX_TOOL_RESULT_CHARS]})
        trace.progress("screener", "Screening loop ended without a submission; using the deterministic screen")
        return self.fallback(analysis, sectors=sectors, limit=limit, trace=trace), True
