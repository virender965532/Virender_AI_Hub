"""Router: classifies a free-form request into an intent (SF_ROUTER_MODEL, temperature 0)."""

from __future__ import annotations

from typing import Any

from ..config import StockFinderSettings
from ..data.market_data import Universe, normalize_symbol
from ..observability import Trace
from ..prompts import ROUTER_SYSTEM
from ..schemas import RouteDecision

CLARIFY_THRESHOLD = 0.5
DEFAULT_QUESTION = ("Could you clarify what you'd like: a market overview, new stock ideas, an analysis of a "
                    "specific stock, or a review of your portfolio?")
_MAX_MESSAGE_CHARS = 4000


def route(llm: Any, settings: StockFinderSettings, message: str, trace: Trace, universe: Universe | None = None) -> RouteDecision:
    """Classify ``message``; symbols are normalized to exchange tickers; low confidence gets one clarifying question."""
    text = message.strip()[:_MAX_MESSAGE_CHARS].replace("</user_request>", "")
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM.format(market=settings.market)},
        {"role": "user", "content": f"<user_request>\n{text}\n</user_request>"},
    ]
    result = llm.parse(model=settings.router_model, messages=messages, schema=RouteDecision, trace=trace,
                       purpose="router", temperature=0.0)
    decision: RouteDecision = result.output  # type: ignore[assignment]
    confidence = max(0.0, min(1.0, float(decision.confidence)))
    symbols = list(dict.fromkeys(normalize_symbol(s, settings, universe) for s in decision.symbols if s.strip()))[:5]
    question = decision.clarifying_question.strip()
    if confidence < CLARIFY_THRESHOLD:
        question = question or DEFAULT_QUESTION
    else:
        question = ""
    decision = decision.model_copy(update={"confidence": confidence, "symbols": symbols, "clarifying_question": question})
    trace.log("route_decision", intent=decision.intent, symbols=symbols, confidence=confidence)
    trace.emit("route", decision.model_dump(), log=False)
    return decision
