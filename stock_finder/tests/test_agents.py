"""Critic rules, the generate-critic-revise loop, router, news analyst, screener and the company analyst."""

from __future__ import annotations

import json

import pytest

from stock_finder.agents.company_analyst import CompanyAnalyst, gather_company_data
from stock_finder.agents.critic import (
    Critic, ReviewContext, deterministic_review, extract_numbers, forbidden_phrases, run_with_critic,
)
from stock_finder.agents.news_analyst import NewsAnalyst
from stock_finder.agents.router import route
from stock_finder.agents.screener import Screener
from stock_finder.analysis.scoring import verdict_consistent
from stock_finder.data.news_data import make_article
from stock_finder.llm_client import ChatResult
from stock_finder.observability import Trace
from stock_finder.schemas import CompanyAnalysisDraft, MarketAnalysis, MarketTheme

from .conftest import CRITIC, FakeLLM, fake_articles, input_json, user_payload

INPUT = {"fundamentals": {"pe": 25.4, "roe_pct": 18.2, "sector_median_pe": 22.0},
         "technicals": {"rsi_14": 61.3, "return_3m_pct": -4.5}, "score_breakdown": {"composite": 68.4}}
ARTICLE_IDS = {"a1b2c3d4e5"}


def draft(**kw) -> CompanyAnalysisDraft:
    base = dict(
        reasoning="ROE of 18.2% is solid and P/E 25.4 is above the sector median of 22.0.",
        thesis="Quality compounder trading at a modest premium.", why_now="RSI 61.3 shows steady momentum.",
        strengths=["ROE 18.2%"], risks=["Valuation premium", "3-month return of -4.5%"],
        valuation_view="P/E 25.4 vs 22.0 median.", technical_view="Above its 200-day average.",
        news_view="No relevant company news in the window.", verdict="STRONG_BUY_CANDIDATE", confidence="medium",
        what_would_change_view="A fall in ROE below the sector.", citation_ids=[],
    )
    base.update(kw)
    return CompanyAnalysisDraft(**base)


def ctx(**kw) -> ReviewContext:
    base = dict(kind="company", label="TCS.NS", input_data=INPUT, allowed_citation_ids=ARTICLE_IDS,
                logic_check=lambda d: verdict_consistent(d.verdict, 68.4, "good"), data_quality="good")
    base.update(kw)
    return ReviewContext(**base)


def pass_verdict(**kw) -> dict:
    base = dict(reasoning="ok", numbers_grounded=True, claims_cited=True, logic_consistent=True,
                risks_balanced=True, overconfident=False, verdict="PASS", feedback="")
    base.update(kw)
    return base


# --- deterministic critic rules -------------------------------------------------------------------

def test_grounded_draft_passes_deterministic_checks():
    v = deterministic_review(draft(), ctx())
    assert v.verdict == "PASS", v.feedback


def test_critic_fails_number_not_in_input():
    v = deterministic_review(draft(valuation_view="P/E 31.7 is cheap relative to peers."), ctx())
    assert v.verdict == "FAIL" and not v.numbers_grounded and "31.7" in v.feedback


def test_rounding_and_conventions_are_allowed():
    v = deterministic_review(draft(reasoning="ROE about 18% and the 50-day and 200-day averages, RSI(14) at 61.3; "
                                             "52-week view; return -4.5 over 3 months; FY25 results"), ctx())
    assert v.verdict == "PASS", v.feedback


def test_critic_fails_uncited_news_claim():
    v = deterministic_review(draft(news_view="Recent news reported a large deal win."), ctx())
    assert v.verdict == "FAIL" and not v.claims_cited
    ok = deterministic_review(draft(news_view="Recent news reported a large deal win.", citation_ids=["a1b2c3d4e5"]), ctx())
    assert ok.verdict == "PASS", ok.feedback


def test_critic_fails_unknown_citation():
    v = deterministic_review(draft(citation_ids=["made-up-id"]), ctx())
    assert v.verdict == "FAIL" and "made-up-id" in v.feedback


@pytest.mark.parametrize("phrase", ["This is a guaranteed winner.", "A sure shot pick.", "You can't lose here."])
def test_critic_fails_overconfident_language(phrase):
    v = deterministic_review(draft(thesis=phrase), ctx())
    assert v.verdict == "FAIL" and v.overconfident


def test_negated_guarantee_is_fine():
    assert forbidden_phrases(["Returns are not guaranteed."]) == []


def test_verdict_must_match_score():
    v = deterministic_review(draft(), ctx(logic_check=lambda d: verdict_consistent(d.verdict, 41.0, "good")))
    assert v.verdict == "FAIL" and not v.logic_consistent


def test_high_confidence_with_poor_data_is_overconfident():
    v = deterministic_review(draft(confidence="high", verdict="WATCHLIST"), ctx(data_quality="poor"))
    assert v.verdict == "FAIL" and v.overconfident


def test_number_extraction():
    assert [v for _, v in extract_numbers("5,00,000 crore and 1,234.5 and -3.2%")] == [500000.0, 1234.5, -3.2]


# --- critic LLM layer and revision loop -----------------------------------------------------------

def test_deterministic_fail_skips_llm(settings):
    llm = FakeLLM(lambda *a: pytest.fail("LLM critic should not be called"))
    verdict, reviewed = Critic(settings, llm).review(draft(thesis="guaranteed"), ctx(), Trace(max_llm_calls=5))
    assert verdict.verdict == "FAIL" and not reviewed


def test_llm_critic_uses_critic_model_and_inconsistent_pass_becomes_fail(settings):
    llm = FakeLLM(lambda purpose, model, messages, schema: pass_verdict(overconfident=True))
    verdict, reviewed = Critic(settings, llm).review(draft(), ctx(), Trace(max_llm_calls=5))
    assert reviewed and verdict.verdict == "FAIL" and llm.calls[0]["model"] == CRITIC
    assert settings.critic_model != settings.analyst_model


def test_revision_then_pass(settings):
    drafts = iter([draft(thesis="guaranteed upside"), draft()])
    llm = FakeLLM(lambda *a: pass_verdict())
    final, status = run_with_critic(lambda fb: next(drafts), Critic(settings, llm), lambda d: ctx(), Trace(max_llm_calls=9))
    assert status.passed and status.revised and not status.downgraded and final.thesis != "guaranteed upside"


def test_still_failing_is_downgraded(settings):
    llm = FakeLLM(lambda *a: pass_verdict())
    feedback_seen = []

    def generate(feedback):
        feedback_seen.append(feedback)
        return draft(thesis="sure shot")

    _, status = run_with_critic(generate, Critic(settings, llm), lambda d: ctx(), Trace(max_llm_calls=9))
    assert status.downgraded and not status.passed and status.concerns
    assert feedback_seen[0] is None and "overconfident" in feedback_seen[1]


# --- router ---------------------------------------------------------------------------------------

def test_router_low_confidence_asks_one_question(settings, market):
    llm = FakeLLM(lambda *a: {"reasoning": "?", "intent": "general_question", "symbols": [], "confidence": 0.3,
                              "clarifying_question": ""})
    decision = route(llm, settings, "hmm", Trace(max_llm_calls=3), market.universe)
    assert decision.clarifying_question and llm.calls[0]["model"] == settings.router_model


def test_router_normalizes_symbols(settings, market):
    llm = FakeLLM(lambda *a: {"reasoning": "r", "intent": "analyze_stock", "symbols": ["tcs", "Infosys"],
                              "confidence": 0.9, "clarifying_question": "should be dropped"})
    decision = route(llm, settings, "analyse tcs and infosys", Trace(max_llm_calls=3), market.universe)
    assert decision.symbols == ["TCS.NS", "INFY.NS"] and decision.clarifying_question == ""


# --- news analyst ---------------------------------------------------------------------------------

def _news_responder(purpose, model, messages, schema):
    text = user_payload(messages)
    ids = [line.split("]")[0].lstrip("[") for line in text.splitlines() if line.startswith("[")]
    if purpose.startswith("news_map"):
        return {"themes": [{"theme": "IT demand recovery", "sectors": ["IT"], "sentiment": "positive",
                            "impact_horizon": "medium", "evidence_ids": ids[:2]},
                           {"theme": "Fake theme", "sectors": ["Energy"], "sentiment": "negative",
                            "impact_horizon": "short", "evidence_ids": ["not-a-real-id"]}]}
    data = input_json(messages)
    return {"reasoning": "r", "overall_sentiment": "positive", "summary": "IT leads the market.",
            "themes": data["candidate_themes"], "key_risks": ["Crude oil volatility"]}


def test_news_analyst_map_reduce_with_citations(settings, market, news, store):
    llm = FakeLLM(_news_responder)
    trace = Trace(max_llm_calls=10)
    analysis, articles = NewsAnalyst(settings, llm, news, store, market).analyze(trace)
    assert [t.theme for t in analysis.themes] == ["IT demand recovery"]  # uncited theme dropped
    assert all(c.url.startswith("https://") and c.published_date for c in analysis.themes[0].evidence)
    assert analysis.index_snapshot and analysis.article_count == len(articles) == 5
    assert llm.purposes() == ["news_map:1", "news_reduce"]
    assert llm.calls[0]["model"] == settings.worker_model and llm.calls[1]["model"] == settings.analyst_model
    prompt = user_payload(llm.calls[0]["messages"])
    injected = prompt.index("Ignore previous instructions")
    assert prompt.rfind("<news>", 0, injected) > prompt.rfind("</news>", 0, injected)  # inside an open <news> block
    assert prompt.find("</news>", injected) > injected
    assert "DATA written by third parties" in llm.calls[0]["messages"][0]["content"]


def test_news_summary_with_invented_number_is_replaced(settings, market, news, store):
    def responder(purpose, model, messages, schema):
        out = _news_responder(purpose, model, messages, schema)
        if purpose == "news_reduce":
            out["summary"] = "Nifty will rise 37.3% this year."
        return out

    analysis, _ = NewsAnalyst(settings, FakeLLM(responder), news, store, market).analyze(Trace(max_llm_calls=10))
    assert "37.3" not in analysis.summary and analysis.data_quality.issues


def test_news_analyst_without_news(settings, market, store):
    from stock_finder.data.news_data import NewsService

    from .conftest import FakeNewsProvider

    empty = NewsService(settings, [FakeNewsProvider([])])
    llm = FakeLLM(lambda *a: pytest.fail("no LLM call expected"))
    analysis, articles = NewsAnalyst(settings, llm, empty, store, market).analyze(Trace(max_llm_calls=5))
    assert not articles and analysis.degraded and analysis.data_quality.level == "poor"


# --- screener -------------------------------------------------------------------------------------

def _analysis() -> MarketAnalysis:
    from stock_finder.schemas import Citation

    return MarketAnalysis(overall_sentiment="positive", themes=[MarketTheme(
        theme="IT demand recovery", sectors=["Information Technology"], sentiment="positive", impact_horizon="medium",
        evidence=[Citation(source="s", title="t", url="https://x")])])


def test_screener_react_loop(settings, market, store):
    steps = iter([
        ChatResult(content="", tool_calls=[{"id": "1", "name": "screen_by_metrics",
                                            "arguments": json.dumps({"sector": "IT", "min_roe_pct": 20})}]),
        ChatResult(content="", tool_calls=[{"id": "2", "name": "submit_candidates", "arguments": json.dumps({"candidates": [
            {"symbol": "TCS", "company_name": "TCS", "linked_theme": "IT demand recovery", "why_candidate": "High ROE"},
            {"symbol": "TCS.NS", "company_name": "dup", "linked_theme": "x", "why_candidate": "dup"},
            {"symbol": "HDFCBANK", "company_name": "HDFC Bank", "linked_theme": "x", "why_candidate": "off-sector"},
        ]})}]),
    ])
    llm = FakeLLM(lambda *a: next(steps))
    trace = Trace(max_llm_calls=10)
    found, fallback = Screener(settings, llm, market, store).screen(_analysis(), risk_profile="moderate",
                                                                    sectors=["Information Technology"], max_candidates=5, trace=trace)
    assert not fallback and [c.symbol for c in found] == ["TCS.NS"]
    tool_msg = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]
    rows = json.loads(tool_msg["content"])
    assert {r["symbol"] for r in rows} == {"TCS.NS", "INFY.NS"}  # ROE >= 20%


def test_screener_falls_back_when_model_never_submits(settings, market, store):
    llm = FakeLLM(lambda *a: ChatResult(content="thinking..."))
    found, fallback = Screener(settings, llm, market, store).screen(_analysis(), risk_profile="moderate", sectors=None,
                                                                    max_candidates=3, trace=Trace(max_llm_calls=10))
    assert fallback and found and len(found) <= 3
    assert all(market.universe.sector_of(c.symbol) == "Information Technology" for c in found)
    assert len(llm.calls) == 2  # one nudge, then stop


def test_screener_rejects_penny_stocks(settings, market, store):
    from stock_finder.data.market_data import Quote

    original = market.get_quote
    market.get_quote = lambda s: Quote(symbol=s, price=3.0) if s == "WIPRO.NS" else original(s)
    found = Screener(settings, None, market, store)._validate(
        [{"symbol": "WIPRO.NS"}, {"symbol": "INFY.NS"}], sectors=None, limit=5, trace=Trace(max_llm_calls=1))
    assert [c.symbol for c in found] == ["INFY.NS"]


# --- company analyst ------------------------------------------------------------------------------

def _company_responder(bad: bool = False):
    def responder(purpose, model, messages, schema):
        if purpose.startswith("critic"):
            return pass_verdict()
        data = input_json(messages)
        pe = data["fundamentals"]["pe"]
        ids = [a["id"] for a in data["news"]][:1]
        return draft(reasoning=f"P/E is {pe}.", valuation_view=f"P/E {pe}.", strengths=["Strong franchise"],
                     risks=["Valuation", "Competition"], why_now="Momentum is constructive.",
                     thesis="guaranteed returns" if bad else "Steady compounder.",
                     news_view="Recent news reported a deal win." if ids else "No relevant news.",
                     citation_ids=ids, verdict="WATCHLIST")
    return responder


def test_company_analysis_end_to_end(settings, market, news, store):
    data = gather_company_data("INFY.NS", settings=settings, market=market, news=news, store=store,
                               risk_profile="moderate", index_history=market.get_history("^NSEI"))
    assert data.metrics.sector_median_pe is not None and data.technicals.sma_200 and data.articles
    llm = FakeLLM(_company_responder())
    result = CompanyAnalyst(settings, llm, Critic(settings, llm), store).analyze(data, Trace(max_llm_calls=10))
    assert result.status == "ok" and result.critic.passed and result.citations and result.citations[0].url
    assert result.score_breakdown.composite is not None and result.price
    assert llm.purposes() == ["company:INFY.NS", "critic:INFY.NS"]


def test_company_analysis_downgraded_when_critic_keeps_failing(settings, market, news, store):
    data = gather_company_data("TCS.NS", settings=settings, market=market, news=news, store=store, risk_profile="moderate")
    llm = FakeLLM(_company_responder(bad=True))
    result = CompanyAnalyst(settings, llm, Critic(settings, llm), store).analyze(data, Trace(max_llm_calls=10))
    assert result.verdict == "WATCHLIST" and result.confidence == "low"
    assert result.critic.downgraded and result.critic.concerns
    assert llm.purposes() == ["company:TCS.NS", "company:TCS.NS:revise"]  # deterministic FAILs: no critic LLM calls


def test_critic_fallback_never_reviews_its_own_draft(settings):
    critic = Critic(settings, None)
    assert critic.fallback_for(settings.worker_model) == settings.analyst_model  # worker == fallback model
    assert critic.fallback_for(settings.analyst_model) == settings.fallback_model
    llm = FakeLLM(lambda *a: pass_verdict())
    _, reviewer = Critic(settings, llm).review(draft(), ctx(generator_model=settings.worker_model), Trace(max_llm_calls=3))
    assert reviewer == settings.critic_model and llm.calls[0]["fallback_model"] == settings.analyst_model
