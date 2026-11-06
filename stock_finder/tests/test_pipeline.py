"""End-to-end pipeline flows with a fake LLM and fake data providers."""

from __future__ import annotations

import json

import pytest

from stock_finder.llm_client import ChatResult
from stock_finder.pipeline import JobManager, StockFinderPipeline, run_workers
from stock_finder.portfolio.models import HoldingsRepository
from stock_finder.schemas import FinalReport, HoldingIn

from .conftest import FakeLLM, input_json, user_payload


def verdict_ok() -> dict:
    return dict(reasoning="ok", numbers_grounded=True, claims_cited=True, logic_consistent=True, risks_balanced=True,
                overconfident=False, verdict="PASS", feedback="")


def company_draft(data: dict, verdict: str = "WATCHLIST") -> dict:
    pe = data["fundamentals"]["pe"]
    composite = data["score_breakdown"]["composite"]
    if verdict == "STRONG_BUY_CANDIDATE" and composite < 60:
        verdict = "WATCHLIST"
    return dict(reasoning=f"Composite score {composite}; P/E {pe}.", thesis="Quality business.", why_now="Momentum.",
                strengths=["Brand"], risks=["Valuation", "Execution"], valuation_view=f"P/E {pe}.",
                technical_view="Trend is up.", news_view="No relevant company news.", verdict=verdict,
                confidence="medium", what_would_change_view="Earnings miss.", citation_ids=[])


def responder(*, fail_company: str | None = None, holding_action: str = "HOLD"):
    def respond(purpose, model, messages, schema):
        text = user_payload(messages)
        if purpose.startswith("news_map"):
            ids = [line.split("]")[0].lstrip("[") for line in text.splitlines() if line.startswith("[")]
            return {"themes": [{"theme": "IT demand recovery", "sectors": ["Information Technology"],
                                "sentiment": "positive", "impact_horizon": "medium", "evidence_ids": ids[:2]}]}
        if purpose == "news_reduce":
            return {"reasoning": "r", "overall_sentiment": "positive", "summary": "IT leads.",
                    "themes": input_json(messages)["candidate_themes"], "key_risks": ["Crude oil"]}
        if purpose.startswith("screener"):
            return ChatResult(content="", tool_calls=[{"id": "1", "name": "submit_candidates", "arguments": json.dumps(
                {"candidates": [{"symbol": s, "company_name": s, "linked_theme": "IT demand recovery",
                                 "why_candidate": "theme fit"} for s in ("TCS", "INFY", "WIPRO")]})}])
        if purpose.startswith("company:"):
            if fail_company and fail_company in purpose:
                raise RuntimeError("worker exploded")
            return company_draft(input_json(messages), "STRONG_BUY_CANDIDATE")
        if purpose.startswith("holding:"):
            data = input_json(messages)
            action = holding_action if ":revise" not in purpose else "TRIM"
            return dict(reasoning=f"Allocation {data['position']['allocation_pct']}%.", action=action,
                        rationale="Position review.", risks=["Concentration"], triggers_to_revisit=["Earnings"],
                        confidence="medium", citation_ids=[])
        if purpose.startswith("critic"):
            return verdict_ok()
        if purpose == "summary":
            return {"executive_summary": "IT names lead the shortlist."}
        if purpose == "router":
            return {"reasoning": "r", "intent": "general_question", "symbols": [], "confidence": 0.9,
                    "clarifying_question": ""}
        if purpose.startswith("chat"):
            return {"answer": "Diversification reduces single-stock risk.", "citation_ids": []}
        raise AssertionError(f"unexpected purpose {purpose}")
    return respond


@pytest.fixture()
def make_pipeline(settings, market, news, store):
    def build(llm: FakeLLM, **kw) -> StockFinderPipeline:
        return StockFinderPipeline(kw.pop("settings", settings), llm=llm, market=market, news=news, store=store, **kw)
    return build


def assert_disclaimer(report: FinalReport) -> None:
    assert "not financial advice" in report.disclaimer and "SEBI-registered" in report.disclaimer


def test_find_opportunities_end_to_end(make_pipeline):
    llm = FakeLLM(responder())
    events: list[tuple[str, dict]] = []
    report = make_pipeline(llm).find_opportunities(risk_profile="moderate", listener=lambda e, d: events.append((e, d)))
    assert not report.error and report.kind == "opportunities"
    ranked = report.opportunities + report.watchlist + report.avoid
    assert {c.symbol for c in ranked} == {"TCS.NS", "INFY.NS", "WIPRO.NS"}
    scores = [c.score_breakdown.composite for c in report.opportunities]
    assert scores == sorted(scores, reverse=True)
    assert report.market_analysis.themes and report.market_analysis.themes[0].evidence
    assert report.executive_summary == "IT names lead the shortlist."
    assert report.metrics["llm_calls"] == len(llm.calls) and report.metrics["trace_id"]
    assert {e for e, _ in events} >= {"progress", "worker_result"}
    assert_disclaimer(report)


def test_one_failed_worker_does_not_fail_the_run(make_pipeline):
    report = make_pipeline(FakeLLM(responder(fail_company="INFY.NS"))).find_opportunities()
    assert not report.error and report.partial
    assert [c.symbol for c in report.unavailable] == ["INFY.NS"]
    assert "unavailable" in report.unavailable[0].error
    assert len(report.opportunities + report.watchlist + report.avoid) == 2


def test_budget_exceeded_returns_partial_results(make_pipeline, settings):
    from dataclasses import replace

    tight = replace(settings, max_llm_calls=5)
    report = make_pipeline(FakeLLM(responder()), settings=tight).find_opportunities()
    assert not report.error and report.partial and report.budget_exceeded
    assert report.metrics["llm_calls"] <= 5
    assert any("budget" in n for n in report.notes)
    assert report.unavailable and all("budget exhausted" in c.error for c in report.unavailable)
    assert len(report.opportunities + report.watchlist + report.avoid) >= 1  # what finished is kept
    assert_disclaimer(report)


def test_report_is_saved_for_audit(make_pipeline):
    pipeline = make_pipeline(FakeLLM(responder()))
    report = pipeline.analyze("TCS", risk_profile="conservative")
    doc = pipeline.reports.get(report.report_id, include_snapshot=True)
    assert doc["prompt_version"] == "v1" and doc["models"]["critic"] == "gpt-4.1"
    assert "TCS.NS" in doc["data_snapshot"]["companies"]
    assert doc["report"]["risk_profile"] == "conservative"


def test_analyze_single_stock(make_pipeline):
    report = make_pipeline(FakeLLM(responder())).analyze("infosys")
    company = (report.opportunities + report.watchlist)[0]
    assert company.symbol == "INFY.NS" and company.critic.passed and company.price
    assert company.data_quality.level in ("good", "partial")
    assert_disclaimer(report)


def test_portfolio_review_blocks_buy_more_on_concentrated_position(make_pipeline):
    repo = HoldingsRepository()
    repo.upsert(HoldingIn(symbol="TCS.NS", quantity=100, avg_cost=50))
    repo.upsert(HoldingIn(symbol="HDFCBANK.NS", quantity=1, avg_cost=50, thesis="Banking leader"))
    llm = FakeLLM(responder(holding_action="BUY_MORE"))
    report = make_pipeline(llm, holdings=repo).review_portfolio(risk_profile="aggressive")
    snap = report.portfolio.snapshot
    assert snap.concentration_warnings and snap.total_value > 0
    by = {r.symbol: r for r in report.portfolio.reviews}
    assert by["TCS.NS"].action == "TRIM" and by["TCS.NS"].critic.revised  # BUY_MORE rejected, revised
    assert by["TCS.NS"].pnl_pct is not None and by["TCS.NS"].allocation_pct > 25
    assert any(w for w in report.key_risks if "TCS.NS" in w)
    assert_disclaimer(report)


def test_portfolio_review_without_holdings(make_pipeline):
    report = make_pipeline(FakeLLM(responder())).review_portfolio()
    assert "No holdings" in report.executive_summary and report.portfolio is None


def test_chat_general_question(make_pipeline):
    llm = FakeLLM(responder())
    report = make_pipeline(llm).chat("What is diversification?")
    assert report.answer.startswith("Diversification") and llm.purposes() == ["router", "chat"]
    assert_disclaimer(report)


def test_chat_clarifying_question(make_pipeline):
    def respond(purpose, model, messages, schema):
        return {"reasoning": "?", "intent": "general_question", "symbols": [], "confidence": 0.2,
                "clarifying_question": "Which stock do you mean?"}
    report = make_pipeline(FakeLLM(respond)).chat("that one?")
    assert report.clarifying_question == "Which stock do you mean?"


def test_chat_answer_with_invented_number_is_revised(make_pipeline):
    answers = iter(["The index will gain 42.7% next year.", "Markets are uncertain."])
    base = responder()

    def respond(purpose, model, messages, schema):
        return {"answer": next(answers), "citation_ids": []} if purpose.startswith("chat") else base(purpose, model, messages, schema)

    llm = FakeLLM(respond)
    report = make_pipeline(llm).chat("Where is the market going?")
    assert report.answer == "Markets are uncertain." and llm.purposes() == ["router", "chat", "chat:revise"]


def test_unexpected_error_gives_friendly_message(make_pipeline, monkeypatch):
    pipeline = make_pipeline(FakeLLM(responder()))
    monkeypatch.setattr(pipeline.screener, "screen", lambda *a, **k: (_ for _ in ()).throw(KeyError("boom")))
    report = pipeline.find_opportunities()
    assert report.error and report.metrics["trace_id"] in report.error and "boom" not in report.error
    assert_disclaimer(report)


def test_invalid_risk_profile(make_pipeline):
    with pytest.raises(ValueError):
        make_pipeline(FakeLLM(responder())).find_opportunities(risk_profile="yolo")


def test_run_workers_keeps_order_and_returns_exceptions():
    def fn(x: int) -> int:
        if x == 2:
            raise ValueError("bad")
        return x * 10

    results = run_workers([1, 2, 3], fn, limit=2)
    assert results[0] == 10 and isinstance(results[1], ValueError) and results[2] == 30


def test_job_manager_streams_events_and_final(settings):
    def run(listener):
        listener("progress", {"message": "step"})
        return FinalReport(report_id="r", kind="analysis", created_at="t", as_of="t", market="IN",
                           disclaimer=settings.disclaimer)

    job = JobManager(1).submit("analysis", run)
    events = [e for e in job.stream(heartbeat=0.05) if e is not None]
    names = [e for e, _ in events]
    assert names[-1] == "final" and "progress" in names and job.status == "done"
    assert events[-1][1]["disclaimer"] == settings.disclaimer
