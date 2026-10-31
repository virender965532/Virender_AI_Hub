"""Flask endpoints via the test client (fake LLM + fake data providers)."""

from __future__ import annotations

import json

import pytest
from flask import Flask

from stock_finder import register_stock_finder
from stock_finder.pipeline import StockFinderPipeline, set_pipeline_for_tests

from .conftest import FakeLLM
from .test_pipeline import responder

PREFIX = "/stock-finder/api"


@pytest.fixture()
def client(settings, market, news, store):
    pipeline = StockFinderPipeline(settings, llm=FakeLLM(responder()), market=market, news=news, store=store)
    set_pipeline_for_tests(pipeline)
    app = Flask(__name__)
    register_stock_finder(app)
    yield app.test_client()
    set_pipeline_for_tests(None)


def events(response) -> list[tuple[str, dict]]:
    out = []
    for block in response.get_data(as_text=True).split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line and not line.startswith(":"))
        if "event" in lines:
            out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_health(client):
    res = client.get(f"{PREFIX}/health")
    body = res.get_json()
    assert res.status_code == 200 and body["mongodb"] == "ok" and body["openai_configured"] is True
    assert "sk-" not in res.get_data(as_text=True) and "not financial advice" in body["disclaimer"]


def test_market_overview(client):
    body = client.get(f"{PREFIX}/market-overview").get_json()
    assert body["ok"] and body["kind"] == "market_overview" and body["market_analysis"]["themes"]
    assert "SEBI" in body["disclaimer"]


def test_find_opportunities_streams_progress_and_final(client):
    res = client.post(f"{PREFIX}/find-opportunities", json={"risk_profile": "aggressive", "max_candidates": 3})
    assert res.mimetype == "text/event-stream"
    evs = events(res)
    names = [e for e, _ in evs]
    assert names[0] == "session" and "disclaimer" in evs[0][1]
    assert "progress" in names and names[-1] == "final"
    final = evs[-1][1]
    assert final["risk_profile"] == "aggressive" and "not financial advice" in final["disclaimer"]
    report = client.get(f"{PREFIX}/reports/{final['report_id']}").get_json()
    assert report["ok"] and report["report"]["report_id"] == final["report_id"] and report["disclaimer"]


def test_find_opportunities_job_mode(client):
    res = client.post(f"{PREFIX}/find-opportunities", json={"stream": False})
    assert res.status_code == 202
    job_id = res.get_json()["job_id"]
    list(events(client.get(f"{PREFIX}/jobs/{job_id}/events")))  # wait for completion
    status = client.get(f"{PREFIX}/jobs/{job_id}").get_json()
    assert status["status"] == "done" and status["report"]["kind"] == "opportunities" and status["disclaimer"]


def test_validation_errors_have_disclaimer(client):
    for res in (client.post(f"{PREFIX}/find-opportunities", json={"risk_profile": "yolo"}),
                client.post(f"{PREFIX}/find-opportunities", json={"sectors": "IT"}),
                client.post(f"{PREFIX}/analyze", json={}),
                client.post(f"{PREFIX}/chat", json={"message": ""})):
        body = res.get_json()
        assert res.status_code == 400 and not body["ok"] and "not financial advice" in body["disclaimer"]
    assert client.get(f"{PREFIX}/reports/nope").status_code == 404
    assert client.get(f"{PREFIX}/jobs/nope").status_code == 404


def test_analyze(client):
    body = client.post(f"{PREFIX}/analyze", json={"symbol": "TCS", "risk_profile": "conservative"}).get_json()
    company = (body["opportunities"] + body["watchlist"] + body["avoid"])[0]
    assert body["ok"] and company["symbol"] == "TCS.NS" and company["score_breakdown"]["composite"] is not None
    assert body["disclaimer"]


def test_chat(client):
    body = client.post(f"{PREFIX}/chat", json={"message": "What is diversification?"}).get_json()
    assert body["ok"] and body["answer"] and body["disclaimer"]


def test_portfolio_crud_csv_and_review(client):
    res = client.post(f"{PREFIX}/portfolio", json={"symbol": "tcs", "quantity": 10, "avg_cost": 3000})
    assert res.status_code == 201 and res.get_json()["saved"][0]["symbol"] == "TCS.NS"
    bad = client.post(f"{PREFIX}/portfolio", json={"symbol": "TCS", "quantity": -1, "avg_cost": 3000})
    assert bad.status_code == 400 and bad.get_json()["disclaimer"]
    csv_text = "symbol,quantity,avg_cost\nINFY,5,1400\nHDFCBANK,3,1500\n"
    res = client.post(f"{PREFIX}/portfolio", data=csv_text, content_type="text/csv")
    assert res.status_code == 201
    holdings = client.get(f"{PREFIX}/portfolio").get_json()["holdings"]
    assert sorted(h["symbol"] for h in holdings) == ["HDFCBANK.NS", "INFY.NS"]  # CSV replaces by default
    assert client.delete(f"{PREFIX}/portfolio?symbol=INFY").get_json()["deleted"] == 1
    assert client.delete(f"{PREFIX}/portfolio?symbol=INFY").status_code == 404
    evs = events(client.post(f"{PREFIX}/portfolio/review", json={"risk_profile": "moderate"}))
    final = evs[-1][1]
    assert evs[-1][0] == "final" and final["portfolio"]["reviews"][0]["symbol"] == "HDFCBANK.NS"
    assert final["disclaimer"]


def test_portfolio_without_mongo_returns_503(client, monkeypatch):
    from stock_finder import mongo

    def not_configured(name):
        raise mongo.MongoNotConfigured("x")

    monkeypatch.setattr("stock_finder.portfolio.models.get_collection", not_configured)
    from stock_finder.pipeline import get_pipeline
    from stock_finder.portfolio.models import HoldingsRepository

    get_pipeline().holdings = HoldingsRepository()
    res = client.get(f"{PREFIX}/portfolio")
    assert res.status_code == 503 and "not configured" in res.get_json()["error"] and res.get_json()["disclaimer"]
