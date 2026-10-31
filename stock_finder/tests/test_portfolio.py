"""Holdings repository (mongomock), CSV parsing, P&L / allocation / concentration, report store."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_finder.data.market_data import Quote
from stock_finder.portfolio.models import HoldingsRepository, parse_holdings_csv
from stock_finder.portfolio.service import compute_snapshot
from stock_finder.reports import ReportStore
from stock_finder.schemas import Citation, FinalReport, Holding, HoldingIn, MarketTheme


def _q(symbol: str, price: float | None, stale: bool = False) -> Quote:
    return Quote(symbol=symbol, price=price, as_of=datetime.now(timezone.utc), stale=stale)


def _h(symbol: str, qty: float, cost: float) -> Holding:
    return Holding(symbol=symbol, quantity=qty, avg_cost=cost)


# --- repository -----------------------------------------------------------------------------------

def test_crud_round_trip():
    repo = HoldingsRepository()
    repo.upsert(HoldingIn(symbol="tcs.ns", quantity=10, avg_cost=3500, buy_date=date(2024, 5, 1), thesis="IT leader"))
    repo.upsert(HoldingIn(symbol="INFY.NS", quantity=5, avg_cost=1500))
    repo.upsert(HoldingIn(symbol="TCS.NS", quantity=12, avg_cost=3400))  # replace
    holdings = repo.list()
    assert [h.symbol for h in holdings] == ["INFY.NS", "TCS.NS"]
    tcs = holdings[1]
    assert tcs.quantity == 12 and tcs.thesis is None and tcs.created_at is not None
    assert repo.delete("TCS.NS") and not repo.delete("TCS.NS")
    assert repo.list("other") == []


def test_replace_all():
    repo = HoldingsRepository()
    repo.upsert(HoldingIn(symbol="A.NS", quantity=1, avg_cost=1))
    repo.replace_all([HoldingIn(symbol="B.NS", quantity=2, avg_cost=2)])
    assert [h.symbol for h in repo.list()] == ["B.NS"]


def test_invalid_holding_rejected():
    with pytest.raises(ValueError):
        HoldingIn(symbol="TCS.NS", quantity=0, avg_cost=10)
    with pytest.raises(ValueError):
        HoldingIn(symbol="TCS.NS", quantity=1, avg_cost=-5)


def test_parse_csv():
    text = "﻿symbol,quantity,avg_cost,buy_date,thesis\nTCS.NS,10,3500,2024-05-01,IT\nINFY.NS,abc,1500,,\n\nHDFCBANK.NS,4,1600,,\n"
    rows, errors = parse_holdings_csv(text)
    assert [r.symbol for r in rows] == ["TCS.NS", "HDFCBANK.NS"] and rows[0].buy_date == date(2024, 5, 1)
    assert len(errors) == 1 and "line 3" in errors[0]
    assert parse_holdings_csv("a,b\n1,2")[1]


def test_sample_csv_parses():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "sample_portfolio.csv").read_text(encoding="utf-8")
    rows, errors = parse_holdings_csv(text)
    assert rows and not errors


# --- P&L / allocation / concentration -------------------------------------------------------------

def test_pnl_allocation_and_concentration():
    holdings = [_h("TCS.NS", 10, 3000), _h("INFY.NS", 10, 1000), _h("HDFCBANK.NS", 10, 1500)]
    quotes = {"TCS.NS": _q("TCS.NS", 3600), "INFY.NS": _q("INFY.NS", 900), "HDFCBANK.NS": _q("HDFCBANK.NS", 1500)}
    sectors = {"TCS.NS": "Information Technology", "INFY.NS": "Information Technology", "HDFCBANK.NS": "Financials"}
    themes = [MarketTheme(theme="IT rally", sectors=["IT"], sentiment="positive", impact_horizon="short",
                          evidence=[Citation(source="s", title="t", url="https://x")])]
    snap = compute_snapshot(holdings, quotes, sectors=sectors, themes=themes, max_stock_pct=25, max_sector_pct=40,
                            sector_matcher=lambda s: "Information Technology" if s == "IT" else s)
    by = {p.symbol: p for p in snap.positions}
    assert by["TCS.NS"].pnl_pct == 20.0 and by["TCS.NS"].pnl_abs == 6000.0
    assert by["INFY.NS"].pnl_pct == -10.0
    assert snap.total_value == 36000 + 9000 + 15000 and snap.total_cost == 55000
    assert snap.total_pnl_abs == 5000.0 and snap.total_pnl_pct == pytest.approx(9.09, abs=0.01)
    assert by["TCS.NS"].allocation_pct == 60.0 and by["HDFCBANK.NS"].allocation_pct == 25.0
    assert snap.sector_allocation["Information Technology"] == 75.0
    assert by["TCS.NS"].breaches_stock_limit and not by["HDFCBANK.NS"].breaches_stock_limit  # 25.0 is not > 25
    assert by["INFY.NS"].breaches_sector_limit and not by["HDFCBANK.NS"].breaches_sector_limit
    assert any("TCS.NS is 60.0%" in w for w in snap.concentration_warnings)
    assert any("Information Technology sector is 75.0%" in w for w in snap.concentration_warnings)
    assert by["TCS.NS"].theme_overlap == ["IT rally"] and by["HDFCBANK.NS"].theme_overlap == []


def test_missing_price_valued_at_cost_and_flagged():
    snap = compute_snapshot([_h("A.NS", 10, 100), _h("B.NS", 10, 100)], {"A.NS": _q("A.NS", 110), "B.NS": None},
                            sectors={})
    by = {p.symbol: p for p in snap.positions}
    assert by["B.NS"].market_value is None and by["B.NS"].pnl_pct is None
    assert by["B.NS"].data_quality.level == "poor" and by["B.NS"].allocation_pct == pytest.approx(47.62, abs=0.01)
    assert snap.total_pnl_abs == 100.0


def test_stale_price_poor_quality():
    snap = compute_snapshot([_h("A.NS", 1, 100)], {"A.NS": _q("A.NS", 100, stale=True)}, sectors={})
    assert snap.positions[0].data_quality.level == "poor"


# --- report store ---------------------------------------------------------------------------------

def _report(rid: str = "r1") -> FinalReport:
    return FinalReport(report_id=rid, kind="analysis", created_at="now", as_of="now", market="IN", disclaimer="d",
                       prompt_version="v1", models={"analyst": "gpt-4o"})


def test_report_store_round_trip():
    store = ReportStore()
    assert store.save(_report(), {"TCS.NS": {"pe": 20}})
    doc = store.get("r1")
    assert doc["report"]["report_id"] == "r1" and doc["prompt_version"] == "v1" and "data_snapshot" not in doc
    assert store.get("r1", include_snapshot=True)["data_snapshot"]["TCS.NS"]["pe"] == 20
    assert store.list_recent()[0]["report_id"] == "r1"


def test_report_store_survives_mongo_outage():
    def broken():
        raise ConnectionError("mongo down")

    store = ReportStore(collection=broken)
    assert store.save(_report("r2")) is False
    assert store.get("r2")["report"]["report_id"] == "r2"
    assert store.list_recent()[0]["report_id"] == "r2"
