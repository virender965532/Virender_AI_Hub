"""Gradio UI smoke test and rendering helpers (no server is started)."""

from __future__ import annotations

import pytest

from stock_finder.pipeline import StockFinderPipeline, set_pipeline_for_tests
from stock_finder.portfolio.models import HoldingsRepository
from stock_finder.schemas import HoldingIn

from .conftest import FakeLLM
from .test_pipeline import responder

gr = pytest.importorskip("gradio")


@pytest.fixture()
def pipeline(settings, market, news, store):
    p = StockFinderPipeline(settings, llm=FakeLLM(responder()), market=market, news=news, store=store)
    set_pipeline_for_tests(p)
    yield p
    set_pipeline_for_tests(None)


def test_build_ui(pipeline, settings):
    from stock_finder.ui import build_ui

    demo = build_ui(settings)
    assert isinstance(demo, gr.Blocks)


def test_render_reports_and_charts(pipeline, settings):
    from stock_finder import ui

    report = pipeline.find_opportunities()
    text = ui.opportunities_md(report)
    assert "Data as of" in text and "Composite" in text and "Critic" in text
    pulse = ui.market_md(pipeline.market_overview())
    assert "Market sentiment" in pulse and "https://news.example.com/" in pulse
    assert not ui.index_frame(pipeline.market_overview()).empty
    fig = ui.price_chart(pipeline.market.get_history("TCS.NS"), "TCS.NS", "INR")
    assert [t.name for t in fig.data] == ["Close", "50-day avg", "200-day avg"]

    HoldingsRepository().upsert(HoldingIn(symbol="TCS.NS", quantity=10, avg_cost=50))
    HoldingsRepository().upsert(HoldingIn(symbol="INFY.NS", quantity=1, avg_cost=50))
    review = pipeline.review_portfolio()
    assert "Concentration warnings" in ui.portfolio_md(review)
    chart = ui.allocation_chart(review, settings.max_stock_pct)
    assert list(chart.data[0].y) == ["INFY.NS", "TCS.NS"]  # sorted ascending, largest on top


def test_model_text_is_escaped():
    from stock_finder.ui import esc

    out = esc("<script>x</script> [click](http://evil) `code`")
    assert "<script>" not in out and "[" not in out and "`" not in out
