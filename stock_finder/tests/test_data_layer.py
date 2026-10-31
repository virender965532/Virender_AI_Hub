"""Cache, validators, market-data fallback / staleness, universe, and news fetching + dedup."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from stock_finder.data.cache import TTLCache
from stock_finder.data.market_data import DataUnavailableError, MarketDataService, Universe, normalize_symbol
from stock_finder.data.news_data import NewsAPIProvider, NewsService, RSSProvider, deduplicate, make_article
from stock_finder.data.validators import build_quality, check_price, sanitize_fundamentals
from stock_finder.observability import Trace

from .conftest import FakeMarketProvider, FakeNewsProvider


def _service(settings, *providers, **kw) -> MarketDataService:
    return MarketDataService(settings, list(providers), sleep=lambda s: None, **kw)


# --- cache ----------------------------------------------------------------------------------------

def test_ttl_cache_expires_and_bounds():
    now = [0.0]
    cache = TTLCache(max_entries=2, clock=lambda: now[0])
    cache.set("a", 1, ttl=10)
    assert cache.get("a") == 1
    now[0] = 11
    assert cache.get("a") is None
    cache.set("b", 2, 100)
    cache.set("c", 3, 100)
    cache.set("d", 4, 100)
    assert cache.get("b") is None and cache.get("d") == 4


# --- validators -----------------------------------------------------------------------------------

def test_stale_price_is_flagged():
    now = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)  # Thursday
    issues, stale = check_price(100.0, datetime(2026, 10, 28, tzinfo=timezone.utc), now=now)
    assert not stale and not issues
    issues, stale = check_price(100.0, datetime(2026, 10, 23, tzinfo=timezone.utc), now=now)
    assert stale and "stale" in issues[0]


def test_weekend_is_not_stale():
    monday = datetime(2026, 11, 2, 9, tzinfo=timezone.utc)
    _, stale = check_price(100.0, datetime(2026, 10, 30, tzinfo=timezone.utc), now=monday)  # Friday close
    assert not stale


def test_invalid_values_rejected():
    clean, issues = sanitize_fundamentals({"pe": 5000.0, "roe": 0.2, "pb": -3, "debt_to_equity": 0.4,
                                           "revenue_growth": 0.1, "profit_margin": 0.1})
    assert clean["pe"] is None and clean["pb"] is None and clean["roe"] == 0.2
    assert any("pe value" in i for i in issues) and any("missing fundamentals" in i for i in issues)
    issues, _ = check_price(-1, None)
    assert "price missing or invalid" in issues


def test_quality_levels():
    assert build_quality().level == "good"
    assert build_quality(history_issues=["only 50 days"]).level == "partial"
    assert build_quality(stale=True).level == "poor"
    assert build_quality(has_fundamentals=False).level == "poor"


# --- market data ----------------------------------------------------------------------------------

def test_quote_and_cache_hit(settings):
    provider = FakeMarketProvider()
    service = _service(settings, provider)
    trace = Trace(max_llm_calls=5)
    with trace:
        quote = service.get_quote("TCS.NS")
        service.get_quote("TCS.NS")
    assert quote.price and not quote.stale and quote.provider == "fake"
    assert trace.cache_hits == 1 and trace.fetches == 2


def test_stale_quote_flagged(settings):
    quote = _service(settings, FakeMarketProvider(stale_days=10)).get_quote("TCS.NS")
    assert quote.stale and any("stale" in i for i in quote.issues)


def test_provider_fallback(settings):
    broken, good = FakeMarketProvider(fail=True), FakeMarketProvider()
    broken.name = "broken"
    fundamentals = _service(settings, broken, good).get_fundamentals("INFY.NS")
    assert fundamentals.provider == "fake" and fundamentals.name == "Infosys"
    assert broken.calls and good.calls


def test_all_providers_fail_raises(settings):
    with pytest.raises(DataUnavailableError):
        _service(settings, FakeMarketProvider(fail=True)).get_quote("TCS.NS")


def test_sector_peers_and_index_snapshot(settings):
    service = _service(settings, FakeMarketProvider())
    peers = service.get_sector_peers("TCS.NS")
    assert "INFY.NS" in peers and "TCS.NS" not in peers and len(peers) <= settings.max_peers
    snapshot = service.get_index_snapshot()
    assert {q.name for q in snapshot} >= {"NIFTY 50", "SENSEX"} and all(q.price for q in snapshot)


def test_universe_resolution(settings):
    universe = Universe.load(settings.universe_path, "IN")
    assert normalize_symbol("tcs", settings, universe) == "TCS.NS"
    assert normalize_symbol("Infosys", settings, universe) == "INFY.NS"
    assert normalize_symbol("XYZCORP", settings, universe) == "XYZCORP.NS"
    assert universe.match_sector("Banking") == "Financials"
    assert universe.match_sector("IT services") == "Information Technology"
    assert universe.match_sector("pharma") == "Healthcare"


# --- news -----------------------------------------------------------------------------------------

def test_dedup_by_url_and_title():
    now = datetime.now(timezone.utc)
    rows = [
        {"source": "A", "title": "Sensex jumps 500 points on bank rally", "url": "https://a.com/x?utm_source=rss", "published": now},
        {"source": "A", "title": "Different title", "url": "https://A.com/x/", "published": now},
        {"source": "B", "title": "Sensex jumps 500 points on bank rally - Mint", "url": "https://b.com/y", "published": now},
        {"source": "C", "title": "Crude oil falls", "url": "https://c.com/z", "published": now},
    ]
    articles = deduplicate([make_article(r, "t") for r in rows])
    assert [a.source for a in articles] == ["A", "C"]


def test_market_news_and_provider_failure(settings):
    down = FakeNewsProvider(fail=True)
    down.name = "down"
    result = NewsService(settings, [FakeNewsProvider(), down], sleep=lambda s: None).fetch_market_news()
    assert len(result.articles) == 5 and result.failed_providers == ["down"]


def test_old_articles_filtered(settings):
    now = datetime.now(timezone.utc)
    rows = [{"source": "A", "title": "Old", "summary": "", "url": "https://a.com/1", "published": now - timedelta(days=30)},
            {"source": "A", "title": "New", "summary": "", "url": "https://a.com/2", "published": now}]
    result = NewsService(settings, [FakeNewsProvider(rows)]).fetch_market_news(lookback_days=7)
    assert [a.title for a in result.articles] == ["New"]


def test_company_news(settings):
    result = NewsService(settings, [FakeNewsProvider()]).fetch_company_news("INFY.NS", "Infosys")
    assert result.articles and all("infosys" in (a.title + a.summary).lower() for a in result.articles)


def test_html_is_stripped():
    article = make_article({"source": "X", "title": "<b>Hello</b> &amp; world", "summary": "<p>a<script>x</script></p>",
                            "url": "https://x.com/a"}, "rss")
    assert article.title == "Hello & world" and "<" not in article.summary


def test_rss_provider_parses_feed():
    xml = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>T</title>
      <item><title>Nifty ends higher</title><link>https://n.com/1</link><description>Banks lead</description>
      <pubDate>Wed, 28 Oct 2026 10:00:00 GMT</pubDate></item></channel></rss>"""

    def http_get(url, **kw):
        return SimpleNamespace(content=xml, raise_for_status=lambda: None)

    rows = RSSProvider([{"name": "Feed", "url": "https://n.com/rss"}], "", http_get=http_get).fetch(None, datetime.now(timezone.utc))
    assert rows[0]["title"] == "Nifty ends higher" and rows[0]["source"] == "Feed" and rows[0]["published"].day == 28


def test_newsapi_key_sent_in_header_not_url():
    seen = {}

    def http_get(url, params=None, headers=None, **kw):
        seen.update(url=url, params=params, headers=headers)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"status": "ok", "articles": [
            {"source": {"name": "Reuters"}, "title": "T", "description": "D", "url": "https://r.com/1",
             "publishedAt": "2026-10-28T10:00:00Z"}]})

    rows = NewsAPIProvider("secret-key", "market", http_get=http_get).fetch(None, datetime.now(timezone.utc))
    assert rows[0]["source"] == "Reuters"
    assert seen["headers"]["X-Api-Key"] == "secret-key" and "secret-key" not in str(seen["params"]) + seen["url"]


def test_quote_pages_are_not_news():
    for title in ("Tata Consultancy Services Limited (TCS.NS) stock price, news, quote and history - Yahoo Finance",
                  "TCS Share Price Today Live NSE/BSE", "INFY stock quote"):
        assert make_article({"source": "s", "title": title, "url": "https://x.com/q"}, "rss") is None
    assert make_article({"source": "s", "title": "TCS shares rise after Q2 results", "url": "https://x.com/n"}, "rss")
