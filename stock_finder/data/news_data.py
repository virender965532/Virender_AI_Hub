"""News fetching from several sources, normalized into deduplicated ``Article`` objects.

Providers (``SF_NEWS_PROVIDERS``, all queried; a failing one is skipped):

- ``rss``: market feeds and a per-company search feed from ``config/stock_finder_feeds.yaml``.
- ``newsapi``: NewsAPI.org ``/v2/everything`` (only when ``NEWSAPI_KEY`` is set).

Articles are deduplicated by normalized URL and by title similarity, filtered
to the lookback window, and keep their source, published date and URL so every
claim can be cited. Article text is untrusted data: it is HTML-stripped and
length-capped here, and only ever shown to models inside ``<news>`` tags.
"""

from __future__ import annotations

import hashlib
import html
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol
from urllib.parse import quote_plus, urlsplit, urlunsplit, parse_qsl, urlencode

import requests
import yaml
from pydantic import BaseModel

from ..config import StockFinderSettings
from ..observability import log_data_fetch
from ..resilience import CircuitBreaker, call_with_retry
from ..schemas import Citation
from .cache import NEWS_TTL, TTLCache

logger = logging.getLogger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_TRACKING_PARAMS = ("utm_", "fbclid", "gclid", "ref", "cmpid", "ocid")
_MAX_SUMMARY = 600
_MAX_TITLE = 300
_TITLE_SIMILARITY = 0.88
_USER_AGENT = "Mozilla/5.0 (compatible; StockFinderResearchBot/1.0)"
# Search feeds also return quote / profile pages, which are not news and must not be cited.
_NOT_NEWS_RE = re.compile(
    r"(stock|share) price(,| today| live| chart| and)|quote(s)? (and|&) (history|news)|price target and|"
    r"stock (quote|forecast)|\bshare price\b.*\b(nse|bse)\b|historical (prices|data)",
    re.IGNORECASE,
)


class Article(BaseModel):
    id: str
    source: str
    title: str
    summary: str = ""
    url: str
    published: Optional[datetime] = None
    provider: str = ""

    def to_citation(self) -> Citation:
        return Citation(id=self.id, source=self.source, title=self.title, url=self.url,
                        published_date=self.published.date().isoformat() if self.published else None)

    def prompt_line(self) -> str:
        """Compact representation for prompts (inside ``<news>`` tags)."""
        date = self.published.date().isoformat() if self.published else "unknown date"
        return f"[{self.id}] {date} | {self.source} | {self.title} — {self.summary[:300]}"


def clean_text(text: Any, limit: int) -> str:
    text = html.unescape(_TAG_RE.sub(" ", str(text or "")))
    text = _WS_RE.sub(" ", text).strip()
    return text[:limit]


def normalize_url(url: str) -> str:
    """Lower-case host, drop fragments, tracking parameters and trailing slashes."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    query = [(k, v) for k, v in parse_qsl(parts.query) if not k.lower().startswith(_TRACKING_PARAMS)]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


def article_id(url: str) -> str:
    return hashlib.sha1(normalize_url(url).encode("utf-8")).hexdigest()[:10]


def _title_key(title: str) -> str:
    title = re.sub(r"\s+-\s+[^-]{2,40}$", "", title)  # Google News appends " - Source"
    return re.sub(r"[^a-z0-9 ]", "", title.lower()).strip()


def make_article(raw: dict[str, Any], provider: str) -> Article | None:
    """Build an ``Article`` from a provider row; rows without a URL or title are dropped."""
    url = str(raw.get("url") or "").strip()
    title = clean_text(raw.get("title"), _MAX_TITLE)
    if not url.startswith(("http://", "https://")) or not title or _NOT_NEWS_RE.search(title):
        return None
    published = raw.get("published")
    if isinstance(published, str):
        published = parse_date(published)
    if isinstance(published, datetime) and published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return Article(id=article_id(url), source=clean_text(raw.get("source"), 80) or urlsplit(url).netloc,
                   title=title, summary=clean_text(raw.get("summary"), _MAX_SUMMARY), url=url,
                   published=published if isinstance(published, datetime) else None, provider=provider)


def parse_date(value: str) -> datetime | None:
    try:
        return parsedate_to_datetime(value)
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def deduplicate(articles: list[Article]) -> list[Article]:
    """Drop repeats by normalized URL, then by near-identical titles (keeps the earliest seen)."""
    seen_urls: set[str] = set()
    kept: list[Article] = []
    keys: list[str] = []
    for article in articles:
        url = normalize_url(article.url)
        if url in seen_urls:
            continue
        key = _title_key(article.title)
        if any(key == other or SequenceMatcher(None, key, other).ratio() >= _TITLE_SIMILARITY for other in keys):
            continue
        seen_urls.add(url)
        keys.append(key)
        kept.append(article)
    return kept


# --- providers ------------------------------------------------------------------------------------

class NewsProvider(Protocol):
    name: str

    def fetch(self, query: str | None, since: datetime) -> list[dict[str, Any]]:
        """Market news when ``query`` is None, else news matching ``query``. Rows: source/title/summary/url/published."""
        ...


HttpGet = Callable[..., Any]


class RSSProvider:
    name = "rss"

    def __init__(self, feeds: list[dict[str, str]], company_search: str, *, timeout: float = 10.0,
                 http_get: HttpGet = requests.get) -> None:
        self.feeds = feeds
        self.company_search = company_search
        self.timeout = timeout
        self._get = http_get

    def _parse_feed(self, url: str, default_source: str) -> list[dict[str, Any]]:
        import feedparser

        # feedparser has no timeout of its own, so download with requests first.
        response = self._get(url, timeout=self.timeout, headers={"User-Agent": _USER_AGENT})
        response.raise_for_status()
        feed = feedparser.parse(response.content)
        rows = []
        for entry in feed.entries:
            published = None
            for key in ("published_parsed", "updated_parsed"):
                if entry.get(key):
                    published = datetime(*entry[key][:6], tzinfo=timezone.utc)
                    break
            source = default_source
            if isinstance(entry.get("source"), dict) and entry["source"].get("title"):
                source = entry["source"]["title"]
            rows.append({"source": source, "title": entry.get("title"), "summary": entry.get("summary", ""),
                         "url": entry.get("link"), "published": published})
        return rows

    def fetch(self, query: str | None, since: datetime) -> list[dict[str, Any]]:
        if query is not None:
            if not self.company_search:
                return []
            return self._parse_feed(self.company_search.format(query=quote_plus(query)), "Google News")
        rows: list[dict[str, Any]] = []
        errors = 0
        for feed in self.feeds:
            try:
                rows.extend(self._parse_feed(feed["url"], feed.get("name", "")))
            except Exception as exc:  # noqa: BLE001 - one broken feed must not hide the others
                errors += 1
                logger.warning("RSS feed %s failed: %s", feed.get("name") or feed.get("url"), type(exc).__name__)
        if errors and errors == len(self.feeds):
            raise ConnectionError("all RSS feeds failed")
        return rows


class NewsAPIProvider:
    name = "newsapi"
    endpoint = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str, market_query: str, *, timeout: float = 10.0, http_get: HttpGet = requests.get) -> None:
        self.api_key = api_key
        self.market_query = market_query
        self.timeout = timeout
        self._get = http_get

    def fetch(self, query: str | None, since: datetime) -> list[dict[str, Any]]:
        params = {"q": query or self.market_query, "from": since.date().isoformat(), "language": "en",
                  "sortBy": "publishedAt", "pageSize": 50}
        # Key in a header, never in the URL, so it cannot leak into logs.
        response = self._get(self.endpoint, params=params, timeout=self.timeout, headers={"X-Api-Key": self.api_key})
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") != "ok":
            raise RuntimeError(f"NewsAPI error: {payload.get('code')}")
        return [{"source": (a.get("source") or {}).get("name", ""), "title": a.get("title"),
                 "summary": a.get("description") or "", "url": a.get("url"), "published": a.get("publishedAt")}
                for a in payload.get("articles", [])]


def load_feed_config(path: Path, market: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        logger.warning("Stock Finder feeds file %s not found; RSS news disabled", path)
        return {}
    return data.get(market) or {}


def build_news_providers(settings: StockFinderSettings) -> list[NewsProvider]:
    config = load_feed_config(settings.feeds_path, settings.market)
    providers: list[NewsProvider] = []
    for name in settings.news_providers:
        if name == "rss" and (config.get("market_feeds") or config.get("company_search")):
            providers.append(RSSProvider(config.get("market_feeds") or [], config.get("company_search") or "",
                                         timeout=settings.data_timeout))
        elif name == "newsapi":
            if settings.newsapi_key:
                providers.append(NewsAPIProvider(settings.newsapi_key, config.get("newsapi_market_query") or "stock market",
                                                 timeout=settings.data_timeout))
            else:
                logger.info("NEWSAPI_KEY not set; NewsAPI provider skipped")
        elif name != "rss":
            logger.warning("Unknown news provider %r ignored", name)
    return providers


# --- service --------------------------------------------------------------------------------------

@dataclass
class NewsResult:
    articles: list[Article] = field(default_factory=list)
    failed_providers: list[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        return bool(self.failed_providers)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, requests.HTTPError):
        status = getattr(exc.response, "status_code", 0) or 0
        return status == 429 or status >= 500
    return isinstance(exc, (requests.ConnectionError, requests.Timeout, ConnectionError, TimeoutError))


class NewsService:
    """Fetches, cleans, deduplicates and caches news across all configured providers."""

    def __init__(
        self,
        settings: StockFinderSettings,
        providers: list[NewsProvider] | None = None,
        *,
        cache: TTLCache | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.settings = settings
        self.providers = build_news_providers(settings) if providers is None else providers
        self.cache = cache or TTLCache()
        self._sleep = sleep
        self._now = now
        self.breakers = {
            p.name: CircuitBreaker(f"news:{p.name}", settings.circuit_failure_threshold, settings.circuit_cooldown_seconds)
            for p in self.providers
        }

    def _collect(self, query: str | None, lookback_days: int) -> NewsResult:
        cache_key = f"news:{query or '__market__'}:{lookback_days}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            log_data_fetch(provider="cache", kind="news", key=query or "market", latency_ms=0, cache_hit=True)
            return NewsResult(list(cached.articles), list(cached.failed_providers))
        since = self._now() - timedelta(days=lookback_days)
        articles: list[Article] = []
        failed: list[str] = []
        for provider in self.providers:
            started = time.monotonic()
            try:
                rows = call_with_retry(lambda p=provider: p.fetch(query, since), breaker=self.breakers[provider.name],
                                       is_transient=_is_transient, max_retries=min(self.settings.max_retries, 2),
                                       sleep=self._sleep, label=f"news:{provider.name}")
            except Exception as exc:  # noqa: BLE001 - continue with the other providers
                failed.append(provider.name)
                log_data_fetch(provider=provider.name, kind="news", key=query or "market",
                               latency_ms=int((time.monotonic() - started) * 1000), cache_hit=False, ok=False,
                               error=type(exc).__name__)
                continue
            log_data_fetch(provider=provider.name, kind="news", key=query or "market",
                           latency_ms=int((time.monotonic() - started) * 1000), cache_hit=False)
            articles.extend(a for a in (make_article(r, provider.name) for r in rows) if a is not None)
        recent = [a for a in articles if a.published is None or a.published >= since]
        recent.sort(key=lambda a: a.published or since, reverse=True)
        result = NewsResult(deduplicate(recent), failed)
        if len(failed) < len(self.providers):  # never cache a total failure
            self.cache.set(cache_key, result, NEWS_TTL)
        return result

    def fetch_market_news(self, lookback_days: int | None = None) -> NewsResult:
        return self._collect(None, lookback_days or self.settings.news_lookback_days)

    def fetch_company_news(self, symbol: str, company_name: str = "", lookback_days: int | None = None) -> NewsResult:
        base = symbol.split(".")[0]
        query = f'"{company_name}" stock' if company_name and company_name != symbol else f"{base} share"
        result = self._collect(query, lookback_days or self.settings.news_lookback_days)
        result.articles = result.articles[: self.settings.max_company_articles * 2]
        return result

    def breaker_states(self) -> dict[str, str]:
        return {name: breaker.state for name, breaker in self.breakers.items()}
