"""Market data: quotes, price history, fundamentals, sector peers and index snapshots.

``MarketDataProvider`` is the adapter interface; ``YFinanceProvider`` is the
default implementation. Paid APIs can be added as further adapters and listed
in ``SF_MARKET_DATA_PROVIDER`` (comma-separated, tried in order).

``MarketDataService`` adds, for every call: a TTL cache, transient-only retries
with backoff, a circuit breaker per provider, provider fallback (first provider
that answers wins), validation (stale / invalid values) and a fetch log line on
the current trace. When every provider fails it raises ``DataUnavailableError``
and the caller continues with partial data.
"""

from __future__ import annotations

import contextvars
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Protocol, TypeVar

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from ..config import StockFinderSettings
from ..observability import log_data_fetch
from ..resilience import CircuitBreaker, call_with_retry
from ..schemas import IndexQuote
from .cache import FUNDAMENTALS_TTL, HISTORY_TTL, PRICE_TTL, TTLCache
from .validators import check_price, is_number, sanitize_fundamentals

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ProviderDataError(RuntimeError):
    """The provider answered but had no usable data (unknown symbol, empty response)."""


class DataUnavailableError(RuntimeError):
    """No configured provider could supply the data."""


def is_transient(exc: Exception) -> bool:
    """Network / rate-limit style errors are retried; data errors are not."""
    if isinstance(exc, (ProviderDataError, KeyError, ValueError, TypeError)):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    name = type(exc).__name__.lower()
    return any(word in name for word in ("ratelimit", "timeout", "connection", "curl", "http", "request"))


# --- models ---------------------------------------------------------------------------------------

class Quote(BaseModel):
    symbol: str
    price: Optional[float] = None
    previous_close: Optional[float] = None
    change_pct: Optional[float] = None
    currency: str = ""
    as_of: Optional[datetime] = None
    provider: str = ""
    stale: bool = False
    issues: list[str] = Field(default_factory=list)


class Fundamentals(BaseModel):
    """Raw fundamentals as fractions (0.12 = 12%); ``fundamentals.py`` turns them into display metrics."""

    symbol: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    currency: str = ""
    market_cap: Optional[float] = None
    pe: Optional[float] = None
    pb: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    free_cash_flow: Optional[float] = None
    total_revenue: Optional[float] = None
    ebit: Optional[float] = None
    total_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    insider_holding: Optional[float] = None
    pledged_pct: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    provider: str = ""
    issues: list[str] = Field(default_factory=list)


class MarketDataProvider(Protocol):
    """Adapter interface. Implementations raise ``ProviderDataError`` for unknown symbols."""

    name: str

    def get_quote(self, symbol: str) -> dict[str, Any]: ...

    def get_history(self, symbol: str, period: str = "1y") -> pd.DataFrame: ...

    def get_fundamentals(self, symbol: str) -> dict[str, Any]: ...


# --- yfinance adapter -----------------------------------------------------------------------------

def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if is_number(number) else None


def _statement_value(frame: Any, *labels: str) -> float | None:
    """Most recent value of the first matching row in a yfinance financial statement."""
    if frame is None or getattr(frame, "empty", True):
        return None
    for label in labels:
        if label in frame.index:
            row = frame.loc[label].dropna()
            if not row.empty:
                return _num(row.iloc[0])
    return None


class YFinanceProvider:
    """Free Yahoo Finance data via ``yfinance`` (delayed quotes; fundamentals may be incomplete for NSE/BSE)."""

    name = "yfinance"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    @staticmethod
    def _ticker(symbol: str) -> Any:
        import yfinance as yf

        return yf.Ticker(symbol)

    def get_history(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        frame = self._ticker(symbol).history(period=period, auto_adjust=True, timeout=self.timeout)
        if frame is None or frame.empty:
            raise ProviderDataError(f"no price history for {symbol}")
        frame = frame.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if frame.index.tz is not None:
            frame.index = frame.index.tz_convert("UTC").tz_localize(None)
        return frame

    def get_quote(self, symbol: str) -> dict[str, Any]:
        frame = self.get_history(symbol, period="5d")
        last = frame.iloc[-1]
        previous = frame["close"].iloc[-2] if len(frame) > 1 else None
        currency = ""
        try:
            currency = str(self._ticker(symbol).fast_info.get("currency") or "")
        except Exception:  # noqa: BLE001 - currency is cosmetic
            pass
        as_of = frame.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
        return {"symbol": symbol, "price": _num(last["close"]), "previous_close": _num(previous),
                "currency": currency, "as_of": as_of}

    def get_fundamentals(self, symbol: str) -> dict[str, Any]:
        ticker = self._ticker(symbol)
        info = ticker.info or {}
        if not info or not (info.get("longName") or info.get("shortName") or info.get("marketCap")):
            raise ProviderDataError(f"no fundamentals for {symbol}")
        dte = _num(info.get("debtToEquity"))
        values: dict[str, Any] = {
            "symbol": symbol,
            "name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector") or "",
            "industry": info.get("industry") or "",
            "currency": info.get("currency") or "",
            "market_cap": _num(info.get("marketCap")),
            "pe": _num(info.get("trailingPE")),
            "pb": _num(info.get("priceToBook")),
            "roe": _num(info.get("returnOnEquity")),
            # Yahoo reports debt/equity as a percentage (45.3 = 0.453x).
            "debt_to_equity": dte / 100 if dte is not None else None,
            "revenue_growth": _num(info.get("revenueGrowth")),
            "earnings_growth": _num(info.get("earningsGrowth")),
            "profit_margin": _num(info.get("profitMargins")),
            "operating_margin": _num(info.get("operatingMargins")),
            "free_cash_flow": _num(info.get("freeCashflow")),
            "total_revenue": _num(info.get("totalRevenue")),
            "ebit": _num(info.get("ebit")),
            # Insider holding is the closest Yahoo field to Indian "promoter holding"; pledging is not available.
            "insider_holding": _num(info.get("heldPercentInsiders")),
            "beta": _num(info.get("beta")),
            "dividend_yield": _num(info.get("dividendYield")),
        }
        if values["dividend_yield"] is not None and values["dividend_yield"] > 1:
            values["dividend_yield"] /= 100  # newer yfinance returns a percentage
        try:  # ROCE inputs; best effort (extra requests, often missing)
            income, balance = ticker.income_stmt, ticker.balance_sheet
            values["ebit"] = values["ebit"] or _statement_value(income, "EBIT", "Operating Income")
            values["total_assets"] = _statement_value(balance, "Total Assets")
            values["current_liabilities"] = _statement_value(balance, "Current Liabilities")
        except Exception:  # noqa: BLE001
            logger.debug("statements unavailable for %s", symbol, exc_info=True)
        return values


PROVIDER_FACTORIES: dict[str, Callable[[StockFinderSettings], MarketDataProvider]] = {
    "yfinance": lambda s: YFinanceProvider(timeout=s.data_timeout),
}


# --- universe -------------------------------------------------------------------------------------

class Universe:
    """Sector -> symbols map from ``config/stock_finder_universe.yaml`` for one market."""

    def __init__(self, market: str, data: dict[str, Any]) -> None:
        section = data.get(market) or {}
        self.market = market
        self.indices: list[dict[str, str]] = list(section.get("indices") or [])
        self.benchmark: str = section.get("benchmark") or (self.indices[0]["symbol"] if self.indices else "")
        self.sectors: dict[str, list[dict[str, str]]] = {
            sector: [dict(entry) for entry in entries or []] for sector, entries in (section.get("sectors") or {}).items()
        }
        self._by_symbol = {e["symbol"].upper(): (sector, e) for sector, es in self.sectors.items() for e in es}

    @classmethod
    def load(cls, path: Path, market: str) -> "Universe":
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            logger.warning("Stock Finder universe file %s not found; screening is limited", path)
            data = {}
        return cls(market, data)

    def sector_names(self) -> list[str]:
        return list(self.sectors)

    def symbols(self, sector: str) -> list[str]:
        return [e["symbol"] for e in self.sectors.get(sector, [])]

    def sector_of(self, symbol: str) -> str | None:
        hit = self._by_symbol.get(symbol.upper())
        return hit[0] if hit else None

    def name_of(self, symbol: str) -> str | None:
        hit = self._by_symbol.get(symbol.upper())
        return hit[1].get("name") if hit else None

    def match_sector(self, text: str) -> str | None:
        """Map a free-text sector (from an LLM or a provider) to a universe sector name."""
        if not text:
            return None
        needle = text.lower().replace("&", "and")
        aliases = {
            "technology": "Information Technology", "it": "Information Technology", "software": "Information Technology",
            "bank": "Financials", "banking": "Financials", "financial services": "Financials", "nbfc": "Financials",
            "insurance": "Financials", "oil": "Energy", "gas": "Energy", "power": "Power & Utilities",
            "utilities": "Power & Utilities" if self.market == "IN" else "Utilities", "fmcg": "Consumer Staples",
            "consumer defensive": "Consumer Staples", "consumer cyclical": "Consumer Discretionary",
            "retail": "Consumer Discretionary", "auto": "Automobiles", "automobile": "Automobiles",
            "pharma": "Healthcare", "pharmaceuticals": "Healthcare", "health care": "Healthcare",
            "capital goods": "Industrials & Capital Goods", "infrastructure": "Industrials & Capital Goods",
            "defence": "Industrials & Capital Goods", "defense": "Industrials", "metals": "Materials & Metals",
            "basic materials": "Materials & Metals" if self.market == "IN" else "Materials", "cement": "Materials & Metals",
            "communication services": "Telecom" if self.market == "IN" else "Communication Services",
            "telecommunications": "Telecom", "realty": "Real Estate",
        }
        for sector in self.sectors:
            if sector.lower().replace("&", "and") == needle:
                return sector
        alias = aliases.get(needle)
        if alias in self.sectors:
            return alias
        for sector in self.sectors:
            words = [w for w in sector.lower().replace("&", "and").split() if w != "and"]
            if needle in sector.lower() or any(w in needle for w in words if len(w) > 3):
                return sector
        tokens = set(re.findall(r"[a-z]+", needle))
        for key, alias in aliases.items():
            if alias in self.sectors and (key in tokens or (" " in key and key in needle)):
                return alias
        return None

    def resolve(self, text: str) -> str | None:
        """Find a universe symbol by ticker (with or without suffix) or company name."""
        raw = text.strip().upper()
        if raw in self._by_symbol:
            return raw
        for symbol in self._by_symbol:
            if symbol.split(".")[0] == raw:
                return symbol
        lowered = text.strip().lower()
        if len(lowered) >= 3:
            for symbol, (_, entry) in self._by_symbol.items():
                if lowered in entry.get("name", "").lower():
                    return symbol
        return None


def normalize_symbol(raw: str, settings: StockFinderSettings, universe: Universe | None = None) -> str:
    """Upper-case ticker with the market's exchange suffix (``TCS`` -> ``TCS.NS`` for IN)."""
    text = raw.strip()
    if universe is not None:
        hit = universe.resolve(text)
        if hit:
            return hit
    symbol = text.upper().replace(" ", "")
    if not symbol or symbol.startswith("^"):
        return symbol
    if settings.market == "IN" and "." not in symbol and settings.exchange_suffix:
        symbol += settings.exchange_suffix
    return symbol


# --- service --------------------------------------------------------------------------------------

def run_parallel(fn: Callable[[str], T], items: Iterable[str], max_workers: int = 4) -> dict[str, T | Exception]:
    """``fn`` over ``items`` in threads (keeping the trace context); exceptions are returned, not raised."""
    items = list(dict.fromkeys(items))
    results: dict[str, T | Exception] = {}
    if not items:
        return results

    def wrapped(item: str) -> T | Exception:
        try:
            return fn(item)
        except Exception as exc:  # noqa: BLE001 - reported per item
            return exc

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(items)))) as pool:
        futures = {item: pool.submit(contextvars.copy_context().run, wrapped, item) for item in items}
        for item, future in futures.items():
            results[item] = future.result()
    return results


class MarketDataService:
    """Cached, validated, fault-tolerant market data across one or more providers."""

    def __init__(
        self,
        settings: StockFinderSettings,
        providers: list[MarketDataProvider] | None = None,
        *,
        cache: TTLCache | None = None,
        universe: Universe | None = None,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.settings = settings
        if providers is None:
            providers = [PROVIDER_FACTORIES[name](settings) for name in settings.market_data_providers
                         if name in PROVIDER_FACTORIES]
            unknown = [n for n in settings.market_data_providers if n not in PROVIDER_FACTORIES]
            if unknown:
                logger.warning("Unknown market data providers ignored: %s", ", ".join(unknown))
        if not providers:
            raise ValueError("no usable market data provider configured")
        self.providers = providers
        self.cache = cache or TTLCache()
        self.universe = universe or Universe.load(settings.universe_path, settings.market)
        self._sleep = sleep
        self._now = now
        self.breakers = {
            p.name: CircuitBreaker(f"market:{p.name}", settings.circuit_failure_threshold, settings.circuit_cooldown_seconds)
            for p in providers
        }

    # --- core fetch with cache + fallback -----------------------------------------------------

    def _fetch(self, kind: str, key: str, ttl: float, call: Callable[[MarketDataProvider], T]) -> tuple[T, str]:
        cache_key = f"{kind}:{key}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            log_data_fetch(provider=cached[1], kind=kind, key=key, latency_ms=0, cache_hit=True)
            return cached
        errors: list[str] = []
        for provider in self.providers:
            started = time.monotonic()
            try:
                value = call_with_retry(
                    lambda p=provider: call(p), breaker=self.breakers[provider.name], is_transient=is_transient,
                    max_retries=self.settings.max_retries, sleep=self._sleep, label=f"{provider.name}:{kind}",
                )
            except Exception as exc:  # noqa: BLE001 - includes CircuitOpenError; try the next provider
                latency = int((time.monotonic() - started) * 1000)
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                log_data_fetch(provider=provider.name, kind=kind, key=key, latency_ms=latency, cache_hit=False,
                               ok=False, error=f"{type(exc).__name__}: {exc}")
                continue
            latency = int((time.monotonic() - started) * 1000)
            log_data_fetch(provider=provider.name, kind=kind, key=key, latency_ms=latency, cache_hit=False)
            self.cache.set(cache_key, (value, provider.name), ttl)
            return value, provider.name
        raise DataUnavailableError(f"{kind} unavailable for {key}: " + "; ".join(errors))

    # --- public API ---------------------------------------------------------------------------

    def normalize(self, raw: str) -> str:
        return normalize_symbol(raw, self.settings, self.universe)

    def get_quote(self, symbol: str) -> Quote:
        raw, provider = self._fetch("quote", symbol, PRICE_TTL, lambda p: p.get_quote(symbol))
        price, previous = _num(raw.get("price")), _num(raw.get("previous_close"))
        as_of = raw.get("as_of")
        issues, stale = check_price(price, as_of, now=self._now())
        change = round((price / previous - 1) * 100, 2) if price and previous else None
        return Quote(symbol=symbol, price=round(price, 2) if price and price > 0 else None,
                     previous_close=round(previous, 2) if previous else None, change_pct=change,
                     currency=raw.get("currency") or self.settings.currency, as_of=as_of, provider=provider,
                     stale=stale, issues=issues)

    def get_history(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        frame, _ = self._fetch("history", f"{symbol}:{period}", HISTORY_TTL, lambda p: p.get_history(symbol, period))
        return frame

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        raw, provider = self._fetch("fundamentals", symbol, FUNDAMENTALS_TTL, lambda p: p.get_fundamentals(symbol))
        clean, issues = sanitize_fundamentals(raw)
        known = {k: v for k, v in clean.items() if k in Fundamentals.model_fields}
        known.setdefault("symbol", symbol)
        if not known.get("name"):
            known["name"] = self.universe.name_of(symbol) or symbol
        known.update(provider=provider, issues=issues)
        return Fundamentals(**known)

    def get_sector_peers(self, symbol: str, sector_hint: str | None = None) -> list[str]:
        """Universe peers in the same sector (excluding ``symbol``), at most ``SF_MAX_PEERS``."""
        sector = self.universe.sector_of(symbol) or self.universe.match_sector(sector_hint or "")
        if not sector:
            return []
        return [s for s in self.universe.symbols(sector) if s.upper() != symbol.upper()][: self.settings.max_peers]

    def get_index_snapshot(self) -> list[IndexQuote]:
        """Level, day change and 1-month return for each configured index; failures are flagged, not raised."""

        def one(entry: dict[str, str]) -> IndexQuote:
            symbol = entry["symbol"]
            try:
                frame = self.get_history(symbol, period="3mo")
            except DataUnavailableError:
                return IndexQuote(symbol=symbol, name=entry.get("name", symbol), stale=True)
            close = frame["close"]
            as_of = frame.index[-1].to_pydatetime().replace(tzinfo=timezone.utc)
            _, stale = check_price(float(close.iloc[-1]), as_of, now=self._now())
            change = (close.iloc[-1] / close.iloc[-2] - 1) * 100 if len(close) > 1 else None
            past = close[close.index <= close.index[-1] - pd.Timedelta(days=30)]
            ret_1m = (close.iloc[-1] / past.iloc[-1] - 1) * 100 if not past.empty else None
            return IndexQuote(symbol=symbol, name=entry.get("name", symbol), price=round(float(close.iloc[-1]), 2),
                              change_pct=round(float(change), 2) if change is not None else None,
                              return_1m_pct=round(float(ret_1m), 2) if ret_1m is not None else None,
                              as_of=as_of.isoformat(), stale=stale)

        by_symbol = {e["symbol"]: e for e in self.universe.indices}
        results = run_parallel(lambda s: one(by_symbol[s]), list(by_symbol), max_workers=4)
        return [r if isinstance(r, IndexQuote) else IndexQuote(symbol=s, name=by_symbol[s].get("name", s), stale=True)
                for s, r in results.items()]

    def get_fundamentals_many(self, symbols: Iterable[str]) -> dict[str, Fundamentals]:
        """Fundamentals for several symbols in parallel; failures are skipped."""
        results = run_parallel(self.get_fundamentals, symbols, max_workers=4)
        return {s: f for s, f in results.items() if isinstance(f, Fundamentals)}

    def breaker_states(self) -> dict[str, str]:
        return {name: breaker.state for name, breaker in self.breakers.items()}
