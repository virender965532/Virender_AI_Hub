"""Thread-safe in-memory TTL cache shared by the market-data and news services.

TTLs follow how fast each kind of data changes: prices for minutes, price
history for an hour, fundamentals for a day, news for 15 minutes.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Callable

PRICE_TTL = 5 * 60
HISTORY_TTL = 60 * 60
FUNDAMENTALS_TTL = 24 * 60 * 60
NEWS_TTL = 15 * 60

_MISSING = object()


class TTLCache:
    """LRU-bounded cache whose entries expire ``ttl`` seconds after being set."""

    def __init__(self, max_entries: int = 2000, clock: Callable[[], float] = time.monotonic) -> None:
        self.max_entries = max_entries
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            entry = self._data.get(key)
            if entry is None or entry[0] <= self._clock():
                if entry is not None:
                    del self._data[key]
                self.misses += 1
                return default
            self._data.move_to_end(key)
            self.hits += 1
            # Shared, not copied: callers must treat cached DataFrames / dicts as read-only.
            return entry[1]

    def set(self, key: str, value: Any, ttl: float) -> None:
        with self._lock:
            self._data[key] = (self._clock() + ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def contains(self, key: str) -> bool:
        return self.get(key, _MISSING) is not _MISSING

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"entries": len(self._data), "hits": self.hits, "misses": self.misses}
