from __future__ import annotations

from typing import Any, Awaitable, Callable

NodeFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
