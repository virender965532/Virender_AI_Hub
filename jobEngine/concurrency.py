"""Shared process pool for the job engine's CPU-bound per-job work.

Scraping, MongoDB and LLM calls are I/O and stay on asyncio (blocking calls in
worker threads); only pure-Python CPU work — today the regex-heavy worldwide
classification in normalization — goes to worker processes, where it isn't
held back by the GIL.

The pool lives for the whole server process and is reused by every run: on
Windows each worker is spawned and re-imports the app, which takes seconds, so
``warm_cpu_pool`` is called when a run starts and the spawn overlaps scraping.
Small batches run inline (process IPC would cost more than it saves), and a
broken pool falls back to inline so a crashed worker never fails a run.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any, Callable, Sequence, TypeVar

from .config.settings import get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

_pool: ProcessPoolExecutor | None = None
_lock = threading.Lock()


def _warm_worker() -> None:
    """Import the modules the workers run, so the first real batch doesn't pay for it."""
    from .graph.nodes import normalization  # noqa: F401


def _run_chunk(fn: Callable[[T], R], chunk: Sequence[T]) -> list[R]:
    return [fn(item) for item in chunk]


def cpu_pool() -> ProcessPoolExecutor | None:
    """The shared pool, created on first use; None when ``JOB_CPU_WORKERS=0``."""
    global _pool
    workers = get_settings().cpu_workers
    if workers <= 0:
        return None
    with _lock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=workers)
            logger.info("CPU process pool started with %s workers", workers)
        return _pool


def _discard_pool(pool: ProcessPoolExecutor) -> None:
    global _pool
    with _lock:
        if _pool is pool:
            _pool = None
    pool.shutdown(wait=False, cancel_futures=True)


def warm_cpu_pool() -> None:
    """Start every worker in the background (fire and forget)."""
    pool = cpu_pool()
    if pool is None:
        return
    try:
        for _ in range(get_settings().cpu_workers):
            pool.submit(_warm_worker)
    except (BrokenProcessPool, RuntimeError):
        logger.warning("CPU process pool unusable; it will be recreated on next use")
        _discard_pool(pool)


async def map_cpu(fn: Callable[[T], R], items: Sequence[T]) -> list[R]:
    """``[fn(item) for item in items]``, in the process pool when the batch is big
    enough. ``fn`` and the items must be picklable (a module-level function or a
    ``functools.partial`` of one). Order is preserved."""
    items = list(items)
    settings = get_settings()
    pool = cpu_pool() if len(items) >= settings.cpu_parallel_min_jobs else None
    if pool is None:
        return _run_chunk(fn, items)

    # A few chunks per worker: balances uneven jobs without paying IPC per item.
    n_chunks = settings.cpu_workers * 4
    size = max(1, -(-len(items) // n_chunks))
    chunks = [items[i : i + size] for i in range(0, len(items), size)]
    loop = asyncio.get_running_loop()
    try:
        parts: list[list[Any]] = await asyncio.gather(
            *(loop.run_in_executor(pool, _run_chunk, fn, chunk) for chunk in chunks)
        )
    except (BrokenProcessPool, RuntimeError):
        logger.exception("CPU process pool failed; processing %s items inline", len(items))
        _discard_pool(pool)
        return _run_chunk(fn, items)
    return [result for part in parts for result in part]
