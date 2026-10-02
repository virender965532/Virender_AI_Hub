"""In-process schedule for the job search (jobEngine.cron.run_once).

Started by ``app.py`` so the schedule ships with the app: the EC2 container
needs no crontab. Missed times are skipped, not caught up — if the app was
down, restarting, or a previous run was still busy at the scheduled time,
the next run is the next scheduled time.

Environment (defaults in brackets):
    JOB_CRON_ENABLED   [true]          false = no scheduled runs from this process
    JOB_CRON_SCHEDULE  [0 * * * *]     crontab expression (minute hour day month weekday); default hourly
    JOB_CRON_TIMEZONE  [UTC]           timezone the schedule is read in
"""

from __future__ import annotations

import asyncio
import logging
import os
import re

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .cron import run_once

logger = logging.getLogger("jobEngine.scheduler")

JOB_ID = "scheduled_job_search"
# Seconds after the scheduled time a delayed run may still start (scheduler
# thread busy, clock jitter). Anything later is dropped until the next time.
_MISFIRE_GRACE_SECONDS = 60

_scheduler: BackgroundScheduler | None = None


def _enabled() -> bool:
    return os.getenv("JOB_CRON_ENABLED", "true").strip().lower() in ("1", "true", "yes", "on")


_WEEKDAYS = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")


def _crontab_weekdays(field: str) -> str:
    """Standard crontab weekday numbers (0/7 = Sunday) as names.

    APScheduler 3 numbers weekdays from Monday = 0, so ``1-5`` would otherwise
    mean Tue-Sat. Step values (``*/2``) are left alone.
    """
    return re.sub(r"(?<!/)\b\d+\b", lambda m: _WEEKDAYS[int(m.group()) % 7], field)


def build_trigger() -> CronTrigger:
    schedule = os.getenv("JOB_CRON_SCHEDULE", "").strip() or "0 * * * *"
    timezone = os.getenv("JOB_CRON_TIMEZONE", "").strip() or "UTC"
    fields = schedule.split()
    if len(fields) == 5:
        fields[4] = _crontab_weekdays(fields[4])
    return CronTrigger.from_crontab(" ".join(fields), timezone=timezone)


def _run_job() -> None:
    logger.info("Scheduled job search starting")
    try:
        code = asyncio.run(run_once())
        logger.info("Scheduled job search finished (exit code %s)", code)
    except Exception:  # noqa: BLE001  (run_once reports its own failures; never kill the scheduler)
        logger.exception("Scheduled job search raised")


def start_scheduler() -> BackgroundScheduler | None:
    """Start the background schedule once per process; no-op when disabled."""
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    if not _enabled():
        logger.info("Scheduled job search disabled (JOB_CRON_ENABLED)")
        return None

    trigger = build_trigger()
    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        _run_job,
        trigger,
        id=JOB_ID,
        replace_existing=True,
        max_instances=1,  # a run still going at the next time -> that time is skipped
        coalesce=True,  # several missed times collapse into one decision, never a burst
        misfire_grace_time=_MISFIRE_GRACE_SECONDS,
    )
    scheduler.start()
    _scheduler = scheduler
    logger.info("Scheduled job search: %s, next run %s", trigger, scheduler.get_job(JOB_ID).next_run_time)
    return scheduler
