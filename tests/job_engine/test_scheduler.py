from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from jobEngine import scheduler


@pytest.fixture(autouse=True)
def fresh_scheduler(monkeypatch):
    monkeypatch.setattr(scheduler, "_scheduler", None)
    for name in ("JOB_CRON_ENABLED", "JOB_CRON_SCHEDULE", "JOB_CRON_TIMEZONE"):
        monkeypatch.delenv(name, raising=False)
    yield
    if scheduler._scheduler is not None:
        scheduler._scheduler.shutdown(wait=False)


def test_default_trigger_is_every_hour_on_the_hour_utc():
    trigger = scheduler.build_trigger()
    utc = ZoneInfo("UTC")
    fire = trigger.get_next_fire_time(None, datetime(2026, 10, 2, 10, 0, tzinfo=utc))
    assert fire == datetime(2026, 10, 2, 10, 0, tzinfo=utc)
    fires = []
    for _ in range(3):
        fire = trigger.get_next_fire_time(fire, fire)
        fires.append(fire.hour)
    assert fires == [11, 12, 13]


def test_hour_already_passed_waits_for_the_next_one():
    trigger = scheduler.build_trigger()
    utc = ZoneInfo("UTC")
    fire = trigger.get_next_fire_time(None, datetime(2026, 10, 2, 23, 30, tzinfo=utc))
    assert fire == datetime(2026, 10, 3, 0, 0, tzinfo=utc)


def test_schedule_and_timezone_come_from_env(monkeypatch):
    monkeypatch.setenv("JOB_CRON_SCHEDULE", "30 4 * * 1-5")
    monkeypatch.setenv("JOB_CRON_TIMEZONE", "UTC")
    utc = ZoneInfo("UTC")
    fire = scheduler.build_trigger().get_next_fire_time(None, datetime(2026, 10, 2, 5, 0, tzinfo=utc))  # Friday
    assert fire == datetime(2026, 10, 5, 4, 30, tzinfo=utc)  # Monday


def test_disabled_starts_nothing(monkeypatch):
    monkeypatch.setenv("JOB_CRON_ENABLED", "false")
    assert scheduler.start_scheduler() is None


def test_started_job_skips_missed_runs_and_never_overlaps():
    started = scheduler.start_scheduler()
    assert scheduler.start_scheduler() is started  # once per process
    job = started.get_job(scheduler.JOB_ID)
    assert job.misfire_grace_time == 60
    assert job.coalesce is True
    assert job.max_instances == 1
    assert job.next_run_time > datetime.now(ZoneInfo("UTC"))
