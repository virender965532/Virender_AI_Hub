"""Flask blueprint for the Global Remote Job Finder module.

New module, registered in ``app.py`` alongside ``main_bp`` — no existing
blueprint or route is touched. The UI here is a thin client of the existing,
unmodified ``job_engine_bp`` JSON API (``/api/jobs/search`` and
``/api/jobs/search/<run_id>``); it does not duplicate scraping or matching
logic, it only presents it differently (full job detail per match instead of
a compact card).
"""

from __future__ import annotations

import logging

from flask import Blueprint, render_template

from jobEngine.config.sources import load_job_sources

logger = logging.getLogger(__name__)

remote_job_finder_bp = Blueprint("remote_job_finder", __name__)


@remote_job_finder_bp.route("/remote-job-finder")
def remote_job_finder_ui():
    """Global Remote Job Finder UI. JSON API lives on job_engine_bp (/api/jobs/*)."""
    try:
        sources = [
            {"id": s.id, "name": s.name, "enabled": s.enabled, "requires_auth": s.requires_auth}
            for s in load_job_sources()
        ]
    except Exception:
        logger.exception("Failed to load job_sources.yaml for remote-job-finder UI")
        sources = []
    return render_template("remote_job_finder.html", sources=sources)
