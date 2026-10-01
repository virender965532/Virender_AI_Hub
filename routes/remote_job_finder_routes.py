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

from flask import Blueprint, jsonify, render_template, request
from pymongo.errors import PyMongoError

from jobEngine.config.sources import DEFAULT_UNCHECKED_SOURCES, load_job_sources
from jobEngine.sources.search_keywords import url_keywords
from jobEngine.persistence.mongo import MongoNotConfigured
from jobEngine.persistence.saved_jobs import (
    SAVED_JOB_STATUSES,
    clear_saved_jobs,
    delete_saved_job,
    list_saved_jobs,
    update_saved_job_status,
)

logger = logging.getLogger(__name__)

remote_job_finder_bp = Blueprint("remote_job_finder", __name__)


@remote_job_finder_bp.errorhandler(PyMongoError)
@remote_job_finder_bp.errorhandler(MongoNotConfigured)
def _database_unavailable(exc: Exception):
    logger.error("MongoDB unavailable: %s", exc)
    return jsonify({"ok": False, "error": "database unavailable"}), 503

# Pre-filled (editable) form values on the Remote Job Finder page. Whatever the
# user leaves or types in the fields is exactly what gets sent to /api/jobs/search.
# The Keywords default is read from the sources' search URLs (``keywords=``).
DEFAULT_SEARCH_FORM = {
    "min_score": 30,
    "max_results": 10,
}


@remote_job_finder_bp.route("/remote-job-finder")
def remote_job_finder_ui():
    """Global Remote Job Finder UI. JSON API lives on job_engine_bp (/api/jobs/*)."""
    try:
        sources = [
            {
                "id": s.id,
                "name": s.name,
                "enabled": s.enabled,
                "requires_auth": s.requires_auth,
                "checked": s.enabled and s.id not in DEFAULT_UNCHECKED_SOURCES,
            }
            for s in load_job_sources()
        ]
    except Exception:
        logger.exception("Failed to load job_sources.yaml for remote-job-finder UI")
        sources = []
    return render_template(
        "remote_job_finder.html", sources=sources, defaults={**DEFAULT_SEARCH_FORM, "keywords": _default_keywords()}
    )


def _default_keywords() -> str:
    """The ``keywords=`` value of the first enabled source search URL that has one."""
    try:
        for source in load_job_sources():
            if not source.enabled:
                continue
            for url in source.search_urls:
                keywords = url_keywords(url)
                if keywords:
                    return keywords
    except Exception:
        logger.exception("Failed to read default keywords from job_sources.yaml")
    return ""


@remote_job_finder_bp.route("/saved-jobs")
def saved_jobs_ui():
    """Every job recommended by past searches, with its full match analysis."""
    return render_template("saved_jobs.html", statuses=list(SAVED_JOB_STATUSES))


@remote_job_finder_bp.route("/api/saved-jobs", methods=["GET"])
def api_list_saved_jobs():
    jobs = list_saved_jobs()
    return jsonify({"ok": True, "count": len(jobs), "jobs": jobs}), 200


@remote_job_finder_bp.route("/api/saved-jobs", methods=["DELETE"])
def api_clear_saved_jobs():
    return jsonify({"ok": True, "removed": clear_saved_jobs()}), 200


@remote_job_finder_bp.route("/api/saved-jobs/<path:key>/status", methods=["POST"])
def api_update_saved_job_status(key: str):
    status = ((request.get_json(silent=True) or {}).get("status") or "").strip()
    try:
        job = update_saved_job_status(key, status)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    if job is None:
        return jsonify({"ok": False, "error": "saved job not found"}), 404
    return jsonify(
        {"ok": True, "saved_key": key, "status": job["status"], "applied_at": job.get("applied_at")}
    ), 200


@remote_job_finder_bp.route("/api/saved-jobs/<path:key>", methods=["DELETE"])
def api_delete_saved_job(key: str):
    if not delete_saved_job(key):
        return jsonify({"ok": False, "error": "saved job not found"}), 404
    return jsonify({"ok": True, "saved_key": key}), 200
