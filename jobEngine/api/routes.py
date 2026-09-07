"""Flask blueprint for the job engine (section 36).

Registered in ``app.py`` alongside the existing ``main_bp`` — no existing
route is touched. Background execution follows the same pattern already
used by ``routes/main_routes.py``'s ``/api/jobs/naukri`` (thread +
``asyncio.run`` + a poll endpoint), since this project has no Celery/Redis
worker to hand the long-running browser workflow off to (section 37).
"""

from __future__ import annotations

import asyncio
import logging
import threading

from flask import Blueprint, jsonify, request
from pydantic import ValidationError

from ..config.sources import load_job_sources
from ..observability.logging import new_run_id
from ..persistence.repository import get_run, list_runs
from ..services.job_engine_service import run_job_search, test_source
from .health import build_health_report
from .run_status import get_status, set_status
from .schemas import JobSearchRequest, SourceTestRequest

logger = logging.getLogger(__name__)

job_engine_bp = Blueprint("job_engine", __name__, url_prefix="/api/jobs")


@job_engine_bp.route("/search", methods=["POST"])
def api_job_search():
    payload = request.get_json(silent=True) or {}
    try:
        parsed = JobSearchRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"ok": False, "error": "invalid request", "details": exc.errors()}), 400

    run_id = new_run_id()
    set_status(run_id, "running")
    request_dict = parsed.model_dump(mode="json")

    def _run() -> None:
        try:
            result = asyncio.run(run_job_search(request_dict, run_id=run_id))
            set_status(run_id, result.get("status", "completed"))
        except Exception:  # noqa: BLE001
            logger.exception("Background job search failed for run_id=%s", run_id)
            set_status(run_id, "failed")

    threading.Thread(target=_run, name=f"job-engine-{run_id[-8:]}", daemon=True).start()

    return jsonify({"ok": True, "run_id": run_id, "status": "running"}), 202


@job_engine_bp.route("/search/<run_id>", methods=["GET"])
def api_job_search_status(run_id: str):
    status = get_status(run_id)
    persisted = get_run(run_id)

    if status is None and persisted is None:
        return jsonify({"ok": False, "error": "run not found"}), 404
    if persisted is not None:
        return jsonify({"ok": True, **persisted}), 200
    return jsonify({"ok": True, "run_id": run_id, "status": status or "unknown"}), 200


@job_engine_bp.route("", methods=["GET"])
def api_list_recent_runs():
    limit = request.args.get("limit", default=20, type=int) or 20
    return jsonify({"ok": True, "runs": list_runs(limit=limit)}), 200


@job_engine_bp.route("/<job_id>", methods=["GET"])
def api_get_job(job_id: str):
    for run_summary in list_runs(limit=100):
        run = get_run(run_summary["run_id"])
        if not run:
            continue
        for matched in run.get("jobs") or []:
            if matched.get("job", {}).get("job_id") == job_id:
                return jsonify({"ok": True, "job": matched, "run_id": run.get("run_id")}), 200
    return jsonify({"ok": False, "error": "job not found in recent runs"}), 404


@job_engine_bp.route("/sources", methods=["GET"])
def api_list_sources():
    try:
        sources = load_job_sources()
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500
    return jsonify(
        {
            "ok": True,
            "sources": [
                {
                    "id": s.id,
                    "name": s.name,
                    "enabled": s.enabled,
                    "adapter": s.adapter,
                    "requires_auth": s.requires_auth,
                    "max_jobs": s.max_jobs,
                }
                for s in sources
            ],
        }
    ), 200


@job_engine_bp.route("/sources/test", methods=["POST"])
def api_test_source():
    payload = request.get_json(silent=True) or {}
    try:
        parsed = SourceTestRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"ok": False, "error": "invalid request", "details": exc.errors()}), 400

    result = asyncio.run(test_source(parsed.source_id))
    status_code = 200 if result.get("ok") else 502
    return jsonify(result), status_code


@job_engine_bp.route("/health", methods=["GET"])
def api_job_engine_health():
    report = build_health_report()
    status_code = 200 if report.get("ok") else 503
    return jsonify({"ok": report.get("ok", False), "checks": report}), status_code
