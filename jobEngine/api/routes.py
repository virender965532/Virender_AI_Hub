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

from flask import Blueprint, Response, jsonify, request
from pydantic import ValidationError
from pymongo.errors import PyMongoError

from ..config.sources import load_job_sources
from ..observability.logging import new_run_id
from ..persistence.mongo import MongoNotConfigured
from ..persistence.repository import find_job_in_runs, get_run, list_runs
from ..persistence.sessions import storage_state_info
from ..services.job_engine_service import run_job_search, test_source
from ..sources.linkedin.remote_login import LoginNotFound, remote_login_manager
from .health import build_health_report
from .run_status import get_status, set_status
from .schemas import JobSearchRequest, SourceTestRequest

logger = logging.getLogger(__name__)

job_engine_bp = Blueprint("job_engine", __name__, url_prefix="/api/jobs")


@job_engine_bp.errorhandler(PyMongoError)
@job_engine_bp.errorhandler(MongoNotConfigured)
def _database_unavailable(exc: Exception):
    logger.error("MongoDB unavailable: %s", exc)
    return jsonify({"ok": False, "error": "database unavailable"}), 503


@job_engine_bp.route("/search", methods=["POST"])
def api_job_search():
    payload = request.get_json(silent=True) or {}
    try:
        parsed = JobSearchRequest.model_validate(payload)
    except ValidationError as exc:
        return jsonify({"ok": False, "error": "invalid request", "details": exc.errors(include_url=False, include_context=False)}), 400

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
    found = find_job_in_runs(job_id)
    if found is not None:
        matched, run_id = found
        return jsonify({"ok": True, "job": matched, "run_id": run_id}), 200
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
        return jsonify({"ok": False, "error": "invalid request", "details": exc.errors(include_url=False, include_context=False)}), 400

    result = asyncio.run(test_source(parsed.source_id))
    status_code = 200 if result.get("ok") else 502
    return jsonify(result), status_code


@job_engine_bp.route("/health", methods=["GET"])
def api_job_engine_health():
    report = build_health_report()
    status_code = 200 if report.get("ok") else 503
    return jsonify({"ok": report.get("ok", False), "checks": report}), status_code


# --- LinkedIn "Connect" popup: the candidate signs in through a server-side browser.


@job_engine_bp.errorhandler(LoginNotFound)
def _login_not_found(exc: LoginNotFound):
    return jsonify({"ok": False, "error": "sign-in not found or already finished"}), 404


@job_engine_bp.route("/linkedin/session", methods=["GET"])
def api_linkedin_session():
    info = storage_state_info("linkedin")
    return jsonify({"ok": True, "connected": info is not None, **(info or {})}), 200


@job_engine_bp.route("/linkedin/login", methods=["POST"])
def api_linkedin_login_start():
    try:
        status = remote_login_manager.start()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not start remote LinkedIn login")
        return jsonify({"ok": False, "error": f"could not start the browser: {exc}"}), 500
    return jsonify({"ok": status.get("state") == "waiting", **status}), 200


@job_engine_bp.route("/linkedin/login/<login_id>", methods=["GET"])
def api_linkedin_login_status(login_id: str):
    return jsonify({"ok": True, **remote_login_manager.status(login_id)}), 200


@job_engine_bp.route("/linkedin/login/<login_id>", methods=["DELETE"])
def api_linkedin_login_cancel(login_id: str):
    remote_login_manager.cancel(login_id)
    return jsonify({"ok": True}), 200


@job_engine_bp.route("/linkedin/login/<login_id>/frame", methods=["GET"])
def api_linkedin_login_frame(login_id: str):
    image = remote_login_manager.frame(login_id)
    return Response(image, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@job_engine_bp.route("/linkedin/login/<login_id>/input", methods=["POST"])
def api_linkedin_login_input(login_id: str):
    try:
        remote_login_manager.send_input(login_id, request.get_json(silent=True) or {})
    except (ValueError, KeyError, TypeError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    return jsonify({"ok": True}), 200
