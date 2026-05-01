from __future__ import annotations

import logging
from flask import Blueprint, jsonify, render_template, request

from services.naukri_service import run_job_detail, run_login_and_fetch_jobs

# 👉 NEW IMPORTS
from interviewSimulator.graph import run_interview_graph
from interviewSimulator.memory import load_state, save_state

logger = logging.getLogger(__name__)

main_bp = Blueprint("main", __name__)


# ================= HOME =================

@main_bp.route("/")
def home():
    return render_template("index.html")


# ================= INTERVIEW SIMULATOR =================

@main_bp.route("/interview")
def interview_ui():
    """Render Interview Simulator UI"""
    return render_template("interviewSimulator.html")


@main_bp.route("/api/interview/start", methods=["POST"])
def start_interview():
    payload = request.get_json(silent=True) or {}

    state = {
        "role": payload.get("role", "Backend Engineer"),
        "difficulty": payload.get("difficulty", "Mid-level"),
        "total_questions": int(payload.get("num_questions", 5)),
        "questions_asked": 0,
        "scores": [],
        "conversation": [],
        "current_question": "",
        "active": True
    }

    try:
        state = run_interview_graph(state, action="start")
        save_state(state)

        return jsonify({"ok": True, "state": state})

    except Exception as e:
        logger.exception("Interview start failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@main_bp.route("/api/interview/answer", methods=["POST"])
def answer_interview():
    payload = request.get_json(silent=True) or {}
    answer = payload.get("answer", "").strip()
    logger.info(f"Answer: {answer}")
    if not answer:
        return jsonify({"ok": False, "error": "Answer is required"}), 400

    try:
        state = load_state()
        logger.info(f"State: {state}")
        if not state:
            return jsonify({"ok": False, "error": "No active session"}), 400

        state["user_answer"] = answer
        logger.info(f"State before running graph: {state}")
        state = run_interview_graph(state, action="answer")
        logger.info(f"State after running graph: {state}")
        save_state(state)

        return jsonify({"ok": True, "state": state})

    except Exception as e:
        logger.exception("Interview answer failed")
        return jsonify({"ok": False, "error": str(e)}), 500


# ================= JOB SEARCH =================

def _parse_headless_param(val: object, *, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s not in ("0", "false", "no")

@main_bp.route("/job-search")
def job_search():
    return render_template("job_search.html")

@main_bp.route("/job-detail")
def job_detail_page():
    url = (request.args.get("url") or "").strip()
    if not url:
        return jsonify({"ok": False, "error": "Missing url query parameter."}), 400

    headless = _parse_headless_param(request.args.get("headless"), default=True)

    try:
        detail = run_job_detail(url=url, headless=headless)
        return render_template("job_detail.html", detail=detail)

    except Exception as e:
        logger.exception("Job detail scrape failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@main_bp.route("/api/jobs/naukri", methods=["POST"])
def api_naukri_jobs():
    payload = request.get_json(silent=True) or {}

    headless = _parse_headless_param(payload.get("headless"), default=True)

    try:
        jobs = run_login_and_fetch_jobs(headless=headless)
        return jsonify({"ok": True, "jobs": jobs})

    except Exception as e:
        logger.exception("Naukri automation failed")
        return jsonify({"ok": False, "error": str(e)}), 500


# ================= STOCK MARKET PREDICTION =================




# ================= ASK VIRENDER =================


