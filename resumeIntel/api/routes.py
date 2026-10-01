"""Flask blueprint for Resume Intelligence.

Registered in ``app.py`` alongside the existing blueprints — no existing
route is touched. Self-contained: this blueprint owns both its page
(``/resume-intelligence``) and its API (``/api/resume-intel/analyze``).
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, render_template, request

from ..analyzer import ResumeGapAnalysisError, ResumeGapAnalyzer
from ..config.settings import get_settings
from ..parsing.resume_text import (
    ResumeTextError,
    extract_text_from_default_resume,
    extract_text_from_upload,
)

logger = logging.getLogger(__name__)

resume_intel_bp = Blueprint("resume_intel", __name__)


@resume_intel_bp.route("/resume-intelligence", methods=["GET"])
def resume_intelligence_ui():
    settings = get_settings()
    return render_template(
        "resume_intelligence.html",
        default_resume_filename=settings.default_resume_filename,
    )


@resume_intel_bp.route("/api/resume-intel/analyze", methods=["POST"])
def api_resume_intel_analyze():
    settings = get_settings()

    jd_text = (request.form.get("jd_text") or "").strip()
    if not jd_text:
        return jsonify({"ok": False, "error": "Job description is required."}), 400

    uploaded_file = request.files.get("resume_file")
    try:
        if uploaded_file and uploaded_file.filename:
            resume_text = extract_text_from_upload(uploaded_file, max_bytes=settings.max_upload_bytes)
            resume_source = uploaded_file.filename
        else:
            resume_text = extract_text_from_default_resume(settings.default_resume_path)
            resume_source = settings.default_resume_filename
    except ResumeTextError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    analyzer = ResumeGapAnalyzer(settings)
    try:
        analysis = analyzer.analyze(resume_text=resume_text, jd_text=jd_text)
    except ResumeGapAnalysisError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except Exception:  # noqa: BLE001
        logger.exception("Resume gap analysis failed unexpectedly")
        return jsonify({"ok": False, "error": "Analysis failed unexpectedly."}), 500

    return jsonify({"ok": True, "resume_source": resume_source, "analysis": analysis.model_dump()}), 200
