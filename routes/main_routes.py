from __future__ import annotations

import logging
from flask import Blueprint, jsonify, render_template, request, Response, send_file

from jobApplyPro import run_recruitment_pipeline
from jobApplyPro.config import RESUME_FILENAME
from jobSearch.graph import run_job_search_workflow
from jobSearch.nodes.fetch_jobs_node import (
    get_naukri_search_defaults,
    parse_ctc_filters,
)
from jobSearch.state import initial_workflow_state
from jobSearch.utils.job_payload import job_record_to_dict
from jobSearch.utils.scrape_progress import create_progress, finish_progress, get_progress
from services.naukri_service import run_job_detail, run_login_and_fetch_jobs

# 👉 NEW IMPORTS
from interviewSimulator.graph import run_interview_graph
from interviewSimulator.memory import load_state, save_state
from interviewSimulator.tts import get_tts_options, synthesize_speech
from pageIndex.basicPageIndex import DOC_ID, PDF_PATH, ask_pageindex_question
from simpleRAG.simpleRAG import PDF_PATH as SIMPLE_RAG_PDF_PATH, ask_simple_rag_question
from dynamicRAG.dynamicRAG import (
    SUPPORTED_EXTENSIONS,
    ask_dynamic_rag_question,
    get_document_preview_text,
    get_uploaded_file,
    get_uploaded_file_info,
    save_uploaded_file,
)
from enterpriseRAG.config.settings import SUPPORTED_EXTENSIONS as ENTERPRISE_RAG_EXTENSIONS
from enterpriseRAG.enterpriseRAG import (
    ask_enterprise_rag_question,
    get_document_preview_text as enterprise_preview_text,
    get_supported_roles,
    get_uploaded_file as enterprise_get_uploaded_file,
    get_uploaded_file_info as enterprise_get_uploaded_file_info,
    save_uploaded_file as enterprise_save_uploaded_file,
)

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


@main_bp.route("/api/interview/tts/options", methods=["GET"])
def interview_tts_options():
    """Models and voice ids for the interview TTS UI."""
    return jsonify(get_tts_options())


@main_bp.route("/api/interview/tts", methods=["POST"])
def interview_tts():
    """Synthesize speech for interview question text (requires HF_TOKEN)."""
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"ok": False, "error": "text is required"}), 400

    model = (payload.get("model") or "").strip() or None
    voice = payload.get("voice")
    if voice is not None and not isinstance(voice, str):
        voice = str(voice)

    result = synthesize_speech(text, model=model, voice=voice)
    if not result.get("ok"):
        return jsonify(
            {
                "ok": False,
                "error": result.get("error")
                or "TTS failed. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in .env and restart the server.",
            }
        ), 503

    audio = result["audio"]
    mime = result["mime"]
    return Response(
        audio,
        mimetype=mime,
        headers={
            "Content-Length": str(len(audio)),
            "Cache-Control": "no-store",
        },
    )


# ================= JOB SEARCH =================

# def _parse_headless_param(val: object, *, default: bool = True) -> bool:
#     if val is None:
#         return default
#     if isinstance(val, bool):
#         return val
#     s = str(val).strip().lower()
#     return s not in ("0", "false", "no")

# @main_bp.route("/job-search")
# def job_search():
#     return render_template("job_search.html")

# @main_bp.route("/job-detail")
# def job_detail_page():
#     url = (request.args.get("url") or "").strip()
#     if not url:
#         return jsonify({"ok": False, "error": "Missing url query parameter."}), 400

#     headless = _parse_headless_param(request.args.get("headless"), default=True)

#     try:
#         detail = run_job_detail(url=url, headless=headless)
#         return render_template("job_detail.html", detail=detail)

#     except Exception as e:
#         logger.exception("Job detail scrape failed")
#         return jsonify({"ok": False, "error": str(e)}), 500


# @main_bp.route("/api/jobs/naukri", methods=["POST"])
# def api_naukri_jobs():
#     payload = request.get_json(silent=True) or {}

#     headless = _parse_headless_param(payload.get("headless"), default=True)

#     try:
#         jobs = run_login_and_fetch_jobs(headless=headless)
#         return jsonify({"ok": True, "jobs": jobs})

#     except Exception as e:
#         logger.exception("Naukri automation failed")
#         return jsonify({"ok": False, "error": str(e)}), 500


# ===================JobSearch with playwriter ==================
def _parse_headless_param(val: object, *, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s not in ("0", "false", "no")

@main_bp.route("/job-search")
def job_search():
    return render_template(
        "job_search.html",
        naukri_defaults=get_naukri_search_defaults(),
    )


@main_bp.route("/job-apply")
def job_apply():
    return render_template("job_apply.html", resume_file=RESUME_FILENAME)


@main_bp.route("/api/job-apply/submit", methods=["POST"])
async def api_job_apply_submit():
    payload = request.get_json(silent=True) or {}
    job_description = (payload.get("job_description") or "").strip()
    hiring_email = (payload.get("hiring_manager_email") or "").strip()

    if not job_description:
        return jsonify({"ok": False, "error": "Job description is required."}), 400
    if not hiring_email:
        return jsonify({"ok": False, "error": "Hiring manager email is required."}), 400
    if "@" not in hiring_email or "." not in hiring_email.split("@")[-1]:
        return jsonify({"ok": False, "error": "Enter a valid hiring manager email."}), 400

    try:
        result = await run_recruitment_pipeline(
            job_requirement_text=job_description,
            hiring_manager_email=hiring_email,
        )
        if not result:
            return jsonify({"ok": False, "error": "Pipeline returned no result."}), 500
        return jsonify({"ok": True, **result}), 200
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Job apply pipeline failed")
        return jsonify({"ok": False, "error": str(e)}), 500

@main_bp.route("/job-detail")
def job_detail_page():
    return jsonify({"ok": True, "message": "Job detail page"}), 200
    

@main_bp.route("/api/jobs/naukri", methods=["POST"])
def api_naukri_jobs():
    """Start scrape in a background thread; client polls /progress/<id> for updates + results."""
    import asyncio
    import threading

    progress_id = None
    try:
        payload = request.get_json(silent=True) or {}
        defaults = get_naukri_search_defaults()
        enrich = payload.get("enrich_jd")

        keyword = (payload.get("keyword") or "").strip() or defaults["keyword"]
        job_age = str(payload.get("job_age") or defaults["job_age"]).strip() or defaults[
            "job_age"
        ]
        if "ctc_filters" in payload:
            ctc_filters = parse_ctc_filters(payload.get("ctc_filters"))
        else:
            ctc_filters = list(defaults["ctc_filters"])

        try:
            if "no_of_jobs" in payload and payload.get("no_of_jobs") is not None:
                no_of_jobs = int(payload.get("no_of_jobs"))
            else:
                no_of_jobs = int(defaults["no_of_jobs"])
        except (TypeError, ValueError):
            no_of_jobs = int(defaults["no_of_jobs"])
        no_of_jobs = max(1, no_of_jobs)

        try:
            if "max_pages" in payload and payload.get("max_pages") is not None:
                max_pages = int(payload.get("max_pages"))
            else:
                max_pages = int(defaults["max_pages"])
        except (TypeError, ValueError):
            max_pages = int(defaults["max_pages"])
        max_pages = max(1, max_pages)

        try:
            if payload.get("relevance_min_pct") is not None:
                relevance_min_pct = float(payload.get("relevance_min_pct"))
            else:
                relevance_min_pct = float(defaults["relevance_min_pct"])
        except (TypeError, ValueError):
            relevance_min_pct = float(defaults["relevance_min_pct"])

        client_progress_id = (payload.get("progress_id") or "").strip() or None
        progress_id = create_progress(
            target=no_of_jobs,
            keyword=keyword,
            run_id=client_progress_id,
        )

        initial = initial_workflow_state()
        initial["job_keyword"] = keyword
        initial["job_age"] = job_age
        initial["ctc_filters"] = ctc_filters
        initial["no_of_jobs"] = no_of_jobs
        initial["max_pages"] = max_pages
        initial["relevance_min_pct"] = relevance_min_pct
        initial["progress_id"] = progress_id
        if enrich is not None:
            initial["enrich_jd"] = bool(enrich)

        def _run_scrape() -> None:
            try:
                result = asyncio.run(run_job_search_workflow(initial_state=initial))
                jobs = result.get("jobs") or []
                workflow_errors = list(result.get("errors") or [])
                display_ok = bool(result.get("display_complete"))
                used_relevance = result.get("relevance_min_pct")
                if used_relevance is None:
                    used_relevance = relevance_min_pct

                failed = (
                    not result.get("login_complete")
                    or not result.get("fetch_complete")
                    or bool(workflow_errors and not jobs)
                )
                err_msg = None
                if failed and workflow_errors:
                    err_msg = workflow_errors[-1]
                elif failed and not jobs:
                    err_msg = "Job scrape did not complete successfully."

                finish_progress(
                    progress_id,
                    found=len(jobs),
                    jobs=[job_record_to_dict(j) for j in jobs],
                    errors=workflow_errors,
                    relevance_min_pct=used_relevance,
                    display_complete=display_ok,
                    error=err_msg if not jobs and err_msg else None,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Background Naukri scrape failed")
                finish_progress(progress_id, error=str(exc))

        threading.Thread(
            target=_run_scrape,
            name=f"naukri-scrape-{progress_id[:8]}",
            daemon=True,
        ).start()

        return jsonify(
            {
                "ok": True,
                "started": True,
                "message": "Naukri scrape started",
                "progress_id": progress_id,
                "target": no_of_jobs,
            }
        ), 202
    except Exception as e:
        logger.exception("Failed to start Naukri job search")
        if progress_id:
            finish_progress(progress_id, error=str(e))
        return jsonify({"ok": False, "error": str(e)}), 500


@main_bp.route("/api/jobs/naukri/progress/<progress_id>", methods=["GET"])
def api_naukri_jobs_progress(progress_id: str):
    row = get_progress((progress_id or "").strip())
    if not row:
        return jsonify({"ok": False, "error": "Progress not found."}), 404
    return jsonify({"ok": True, **row}), 200


# ===================Simple RAG Search ==================

@main_bp.route("/basic-rag-search")
def basic_rag_search():
    return render_template(
        "simple_rag_search.html",
        pdf_name=SIMPLE_RAG_PDF_PATH.name,
    )


@main_bp.route("/api/simple-rag-search/pdf")
def simple_rag_pdf():
    if not SIMPLE_RAG_PDF_PATH.exists():
        return jsonify({"ok": False, "error": "PDF file not found on server."}), 404
    return send_file(
        SIMPLE_RAG_PDF_PATH,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=SIMPLE_RAG_PDF_PATH.name,
    )


@main_bp.route("/api/simple-rag-search/ask", methods=["POST"])
def simple_rag_ask():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"ok": False, "error": "Question is required."}), 400

    try:
        result = ask_simple_rag_question(query)
        return jsonify({"ok": True, **result}), 200
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Simple RAG question failed")
        return jsonify({"ok": False, "error": str(e)}), 500


# ===================Dynamic RAG Search ==================

_DYNAMIC_RAG_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain; charset=utf-8",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
}


@main_bp.route("/dynamic-rag-search")
def dynamic_rag_search():
    uploaded = get_uploaded_file_info()
    return render_template(
        "dynamic_rag_search.html",
        uploaded_file=uploaded,
        supported_extensions=sorted(SUPPORTED_EXTENSIONS),
    )


@main_bp.route("/api/dynamic-rag-search/upload", methods=["POST"])
def dynamic_rag_upload():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"ok": False, "error": "No file was uploaded."}), 400

    try:
        saved_path = save_uploaded_file(uploaded_file)
        return jsonify(
            {
                "ok": True,
                "file": {
                    "name": saved_path.name,
                    "extension": saved_path.suffix.lower(),
                },
            }
        ), 200
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Dynamic RAG file upload failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@main_bp.route("/api/dynamic-rag-search/document")
def dynamic_rag_document():
    uploaded = get_uploaded_file()
    if not uploaded:
        return jsonify({"ok": False, "error": "No document uploaded yet."}), 404

    mimetype = _DYNAMIC_RAG_MIME_TYPES.get(
        uploaded.suffix.lower(), "application/octet-stream"
    )
    return send_file(
        uploaded,
        mimetype=mimetype,
        as_attachment=False,
        download_name=uploaded.name,
    )


@main_bp.route("/api/dynamic-rag-search/preview")
def dynamic_rag_preview():
    try:
        preview_text = get_document_preview_text()
        uploaded = get_uploaded_file()
        return jsonify(
            {
                "ok": True,
                "preview": preview_text,
                "file": {
                    "name": uploaded.name if uploaded else "",
                    "extension": uploaded.suffix.lower() if uploaded else "",
                },
            }
        ), 200
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    except Exception as e:
        logger.exception("Dynamic RAG preview failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@main_bp.route("/api/dynamic-rag-search/ask", methods=["POST"])
def dynamic_rag_ask():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"ok": False, "error": "Question is required."}), 400

    try:
        result = ask_dynamic_rag_question(query)
        return jsonify({"ok": True, **result}), 200
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    except Exception as e:
        logger.exception("Dynamic RAG question failed")
        return jsonify({"ok": False, "error": str(e)}), 500


# ===================Page Index RAG Search ==================
@main_bp.route("/page-index-rag-search")
def page_index_rag_search():
    return render_template(
        "page_index_rag_search.html",
        doc_id=DOC_ID,
    )


@main_bp.route("/api/page-index-rag-search/pdf")
def page_index_rag_pdf():
    if not PDF_PATH.exists():
        return jsonify({"ok": False, "error": "PDF file not found on server."}), 404
    return send_file(
        PDF_PATH,
        mimetype="application/pdf",
        as_attachment=False,
        download_name=PDF_PATH.name,
    )


@main_bp.route("/api/page-index-rag-search/ask", methods=["POST"])
def page_index_rag_ask():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    if not query:
        return jsonify({"ok": False, "error": "Question is required."}), 400

    try:
        result = ask_pageindex_question(query)
        return jsonify({"ok": True, **result}), 200
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Page Index RAG question failed")
        return jsonify({"ok": False, "error": str(e)}), 500

# ================= ENTERPRISE MULTI-AGENT RAG =================

_ENTERPRISE_RAG_MIME_TYPES = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
}


@main_bp.route("/enterprise-rag-search")
def enterprise_rag_search():
    return render_template(
        "enterprise_rag_search.html",
        roles=get_supported_roles(),
        supported_extensions=sorted(ENTERPRISE_RAG_EXTENSIONS),
    )


@main_bp.route("/api/enterprise-rag-search/upload-status")
def enterprise_rag_upload_status():
    info = enterprise_get_uploaded_file_info()
    if not info:
        return jsonify({"ok": True, "file": None}), 200
    return jsonify({"ok": True, "file": info}), 200


@main_bp.route("/api/enterprise-rag-search/upload", methods=["POST"])
def enterprise_rag_upload():
    uploaded_file = request.files.get("file")
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"ok": False, "error": "No file was uploaded."}), 400

    try:
        saved_path = enterprise_save_uploaded_file(uploaded_file)
        return jsonify(
            {
                "ok": True,
                "file": {
                    "name": saved_path.name,
                    "extension": saved_path.suffix.lower(),
                },
            }
        ), 200
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        logger.exception("Enterprise RAG file upload failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@main_bp.route("/api/enterprise-rag-search/document")
def enterprise_rag_document():
    uploaded = enterprise_get_uploaded_file()
    if not uploaded:
        return jsonify({"ok": False, "error": "No document uploaded yet."}), 404

    mimetype = _ENTERPRISE_RAG_MIME_TYPES.get(
        uploaded.suffix.lower(), "application/octet-stream"
    )
    return send_file(
        uploaded,
        mimetype=mimetype,
        as_attachment=False,
        download_name=uploaded.name,
    )


@main_bp.route("/api/enterprise-rag-search/preview")
def enterprise_rag_preview():
    try:
        preview_text = enterprise_preview_text()
        uploaded = enterprise_get_uploaded_file()
        return jsonify(
            {
                "ok": True,
                "preview": preview_text,
                "file": {
                    "name": uploaded.name if uploaded else "",
                    "extension": uploaded.suffix.lower() if uploaded else "",
                },
            }
        ), 200
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    except Exception as e:
        logger.exception("Enterprise RAG preview failed")
        return jsonify({"ok": False, "error": str(e)}), 500


@main_bp.route("/api/enterprise-rag-search/roles")
def enterprise_rag_roles():
    return jsonify({"ok": True, "roles": get_supported_roles()}), 200


@main_bp.route("/api/enterprise-rag-search/ask", methods=["POST"])
def enterprise_rag_ask():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    role = (payload.get("role") or "Enterprise Architect").strip()
    session_id = (payload.get("session_id") or "default").strip()

    if not query:
        return jsonify({"ok": False, "error": "Question is required."}), 400

    try:
        result = ask_enterprise_rag_question(
            query, role=role, session_id=session_id
        )
        return jsonify({"ok": True, **result}), 200
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"ok": False, "error": str(e)}), 404
    except Exception as e:
        logger.exception("Enterprise RAG question failed")
        return jsonify({"ok": False, "error": str(e)}), 500


# ================= STOCK MARKET PREDICTION =================




# ================= ASK VIRENDER =================


