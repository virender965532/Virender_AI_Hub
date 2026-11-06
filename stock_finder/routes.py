"""Flask blueprint for Stock Finder (JSON + Server-Sent Events).

The hub has no authentication layer, so these routes are open to anyone who can
reach the server, like the other hub modules (see README "Security"). They are
read/research-only: nothing here can place a trade.

Long runs (find-opportunities, portfolio review) execute as background jobs
and stream ``text/event-stream`` events: ``session``, ``job``, ``progress``,
``route``, ``worker_result``, ``portfolio_snapshot``, ``final`` (the full
report) or ``error``. Send ``"stream": false`` to get ``202 {job_id}`` instead
and poll ``GET /api/jobs/<job_id>``. Every JSON response carries the disclaimer.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from flask import Blueprint, Response, jsonify, render_template, request, stream_with_context
from pydantic import ValidationError

from .config import StockFinderSettings

logger = logging.getLogger(__name__)


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str, ensure_ascii=False)}\n\n"


def build_blueprint(settings: StockFinderSettings) -> Blueprint:
    """Create the ``stock_finder`` blueprint mounted at ``settings.url_prefix``."""
    bp = Blueprint("stock_finder", __name__, url_prefix=settings.url_prefix, template_folder="templates")

    def pipe():
        """The shared pipeline, built (and its heavy imports loaded) on the first API call."""
        from .pipeline import get_pipeline

        return get_pipeline()

    def error(message: str, status: int):
        return jsonify({"ok": False, "error": message, "disclaimer": settings.disclaimer}), status

    def body() -> dict[str, Any]:
        payload = request.get_json(silent=True)
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def report_json(report: Any, status: int = 200):
        return jsonify({"ok": not report.error, **report.model_dump(mode="json")}), status

    def stream_job(job: Any) -> Response:
        def generate() -> Iterator[str]:
            yield sse("session", {"job_id": job.id, "kind": job.kind, "disclaimer": settings.disclaimer})
            for item in job.stream():
                if item is None:
                    yield ": keep-alive\n\n"
                    continue
                event, data = item
                if event == "error":
                    data = {**data, "disclaimer": settings.disclaimer}
                yield sse(event, data)

        return Response(stream_with_context(generate()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    def start_job(kind: str, run: Any, payload: dict[str, Any]):
        job = pipe().jobs.submit(kind, run)
        if payload.get("stream", True) is False:
            return jsonify({"ok": True, "job_id": job.id, "status_url": f"{settings.url_prefix}/api/jobs/{job.id}",
                            "disclaimer": settings.disclaimer}), 202
        return stream_job(job)

    @bp.after_request
    def add_disclaimer(response: Response) -> Response:
        """Guarantee the disclaimer on every JSON object response."""
        if response.mimetype == "application/json" and not response.direct_passthrough:
            try:
                data = json.loads(response.get_data(as_text=True))
            except ValueError:
                return response
            if isinstance(data, dict) and "disclaimer" not in data:
                data["disclaimer"] = settings.disclaimer
                response.set_data(json.dumps(data, default=str, ensure_ascii=False))
        return response

    @bp.errorhandler(ValueError)
    def bad_request(exc: ValueError):
        return error(str(exc), 400)

    # --- page ---------------------------------------------------------------------------------

    @bp.route("/", methods=["GET"])
    def stock_finder_ui():
        """Hub-styled page (extends ``base.html``) that embeds the Gradio UI."""
        return render_template("stock_finder/stock_finder.html", title="Stock Finder",
                               ui_url=f"{settings.url_prefix}/ui/?__theme=dark", api_url=f"{settings.url_prefix}/api",
                               disclaimer=settings.disclaimer, market=settings.market, ui_enabled=settings.ui_enabled)

    # --- research -----------------------------------------------------------------------------

    @bp.route("/api/health", methods=["GET"])
    def api_health():
        return jsonify({"ok": True, **pipe().health()})

    @bp.route("/api/market-overview", methods=["GET"])
    def api_market_overview():
        refresh = request.args.get("refresh", "").lower() in ("1", "true", "yes")
        return report_json(pipe().market_overview(refresh=refresh))

    @bp.route("/api/find-opportunities", methods=["POST"])
    def api_find_opportunities():
        payload = body()
        sectors = payload.get("sectors")
        if sectors is not None and (not isinstance(sectors, list) or not all(isinstance(s, str) for s in sectors)):
            return error("'sectors' must be a list of strings", 400)
        max_candidates = payload.get("max_candidates")
        if max_candidates is not None and (not isinstance(max_candidates, int) or max_candidates < 1):
            return error("'max_candidates' must be a positive integer", 400)
        profile = pipe().risk_profile(payload.get("risk_profile"))
        return start_job("opportunities", lambda listener: pipe().find_opportunities(
            risk_profile=profile, sectors=sectors, max_candidates=max_candidates, listener=listener), payload)

    @bp.route("/api/analyze", methods=["POST"])
    def api_analyze():
        payload = body()
        symbol = payload.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            return error("'symbol' must be a non-empty string", 400)
        return report_json(pipe().analyze(symbol, risk_profile=payload.get("risk_profile")))

    @bp.route("/api/chat", methods=["POST"])
    def api_chat():
        payload = body()
        message = payload.get("message")
        if not isinstance(message, str) or not message.strip() or len(message) > 4000:
            return error("'message' must be a non-empty string of at most 4000 characters", 400)
        return report_json(pipe().chat(message, risk_profile=payload.get("risk_profile")))

    @bp.route("/api/jobs/<job_id>", methods=["GET"])
    def api_job(job_id: str):
        job = pipe().jobs.get(job_id)
        if job is None:
            return error("unknown job", 404)
        info = job.info()
        if job.report is not None:
            info["report"] = job.report.model_dump(mode="json")
        return jsonify({"ok": True, **info})

    @bp.route("/api/jobs/<job_id>/events", methods=["GET"])
    def api_job_events(job_id: str):
        job = pipe().jobs.get(job_id)
        if job is None:
            return error("unknown job", 404)
        return stream_job(job)

    @bp.route("/api/reports/<report_id>", methods=["GET"])
    def api_report(report_id: str):
        include = request.args.get("snapshot", "").lower() in ("1", "true", "yes")
        doc = pipe().reports.get(report_id, include_snapshot=include)
        if doc is None:
            return error("unknown report", 404)
        return jsonify({"ok": True, **doc})

    # --- portfolio ----------------------------------------------------------------------------

    def portfolio_id() -> str:
        value = (request.args.get("portfolio_id") or "default").strip()
        if not value.replace("-", "").replace("_", "").isalnum() or len(value) > 40:
            raise ValueError("portfolio_id may contain letters, digits, '-' and '_' (max 40)")
        return value

    def holding_from(raw: Any) -> Any:
        from .schemas import HoldingIn

        if not isinstance(raw, dict):
            raise ValueError("each holding must be an object")
        holding = HoldingIn.model_validate(raw)
        return holding.model_copy(update={"symbol": pipe().market.normalize(holding.symbol)})

    def mongo_errors(fn):
        from functools import wraps

        @wraps(fn)
        def wrapper(*args, **kwargs):
            from .mongo import MongoNotConfigured

            try:
                return fn(*args, **kwargs)
            except MongoNotConfigured:
                return error("portfolio storage is not configured (set SF_MONGODB_URI or MONGODB_URI)", 503)
            except ValidationError as exc:
                first = exc.errors()[0]
                return error(f"{'.'.join(str(p) for p in first['loc'])}: {first['msg']}", 400)
            except ValueError as exc:
                return error(str(exc), 400)
            except Exception:  # noqa: BLE001
                logger.exception("Stock Finder portfolio storage error")
                return error("portfolio storage is temporarily unavailable", 503)
        return wrapper

    @bp.route("/api/portfolio", methods=["GET"])
    @mongo_errors
    def api_portfolio_list():
        holdings = pipe().holdings.list(portfolio_id())
        return jsonify({"ok": True, "holdings": [h.model_dump(mode="json") for h in holdings]})

    @bp.route("/api/portfolio", methods=["POST"])
    @mongo_errors
    def api_portfolio_save():
        """Body: one holding, ``{"holdings": [...], "replace": bool}``, or a CSV (``Content-Type: text/csv``)."""
        from .portfolio.models import parse_holdings_csv

        pid = portfolio_id()
        if request.mimetype in ("text/csv", "text/plain"):
            rows, errors = parse_holdings_csv(request.get_data(as_text=True))
            if errors and not rows:
                return error("; ".join(errors[:5]), 400)
            items, replace = [r.model_dump() for r in rows], request.args.get("replace", "true").lower() != "false"
        else:
            payload = body()
            if "holdings" in payload:
                if not isinstance(payload["holdings"], list):
                    return error("'holdings' must be a list", 400)
                items, replace, errors = payload["holdings"], bool(payload.get("replace", False)), []
            else:
                items, replace, errors = [payload], False, []
        holdings = [holding_from(item) for item in items]
        repo = pipe().holdings
        saved = repo.replace_all(holdings, pid) if replace else [repo.upsert(h, pid) for h in holdings]
        return jsonify({"ok": True, "saved": [h.model_dump(mode="json") for h in saved], "warnings": errors}), 201

    @bp.route("/api/portfolio", methods=["DELETE"])
    @mongo_errors
    def api_portfolio_delete():
        payload = body() if request.data else {}
        symbol = request.args.get("symbol") or payload.get("symbol")
        if payload.get("all") is True or request.args.get("all") == "true":
            return jsonify({"ok": True, "deleted": pipe().holdings.clear(portfolio_id())})
        if not isinstance(symbol, str) or not symbol.strip():
            return error("pass 'symbol' (query or body), or all=true", 400)
        deleted = pipe().holdings.delete(pipe().market.normalize(symbol), portfolio_id())
        if not deleted:
            return error("holding not found", 404)
        return jsonify({"ok": True, "deleted": 1})

    @bp.route("/api/portfolio/review", methods=["POST"])
    def api_portfolio_review():
        payload = body()
        profile = pipe().risk_profile(payload.get("risk_profile"))
        pid = portfolio_id()
        return start_job("portfolio_review", lambda listener: pipe().review_portfolio(
            risk_profile=profile, portfolio_id=pid, listener=listener), payload)

    return bp
