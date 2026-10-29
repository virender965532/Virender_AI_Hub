from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from flask import Flask

# Scraped job text is full of non-cp1252 characters (₹, ē, emoji); on a Windows
# console a stray print() of it would otherwise raise UnicodeEncodeError mid-run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from routes.main_routes import main_bp
from routes.remote_job_finder_routes import remote_job_finder_bp
from interviewSimulator.graph import show_graph
from jobEngine.api.routes import job_engine_bp
from resumeIntel.api.routes import resume_intel_bp
from profile_assistant import register_profile_assistant
from code_assistant import register_code_assistant

load_dotenv()


def setup_logging() -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",  # keep your main templates
        static_folder="static"
    )

    app.config["JSON_SORT_KEYS"] = False

    app.register_blueprint(main_bp)
    app.register_blueprint(remote_job_finder_bp)
    app.register_blueprint(job_engine_bp)
    app.register_blueprint(resume_intel_bp)
    register_profile_assistant(app)
    register_code_assistant(app)

    return app


setup_logging()
app = create_app()


if __name__ == "__main__":
    use_reloader = os.environ.get("DEV_WATCH") != "1"
    try:
        show_graph()
    except Exception:
        logging.getLogger(__name__).exception("Skipping interview graph export")

    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "8000"))
    debug = os.environ.get("FLASK_DEBUG", "1").strip().lower() not in ("0", "false", "no")

    # With the reloader on, this block also runs in the watcher parent, which never
    # serves requests; schedule only in the serving process so a run never doubles.
    if not (use_reloader and debug) or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        from jobEngine.scheduler import start_scheduler

        try:
            start_scheduler()
        except Exception:
            logging.getLogger(__name__).exception("Scheduled job search not started")

    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=use_reloader and debug,
        threaded=True,
    )