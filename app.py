from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from flask import Flask

from llm_provider import log_startup_status

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
from stock_finder import register_stock_finder

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
    register_stock_finder(app)

    return app


debug = os.environ.get("FLASK_DEBUG", "1").strip().lower() not in ("0", "false", "no")
use_reloader = debug and os.environ.get("DEV_WATCH") != "1"
# With `python app.py` and the reloader on, this file runs twice: in a watcher parent that
# only restarts the server on code changes (it never serves requests), and in the serving
# child (WERKZEUG_RUN_MAIN=true). Build the app and log startup only in the serving process.
is_reloader_watcher = __name__ == "__main__" and use_reloader and os.environ.get("WERKZEUG_RUN_MAIN") != "true"

setup_logging()
if is_reloader_watcher:
    app = Flask(__name__)  # placeholder; the serving child builds the real app
else:
    # Which LLM backend (LLM_PROVIDER) is active; warns if Ollama or a model is missing.
    log_startup_status()
    app = create_app()


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "8000"))

    if not is_reloader_watcher:
        try:
            show_graph()
        except Exception:
            logging.getLogger(__name__).exception("Skipping interview graph export")

        from jobEngine.scheduler import start_scheduler

        try:
            start_scheduler()
        except Exception:
            logging.getLogger(__name__).exception("Scheduled job search not started")

    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=use_reloader,
        threaded=True,
    )