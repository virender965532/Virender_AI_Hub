from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from flask import Flask

from routes.main_routes import main_bp
from interviewSimulator.graph import show_graph

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

    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=use_reloader and debug,
        threaded=True,
    )