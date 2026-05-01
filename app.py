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
    show_graph()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=use_reloader,
    )