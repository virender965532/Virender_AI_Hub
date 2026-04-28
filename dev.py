"""
Development entry: restart Flask when project files change (watchdog watchmedo).

Run once and keep the terminal open: python dev.py
"""

from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    os.environ["DEV_WATCH"] = "1"
    root = os.path.dirname(os.path.abspath(__file__))
    cmd = [
        sys.executable,
        "-m",
        "watchdog.watchmedo",
        "auto-restart",
        "-d",
        root,
        "-p",
        "*.py;*.html;*.css;*.js",
        "-i",
        "**/venv/**/*;**/.venv/**/*;**/__pycache__/*;**/scraper_debug/*",
        "-R",
        "--debounce-interval",
        "0.5",
        "--",
        sys.executable,
        os.path.join(root, "app.py"),
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
