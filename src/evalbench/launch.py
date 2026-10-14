"""`uv run evalbench-ui`: start the Streamlit app so that code changes in src/ are picked up too.

Streamlit only reloads modules under the app's own folder or on PYTHONPATH; the evalbench package lives in
src/, so without this a running server keeps stale code after an update (and pages crash on new columns).
"""
from __future__ import annotations

import os
import subprocess
import sys

from .items import ROOT


def build_command(port: int = 8501) -> list[str]:
    return [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
        "--server.folderWatchList", str(ROOT / "src"),
        "--server.port", str(port),
        "--browser.gatherUsageStats", "false",
    ]


def build_env(base: dict | None = None) -> dict:
    env = dict(os.environ if base is None else base)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = os.pathsep.join([src] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    return env


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8501
    raise SystemExit(subprocess.call(build_command(port), cwd=ROOT, env=build_env()))
