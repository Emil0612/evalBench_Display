"""Keep a running Streamlit server on the current evalbench code.

Streamlit's own module watcher did not reliably re-import src/ modules on Windows (verified: the watcher logged the
change, yet pages kept running the old code). The app therefore calls refresh() at the top of every run: if any
source file changed since the last run, every evalbench module is dropped from sys.modules and imported fresh.
"""
from __future__ import annotations

import sys
from pathlib import Path


def _stamp(src: Path) -> tuple:
    return tuple(sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in Path(src).rglob("*.py")))


def refresh(state: dict, src: Path, package: str = "evalbench") -> bool:
    """Purge `package` and its submodules if its sources changed since the previous call; True if purged."""
    stamp = _stamp(src)
    previous, state["stamp"] = state.get("stamp"), stamp
    if previous is None or previous == stamp:
        return False
    for name in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        del sys.modules[name]
    return True
