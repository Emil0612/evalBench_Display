"""Background runs: the Run page saves the approved calls as a job file and starts this module as its own process,
so a run keeps going when the page changes, the tab closes or the app restarts. Progress, Stop and failed calls go
through the database (store.run_progress / request_stop / run_errors).

python -m evalbench.worker RUN_ID [--db PATH] [--jobs DIR]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

from . import runner, store
from .items import ROOT, Item

JOBS_DIR = ROOT / "results" / "jobs"


def save_job(jobs_dir: Path, run_id: str, calls: list[runner.CallSpec], meta: dict, spend_cap: float, privacy: str,
             concurrency: int = 8) -> Path:
    """Exactly what was approved: the calls, the prices they were approved at, the cap and the privacy setting."""
    jobs_dir = Path(jobs_dir)
    jobs_dir.mkdir(parents=True, exist_ok=True)
    used = {c.model for c in calls}
    job = {"run_id": run_id, "spend_cap": spend_cap, "privacy": privacy, "concurrency": concurrency,
           "meta": {m: v for m, v in meta.items() if m in used},
           "calls": [dataclasses.asdict(c) for c in calls]}
    path = jobs_dir / f"{run_id}.json"
    path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    return path


def load_job(path: Path) -> dict:
    job = json.loads(Path(path).read_text(encoding="utf-8"))
    job["calls"] = [runner.CallSpec(**{**c, "item": Item(**c["item"])}) for c in job["calls"]]
    return job


def launch(run_id: str, db_path: Path, jobs_dir: Path) -> subprocess.Popen:
    """Start the worker detached from the app: it survives the page, the tab and the app's own process."""
    log = open(Path(jobs_dir) / f"{run_id}.log", "ab")
    cmd = [sys.executable, "-m", "evalbench.worker", run_id, "--db", str(db_path), "--jobs", str(jobs_dir)]
    # a hidden console of its own (not DETACHED_PROCESS: then the venv's python launcher opens a visible window)
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} \
        if sys.platform == "win32" else {"start_new_session": True}
    try:
        return subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                close_fds=True, **flags)
    finally:
        log.close()   # the child has its own handle


def submit(conn, calls: list[runner.CallSpec], meta: dict, spend_cap: float, privacy: str, concurrency: int = 8,
           jobs_dir: Path = JOBS_DIR, db_path: Path = store.DB_PATH,
           start: bool = True) -> tuple[str | None, str | None]:
    """Register the run (refused while another run is active), save its job and start the worker.
    Returns (run_id, None), or (None, id of the active run) when refused."""
    run_id = uuid.uuid4().hex[:10]
    other = store.start_run_exclusive(conn, runner.run_info(calls, meta, run_id))
    if other:
        return None, other
    save_job(jobs_dir, run_id, calls, meta, spend_cap, privacy, concurrency)
    if start:
        try:
            launch(run_id, db_path, jobs_dir)
        except Exception:
            store.upsert_run(conn, {**runner.run_info(calls, meta, run_id), "status": "failed to start"})
            raise
    return run_id, None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run one saved job (started by the Run page).")
    ap.add_argument("run_id")
    ap.add_argument("--db", default=str(store.DB_PATH))
    ap.add_argument("--jobs", default=str(JOBS_DIR))
    args = ap.parse_args(argv)
    conn = store.connect(Path(args.db))
    try:
        job = load_job(Path(args.jobs) / f"{args.run_id}.json")
    except Exception:   # say so at once, instead of leaving the run queued until it times out
        traceback.print_exc()
        conn.execute("UPDATE runs SET status = 'failed to start' WHERE run_id = ?", (args.run_id,))
        conn.commit()
        return 1
    print(f"run {args.run_id}: {len(job['calls'])} calls, cap ${job['spend_cap']:.2f}", flush=True)
    try:
        _, state = runner.run(job["calls"], conn, job["meta"], spend_cap=job["spend_cap"], privacy=job["privacy"],
                              concurrency=job["concurrency"], run_id=args.run_id)
    except Exception:
        traceback.print_exc()
        return 1
    print(f"run {args.run_id}: {state.done} stored, {len(state.errors)} failed, ${state.spent:.4f} spent", flush=True)
    (Path(args.jobs) / f"{args.run_id}.json").unlink(missing_ok=True)   # kept only when a run crashed
    return 0


if __name__ == "__main__":
    sys.exit(main())
