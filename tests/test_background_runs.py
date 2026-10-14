"""Background runs: a run is a separate worker process that reports progress through the database, can be stopped
from any tab, and never pays twice for the same question, even across processes."""
from evalbench import store


def _info(run_id, n=10):
    return {"run_id": run_id, "models": "x/m", "methods": "price_recall", "n_calls": n, "est_cost": 1.0, "cost": 0.0}


# ---------------------------------------------------------------- the database
def test_the_database_uses_wal_and_waits_for_a_busy_writer(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"      # readers never block the run's writes
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 30000


def test_only_one_run_can_be_active(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    assert store.start_run_exclusive(conn, _info("r1")) is None
    assert store.start_run_exclusive(conn, _info("r2")) == "r1"            # refused: r1 is still going
    assert store.active_run(conn)["run_id"] == "r1"
    assert store.load_runs(conn).set_index("run_id").status["r1"] == "queued"


def test_a_run_whose_heartbeat_stopped_is_interrupted_and_frees_the_slot(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    store.start_run_exclusive(conn, _info("r1"))
    store.heartbeat(conn, "r1", done=3, failed=1, spent=0.5, now="2026-01-01T00:00:00+00:00")
    assert store.active_run(conn, now="2026-01-01T00:01:00+00:00")["run_id"] == "r1"
    assert store.active_run(conn, now="2026-01-01T00:05:00+00:00") is None  # silent for 5 minutes
    assert store.load_runs(conn).set_index("run_id").status["r1"] == "interrupted"
    assert store.start_run_exclusive(conn, _info("r2")) is None


def test_a_heartbeat_after_sleep_revives_an_interrupted_run(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    store.start_run_exclusive(conn, _info("r1"))
    store.heartbeat(conn, "r1", done=0, failed=0, spent=0.0, now="2026-01-01T00:00:00+00:00")
    store.active_run(conn, now="2026-01-01T00:05:00+00:00")                 # marked interrupted
    store.heartbeat(conn, "r1", done=1, failed=0, spent=0.1, now="2026-01-01T00:05:01+00:00")
    row = store.active_run(conn, now="2026-01-01T00:05:02+00:00")
    assert row["run_id"] == "r1" and row["status"] == "running" and row["done"] == 1


def test_stop_is_requested_through_the_database(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    store.start_run_exclusive(conn, _info("r1"))
    assert not store.stop_requested(conn, "r1")
    store.request_stop(conn, "r1")
    assert store.stop_requested(conn, "r1")


def test_a_question_is_claimed_by_one_run_at_a_time(tmp_path):
    path = tmp_path / "t.sqlite"
    a, b = store.connect(path), store.connect(path)                        # two processes, one database
    key = ("x/m", "price_recall", "pr-1", "main", "forced", 0.0, "auto", -1, 0, "price_recall/t", "p1")
    assert store.claim_call(a, key, "r1")
    assert not store.claim_call(b, key, "r2")                               # r2 must not pay for it too
    store.release_call(a, key)
    assert store.claim_call(b, key, "r2")


def test_claims_left_by_a_dead_run_are_cleared(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    key = ("x/m", "price_recall", "pr-1", "main", "forced", 0.0, "auto", -1, 0, "price_recall/t", "p1")
    store.claim_call(conn, key, "dead")
    store.clear_claims(conn, keep={"r2"})
    assert store.claim_call(conn, key, "r2")


def test_failed_calls_are_kept_per_run(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    store.add_run_error(conn, "r1", "x/m | price_recall | pr-1 | main: 502")
    assert store.run_errors(conn, "r1") == ["x/m | price_recall | pr-1 | main: 502"]
    assert store.run_errors(conn, "r2") == []


# ---------------------------------------------------------------- the runner
from evalbench import runner  # noqa: E402
from test_runner import FakeClient, _testsets  # noqa: E402


class StopAfter(FakeClient):
    """Presses Stop (through the database, like the page does) after its n-th paid call."""
    def __init__(self, conn, run_id, n):
        super().__init__(cost=0.1)
        self.conn, self.run_id, self.n = conn, run_id, n

    async def chat(self, body):
        res = await super().chat(body)
        if len(self.bodies) == self.n:
            store.request_stop(self.conn, self.run_id)
        return res


def test_a_registered_run_reports_progress_and_its_final_status(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 4, "forced", _testsets(data))
    store.start_run_exclusive(conn, _info("r1", len(calls)))
    run_id, state = runner.run(calls, conn, {}, client=FakeClient(cost=0.1), run_id="r1")
    assert run_id == "r1" and state.done == 4
    assert store.run_progress(conn, "r1")["done"] == 4 and abs(store.run_progress(conn, "r1")["spent"] - 0.4) < 1e-9
    runs = store.load_runs(conn).set_index("run_id")
    assert runs.status["r1"] == "done" and abs(runs.cost["r1"] - 0.4) < 1e-9 and store.active_run(conn) is None


def test_stop_from_another_tab_ends_the_run_and_keeps_paid_answers(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 8, "forced", _testsets(data))
    store.start_run_exclusive(conn, _info("r1", len(calls)))
    _, state = runner.run(calls, conn, {}, client=StopAfter(conn, "r1", 3), run_id="r1", concurrency=1, poll_s=0)
    assert state.done == 3 and len(runner.pending(calls, conn)) == 5
    assert store.load_runs(conn).set_index("run_id").status["r1"] == "stopped by user"


def test_failed_calls_are_recorded_for_the_run(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 2, "forced", _testsets(data))
    runner.run(calls, conn, {}, client=FakeClient(fail=True), run_id="r1")
    assert len(store.run_errors(conn, "r1")) == 2 and "402" in store.run_errors(conn, "r1")[0]


def test_a_question_another_live_run_is_paying_for_is_skipped(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 3, "forced", _testsets(data))
    store.start_run_exclusive(conn, _info("other"))
    store.heartbeat(conn, "other", done=0, failed=0, spent=0.0)
    store.claim_call(conn, calls[0].key, "other")
    client = FakeClient(cost=0.1)
    _, state = runner.run(calls, conn, {}, client=client, run_id="r1")
    assert state.done == 2 and state.skipped == 1 and len(client.bodies) == 2


def test_claims_are_released_when_a_run_ends(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 3, "forced", _testsets(data))
    runner.run(calls, conn, {}, client=FakeClient(fail=True))
    assert conn.execute("SELECT COUNT(*) FROM inflight").fetchone()[0] == 0


# ---------------------------------------------------------------- the worker process
import time  # noqa: E402

from evalbench import worker  # noqa: E402

CONTROLS = ["mock/oracle@2024-06-30", "mock/random@2024-06-30"]


def test_a_job_file_keeps_exactly_the_approved_calls(data, tmp_path):
    calls = runner.plan(["x/m"], ["price_recall", "decop"], 3, "forced", _testsets(data), temperature=0.3, seed=7,
                        pins={"x/m": "novita/bf16"})
    meta = {"x/m": {"pricing": {"prompt": "0.000001", "completion": "0.000002"}}}
    path = worker.save_job(tmp_path, "r1", calls, meta, spend_cap=1.25, privacy="no_training", concurrency=4)
    job = worker.load_job(path)
    assert job["calls"] == calls and job["meta"] == meta
    assert (job["spend_cap"], job["privacy"], job["concurrency"]) == (1.25, "no_training", 4)


def test_the_worker_runs_a_saved_job_to_the_end(data, tmp_path):
    db = tmp_path / "t.sqlite"
    conn = store.connect(db)
    calls = runner.plan(CONTROLS, ["price_recall"], 5, "forced", _testsets(data))
    run_id, other = worker.submit(conn, calls, {}, spend_cap=0.0, privacy="standard", jobs_dir=tmp_path,
                                  start=False)
    assert other is None
    assert worker.main([run_id, "--db", str(db), "--jobs", str(tmp_path)]) == 0
    assert store.load_runs(conn).set_index("run_id").status[run_id] == "done"
    assert runner.pending(calls, conn) == []


def test_submit_refuses_while_another_run_is_active(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(CONTROLS, ["price_recall"], 2, "forced", _testsets(data))
    first, _ = worker.submit(conn, calls, {}, 0.0, "standard", jobs_dir=tmp_path, start=False)
    second, other = worker.submit(conn, calls, {}, 0.0, "standard", jobs_dir=tmp_path, start=False)
    assert second is None and other == first


def test_a_detached_worker_process_finishes_on_its_own(data, tmp_path):
    db = tmp_path / "t.sqlite"
    conn = store.connect(db)
    calls = runner.plan(CONTROLS, ["price_recall"], 5, "forced", _testsets(data))
    run_id, _ = worker.submit(conn, calls, {}, 0.0, "standard", jobs_dir=tmp_path, db_path=db)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and store.load_runs(conn).set_index("run_id").status[run_id] in store.ACTIVE:
        time.sleep(0.5)
    log = (tmp_path / f"{run_id}.log").read_text(encoding="utf-8", errors="replace")
    assert store.load_runs(conn).set_index("run_id").status[run_id] == "done", log
    assert runner.pending(calls, conn) == []


def test_a_worker_without_its_job_file_reports_failed_to_start(tmp_path):
    db = tmp_path / "t.sqlite"
    conn = store.connect(db)
    store.start_run_exclusive(conn, _info("r1"))
    assert worker.main(["r1", "--db", str(db), "--jobs", str(tmp_path)]) == 1
    assert store.load_runs(conn).set_index("run_id").status["r1"] == "failed to start" and not store.active_run(conn)


def test_the_worker_starts_with_a_hidden_console_on_windows(tmp_path, monkeypatch):
    import subprocess
    import sys
    seen = {}
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: seen.update(kw))
    worker.launch("r1", tmp_path / "t.sqlite", tmp_path)
    if sys.platform == "win32":
        # a console that is never shown: DETACHED_PROCESS (no console at all) made the venv's python launcher open a
        # visible window for the real interpreter
        assert seen["creationflags"] & subprocess.CREATE_NO_WINDOW
        assert not seen["creationflags"] & subprocess.DETACHED_PROCESS
        assert seen["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert seen["start_new_session"]


# ---------------------------------------------------------------- what the pages say
from evalbench import ui  # noqa: E402


def _row(**kw):
    return {"run_id": "r1", "status": "running", "n_calls": 9648, "done": 637, "failed": 0, "spent": 0.01,
            "stop_requested": 0, **kw}


def test_progress_line_counts_finished_calls_and_spend():
    assert ui.run_progress_text(_row()) == "637 / 9,648 calls · $0.0100 spent"
    assert ui.run_progress_text(_row(failed=3)) == "640 / 9,648 calls · $0.0100 spent · 3 failed"
    assert ui.run_progress_text(_row(stop_requested=1)).endswith("· stopping…")


def test_the_last_run_reads_like_the_old_end_message():
    msg, kind = ui.run_outcome(_row(status="done", done=9648, spent=0.35))
    assert msg == "<b>Run r1 finished.</b> 9,648 answers stored, $0.35 spent." and kind == "good"
    msg, kind = ui.run_outcome(_row(status="done with 70 errors", done=9578, failed=70, spent=0.35))
    assert "70 calls failed" in msg and "run the same selection again" in msg and kind == "warn"
    assert ui.run_outcome(_row(status="stopped at cap"))[1] == "warn"
    assert "Stopped" in ui.run_outcome(_row(status="stopped by user"))[0]
    msg, kind = ui.run_outcome(_row(status="interrupted"))
    assert "interrupted" in msg and "never paid twice" in msg and kind == "bad"
    assert "results/jobs/r1.log" in ui.run_outcome(_row(status="failed to start"))[0]


def test_last_run_includes_its_progress(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    assert store.last_run(conn) is None
    store.start_run_exclusive(conn, _info("r1"), now="2026-01-01T00:00:00+00:00")
    store.heartbeat(conn, "r1", done=2, failed=1, spent=0.2)
    store.upsert_run(conn, {**_info("r1"), "cost": 0.2, "status": "done with 1 errors"})
    row = store.last_run(conn)
    assert (row["run_id"], row["status"], row["done"], row["failed"]) == ("r1", "done with 1 errors", 2, 1)


def test_progress_reaches_the_database_during_a_run_that_never_waits(data, tmp_path):
    # the free controls answer instantly, so the event loop never yields: progress must not depend on that
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(CONTROLS, ["price_recall"], 10, "forced", _testsets(data))
    seen = []
    runner.run(calls, conn, {}, run_id="r1", heartbeat_s=0,
               on_progress=lambda state, n: seen.append(store.run_progress(conn, "r1").get("done", 0)))
    assert max(seen) > 0


def test_the_job_file_is_removed_once_the_run_has_ended(data, tmp_path):
    db = tmp_path / "t.sqlite"
    conn = store.connect(db)
    calls = runner.plan(CONTROLS, ["price_recall"], 2, "forced", _testsets(data))
    run_id, _ = worker.submit(conn, calls, {}, 0.0, "standard", jobs_dir=tmp_path, start=False)
    worker.main([run_id, "--db", str(db), "--jobs", str(tmp_path)])
    assert not (tmp_path / f"{run_id}.json").exists()      # the answers are in the database now
