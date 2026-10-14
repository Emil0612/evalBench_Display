"""SQLite store: one row per model call (raw request + raw response + parsed answer), plus run log."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .items import ROOT

DB_PATH = ROOT / "results" / "evalbench.sqlite"
KEY_COLS = ("model", "method", "item_id", "variant", "mode", "temperature", "reasoning", "seed", "sample", "testset",
            "prompt_version")
CALL_COLS = KEY_COLS + ("run_id", "messages", "response", "parsed", "refused", "cost", "prompt_tokens",
                        "completion_tokens", "provider", "model_returned", "latency_s", "created", "finish_reason",
                        "privacy", "route", "logprobs")
RUN_COLS = ("run_id", "created", "models", "methods", "n_calls", "est_cost", "cost", "status")
CALLS_TABLE = """
CREATE TABLE IF NOT EXISTS {name} (
    id INTEGER PRIMARY KEY,
    model TEXT, method TEXT, item_id TEXT, variant TEXT, mode TEXT, temperature REAL NOT NULL DEFAULT 0,
    reasoning TEXT NOT NULL DEFAULT 'auto', seed INTEGER NOT NULL DEFAULT -1, sample INTEGER NOT NULL DEFAULT 0,
    testset TEXT, prompt_version TEXT, privacy TEXT,
    run_id TEXT, messages TEXT, response TEXT, parsed TEXT, refused INTEGER, cost REAL,
    prompt_tokens INTEGER, completion_tokens INTEGER, provider TEXT, model_returned TEXT,
    latency_s REAL, created TEXT, finish_reason TEXT, route TEXT NOT NULL DEFAULT '', logprobs TEXT,
    UNIQUE (model, method, item_id, variant, mode, temperature, reasoning, seed, sample, testset, prompt_version)
);
"""
# answers replaced by a new answer to the same question (e.g. re-asked on a pinned provider): kept, never deleted
SUPERSEDED_TABLE = ("CREATE TABLE IF NOT EXISTS calls_superseded (id INTEGER, "
                    + ", ".join(CALL_COLS) + ", superseded_at TEXT);")
SCHEMA = CALLS_TABLE.format(name="calls") + SUPERSEDED_TABLE + """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, created TEXT, models TEXT, methods TEXT, n_calls INTEGER,
    est_cost REAL, cost REAL, status TEXT
);
CREATE TABLE IF NOT EXISTS run_progress (
    run_id TEXT PRIMARY KEY, pid INTEGER, done INTEGER NOT NULL DEFAULT 0, failed INTEGER NOT NULL DEFAULT 0,
    spent REAL NOT NULL DEFAULT 0, heartbeat TEXT, stop_requested INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS run_errors (run_id TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS inflight (key TEXT PRIMARY KEY, run_id TEXT, claimed TEXT);
CREATE TABLE IF NOT EXISTS reparse_log (id INTEGER PRIMARY KEY, call_id INTEGER, old_parsed TEXT, new_parsed TEXT,
                                        created TEXT);
"""
ACTIVE = ("queued", "running")
STALE_S = 120   # a run silent this long (no heartbeat) has died: its worker was killed, or the computer slept


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_READY: set[tuple] = set()   # (path, schema version) of databases whose schema and migrations this process applied


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # timeout: a writer waits up to 30 s for a lock instead of failing (a paid answer must never be lost to a busy
    # reader); WAL: pages read while a run writes, without blocking each other
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA busy_timeout = 30000")
    # schema and migrations already done in this process: a cheap connect (the app opens one per thread, and
    # Streamlit starts a thread per rerun). The schema version tells a replaced or restored file from the one set up.
    key = str(path.resolve())
    if (key, conn.execute("PRAGMA schema_version").fetchone()[0]) in _READY:
        return conn
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:   # another connection holds the file; WAL is set by the next connect
        pass
    conn.executescript(SCHEMA)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(calls)")}
    if "finish_reason" not in columns:  # databases created before the column existed
        conn.execute("ALTER TABLE calls ADD COLUMN finish_reason TEXT")
        conn.commit()
        columns.add("finish_reason")
    if not {"temperature", "reasoning", "seed", "sample", "privacy"} <= columns:
        _migrate_key(conn, columns)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(calls)")}
    if "route" not in columns:  # answers stored before provider pinning existed used default routing
        conn.execute("ALTER TABLE calls ADD COLUMN route TEXT NOT NULL DEFAULT ''")
    for table in ("calls", "calls_superseded"):   # token log-probabilities, for the probability probes (2026-10-01)
        if "logprobs" not in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN logprobs TEXT")
    # answers stored before the privacy setting existed were routed without restrictions
    conn.execute("UPDATE calls SET privacy = CASE WHEN provider = 'local' THEN 'local' ELSE 'standard' END "
                 "WHERE privacy IS NULL")
    conn.commit()
    _READY.add((key, conn.execute("PRAGMA schema_version").fetchone()[0]))
    return conn


def _migrate_key(conn: sqlite3.Connection, columns: set[str]) -> None:
    """Older databases: rebuild the calls table with the current key. Columns that did not exist yet take their
    defaults, which describe how every earlier answer was requested: temperature 0, reasoning 'auto', no seed,
    first sample."""
    keep = ", ".join(c for c in ("id", *CALL_COLS) if c in columns)
    conn.executescript("BEGIN; DROP TABLE IF EXISTS calls_new;"   # leftover from an interrupted earlier attempt
                       + CALLS_TABLE.format(name="calls_new")
                       + f"INSERT INTO calls_new ({keep}) SELECT {keep} FROM calls;"
                       + "DROP TABLE calls; ALTER TABLE calls_new RENAME TO calls; COMMIT;")


def db_stamp(conn: sqlite3.Connection) -> tuple:
    """Changes whenever an answer is added, replaced or re-read (reparse): lets the app cache the answers until then.
    Rows are never deleted (a replaced answer is archived in calls_superseded first; a re-read is logged), so the
    highest row numbers of the three tables see every change, without the full-table scan a sum would need (80 ms,
    which the Overview paid 108 times)."""
    return tuple(conn.execute("SELECT (SELECT max(id) FROM calls), (SELECT max(rowid) FROM calls_superseded), "
                              "(SELECT max(id) FROM reparse_log)").fetchone())


def first_seen(conn: sqlite3.Connection) -> dict[str, str]:
    """model -> start time of the first run it appeared in, counting archived (replaced) answers too, so the value
    never moves later when answers are re-asked."""
    rows = conn.execute(
        "SELECT a.model, MIN(COALESCE(r.created, a.created)) FROM "
        "(SELECT model, run_id, created FROM calls UNION ALL SELECT model, run_id, created FROM calls_superseded) a "
        "LEFT JOIN runs r ON r.run_id = a.run_id GROUP BY a.model").fetchall()
    return {m: t or "" for m, t in rows}


def cached_keys(conn: sqlite3.Connection) -> set[tuple]:
    return set(conn.execute(f"SELECT {', '.join(KEY_COLS)} FROM calls").fetchall())


def cached_routes(conn: sqlite3.Connection) -> dict[tuple, str]:
    """Stored question key -> the provider route its answer came from ('' = default routing)."""
    return {tuple(r[:-1]): r[-1] or "" for r in conn.execute(f"SELECT {', '.join(KEY_COLS)}, route FROM calls")}


def has_call(conn: sqlite3.Connection, key: tuple, route: str = "") -> bool:
    """A stored answer to this question; for a pinned call only one that came from the same route counts."""
    where = " AND ".join(f"{c} = ?" for c in KEY_COLS)
    row = conn.execute(f"SELECT route FROM calls WHERE {where} LIMIT 1", key).fetchone()
    return row is not None and (not route or (row[0] or "") == route)


def insert_call(conn: sqlite3.Connection, rec: dict) -> None:
    # key fields left out take the table's defaults here too, so the archive below finds the row it replaces
    defaults = {"temperature": 0.0, "reasoning": "auto", "seed": -1, "sample": 0}
    rec = {**rec, **{k: v for k, v in defaults.items() if rec.get(k) is None},
           "created": rec.get("created") or _now(), "route": rec.get("route") or ""}
    cols = ", ".join(CALL_COLS)
    marks = ", ".join("?" for _ in CALL_COLS)
    where = " AND ".join(f"{c} = ?" for c in KEY_COLS)
    # an earlier answer to the same question is archived before it is replaced
    conn.execute(f"INSERT INTO calls_superseded (id, {cols}, superseded_at) SELECT id, {cols}, ? FROM calls "
                 f"WHERE {where}", [_now(), *(rec.get(c) for c in KEY_COLS)])
    conn.execute(f"INSERT OR REPLACE INTO calls ({cols}) VALUES ({marks})", [rec.get(c) for c in CALL_COLS])
    conn.commit()


def upsert_run(conn: sqlite3.Connection, run: dict) -> None:
    run = {**run, "created": run.get("created") or _now()}
    existing = conn.execute("SELECT created FROM runs WHERE run_id = ?", (run["run_id"],)).fetchone()
    if existing:
        run["created"] = existing[0]
    cols = ", ".join(RUN_COLS)
    marks = ", ".join("?" for _ in RUN_COLS)
    conn.execute(f"INSERT OR REPLACE INTO runs ({cols}) VALUES ({marks})", [run.get(c) for c in RUN_COLS])
    conn.commit()


def _ts(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _reap(conn: sqlite3.Connection, now: str | None, stale_s: int) -> int:
    """Mark active runs whose last sign of life (heartbeat, else start) is older than stale_s as interrupted; returns
    how many. No commit: the caller decides."""
    t = _ts(now or _now())
    rows = conn.execute("SELECT r.run_id, COALESCE(p.heartbeat, r.created) FROM runs r LEFT JOIN run_progress p "
                        f"ON p.run_id = r.run_id WHERE r.status IN {ACTIVE}").fetchall()
    dead = [run_id for run_id, last in rows if last is None or (t - _ts(last)).total_seconds() > stale_s]
    for run_id in dead:
        conn.execute("UPDATE runs SET status = 'interrupted' WHERE run_id = ?", (run_id,))
    return len(dead)


def _run_row(conn: sqlite3.Connection, where: str) -> dict | None:
    """The newest run matching where, with its progress."""
    row = conn.execute(
        "SELECT r.run_id, r.status, r.created, r.models, r.methods, r.n_calls, r.est_cost, COALESCE(p.done, 0), "
        "COALESCE(p.failed, 0), COALESCE(p.spent, 0), p.heartbeat, COALESCE(p.stop_requested, 0), p.pid "
        f"FROM runs r LEFT JOIN run_progress p ON p.run_id = r.run_id WHERE {where} "
        "ORDER BY r.created DESC, r.rowid DESC LIMIT 1").fetchone()
    keys = ("run_id", "status", "created", "models", "methods", "n_calls", "est_cost", "done", "failed", "spent",
            "heartbeat", "stop_requested", "pid")
    return dict(zip(keys, row)) if row else None


def _active(conn: sqlite3.Connection) -> dict | None:
    return _run_row(conn, f"r.status IN {ACTIVE}")


def last_run(conn: sqlite3.Connection) -> dict | None:
    """The newest run of any status, with its progress (the Run page's result card)."""
    return _run_row(conn, "1 = 1")


def active_run(conn: sqlite3.Connection, now: str | None = None, stale_s: int = STALE_S) -> dict | None:
    """The run going right now (queued or running, and alive), with its progress; None if there is none."""
    if _reap(conn, now, stale_s):
        conn.commit()
    return _active(conn)


def start_run_exclusive(conn: sqlite3.Connection, info: dict, now: str | None = None,
                        stale_s: int = STALE_S) -> str | None:
    """Register a queued run, unless another run is active: then nothing is written and that run's id is returned.
    Atomic (BEGIN IMMEDIATE), so two tabs pressing Run at the same moment cannot both start one."""
    conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        _reap(conn, now, stale_s)
        other = _active(conn)
        if other:
            conn.rollback()
            return other["run_id"]
        created = now or _now()
        run = {**info, "created": created, "status": "queued", "cost": info.get("cost", 0.0)}
        conn.execute(f"INSERT OR REPLACE INTO runs ({', '.join(RUN_COLS)}) VALUES ({', '.join('?' for _ in RUN_COLS)})",
                     [run.get(c) for c in RUN_COLS])
        conn.execute("INSERT OR REPLACE INTO run_progress (run_id, heartbeat) VALUES (?, ?)", (info["run_id"], created))
        conn.commit()
        return None
    except Exception:
        conn.rollback()
        raise


def heartbeat(conn: sqlite3.Connection, run_id: str, done: int, failed: int, spent: float, now: str | None = None,
              pid: int | None = None) -> None:
    """The run's sign of life and progress. A run marked interrupted while its computer slept comes back to running."""
    conn.execute("INSERT OR IGNORE INTO run_progress (run_id) VALUES (?)", (run_id,))
    conn.execute("UPDATE run_progress SET done = ?, failed = ?, spent = ?, heartbeat = ?, pid = COALESCE(?, pid) "
                 "WHERE run_id = ?", (done, failed, spent, now or _now(), pid, run_id))
    conn.execute("UPDATE runs SET status = 'running' WHERE run_id = ? AND status IN ('queued', 'interrupted')",
                 (run_id,))
    conn.execute("UPDATE runs SET cost = ? WHERE run_id = ?", (spent, run_id))
    conn.commit()


def request_stop(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute("INSERT OR IGNORE INTO run_progress (run_id) VALUES (?)", (run_id,))
    conn.execute("UPDATE run_progress SET stop_requested = 1 WHERE run_id = ?", (run_id,))
    conn.commit()


def stop_requested(conn: sqlite3.Connection, run_id: str) -> bool:
    row = conn.execute("SELECT stop_requested FROM run_progress WHERE run_id = ?", (run_id,)).fetchone()
    return bool(row and row[0])


def run_progress(conn: sqlite3.Connection, run_id: str) -> dict:
    row = conn.execute("SELECT done, failed, spent, heartbeat FROM run_progress WHERE run_id = ?", (run_id,)).fetchone()
    return dict(zip(("done", "failed", "spent", "heartbeat"), row)) if row else {}


def _key_text(key: tuple) -> str:
    return json.dumps(list(key))


def claim_call(conn: sqlite3.Connection, key: tuple, run_id: str) -> bool:
    """Take a question before paying for it; False if another run holds it (never pay twice, across processes)."""
    cur = conn.execute("INSERT OR IGNORE INTO inflight (key, run_id, claimed) VALUES (?, ?, ?)",
                       (_key_text(key), run_id, _now()))
    conn.commit()
    return cur.rowcount == 1


def release_call(conn: sqlite3.Connection, key: tuple) -> None:
    conn.execute("DELETE FROM inflight WHERE key = ?", (_key_text(key),))
    conn.commit()


def clear_claims(conn: sqlite3.Connection, keep: set[str]) -> None:
    """Drop claims left behind by runs that died (all but the runs in keep)."""
    marks = ", ".join("?" for _ in keep)
    conn.execute(f"DELETE FROM inflight WHERE run_id NOT IN ({marks})" if keep else "DELETE FROM inflight", list(keep))
    conn.commit()


def add_run_error(conn: sqlite3.Connection, run_id: str, message: str) -> None:
    conn.execute("INSERT INTO run_errors (run_id, message) VALUES (?, ?)", (run_id, message))
    conn.commit()


def run_errors(conn: sqlite3.Connection, run_id: str) -> list[str]:
    return [r[0] for r in conn.execute("SELECT message FROM run_errors WHERE run_id = ? ORDER BY rowid", (run_id,))]


def load_calls(conn: sqlite3.Connection, mode: str | None = None, temperature: float | None = None,
               reasoning: str | None = None, seed: int | None = None) -> pd.DataFrame:
    where, params = [], []
    for col, value in (("mode", mode), ("temperature", None if temperature is None else round(float(temperature), 2)),
                       ("reasoning", reasoning), ("seed", seed)):
        if value is not None:
            where.append(f"{col} = ?")
            params.append(value)
    sql = "SELECT * FROM calls" + (" WHERE " + " AND ".join(where) if where else "")
    return pd.read_sql_query(sql, conn, params=params)


def temperatures(conn: sqlite3.Connection) -> list[float]:
    return [r[0] for r in conn.execute("SELECT DISTINCT temperature FROM calls ORDER BY temperature")]


def run_settings(conn: sqlite3.Connection, mode: str | None = None) -> list[tuple]:
    """Distinct (temperature, reasoning, seed) combinations that have stored answers (optionally for one mode)."""
    where, params = ("WHERE mode = ?", (mode,)) if mode else ("", ())
    return [tuple(r) for r in conn.execute(
        f"SELECT DISTINCT temperature, reasoning, seed FROM calls {where} ORDER BY temperature, reasoning, seed",
        params)]


def stats(conn: sqlite3.Connection) -> dict:
    """Counts for status displays, computed in the database (no raw texts are loaded)."""
    answers, models = conn.execute("SELECT COUNT(*), COUNT(DISTINCT model) FROM calls").fetchone()
    api_models, spend = conn.execute("SELECT COUNT(DISTINCT model), COALESCE(SUM(cost), 0) FROM calls "
                                     "WHERE provider IS NOT 'local'").fetchone()
    return {"answers": answers, "models": models, "api_models": api_models, "api_spend": float(spend)}


def load_runs(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM runs ORDER BY created DESC", conn)
