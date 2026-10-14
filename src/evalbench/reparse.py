"""Re-read stored replies with a method's current parser, after a parser fix. Nothing is asked again and nothing is
paid; the reply text stays as stored. Every changed reading is logged in reparse_log (old and new), so it can be
traced and undone.

uv run python -m evalbench.reparse index_level index_return us_index_level us_index_return           # dry run
uv run python -m evalbench.reparse index_level index_return us_index_level us_index_return --apply
"""
from __future__ import annotations

import argparse
import json
import sqlite3

from . import analysis, store
from .methods import METHODS


def reparse(conn: sqlite3.Connection, methods: list[str], apply: bool = False) -> list[dict]:
    """The rows whose reading changes (id, model, method, item_id, old, new); written only with apply=True."""
    changes = []
    for method in methods:
        mod = METHODS[method]
        rows = conn.execute("SELECT id, model, item_id, response, parsed FROM calls WHERE method = ?", (method,))
        for call_id, model, item_id, response, parsed in rows.fetchall():
            old = json.loads(parsed)
            if "letter_probs" in old or "parse_error" in old:     # probes and crashed parses: leave as stored
                continue
            new = mod.parse(response or "")
            seen = analysis._with_outcome(mod, old, response)   # rows stored before outcome labels: as analysed
            keys = ("value", "refused", "outcome")
            if {k: seen.get(k) for k in keys} != {k: new.get(k) for k in keys}:
                changes.append({"id": call_id, "model": model, "method": method, "item_id": item_id, "old": old,
                                "new": new})
    if apply and changes:
        if store.active_run(conn):
            raise RuntimeError("a run is active: re-parse after it has finished")
        now = store._now()
        with conn:
            for c in changes:
                conn.execute("UPDATE calls SET parsed = ?, refused = ? WHERE id = ?",
                             (json.dumps(c["new"], ensure_ascii=False), int(c["new"]["refused"]), c["id"]))
                conn.execute("INSERT INTO reparse_log (call_id, old_parsed, new_parsed, created) VALUES (?, ?, ?, ?)",
                             (c["id"], json.dumps(c["old"], ensure_ascii=False), json.dumps(c["new"], ensure_ascii=False),
                              now))
    return changes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="+")
    ap.add_argument("--apply", action="store_true", help="write the new readings (default: only list them)")
    args = ap.parse_args()
    conn = store.connect()
    changes = reparse(conn, args.methods, apply=args.apply)
    for c in changes:
        print(f"{c['method']:<16} {c['model']:<36} {c['item_id']:<8} "
              f"{c['old'].get('value')!s:>10} ({c['old'].get('outcome')}) -> {c['new'].get('value')!s:>10} "
              f"({c['new'].get('outcome')})")
    print(f"{len(changes)} readings {'changed' if args.apply else 'would change (dry run; --apply to write)'}")


if __name__ == "__main__":
    main()
