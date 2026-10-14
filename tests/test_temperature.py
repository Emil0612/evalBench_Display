"""Temperature is part of an answer's identity: answers at different temperatures are never reused or mixed."""
import sqlite3

import pytest

from evalbench import runner, selections, store
from evalbench.llm import openrouter as orr
from evalbench.methods import price_recall


def test_build_body_uses_requested_temperature_only_when_supported():
    meta = {"supported_parameters": ["temperature"]}
    assert orr.build_body(meta, "x/m", [], 60, temperature=0.7)["temperature"] == 0.7
    assert orr.build_body(meta, "x/m", [], 60)["temperature"] == 0  # default stays 0
    assert "temperature" not in orr.build_body({"supported_parameters": []}, "x/m", [], 60, temperature=0.7)


def test_answers_at_another_temperature_are_not_reused(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    ts = {"price_recall": ("price_recall/t", price_recall.build_items(data, 4, 1))}
    cold = runner.plan(["mock/random@2024-06-30"], ["price_recall"], 4, "forced", ts)
    runner.run(cold, conn, {})
    warm = runner.plan(["mock/random@2024-06-30"], ["price_recall"], 4, "forced", ts, temperature=0.7)
    assert runner.pending(cold, conn) == [] and len(runner.pending(warm, conn)) == 4
    runner.run(warm, conn, {})
    assert sorted(store.load_calls(conn)["temperature"].unique()) == [0.0, 0.7]
    assert len(store.load_calls(conn, mode="forced", temperature=0.7)) == 4


def test_old_database_is_migrated_with_temperature_zero(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE calls (id INTEGER PRIMARY KEY, model TEXT, method TEXT, item_id TEXT, variant TEXT, mode TEXT,
            testset TEXT, prompt_version TEXT, run_id TEXT, messages TEXT, response TEXT, parsed TEXT,
            refused INTEGER, cost REAL, prompt_tokens INTEGER, completion_tokens INTEGER, provider TEXT,
            model_returned TEXT, latency_s REAL, created TEXT, finish_reason TEXT,
            UNIQUE (model, method, item_id, variant, mode, testset, prompt_version));
        INSERT INTO calls (model, method, item_id, variant, mode, testset, prompt_version, parsed, refused, cost)
        VALUES ('m', 'price_recall', 'i1', 'main', 'forced', 'price_recall/v1', 'p1', '{}', 0, 0.1);
    """)
    old.commit()
    old.close()
    conn = store.connect(path)
    calls = store.load_calls(conn)
    assert len(calls) == 1 and calls["temperature"].iloc[0] == 0.0
    assert ("m", "price_recall", "i1", "main", "forced", 0.0, "auto", -1, 0, "price_recall/v1", "p1") \
        in store.cached_keys(conn)


def test_saved_selection_keeps_temperature(tmp_path):
    path = tmp_path / "s.yaml"
    selections.save("warm", ["a/m"], ["price_recall"], 10, "forced", temperature=0.7, path=path)
    assert selections.load(path)["warm"]["temperature"] == 0.7
    path.write_text("old:\n  models: [a/m]\n  methods: [decop]\n  n: 5\n  mode: forced\n", encoding="utf-8")
    assert selections.load(path)["old"]["temperature"] == 0.0  # entries saved before the setting existed


def test_temperature_range_is_validated(tmp_path):
    with pytest.raises(ValueError):
        selections.save("hot", ["a/m"], ["decop"], 5, "forced", temperature=3.0, path=tmp_path / "s.yaml")
