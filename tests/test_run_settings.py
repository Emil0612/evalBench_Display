"""Run settings: data privacy routing, reasoning effort, samples per item and seed."""
import json
import sqlite3
from datetime import date

import pandas as pd
import pytest

from evalbench import analysis, runner, selections, store
from evalbench.items import Item
from evalbench.llm import openrouter as orr
from evalbench.llm.openrouter import ChatResult
from evalbench.methods import price_recall

REASONER = {"supported_parameters": ["reasoning", "temperature", "seed"],
            "reasoning": {"mandatory": False, "supported_efforts": ["low", "medium", "high"]},
            "pricing": {"prompt": "0.000001", "completion": "0.00001"}}
MANDATORY = {"supported_parameters": ["reasoning"], "reasoning": {"mandatory": True, "supported_efforts": ["low", "high"]}}


# ---------------------------------------------------------------- privacy
def test_privacy_standard_routing_by_default():
    assert orr.build_body({}, "x/m", [], 60)["provider"] == {"require_parameters": True}
    assert orr.PRIVACY[0] == "standard"   # listed first, so the Run page offers it first


def test_privacy_levels():
    assert orr.build_body({}, "x/m", [], 60, privacy="no_training")["provider"] == \
        {"require_parameters": True, "data_collection": "deny"}
    assert orr.build_body({}, "x/m", [], 60, privacy="zdr")["provider"] == \
        {"require_parameters": True, "zdr": True, "data_collection": "deny"}
    with pytest.raises(ValueError):
        orr.build_body({}, "x/m", [], 60, privacy="loose")


# ---------------------------------------------------------------- reasoning
def test_reasoning_auto_keeps_old_behaviour():
    assert orr.build_body(REASONER, "x/r", [], 60)["reasoning"] == {"enabled": False}  # optional reasoning off
    assert orr.build_body(MANDATORY, "x/r", [], 60)["reasoning"]["effort"] == "low"


def test_reasoning_effort_is_sent_with_matching_headroom():
    body = orr.build_body(REASONER, "x/r", [], 60, reasoning="high")
    assert body["reasoning"] == {"effort": "high", "exclude": True}
    assert body["max_tokens"] == 60 + orr.REASONING_HEADROOM["high"]
    assert orr.REASONING_HEADROOM["high"] > orr.REASONING_HEADROOM["low"]


def test_reasoning_none_on_a_model_that_must_reason_uses_its_lowest_effort():
    assert orr.build_body(MANDATORY, "x/r", [], 60, reasoning="none")["reasoning"]["effort"] == "low"


def test_reasoning_not_sent_to_models_without_it():
    assert "reasoning" not in orr.build_body({"supported_parameters": ["temperature"]}, "x/m", [], 60, reasoning="high")


# ---------------------------------------------------------------- seed
def test_seed_only_when_supported():
    assert orr.build_body(REASONER, "x/r", [], 60, seed=42)["seed"] == 42
    assert "seed" not in orr.build_body({"supported_parameters": []}, "x/m", [], 60, seed=42)
    assert "seed" not in orr.build_body(REASONER, "x/r", [], 60)


# ---------------------------------------------------------------- plan, key, runner
def _ts(data, n=3):
    return {"price_recall": ("price_recall/t", price_recall.build_items(data, n, 1))}


def test_samples_multiply_calls_with_distinct_keys(data):
    calls = runner.plan(["x/m"], ["price_recall"], 3, "forced", _ts(data), samples=3)
    assert len(calls) == 9 and sorted({c.sample for c in calls}) == [0, 1, 2]
    assert len({c.key for c in calls}) == 9


class Recorder:
    def __init__(self):
        self.bodies = []

    async def chat(self, body):
        self.bodies.append(body)
        return ChatResult(text='{"price": 1}', cost=0.0, prompt_tokens=1, completion_tokens=1, provider="p",
                          model=body["model"], finish_reason="stop")


def test_each_sample_gets_its_own_seed_and_privacy_is_recorded(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    rec = Recorder()
    meta = {"x/m": {"supported_parameters": ["seed", "temperature"]}}
    calls = runner.plan(["x/m"], ["price_recall"], 1, "forced", _ts(data), temperature=0.7, seed=100, samples=3)
    runner.run(calls, conn, meta, client=rec, privacy="no_training")
    assert sorted(b["seed"] for b in rec.bodies) == [100, 101, 102]
    assert all(b["temperature"] == 0.7 and b["provider"]["data_collection"] == "deny" for b in rec.bodies)
    stored = store.load_calls(conn)
    assert set(stored["privacy"]) == {"no_training"} and sorted(stored["sample"]) == [0, 1, 2]


def test_runner_routes_standard_by_default(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    rec = Recorder()
    runner.run(runner.plan(["x/m"], ["price_recall"], 1, "forced", _ts(data)), conn, {}, client=rec)
    assert rec.bodies[0]["provider"] == {"require_parameters": True}
    assert set(store.load_calls(conn)["privacy"]) == {"standard"}


def test_other_reasoning_or_seed_is_not_reused(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    base = runner.plan(["mock/random@2024-06-30"], ["price_recall"], 3, "forced", _ts(data))
    runner.run(base, conn, {})
    for kw in ({"reasoning": "high"}, {"seed": 7}):
        other = runner.plan(["mock/random@2024-06-30"], ["price_recall"], 3, "forced", _ts(data), **kw)
        assert len(runner.pending(other, conn)) == 3, kw
    assert runner.pending(base, conn) == []


def test_estimate_grows_with_reasoning_effort(data):
    meta = {"x/r": REASONER}
    low = runner.estimate_cost(runner.plan(["x/r"], ["price_recall"], 3, "forced", _ts(data)), meta)
    high = runner.estimate_cost(runner.plan(["x/r"], ["price_recall"], 3, "forced", _ts(data), reasoning="high"), meta)
    assert high > low > 0


def test_database_from_before_these_settings_is_migrated(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE calls (id INTEGER PRIMARY KEY, model TEXT, method TEXT, item_id TEXT, variant TEXT, mode TEXT,
            temperature REAL NOT NULL DEFAULT 0, testset TEXT, prompt_version TEXT, run_id TEXT, messages TEXT,
            response TEXT, parsed TEXT, refused INTEGER, cost REAL, prompt_tokens INTEGER, completion_tokens INTEGER,
            provider TEXT, model_returned TEXT, latency_s REAL, created TEXT, finish_reason TEXT,
            UNIQUE (model, method, item_id, variant, mode, temperature, testset, prompt_version));
        INSERT INTO calls (model, method, item_id, variant, mode, temperature, testset, prompt_version, parsed)
        VALUES ('m', 'price_recall', 'i1', 'main', 'forced', 0.7, 'price_recall/v1', 'p1', '{}');
    """)
    old.commit()
    old.close()
    row = store.load_calls(store.connect(path)).iloc[0]
    assert (row["temperature"], row["reasoning"], row["seed"], row["sample"]) == (0.7, "auto", -1, 0)


# ---------------------------------------------------------------- first attempt vs best of k
def test_first_attempt_and_best_of_k_are_reported_separately():
    items = {f"i{k}": Item(f"i{k}", "price_recall", "2021-06-01" if k < 20 else "2025-06-01", f"c{k}", {},
                           {"close": 100.0, "adj_close": 100.0}) for k in range(40)}
    rows = []
    for k in range(40):
        for s in range(3):
            right = k < 20 and s == 2  # before the cutoff: only the third attempt is right
            parsed = {"value": 100.0 if right else 150.0, "refused": False, "outcome": "answer"}
            rows.append({"model": "m", "method": "price_recall", "item_id": f"i{k}", "variant": "main", "sample": s,
                         "testset": "t", "parsed": json.dumps(parsed), "response": ""})
    df = analysis.item_scores(pd.DataFrame(rows), items, "price_recall", date(2023, 6, 30))
    assert len(df) == 40 and set(df["n_samples"]) == {3}
    first = df[df["item_id"] == "i0"].iloc[0]
    assert first["score"] == 0.0 and first["score_best"] == 1.0
    s = analysis.summarize(df, "price_recall", B=50)
    assert s["dkappa"] == 0.0 and s["dkappa_best"] == pytest.approx(1.0)


# ---------------------------------------------------------------- saved selections
def test_selections_keep_all_run_settings(tmp_path):
    path = tmp_path / "s.yaml"
    selections.save("full", ["a/m"], ["price_recall"], 10, "forced", temperature=0.7, reasoning="high", samples=3,
                    seed=42, privacy="no_training", path=path)
    got = selections.load(path)["full"]
    assert (got["reasoning"], got["samples"], got["seed"], got["privacy"]) == ("high", 3, 42, "no_training")
    path.write_text("old:\n  models: [a/m]\n  methods: [decop]\n  n: 5\n  mode: forced\n", encoding="utf-8")
    old = selections.load(path)["old"]
    assert (old["reasoning"], old["samples"], old["seed"], old["privacy"]) == ("auto", 1, None, "standard")
    with pytest.raises(ValueError):
        selections.save("bad", ["a/m"], ["decop"], 5, "forced", samples=0, path=path)
