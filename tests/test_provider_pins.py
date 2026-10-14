"""Open-weight models can be pinned to one provider endpoint, so each model in the results is one exact
configuration (the same weights at the same precision), not a mix of providers."""
import sqlite3

import pytest

from evalbench import runner, store
from evalbench.llm import openrouter as orr
from evalbench.llm.openrouter import ChatResult
from evalbench.methods import price_recall

PINS = {"x/open": "novita/bf16"}


def _ts(data, n=2):
    return {"price_recall": ("price_recall/t", price_recall.build_items(data, n, 1))}


class Recorder:
    def __init__(self, provider="Novita"):
        self.bodies, self.provider = [], provider

    async def chat(self, body):
        self.bodies.append(body)
        return ChatResult(text='{"price": 1}', cost=0.0, prompt_tokens=1, completion_tokens=1,
                          provider=self.provider, model=body["model"], finish_reason="stop")


def test_a_pinned_request_goes_only_to_that_endpoint():
    body = orr.build_body({}, "x/open", [], 60, route="novita/bf16")
    assert body["provider"] == {"require_parameters": True, "only": ["novita/bf16"], "allow_fallbacks": False}
    assert "only" not in orr.build_body({}, "x/open", [], 60)["provider"]


def test_pins_are_read_from_config(tmp_path):
    path = tmp_path / "pins.yaml"
    path.write_text("# comment\nx/open: novita/bf16\n", encoding="utf-8")
    assert orr.load_pins(path) == PINS
    assert orr.load_pins(tmp_path / "missing.yaml") == {}


def test_plan_pins_only_the_models_listed(data):
    calls = runner.plan(["x/open", "x/closed", "mock/random@2024-06-30"], ["price_recall"], 1, "forced", _ts(data),
                        pins=PINS)
    assert {c.model: c.route for c in calls} == {"x/open": "novita/bf16", "x/closed": "", "mock/random@2024-06-30": ""}


def test_an_answer_from_another_route_does_not_count_for_a_pinned_call(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    unpinned = runner.plan(["x/open"], ["price_recall"], 2, "forced", _ts(data))
    runner.run(unpinned, conn, {}, client=Recorder("SomeHost"))
    pinned = runner.plan(["x/open"], ["price_recall"], 2, "forced", _ts(data), pins=PINS)
    assert len(runner.pending(pinned, conn)) == 2          # mixed-provider answers are not reused
    assert runner.pending(unpinned, conn) == []            # an unpinned selection still reuses them

    rec = Recorder()
    runner.run(pinned, conn, {}, client=rec)
    assert all(b["provider"]["only"] == ["novita/bf16"] for b in rec.bodies) and len(rec.bodies) == 2
    stored = store.load_calls(conn)
    assert len(stored) == 2 and set(stored["route"]) == {"novita/bf16"}
    assert runner.pending(pinned, conn) == []
    # the replaced answers are archived, never deleted
    archived = conn.execute("SELECT provider, route FROM calls_superseded").fetchall()
    assert len(archived) == 2 and {a[0] for a in archived} == {"SomeHost"}


def test_ceiling_uses_the_pinned_endpoints_price():
    endpoints = [{"tag": "novita/bf16", "pricing": {"prompt": "0.00000014", "completion": "0.0000004"}},
                 {"tag": "cloudflare/fp8", "pricing": {"prompt": "0.0000003", "completion": "0.00000225"}}]
    meta = {"x/open": {"pricing": {"prompt": "0.0000001", "completion": "0.00000032"}}}
    priced, info = orr.with_worst_case_prices(meta, ["x/open"], fetch=lambda m: endpoints, pins=PINS)
    assert priced["x/open"]["pricing"]["completion"] == pytest.approx(4e-7)
    assert info["x/open"]["providers"] == 1 and info["x/open"]["route"] == "novita/bf16"
    # a base slug matches every endpoint of that provider
    assert orr.route_matches("deepinfra/turbo", "deepinfra") and not orr.route_matches("novita/bf16", "deepinfra")


def test_a_pin_that_matches_no_current_endpoint_is_reported(data):
    meta = {"x/open": {"pricing": {"prompt": "0.0000001", "completion": "0.00000032"}}}
    _, info = orr.with_worst_case_prices(meta, ["x/open"], fetch=lambda m: [{"tag": "other", "pricing": {}}], pins=PINS)
    assert not info["x/open"]["ok"] and info["x/open"]["route"] == "novita/bf16"


def test_older_databases_gain_the_route_column(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE calls (id INTEGER PRIMARY KEY, model TEXT, method TEXT, item_id TEXT, variant TEXT, mode TEXT,
            temperature REAL NOT NULL DEFAULT 0, reasoning TEXT NOT NULL DEFAULT 'auto',
            seed INTEGER NOT NULL DEFAULT -1, sample INTEGER NOT NULL DEFAULT 0, testset TEXT, prompt_version TEXT,
            privacy TEXT, run_id TEXT, messages TEXT, response TEXT, parsed TEXT, refused INTEGER, cost REAL,
            prompt_tokens INTEGER, completion_tokens INTEGER, provider TEXT, model_returned TEXT, latency_s REAL,
            created TEXT, finish_reason TEXT,
            UNIQUE (model, method, item_id, variant, mode, temperature, reasoning, seed, sample, testset,
                    prompt_version));
        INSERT INTO calls (model, method, item_id, variant, mode, testset, prompt_version, parsed, provider, privacy)
        VALUES ('x/open', 'price_recall', 'i1', 'main', 'forced', 't', 'p1', '{}', 'SomeHost', 'standard');
    """)
    old.commit()
    old.close()
    conn = store.connect(path)
    assert store.load_calls(conn)["route"].tolist() == [""]      # earlier answers used default routing
    assert conn.execute("SELECT count(*) FROM calls_superseded").fetchone()[0] == 0
