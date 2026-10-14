"""Regression tests for the review of the run-settings change (e34c693..4477c0c)."""
import json
import sqlite3
from datetime import date

import pandas as pd
import pytest

from evalbench import analysis, runner, store
from evalbench.items import Item
from evalbench.llm import openrouter as orr
from evalbench.llm.openrouter import ChatResult
from evalbench.methods import decop, number_cloze, price_recall
from evalbench.textutil import classify_non_answer

OPTIONAL_ON = {"supported_parameters": ["reasoning"], "reasoning": {"mandatory": False,
                                                                    "supported_efforts": ["low", "medium", "high"]}}


# 1: "no reasoning" must actually switch reasoning off on models where it is optional but on by default
@pytest.mark.parametrize("requested", ["auto", "none"])
def test_optional_reasoning_is_switched_off_explicitly(requested):
    body = orr.build_body(OPTIONAL_ON, "x/r", [], 60, reasoning=requested)
    assert body["reasoning"] == {"enabled": False} and body["max_tokens"] == 60


def test_max_tokens_never_exceeds_the_endpoint_limit():
    meta = {"supported_parameters": ["reasoning"], "top_provider": {"max_completion_tokens": 4000},
            "reasoning": {"mandatory": True, "supported_efforts": ["high"]}}
    assert orr.build_body(meta, "x/r", [], 60, reasoning="high")["max_tokens"] == 4000


class Truncating:
    async def chat(self, body):
        return ChatResult(text="", cost=0.01, prompt_tokens=10, completion_tokens=60, provider="p",
                          model=body["model"], finish_reason="length")


def test_reply_cut_off_before_any_answer_is_an_error_not_a_cached_answer(data, tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["x/m"], ["price_recall"], 2, "forced",
                        {"price_recall": ("t", price_recall.build_items(data, 2, 1))})
    _, state = runner.run(calls, conn, {}, client=Truncating())
    assert state.done == 0 and len(state.errors) == 2 and "cut off" in state.errors[0]
    assert state.spent == pytest.approx(0.02)          # it was billed, so it counts against the cap
    assert len(runner.pending(calls, conn)) == 2       # and a re-run retries it


# 2: DE-COP letters written with punctuation or a label
@pytest.mark.parametrize("value", ["B.", "B)", "b) The second", "B: Nokia", "(B)", "Option B", " b "])
def test_decop_json_letter_variants(value):
    parsed = decop.parse(json.dumps({"answer": value}))
    assert parsed["letter"] == "B" and parsed["outcome"] == "answer"


@pytest.mark.parametrize("value", ["E", "", "Both"])
def test_decop_json_non_letters(value):
    assert decop.parse(json.dumps({"answer": value}))["outcome"] != "answer"


# 3: prices (and masked numbers) that look like years
@pytest.mark.parametrize("text,value", [("It closed at 2010.", 2010.0), ("Closing price: 1999 SEK", 1999.0)])
def test_year_like_prices_in_prose(text, value):
    parsed = price_recall.parse(text)
    assert parsed["outcome"] == "answer" and parsed["value"] == value


def test_year_like_text_that_declines_is_still_a_non_answer():
    assert price_recall.parse("I'm sorry, I can't provide 2023 prices.")["outcome"] == "refusal"
    assert number_cloze.parse("I don't know the 2021 figure")["outcome"] == "abstain"
    assert number_cloze.parse("The number was 2040")["answer"] == "2040"


# 4: best of k only over items that have all k samples; headline only from sample 0
def _rows(n_items, samples, right_sample):
    rows, items = [], {}
    for k in range(n_items):
        d = "2021-06-01" if k % 2 == 0 else "2025-06-01"
        items[f"i{k}"] = Item(f"i{k}", "price_recall", d, f"c{k}", {}, {"close": 100.0, "adj_close": 100.0})
        for s in range(samples(k)):
            right = d < "2023" and s == right_sample
            rows.append({"model": "m", "method": "price_recall", "item_id": f"i{k}", "variant": "main", "sample": s,
                         "testset": "t", "response": "",
                         "parsed": json.dumps({"value": 100.0 if right else 150.0, "refused": False,
                                               "outcome": "answer"})})
    return pd.DataFrame(rows), items


def test_best_of_k_uses_only_items_with_all_k_samples():
    calls, items = _rows(60, lambda k: 3 if k < 20 else 1, right_sample=2)
    df = analysis.item_scores(calls, items, "price_recall", date(2023, 6, 30))
    s = analysis.summarize(df, "price_recall", B=50)
    assert s["n_samples"] == 3 and s["n_best"] == 20
    assert s["dkappa_best"] == pytest.approx(1.0)      # computed on the 20 items that had 3 attempts


def test_headline_comes_only_from_sample_zero():
    calls, items = _rows(4, lambda k: 3, right_sample=0)
    calls = calls[~((calls["item_id"] == "i0") & (calls["sample"] == 0))]   # sample 0 missing for i0
    df = analysis.item_scores(calls, items, "price_recall", date(2023, 6, 30))
    assert "i0" not in set(df["item_id"])


# 5: a stored parse failure must not crash the analysis
def test_stored_parse_failure_does_not_crash_analysis(monkeypatch):
    items = {"i0": Item("i0", "price_recall", "2021-06-01", "c0", {}, {"close": 100.0, "adj_close": 100.0})}
    calls = pd.DataFrame([{"model": "m", "method": "price_recall", "item_id": "i0", "variant": "main", "sample": 0,
                           "testset": "t", "response": "boom",
                           "parsed": json.dumps({"refused": True, "parse_error": "x"})}])

    def explode(text):
        raise RuntimeError("parser bug")

    monkeypatch.setattr(price_recall, "parse", explode)
    df = analysis.item_scores(calls, items, "price_recall", date(2023, 6, 30))
    assert df["outcome"].iloc[0] == "malformed"


def test_runner_records_malformed_when_parse_raises(data, tmp_path, monkeypatch):
    conn = store.connect(tmp_path / "t.sqlite")
    ts = {"price_recall": ("t", price_recall.build_items(data, 1, 1))}
    monkeypatch.setattr(price_recall, "parse", lambda text: 1 / 0)
    runner.run(runner.plan(["mock/random@2024-06-30"], ["price_recall"], 1, "forced", ts), conn, {})
    assert json.loads(store.load_calls(conn)["parsed"].iloc[0])["outcome"] == "malformed"


# minor: migration atomic + privacy backfilled; settings listed per answer mode; N/A abstains
def test_migration_cleans_up_a_half_finished_rebuild_and_backfills_privacy(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE calls (id INTEGER PRIMARY KEY, model TEXT, method TEXT, item_id TEXT, variant TEXT, mode TEXT,
            testset TEXT, prompt_version TEXT, parsed TEXT, provider TEXT, cost REAL,
            UNIQUE (model, method, item_id, variant, mode, testset, prompt_version));
        CREATE TABLE calls_new (id INTEGER PRIMARY KEY);
        INSERT INTO calls_new (id) VALUES (1);
        INSERT INTO calls (id, model, method, item_id, variant, mode, testset, prompt_version, parsed, provider)
        VALUES (1, 'mock/oracle@2024-06-30', 'price_recall', 'i1', 'main', 'forced', 't', 'p1', '{}', 'local'),
               (2, 'x/m', 'price_recall', 'i1', 'main', 'forced', 't', 'p1', '{}', 'SomeHost');
    """)
    old.commit()
    old.close()
    calls = store.load_calls(store.connect(path)).set_index("model")
    assert len(calls) == 2
    assert calls.loc["mock/oracle@2024-06-30", "privacy"] == "local" and calls.loc["x/m", "privacy"] == "standard"


def test_run_settings_can_be_listed_per_mode(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    base = {"model": "m", "method": "price_recall", "item_id": "i", "variant": "main", "testset": "t",
            "prompt_version": "p1", "parsed": "{}"}
    store.insert_call(conn, {**base, "mode": "forced", "temperature": 0.7})
    store.insert_call(conn, {**base, "mode": "abstain", "temperature": 0.0})
    assert store.run_settings(conn, mode="forced") == [(0.7, "auto", -1)]


def test_na_is_an_abstention():
    assert classify_non_answer("N/A") == "abstain"
    assert classify_non_answer("n/a - no data") == "abstain"


def test_seed_plus_sample_stays_within_32_bits(data):
    with pytest.raises(ValueError):
        runner.plan(["x/m"], ["price_recall"], 1, "forced", {"price_recall": ("t", [])}, seed=2**31 - 1, samples=3)
