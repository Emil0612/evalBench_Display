"""Probability probes (M12p/M13p): the same multiple choice, scored by the probability the model gives the correct
letter (from the provider's top token log-probabilities), averaged over the four orderings."""
import json
import math
import sqlite3

import pytest

from evalbench import runner, store
from evalbench.items import Item
from evalbench.llm.openrouter import ChatResult, build_body
from evalbench.methods import LABELS, MAIN_METHODS, METHODS, PROBES
from evalbench.methods import price_choice as pc
from evalbench.methods import price_choice_prob as pcp


def tok(token, lp, top=()):
    return {"token": token, "logprob": lp, "top_logprobs": [{"token": t, "logprob": v} for t, v in top]}


ANSWER_B = [tok('{"', 0.0), tok("answer", 0.0), tok('":', 0.0), tok(' "', 0.0),
            tok("B", math.log(0.5), [("B", math.log(0.5)), ("C", math.log(0.3)), (" A", math.log(0.1)),
                                     ("D", math.log(0.05)), ("Sorry", math.log(0.05))]),
            tok('"}', 0.0)]


def test_letter_probabilities_are_read_at_the_answer_token_and_renormalised():
    lp = pc.letter_probs(ANSWER_B)
    assert lp["probs"] == pytest.approx({"A": 0.1 / 0.95, "B": 0.5 / 0.95, "C": 0.3 / 0.95, "D": 0.05 / 0.95})
    assert lp["mass"] == pytest.approx(0.95)
    assert pc.letter_probs(None) is None and pc.letter_probs([tok("Hello", 0.0)]) is None


def _item():
    return Item("pcp-0000", "price_choice_prob", "2023-06-30", "c1",
                {"name": "X", "symbol": "X", "currency": "SEK", "month": "2023-06", "month_name": "June 2023",
                 "answer_text": "100", "decoy_texts": ["140", "75", "60"]},
                {"level": 100.0, "decoy_levels": [140.0, 75.0, 60.0], "baseline_level": 120.0})


def test_the_probe_scores_the_mean_probability_of_the_correct_letter():
    parsed = {}
    for k, v in enumerate(pc.VARIANTS):
        probs = {letter: (0.4 if letter == pc.LETTERS[k] else 0.2) for letter in pc.LETTERS}
        parsed[v] = {"letter": "A", "refused": False, "outcome": "answer", "letter_probs": {"probs": probs, "mass": 1}}
    sc = pcp.item_score(_item(), parsed)
    assert sc.score == pytest.approx(0.4) and sc.extra["from_logprobs"] == 4
    assert sc.extra["n_correct"] == 1                   # the chosen letters: "A" is right only in ordering r0


def test_without_probabilities_the_chosen_letter_counts_fully():
    parsed = {v: {"letter": pc.correct_letter(v) if v != "r3" else "A", "refused": False, "outcome": "answer"}
              for v in pc.VARIANTS}
    sc = pcp.item_score(_item(), parsed)
    assert sc.score == pytest.approx(0.75) and sc.extra["from_logprobs"] == 0


def test_probability_requests_only_go_to_models_that_support_them():
    meta = {"supported_parameters": ["temperature", "logprobs", "top_logprobs"]}
    body = build_body(meta, "x/m", [{"role": "user", "content": "q"}], 60, logprobs=5)
    assert body["logprobs"] is True and body["top_logprobs"] == 5
    assert "logprobs" not in build_body({"supported_parameters": ["temperature"]}, "x/m", [], 60, logprobs=5)
    assert "logprobs" not in build_body(meta, "x/m", [], 60)


class LogprobClient:
    def __init__(self):
        self.bodies = []

    async def chat(self, body):
        self.bodies.append(body)
        return ChatResult(text='{"answer": "B"}', cost=0.001, prompt_tokens=10, completion_tokens=5, provider="p",
                          model=body["model"], finish_reason="stop", logprobs=ANSWER_B)


def test_the_runner_requests_stores_and_parses_probabilities(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    ts = {"price_choice_prob": ("price_choice_prob/t", [_item()])}
    meta = {"x/m": {"supported_parameters": ["logprobs", "top_logprobs"]}}
    calls = runner.plan(["x/m"], ["price_choice_prob"], 1, "forced", ts)
    client = LogprobClient()
    runner.run(calls, conn, meta, client=client)
    assert all(b["top_logprobs"] == 5 for b in client.bodies) and len(client.bodies) == 4
    df = store.load_calls(conn)
    assert json.loads(df["parsed"].iloc[0])["letter_probs"]["probs"]["B"] == pytest.approx(0.5 / 0.95)
    assert json.loads(df["logprobs"].iloc[0])[4]["token"] == "B"


def test_older_databases_gain_the_logprobs_column(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    cols = ", ".join(c for c in store.CALL_COLS if c != "logprobs")
    old.executescript(f"CREATE TABLE calls (id INTEGER PRIMARY KEY, {cols}, UNIQUE (model, method, item_id, variant, "
                      "mode, temperature, reasoning, seed, sample, testset, prompt_version));"
                      f"CREATE TABLE calls_superseded (id INTEGER, {cols}, superseded_at TEXT);")
    old.close()
    conn = store.connect(path)
    for table in ("calls", "calls_superseded"):
        assert "logprobs" in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}


def test_probes_are_registered_apart_from_the_main_methods():
    assert LABELS["price_choice_prob"].startswith("M12p") and LABELS["known_choice_prob"].startswith("M13p")
    assert PROBES == {"price_choice_prob", "known_choice_prob"} and not PROBES & set(MAIN_METHODS)
    assert METHODS["price_choice_prob"].LOGPROBS == 5 and METHODS["known_choice_prob"].P0 == 0.25
