"""Fixes from the fresh review of 02aeb34..badb0bf (2026-10-01)."""
import json
import os
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from evalbench import analysis, index_report as ir, market, store
from evalbench.methods import index_level, index_return
from evalbench.reparse import reparse
from evalbench.textutil import classify_non_answer


# 1 (Important): every change answer in a cell refused
def test_a_change_cell_with_every_answer_refused_does_not_crash():
    df = pd.DataFrame({"item_id": ["a", "b", "c", "d"], "date": ["2020-01-31"] * 2 + ["2026-01-30"] * 2,
                       "cluster": ["m2020-01"] * 2 + ["m2026-01"] * 2, "group": [1, 1, 0, 0], "fabricated": False,
                       "refused": True, "outcome": "refusal", "score": 0.0})
    cell = analysis.registered_index_cell(df, "index_return", B=100)
    assert np.isnan(cell["dkappa"]) and cell["coverage_pre"] == 0


# 2 (Important): the app's cache key sees a re-read answer
def test_fingerprint_changes_with_the_database_stamp(monkeypatch):
    from evalbench import ui
    calls = pd.DataFrame({"cost": [0.0]})
    monkeypatch.setattr(ui.store, "db_stamp", lambda conn: (1, 2, 3))
    a = ui.fingerprint(calls, {})
    monkeypatch.setattr(ui.store, "db_stamp", lambda conn: (1, 2, 4))
    assert ui.fingerprint(calls, {}) != a


# 3: "cannot answer with certainty, but ..." is an answer, not a refusal
def test_a_hedged_estimate_is_not_a_refusal():
    assert classify_non_answer("I cannot answer with certainty, but my best estimate is 12.5%") != "refusal"
    assert classify_non_answer("I cannot answer questions about future stock prices.") == "refusal"


# 3b: re-parse does not count old rows stored before outcomes existed as changes
def test_reparse_ignores_rows_stored_before_outcome_labels(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    rec = {"mode": "forced", "variant": "main", "testset": "index_level/v1", "prompt_version": "p1", "run_id": "r",
           "messages": "[]", "cost": 0.0, "model": "m", "method": "index_level", "item_id": "il-0100",
           "response": '{"level": 2214.5}', "parsed": json.dumps({"value": 2214.5, "raw": "2214.5", "refused": False}),
           "refused": 0}
    store.insert_call(conn, rec)
    assert reparse(conn, ["index_level"]) == []


# 4: the key must be the JSON key, not the word in prose
@pytest.mark.parametrize("text", [
    "I'm sorry, I don't have the exact closing level: 2024 data is beyond my training.",
    "The OMXS30 level: 30 large caps",
    '{"confidence_level": 95',
])
def test_a_level_key_in_prose_is_not_an_answer(text):
    assert index_level.parse(text)["value"] is None


def test_a_change_key_in_prose_is_not_an_answer():
    assert index_return.parse("The change percent: about 2024 levels") ["value"] is None


# 5: non-finite numbers and index names inside the answer string
@pytest.mark.parametrize("text", ['{"change_percent": NaN}', '{"change_percent": Infinity}'])
def test_non_finite_change_answers_are_not_answers(text):
    assert index_return.parse(text)["value"] is None


def test_non_finite_level_answers_are_not_answers():
    assert index_level.parse('{"level": NaN}')["value"] is None


def test_an_index_name_inside_the_answer_string_is_not_read_as_digits():
    p = index_level.parse('{"level": "OMXS30 was about 2200"}')
    assert p["value"] == 2200.0


# 6: partial correlation edge cases
def test_partial_with_a_constant_us_move_is_the_plain_correlation():
    m = pd.Period("2025-01", "M")
    df = pd.DataFrame({"month": [m] * 6, "truth": [1.0, 2, 3, 4, 5, 6], "answer": [1.2, 1.9, 3.3, 3.8, 5.1, 6.4]})
    move = pd.Series({m: 2.0})
    assert ir.partial_corr(df, move) == pytest.approx(market.corr(df["answer"], df["truth"]))


def test_partial_with_a_truth_that_never_varies_is_zero():
    months = list(pd.period_range("2020-01", periods=6, freq="M"))
    df = pd.DataFrame({"month": months, "truth": [1.0] * 6, "answer": [1.0, 2, 3, 4, 5, 7]})
    move = pd.Series({m: float(k) for k, m in enumerate(months)})
    assert ir.partial_corr(df, move) == 0.0


# 7: empty bootstrap draws give NaN, not a crash
def test_report_tests_survive_an_empty_side():
    months = list(pd.period_range("2020-01", periods=12, freq="M"))
    df = pd.DataFrame({"month": [m for m in months for _ in range(3)], "truth": np.arange(36.0),
                       "answer": np.arange(36.0) * 0.5})
    r = ir.corr_drop(df, months[-1], months[-1] + 5, B=50)          # no AFTER months
    assert np.isnan(r["lo"]) and np.isnan(r["p"])
    other = df.assign(month=df["month"] + 100)                        # no shared months
    assert np.isnan(ir.paired_corr_diff(df, other, B=50)["p"])
    assert np.isnan(ir.partial_test(df.iloc[0:0], pd.Series(dtype=float), B=50)["p"])
    assert np.isnan(ir._mean_test(pd.Series([], dtype=float), pd.Series([], dtype=str), B=50)[3])


# 8: on the cutoff line the index cells use the plans' whole-month windows around the cutoff
def test_registered_windows_on_the_cutoff_line():
    cut, rel = date(2024, 6, 30), date(2025, 4, 14)
    assert analysis.registered_window(cut, rel, "release") == (cut, rel)
    c, r = analysis.registered_window(cut, rel, "cutoff")
    assert c == cut and r == date(2024, 9, 29)                       # cutoff + 91 days: AFTER from October


# 9: a database file replaced at the same path gets its schema again
def test_a_recreated_database_file_is_set_up_again(tmp_path):
    p = tmp_path / "t.sqlite"
    store.connect(p).close()
    os.remove(p)
    raw = sqlite3.connect(p)
    raw.execute("CREATE TABLE something (a)")
    raw.commit()
    raw.close()
    c = store.connect(p)
    assert c.execute("SELECT count(*) FROM sqlite_master WHERE name = 'reparse_log'").fetchone()[0] == 1
    store.db_stamp(c)


# 10: the refusal guard does not relabel a registered index cell
def test_the_refusal_guard_leaves_registered_cells_alone():
    from evalbench import ui
    row = {"dkappa": 0.16, "p_holm": 0.001, "n_pre": 350, "n_post": 60, "malformed_rate": 0.0, "coverage_pre": 1.0,
           "coverage_post": 0.5, "answered_dkappa": 0.0, "answered_p_holm": 0.9, "registered": "model − blind, before"}
    assert ui.cell_status(row) == "memorization detected"
