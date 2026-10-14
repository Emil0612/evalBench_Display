"""Every answer is one of four outcomes; abstain, refusal and malformed stay distinct."""
import json
from datetime import date

import pandas as pd
import pytest

from evalbench import analysis
from evalbench.items import Item
from evalbench.methods import completion, decop, number_cloze, price_recall
from evalbench.textutil import classify_non_answer


@pytest.mark.parametrize("text,outcome", [
    ("I don't know.", "abstain"),
    ("I'm not sure what the price was.", "abstain"),
    ("I do not have that information.", "abstain"),
    ("Jag vet inte.", "abstain"),
    ("I'm sorry, but I can't help with that request.", "refusal"),
    ("I cannot provide financial data.", "refusal"),
    ("", "malformed"),
    ("   ", "malformed"),
    ("{broken json", "malformed"),
    ("Banana.", "malformed"),
])
def test_classify_non_answer(text, outcome):
    assert classify_non_answer(text) == outcome


@pytest.mark.parametrize("mod,text,outcome", [
    (price_recall, '{"price": 210.5}', "answer"),
    (price_recall, '{"price": null}', "abstain"),
    (price_recall, '{"price": "about a lot"}', "malformed"),
    (price_recall, "I'm sorry, I can't provide stock prices.", "refusal"),
    (number_cloze, '{"answer": "1 234"}', "answer"),
    (number_cloze, '{"answer": null}', "abstain"),
    (number_cloze, "I don't know", "abstain"),
    (decop, '{"answer": "B"}', "answer"),
    (decop, '{"answer": null}', "abstain"),
    (decop, '{"answer": "E"}', "malformed"),
    (completion, "Operating profit rose.", "answer"),
    (completion, "UNKNOWN", "abstain"),
    (completion, "I'm sorry, I cannot reproduce that text.", "refusal"),
    (completion, "", "malformed"),
])
def test_parse_reports_outcome(mod, text, outcome):
    parsed = mod.parse(text)
    assert parsed["outcome"] == outcome
    assert parsed["refused"] is (outcome != "answer")  # scoring still treats every non-answer as 0


def _calls(outcomes):
    rows, items = [], {}
    for k, out in enumerate(outcomes):
        d = "2021-06-01" if k % 2 == 0 else "2025-06-01"
        items[f"i{k}"] = Item(f"i{k}", "price_recall", d, f"c{k}", {}, {"close": 100.0, "adj_close": 100.0})
        parsed = {"value": 100.0 if out == "answer" else None, "refused": out != "answer", "outcome": out}
        rows.append({"model": "m", "method": "price_recall", "item_id": f"i{k}", "variant": "main",
                     "testset": "t", "parsed": json.dumps(parsed), "response": ""})
    return pd.DataFrame(rows), items


def test_item_scores_and_summary_keep_outcomes_apart():
    calls, items = _calls(["answer", "abstain", "refusal", "malformed", "abstain", "answer"])
    df = analysis.item_scores(calls, items, "price_recall", date(2023, 6, 30))
    assert list(df["outcome"]) == ["answer", "abstain", "refusal", "malformed", "abstain", "answer"]
    s = analysis.summarize(df, "price_recall", B=20)
    assert s["abstain_rate"] == pytest.approx(2 / 6)
    assert s["refusal_rate"] == pytest.approx(1 / 6)
    assert s["malformed_rate"] == pytest.approx(1 / 6)


def test_old_rows_without_outcome_are_reparsed_from_the_response():
    calls, items = _calls(["answer"])
    calls.loc[0, "parsed"] = json.dumps({"value": None, "refused": True})
    calls.loc[0, "response"] = "I'm sorry, I can't provide that."
    df = analysis.item_scores(calls, items, "price_recall", date(2023, 6, 30))
    assert df["outcome"].iloc[0] == "refusal"


def test_decop_item_outcome_is_answer_if_any_rotation_answered():
    it = Item("dc", "decop", "2022-01-01", "c", {}, {"passage": "O", "paraphrases": ["a", "b", "c"]})
    parsed = {v: {"letter": None, "refused": True, "outcome": "abstain"} for v in decop.VARIANTS}
    assert analysis.item_outcome("decop", parsed) == "abstain"
    parsed["r2"] = {"letter": "C", "refused": False, "outcome": "answer"}
    assert analysis.item_outcome("decop", parsed) == "answer"
    assert it  # item itself is not needed for the outcome
