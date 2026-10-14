import pytest

from evalbench.items import Item
from evalbench.methods import completion as cp


def test_split_prefix_cuts_at_sentence_end():
    text = "First sentence is here and long enough to count. Second sentence follows. " + "x " * 300
    prefix, rest = cp.split_prefix(text)
    assert prefix.endswith("Second sentence follows.")
    assert rest.startswith("x")


def test_build_items(data):
    items = cp.build_items(data, 5, seed=1)
    assert len(items) == 5
    for it in items:
        assert it.payload["prefix"] and 300 <= len(it.truth["reference"]) <= 500


def _item():
    return Item("cp-0", "completion", "2022-02-01", "c1",
                {"headline": "H", "prefix": "Company 1 reported net sales of SEK 1037 million.", "distributor": "Cision",
                 "company": "Company 1 B", "source_group": "press_release", "language": "en"},
                {"reference": "Operating profit amounted to SEK 150 million and the order intake grew."})


def test_render_guided_vs_general():
    guided = cp.render(_item(), "abstain", "guided")[-1]["content"]
    general = cp.render(_item(), "abstain", "general")[-1]["content"]
    assert "Cision" in guided and "2022-02-01" in guided and "UNKNOWN" in guided
    assert "Cision" not in general and "2022-02-01" not in general


def test_parse_empty_is_refusal():
    assert cp.parse("")["refused"]
    assert cp.parse("UNKNOWN")["refused"]
    assert cp.parse("I'm sorry, I cannot reproduce that.")["refused"]
    assert not cp.parse("Operating profit rose.")["refused"]


def test_item_score_and_echo_strip():
    it = _item()
    ref = it.truth["reference"]
    parsed = {"guided": {"text": it.payload["prefix"] + " " + ref, "refused": False},
              "general": {"text": "", "refused": True}}
    s = cp.item_score(it, parsed)
    assert s.score == pytest.approx(1.0)
    assert s.extra["rouge_guided"] == pytest.approx(1.0) and s.extra["number_recall"] == 1.0


def test_guided_refusal_scores_zero_not_negative():
    it = _item()
    parsed = {"guided": {"text": "UNKNOWN", "refused": True},
              "general": {"text": it.truth["reference"], "refused": False}}
    s = cp.item_score(it, parsed)
    assert s.score == 0.0 and s.refused
