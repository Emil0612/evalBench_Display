from evalbench.items import Item
from evalbench.methods import number_cloze as nc


def test_find_mask_candidates():
    text = "Sales were SEK 1 234 million in 2023, see page 450. Margin 12%."
    got = [m.group() for m in nc.find_mask_candidates(text, "Q3 report")]
    assert got == ["1 234"]


def test_build_items(data):
    items = nc.build_items(data, 10, seed=1)
    assert len(items) == 10
    for it in items:
        ctx = it.payload["context"]
        assert ctx.count("[MASK]") == 1
        assert it.truth["number"] not in ctx
        assert it.truth["number"] in it.truth["original"]
        assert it.truth["newsid"] != "tpl"
        assert it.payload["language"] in ("sv", "en")


def test_render_contains_context():
    it = Item("nc-0", "number_cloze", "2023-01-01", "c1",
              {"headline": "H", "distributor": "Cision", "company": "Volvo B", "language": "sv",
               "context": "sales of SEK [MASK] million", "cut_start": True, "cut_end": False}, {})
    msg = nc.render(it, "abstain", "main")[-1]["content"]
    assert "…sales of SEK [MASK] million" in msg and "null" in msg and "about Volvo B" in msg


def test_parse():
    assert nc.parse('{"answer": "1 234"}') == {"answer": "1 234", "refused": False, "outcome": "answer"}
    assert nc.parse('{"answer": 1234}') == {"answer": "1234", "refused": False, "outcome": "answer"}
    assert nc.parse('{"answer": null}')["refused"]
    assert nc.parse("It was 1 234 I think")["answer"] == "1 234"
    assert nc.parse("no idea")["refused"]


def test_item_score():
    it = Item("nc-0", "number_cloze", "2023-01-01", "c1", {"language": "sv"}, {"number": "1 234,5"})
    assert nc.item_score(it, {"main": {"answer": "1234.5", "refused": False}}).score == 1.0
    assert nc.item_score(it, {"main": {"answer": "1234", "refused": False}}).score == 0.0
    assert nc.item_score(it, {"main": {"answer": None, "refused": True}}).refused


def test_find_mask_candidates_skips_split_numbers():
    text = "Result was -1 730 288 kr while profit reached SEK 450 million."
    assert [m.group() for m in nc.find_mask_candidates(text, "Q3")] == ["450"]


def test_find_mask_candidates_skips_table_rows():
    text = "| Sales, MEUR | 44.3 | 33.5 | 32.1 % |\nNet sales rose to SEK 900 million."
    assert [m.group() for m in nc.find_mask_candidates(text, "Q3")] == ["900"]


def test_prose_fallback_skips_years():
    assert nc.parse("In 2023 the figure was 1 234")["answer"] == "1 234"


# ---------------------------------------------------------------- v2: numbers that can't be worked out from the text
def test_derived_numbers_are_not_masked():
    growth = "Sales rose to SEK 1 150 million from SEK 1 000 million, an increase of 15,0 percent to a record."
    total = "Region North had sales of SEK 521 million and Region South SEK 516 million, SEK 1 037 million in total."
    share = "Exports were SEK 312 million of total sales of SEK 1 248 million, or 25,0 percent."
    for text, derived in ((growth, "15,0"), (total, "1 037"), (share, "25,0")):
        got = [m.group() for m in nc.find_mask_candidates(text, "H", lang="sv")]
        assert derived not in got, (text, got)


def test_independent_numbers_stay_maskable():
    text = "Sales were SEK 1 234 million and the order intake SEK 987 million; operating profit was SEK 143 million."
    assert set(m.group() for m in nc.find_mask_candidates(text, "H", lang="en")) == {"1 234", "987", "143"}


def test_inferable_helper():
    assert nc.inferable(15.0, 1, [1150.0, 1000.0])          # growth rate
    assert nc.inferable(1037.0, 0, [521.0, 516.0])          # sum
    assert nc.inferable(10.0, 0, [1000.0, 100.0]) is True   # share in percent
    assert not nc.inferable(143.0, 0, [1234.0, 987.0])
