import numpy as np

from evalbench.items import Item
from evalbench.methods import decop


def test_pick_passage_length(data):
    rng = np.random.default_rng(0)
    p = decop.pick_passage(data.news.text.iloc[0], rng)
    assert p is not None and 40 <= len(p.split()) <= 80


def test_build_items_with_mock_paraphrases(data):
    items = decop.build_items(data, 8, seed=1, paraphrase_fn=decop.mock_paraphrases)
    assert len(items) == 8
    for it in items:
        assert len(it.truth["paraphrases"]) == 3
        assert it.truth["passage"] not in it.truth["paraphrases"]


def _item():
    return Item("dc-0", "decop", "2022-01-01", "c1",
                {"distributor": "Cision", "company": None, "language": "en", "source_group": "press_release"},
                {"passage": "ORIG", "paraphrases": ["P1", "P2", "P3"]})


def test_options_rotate_true_answer():
    it = _item()
    for r in range(4):
        opts = decop.options(it, f"r{r}")
        assert opts[r] == "ORIG" and sorted(opts) == ["ORIG", "P1", "P2", "P3"]
    assert decop.correct_letter("r2") == "C"


def test_render():
    msg = decop.render(_item(), "forced", "r1")[-1]["content"]
    assert "A: P1" in msg and "B: ORIG" in msg and "D: P3" in msg


def test_parse():
    assert decop.parse('{"answer": "C"}') == {"letter": "C", "refused": False, "outcome": "answer"}
    assert decop.parse("The original is option B.")["letter"] == "B"
    assert decop.parse('{"answer": null}')["refused"]
    assert decop.parse('{"answer": ""}')["refused"]
    assert decop.parse("")["refused"]


def test_item_score():
    parsed = {"r0": {"letter": "A", "refused": False}, "r1": {"letter": "B", "refused": False},
              "r2": {"letter": "C", "refused": False}, "r3": {"letter": "A", "refused": False}}
    s = decop.item_score(_item(), parsed)
    assert s.score == 0.75 and s.extra == {"n_correct": 3, "known": True} and not s.refused


def test_pick_passage_skips_markdown_paragraphs():
    words = " ".join(f"word{i}" for i in range(50))
    text = "1" + "\\" + f". **Software** platform {words}\n\n| a | b | {words}\n\nClean paragraph {words}."
    for seed in range(5):
        assert decop.pick_passage(text, np.random.default_rng(seed)).startswith("Clean paragraph")


def test_pick_passage_skips_boilerplate():
    words = " ".join(f"word{i}" for i in range(50))
    text = (f"This information is information that X plc is obliged to make public pursuant to the EU Market Abuse "
            f"Regulation. {words}\n\nClean paragraph {words}.")
    for seed in range(5):
        assert decop.pick_passage(text, np.random.default_rng(seed)).startswith("Clean paragraph")


def test_pick_passage_skips_markdown_links():
    words = " ".join(f"word{i}" for i in range(50))
    text = f"Read [more](https://example.com) {words}\n\nClean paragraph {words}."
    for seed in range(5):
        assert decop.pick_passage(text, np.random.default_rng(seed)).startswith("Clean paragraph")


def test_prose_fallback_takes_the_stated_answer():
    assert decop.parse("A reasonable guess is C")["letter"] == "C"
    assert decop.parse("Answer: B. Option A is a paraphrase.")["letter"] == "B"


# ---------------------------------------------------------------- v2: rewrites that can't be spotted by their form
def test_style_tells_catch_rewrites_that_differ_in_form():
    orig = "Omsättningen ökade till 507,8 MSEK, jämfört med 412,3 MSEK föregående år, enligt vd Anna Berg."
    same_form = "Enligt vd Anna Berg steg omsättningen till 507,8 MSEK, mot 412,3 MSEK året innan, vilket var bra."
    assert decop.style_tells(orig, [same_form]) == []
    assert "numbers" in decop.style_tells(orig, [same_form.replace("507,8", "507.8")])[0]
    assert "numbers" in decop.style_tells(orig, [same_form.replace("412,3 MSEK", "cirka 400 MSEK")])[0]
    assert "length" in decop.style_tells(orig, ["Omsättningen var 507,8 MSEK, mot 412,3 MSEK."])[0]
    assert "dash" in decop.style_tells(orig, [same_form.replace(", vilket", " — vilket")])[0]
    assert "quote" in decop.style_tells(orig, ["Anna Berg: “omsättningen blev 507,8 MSEK mot 412,3 MSEK ifjol”, sa hon."])[0]


def test_build_items_drops_rewrite_sets_with_tells(data):
    def one_bad(batch):
        good = decop.mock_paraphrases(batch)
        return [ps if k else [ps[0], ps[1], ps[2] + " — extra words to change the form"] for k, ps in enumerate(good)]
    items = decop.build_items(data, 6, 1, paraphrase_fn=one_bad)
    assert len(items) == 5


def test_style_tells_catch_a_changed_company_name():
    orig = "Metalloprints teknik tar 3D-utskrift av metall till en ny nivå, med en omsättning om 12,5 MSEK i år."
    ok = "Med sin teknik lyfter Metalloprints 3D-utskrift av metall till en ny nivå, med 12,5 MSEK i omsättning."
    bad = ok.replace("Metalloprints", "Metalloprunts")
    assert decop.style_tells(orig, [ok], company="Metalloprint AB") == []
    assert "company name" in decop.style_tells(orig, [bad], company="Metalloprint AB")[0]
