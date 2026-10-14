import pytest

from evalbench.textutil import highlight_overlap, lcs_mask, number_recall, rouge_l_f1, tokens


def test_tokens_lowercase_unicode():
    assert tokens("Årets Vinst, 12,5%!") == ["årets", "vinst", "12", "5"]


def test_rouge_l_known_value():
    assert rouge_l_f1("the cat on the mat", "the cat sat on the mat") == pytest.approx(10 / 11)


def test_rouge_l_truncates_candidate_to_reference_length():
    assert rouge_l_f1("a b c x y z", "a b c") == pytest.approx(1.0)


def test_rouge_l_empty():
    assert rouge_l_f1("", "a b") == 0.0


def test_lcs_mask():
    assert lcs_mask(["a", "x", "b"], ["a", "b"]) == [True, False, True]


def test_highlight_overlap_escapes_and_marks():
    assert highlight_overlap("A x b", "a b") == "<mark>A</mark> x <mark>b</mark>"
    assert "&lt;" in highlight_overlap("<script> a", "a")


def test_number_recall():
    assert number_recall("sales were 1 234,5 and 99", "Net 1 234,5 and 7 000", "sv") == pytest.approx(0.5)
    assert number_recall("anything", "no numbers here", "en") is None
