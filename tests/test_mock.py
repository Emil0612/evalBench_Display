import pytest

from evalbench.llm.mock import is_local, local_answer
from evalbench.methods import METHODS, decop

ORACLE = "mock/oracle@2024-06-30"
RANDOM = "mock/random@2024-06-30"
BLIND = "baseline/blind@2024-06-30"


def test_is_local():
    assert is_local(ORACLE) and is_local(BLIND) and not is_local("openai/gpt-5.6-luna")


def _items(data):
    return {
        "price_recall": [i for i in METHODS["price_recall"].build_items(data, 20, 1) if not i.fabricated],
        "number_cloze": METHODS["number_cloze"].build_items(data, 20, 1),
        "decop": decop.build_items(data, 20, 1, decop.mock_paraphrases),
        "completion": METHODS["completion"].build_items(data, 20, 1),
    }


@pytest.mark.parametrize("method", ["price_recall", "number_cloze", "decop", "completion"])
def test_oracle_is_perfect_before_cutoff(data, method):
    mod = METHODS[method]
    pre = [it for it in _items(data)[method] if it.date < "2024-01-01"]
    assert pre
    for it in pre:
        parsed = {v: mod.parse(local_answer(ORACLE, method, it, v, "forced")) for v in mod.VARIANTS}
        assert mod.item_score(it, parsed).score > 0.7


@pytest.mark.parametrize("method", ["price_recall", "number_cloze", "decop", "completion"])
def test_random_and_blind_answers_parse(data, method):
    mod = METHODS[method]
    for it in _items(data)[method][:5]:
        for model in (RANDOM, BLIND):
            parsed = {v: mod.parse(local_answer(model, method, it, v, "forced")) for v in mod.VARIANTS}
            mod.item_score(it, parsed)  # must not raise


def test_answers_are_deterministic(data):
    it = _items(data)["number_cloze"][0]
    assert local_answer(RANDOM, "number_cloze", it, "main", "forced") == local_answer(RANDOM, "number_cloze", it, "main", "forced")
