from evalbench import analysis, runner, store
from evalbench.cutoffs import resolve_cutoff
from evalbench.methods import completion, decop, number_cloze, price_recall

ORACLE = "mock/oracle@2023-06-30"
RANDOM = "mock/random@2023-06-30"


def test_oracle_is_detected_and_random_is_not(data, tmp_path):
    testsets = {
        "price_recall": ("price_recall/t", price_recall.build_items(data, 90, 1)),
        "number_cloze": ("number_cloze/t", number_cloze.build_items(data, 60, 1)),
        "decop": ("decop/t", decop.build_items(data, 60, 1, decop.mock_paraphrases)),
        "completion": ("completion/t", completion.build_items(data, 60, 1)),
    }
    conn = store.connect(tmp_path / "t.sqlite")
    models = [ORACLE, RANDOM]
    calls = runner.plan(models, list(testsets), 500, "forced", testsets)
    _, state = runner.run(calls, conn, {})
    assert state.done == len(calls) and not state.errors
    cutoffs = {m: resolve_cutoff(m) for m in models}
    table = analysis.summary_table(store.load_calls(conn, mode="forced"), testsets, models, list(testsets), cutoffs, B=500)
    oracle = table[table.model == ORACLE].set_index("method")
    rnd = table[table.model == RANDOM].set_index("method")
    for method in testsets:
        assert oracle.loc[method, "dkappa"] > 0.5, method
        assert oracle.loc[method, "p_value"] < 0.05, method
        assert rnd.loc[method, "ci_lo"] <= 0 <= rnd.loc[method, "ci_hi"], method
    assert oracle.loc["price_recall", "hallucination_rate"] == 0.0
