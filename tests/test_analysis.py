import json
import math
from datetime import date

import pandas as pd

from evalbench import analysis
from evalbench.items import Item


def _calls_and_items():
    items, rows = {}, []
    for k in range(40):
        d = f"{2021 + k // 10}-01-01"
        it = Item(f"i{k}", "price_recall", d, f"c{k}", {}, {"close": 100.0, "adj_close": 100.0})
        items[it.item_id] = it
        value = 100.0 if d < "2023-01-01" else 150.0
        rows.append({"model": "m", "method": "price_recall", "item_id": it.item_id, "variant": "main",
                     "testset": "price_recall/t", "parsed": json.dumps({"value": value, "refused": False})})
    return pd.DataFrame(rows), items


def test_item_scores_and_summary():
    calls, items = _calls_and_items()
    df = analysis.item_scores(calls, items, "price_recall", date(2023, 1, 1))
    assert len(df) == 40 and set(df.group) == {1, 0, -1}
    s = analysis.summarize(df, "price_recall", B=200)
    assert s["delta"] == 1.0 and s["dkappa"] == 1.0 and s["p_value"] < 0.01
    assert s["coverage_pre"] == 1.0


def test_summarize_without_cutoff():
    calls, items = _calls_and_items()
    df = analysis.item_scores(calls, items, "price_recall", None)
    s = analysis.summarize(df, "price_recall", B=50)
    assert s["n_pre"] == 0 and math.isnan(s["delta"]) and math.isnan(s["p_value"])


def test_summarize_empty():
    s = analysis.summarize(pd.DataFrame(), "decop", B=10)
    assert s["n_pre"] == 0 and math.isnan(s["dkappa"])


def test_by_quarter():
    calls, items = _calls_and_items()
    q = analysis.by_quarter(analysis.item_scores(calls, items, "price_recall", date(2023, 1, 1)))
    assert list(q.columns) == ["quarter", "mean", "n"] and q.n.sum() == 40


def _decop_scores(pre_known, post_known, n=10):
    rows = []
    for k in range(2 * n):
        pre = k < n
        known = (k < pre_known) if pre else (k - n < post_known)
        rows.append({"item_id": f"d{k}", "date": "2022-01-01" if pre else "2025-01-01", "cluster": f"c{k}",
                     "fabricated": False, "group": 1 if pre else 0, "score": 1.0 if known else 0.25,
                     "refused": False, "n_correct": 4 if known else 1, "known": known})
    return pd.DataFrame(rows)


def test_decop_known_test_compares_pre_with_post():
    assert analysis.summarize(_decop_scores(10, 0), "decop", B=50)["known_p"] < 0.001
    # a model that consistently "knows" everything regardless of date is not evidence of memorization
    assert analysis.summarize(_decop_scores(10, 10), "decop", B=50)["known_p"] == 1.0


def test_accuracy_among_answered():
    calls, items = _calls_and_items()
    df = analysis.item_scores(calls, items, "price_recall", date(2023, 1, 1))
    df.loc[df.group == 0, "refused"] = True
    s = analysis.summarize(df, "price_recall", B=50)
    assert s["answered_mean_pre"] == 1.0 and math.isnan(s["answered_mean_post"]) and s["coverage_post"] == 0.0


def test_item_scores_carry_strata_and_breakdown():
    calls, items = _calls_and_items()
    for k, it in enumerate(items.values()):
        it.strata = {"size": "large" if k % 2 else "small"}
    df = analysis.item_scores(calls, items, "price_recall", date(2023, 1, 1))
    assert set(df["s_size"]) == {"large", "small"}
    b = analysis.breakdown(df, "price_recall", "size", B=50)
    assert set(b["stratum"]) == {"large", "small"} and (b["dkappa"] == 1.0).all()
    assert {"n_pre", "n_post", "ci_lo", "ci_hi"} <= set(b.columns)


def test_reading_gives_upper_bound_when_nothing_detected():
    sig = {"dkappa": 0.42, "ci_lo": 0.20, "ci_hi": 0.61, "p_holm": 0.001}
    none = {"dkappa": 0.03, "ci_lo": -0.04, "ci_hi": 0.08, "p_holm": 0.4}
    na = {"dkappa": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_holm": float("nan")}
    assert analysis.reading(sig) == "memorization detected: Δκ +0.42 [+0.20, +0.61]"
    assert analysis.reading(none) == "none detected: at most +0.08"
    assert analysis.reading(na) == "not scorable"


def test_extra_strata_are_added_with_unknown_default():
    calls, items = _calls_and_items()
    df = analysis.item_scores(calls, items, "price_recall", date(2023, 1, 1),
                              extra_strata={"company_size": {"c1": "large"}})
    assert df.set_index("item_id").loc["i1", "s_company_size"] == "large"
    assert df.set_index("item_id").loc["i2", "s_company_size"] == "unknown"


def test_effective_cutoff_pools_methods_and_detects_drop():
    calls, items = _calls_and_items()
    df = analysis.item_scores(calls, items, "price_recall", date(2023, 1, 1))
    r = analysis.effective_cutoff([(df, 0.0)], min_side=5, B=100)
    assert r["detected"] and date(2022, 1, 1) < r["breakpoint"] <= date(2023, 1, 1)
    assert analysis.effective_cutoff([(pd.DataFrame(), 0.0)])["detected"] is False


def test_effective_cutoff_is_compared_with_the_gap_not_the_first_date_after():
    calls, items = _calls_and_items()   # known on 2021-01-01 and 2022-01-01, not known from 2023-01-01
    df = analysis.item_scores(calls, items, "price_recall", date(2022, 6, 30))
    eff = analysis.effective_cutoff([(df, 0.0)], min_side=5, B=100)
    assert eff["last_known"] == date(2022, 1, 1) and eff["first_unknown"] == date(2023, 1, 1)
    assert analysis.versus_stated(eff, date(2022, 6, 30)) == "consistent with stated"
    assert analysis.versus_stated(eff, date(2021, 6, 30)) == "later than stated"
    assert analysis.versus_stated(eff, date(2023, 6, 30)) == "earlier than stated"
    assert analysis.versus_stated({"detected": False}, date(2023, 6, 30)) is None


def test_summary_from_frames_matches_summary_table(data):
    from evalbench import runner, store
    from evalbench.methods import price_recall
    import tempfile, pathlib
    conn = store.connect(pathlib.Path(tempfile.mkdtemp()) / "t.sqlite")
    ts = {"price_recall": ("price_recall/t", price_recall.build_items(data, 30, 1))}
    models = ["mock/oracle@2024-06-30", "mock/random@2024-06-30"]
    runner.run(runner.plan(models, ["price_recall"], 30, "forced", ts), conn, {})
    calls = store.load_calls(conn)
    from evalbench.cutoffs import resolve_cutoff
    cut = {m: resolve_cutoff(m) for m in models}
    table = analysis.summary_table(calls, ts, models, ["price_recall"], cut, B=200)
    items = {it.item_id: it for it in ts["price_recall"][1]}
    frames = {(m, "price_recall"): analysis.item_scores(calls[calls.model == m], items, "price_recall", cut[m])
              for m in models}
    again = analysis.summary_from_frames(frames, B=200)
    pd.testing.assert_frame_equal(table.reset_index(drop=True), again.reset_index(drop=True))
