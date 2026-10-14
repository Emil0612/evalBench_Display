"""The app scores price levels by closeness everywhere (change of 2026-10-01, spec §7), keeping the ±10% hit for the
correct/wrong labels; the rolling view can be cut or compared by company size."""
import pandas as pd

from evalbench import ui


def _df(method_closeness=0.4, n=60, sizes=("large", "mid", "small")):
    rows = []
    for k in range(n):
        size = sizes[k % len(sizes)]
        rows.append({"item_id": f"i{k}", "date": f"2024-{1 + k % 12:02d}-15", "cluster": f"c{k % 9}",
                     "fabricated": False, "refused": False, "score": 1.0,
                     "closeness": {"large": 0.5, "mid": 0.3, "small": 0.1}[size] if method_closeness is None
                     else method_closeness, "s_size": size})
    return pd.DataFrame(rows)


def test_price_levels_are_scored_by_closeness_and_keep_the_10_percent_hit():
    out = ui.app_scores(_df(0.4), "price_level")
    assert (out["score"] == 0.4).all() and (out["hit_10pct"] == 1.0).all()
    for method in ("known_level", "us_level"):
        assert (ui.app_scores(_df(0.4), method)["score"] == 0.4).all()


def test_other_methods_keep_their_score():
    df = _df(0.4).drop(columns="closeness")
    out = ui.app_scores(df, "annual_return")
    assert (out["score"] == 1.0).all() and "hit_10pct" not in out


def test_closeness_is_shown_as_a_number_and_rates_as_percentages():
    assert ui.value_fmt("price_level")(0.234) == "0.23"
    assert ui.value_fmt("annual_return")(0.5) == "50%"
    assert ui.value_fmt("price_level")(float("nan")) == "–"


def test_company_size_comes_from_the_question_or_else_the_company():
    df = pd.DataFrame({"s_size": ["large", None, "unknown"], "s_company_size": ["small", "mid", "mid"]})
    assert list(ui.size_of(df, "price_level")) == ["large", "mid", "mid"]
    assert list(ui.size_of(df.drop(columns="s_size"), "price_level")) == ["small", "mid", "mid"]


def test_rolling_by_size_gives_one_curve_per_size():
    curves = ui.rolling_by_size(ui.app_scores(_df(None, n=180), "price_level"), "price_level")
    means = curves.groupby("size")["mean"].mean().round(6).to_dict()
    assert means == {"large": 0.5, "mid": 0.3, "small": 0.1}
    one = ui.rolling_from_scores(ui.app_scores(_df(None, n=180), "price_level"), "price_level", size="large")
    assert (one["mean"].round(6) == 0.5).all()


def test_compare_tests_price_levels_on_the_10_percent_hit_not_on_closeness():
    from evalbench import analysis
    base = {"date": "2024-01-15", "fabricated": False, "refused": False, "group": 1}
    a = pd.DataFrame([{**base, "item_id": f"i{k}", "cluster": f"c{k}", "score": 0.4, "hit_10pct": 1.0}
                      for k in range(30)])
    b = a.assign(score=0.45, hit_10pct=0.0)
    res, _ = analysis.compare(a, b, "price_level", B=200)
    assert res["mcnemar_p"] < 0.05          # A within ±10% on all 30, B on none: a clear difference


def test_the_compare_sizes_chart_draws_every_model_panel():
    rows = [{"model": m, "size": s, "month": f"2024-{k:02d}", "mean": 0.3, "lo": 0.2, "hi": 0.4, "n": 30}
            for m in ("a/one", "b/two") for s in ui.SIZES for k in range(1, 13)]
    cuts = [{"model": "a/one", "month": "2024-06"}, {"model": "b/two", "month": "2024-06"}]
    rels = [{"model": "a/one", "month": "2024-09"}, {"model": "b/two", "month": "2024-09"}]
    spec = ui.rolling_size_chart(pd.DataFrame(rows), ["a/one", "b/two"], cuts, rels).to_dict()
    assert len(spec["concat"]) == 2


def test_the_refusal_guard_only_judges_cells_whose_answer_rates_differ():
    base = {"dkappa": 0.07, "p_holm": 0.046, "n_pre": 400, "n_post": 300, "malformed_rate": 0.0,
            "answered_dkappa": 0.07, "answered_p_holm": 0.052}
    # answered everything on both sides: the answered-only test is the same test (its Holm rank can shift)
    assert ui.cell_status({**base, "coverage_pre": 1.0, "coverage_post": 1.0}) == "memorization detected"
    # refused half the questions after release, and not significant without them: refusal-driven
    assert ui.cell_status({**base, "coverage_pre": 0.99, "coverage_post": 0.54, "answered_dkappa": 0.04,
                           "answered_p_holm": 0.65}) == "refusal-driven"
