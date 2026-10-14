"""Market page fixes (2026-10-01, after the US run): index answers are read only from the JSON answer, never from a
number in a plain-text refusal ("S&P 500", "OMXS30", "my cutoff is 2024"); the rolling bootstrap is vectorised; the
answers-vs-path chart keeps its scale on the real path and marks answers off that scale."""
import json
import time

import numpy as np
import pandas as pd
import pytest

from evalbench import market, store
from evalbench.methods import index_level, index_return, us_index_level, us_index_return
from evalbench.reparse import reparse

# ---------------------------------------------------------------- parsing

LEVEL_NONE = [
    "I cannot provide the closing level of the OMX Stockholm 30 index (OMXS30) on the last trading day of April 2026. "
    "My knowledge cutoff is **June 2024**, and I do not have access to future market data.",
    "I cannot provide the closing level of the S&P 500 index for November 2024, as that date is in the future.",
    "I cannot answer your question as it refers to future events. My knowledge cutoff is **June 2024**.",
    "I cannot provide the exact closing level of the Dow Jones Industrial Average (DJIA) on the last trading day of "
    "June 2024, as that date is in the future. My knowledge cutoff is **June 2",
]
RETURN_NONE = [
    "I cannot answer this question. The S&P 500 index for November 2024 has not occurred yet. My knowledge cutoff is 2023.",
    "!fopen nasdaq composite index historical prices aug 2023\nThe Nasdaq Composite index (COMP) closed at 15,257.76 on "
    "July 31, 2023, and at 15,363.78 on August 31, 2023. \n{\"change_percent",
]


@pytest.mark.parametrize("text", LEVEL_NONE)
def test_a_level_refusal_never_reads_a_number_from_the_index_name_or_a_date(text):
    p = index_level.parse(text)
    assert p["value"] is None and p["refused"]


@pytest.mark.parametrize("text", RETURN_NONE)
def test_a_change_refusal_or_cut_off_reply_is_not_an_answer(text):
    p = index_return.parse(text)
    assert p["value"] is None and p["refused"]


def test_a_plain_refusal_is_classified_as_a_refusal():
    assert index_return.parse(RETURN_NONE[0])["outcome"] == "refusal"
    assert index_level.parse(LEVEL_NONE[2])["outcome"] == "refusal"


@pytest.mark.parametrize("text, value", [
    ('{"level": 2214.5}', 2214.5),
    ('{"level":!!!!!3635}', 3635.0),              # Novita's "!" noise inside the JSON
    ('{"level": 13,200}', 13200.0),               # thousands separator breaks the JSON
    ('Sure. {"level": "2 214,53"}', 2214.53),
])
def test_broken_json_level_answers_are_still_read(text, value):
    p = index_level.parse(text)
    assert not p["refused"]
    assert value in {p["value"], *index_level.price_recall._readings(p)}


@pytest.mark.parametrize("text, value", [
    ('{"change_percent": -2.1}', -2.1),
    ('{"change!!!!!_percent": 4.3}', 4.3),
    ('{"change_percent": 6.95!!!!!!!!!!!!!!!!!!!!!!!', 6.95),
    ('```json\n{"change_percent": -4.4"}\n```', -4.4),
    ('{"change_percent": 5.5"}', 5.5),
    ('{"change_percent": 0}', 0.0),
])
def test_broken_json_change_answers_are_still_read(text, value):
    p = index_return.parse(text)
    assert not p["refused"] and p["value"] == pytest.approx(value)


def test_null_is_an_abstention():
    assert index_level.parse('{"level": null}')["outcome"] == "abstain"
    assert index_return.parse('{"change_percent": null}')["outcome"] == "abstain"


def test_the_us_methods_use_the_same_parsers():
    assert us_index_level.parse is index_level.parse
    assert us_index_return.parse is index_return.parse


# ---------------------------------------------------------------- answers_frame


def _item(method, month, level, ret, symbol="SPX"):
    from evalbench.items import Item
    return Item(item_id=f"x-{month}", method=method, date=f"{month}-28", cluster=f"m{month}",
                payload={"symbol": symbol, "index_name": "S&P 500", "month": month, "month_name": month},
                truth={"level": level, "return_pct": ret, "prev_level": level, "baseline_level": None,
                       "prev_return_pct": None})


def test_answers_frame_uses_the_level_for_every_level_method():
    it = _item("us_index_level", "2020-03", 2584.59, -12.5)
    calls = pd.DataFrame({"model": ["m"], "method": ["us_index_level"], "item_id": [it.item_id],
                          "parsed": [json.dumps({"value": 2500.0})]})
    df = market.answers_frame(calls, [it], "m", "us_index_level")
    assert df["truth"].iloc[0] == 2584.59


# ---------------------------------------------------------------- rolling bootstrap


def _frame(months=128, per=4, seed=0, constant_after=None):
    rng = np.random.default_rng(seed)
    rows = []
    for k, m in enumerate(pd.period_range("2016-01", periods=months, freq="M")):
        move = rng.normal(0, 4)
        for s in range(per):
            t = move + rng.normal(0, 1)
            a = 0.0 if constant_after is not None and k >= constant_after else 0.7 * t + rng.normal(0, 2)
            rows.append({"item_id": f"{m}-{s}", "month": m, "symbol": f"S{s}", "truth": t, "answer": a})
    return pd.DataFrame(rows)


def _slow_band(part, B, seed=1):
    rng = np.random.default_rng(seed)
    groups = [g for _, g in part.groupby("month")]
    draws = [market.corr(p["answer"], p["truth"]) for p in
             (pd.concat([groups[int(i)] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(B))]
    return np.percentile(draws, [2.5, 97.5])


def test_rolling_corr_matches_the_month_bootstrap_and_is_fast():
    df = _frame()
    t0 = time.perf_counter()
    out = market.rolling_corr(df, window=12, min_n=20, B=500)
    assert time.perf_counter() - t0 < 2.0
    row = out.iloc[40]
    m = pd.Period(row["month"], "M")
    part = df[df["month"].isin([m + k for k in range(-5, 7)])]
    assert row["r"] == pytest.approx(market.corr(part["answer"], part["truth"]))
    assert row["n"] == len(part)
    lo, hi = _slow_band(part, B=3000)
    assert row["lo"] == pytest.approx(lo, abs=0.06) and row["hi"] == pytest.approx(hi, abs=0.06)


def test_rolling_corr_answers_that_never_vary_give_zero():
    out = market.rolling_corr(_frame(constant_after=60), window=12, min_n=20, B=200)
    late = out[out["month"] >= "2022-01"]
    assert (late["r"] == 0).all() and (late["lo"] == 0).all() and (late["hi"] == 0).all()


def test_boot_corr_is_the_shared_vectorised_bootstrap():
    df = _frame(months=30)
    draws = market.boot_corr(df, B=400, seed=3)
    assert draws.shape == (400,) and np.isfinite(draws).all() and -1 <= draws.min() <= draws.max() <= 1


# ---------------------------------------------------------------- chart scale


def test_path_scale_follows_the_real_path_and_flags_wild_answers():
    truth = pd.Series([3000.0, 4000.0, 6000.0])
    answers = pd.DataFrame({"answer": [3100.0, 30.0, 500000.0, 5900.0]})
    shown, lo, hi = market.path_scale(truth, answers, kind="level")
    assert lo > 30 and hi < 500000 and lo <= 3000 and hi >= 6000
    assert shown["off"].tolist() == ["", "below", "above", ""]
    assert shown["shown"].between(lo, hi).all()
    assert shown["answer"].tolist() == [3100.0, 30.0, 500000.0, 5900.0]    # the tooltip keeps the real answer


def test_path_scale_for_changes():
    truth = pd.Series([-12.0, 3.0, 9.0])
    answers = pd.DataFrame({"answer": [-30.5, 500.0, 2.0]})
    shown, lo, hi = market.path_scale(truth, answers, kind="change")
    assert -30 < lo <= -12 and 9 <= hi < 50
    assert shown["off"].tolist() == ["below", "above", ""]


def test_path_scale_keeps_moderate_misses_on_the_scale():
    truth = pd.Series([1400.0, 2600.0])
    answers = pd.DataFrame({"answer": [1300.0, 3200.0]})
    shown, lo, hi = market.path_scale(truth, answers, kind="level")
    assert (shown["off"] == "").all() and lo <= 1300 and hi >= 3200


# ---------------------------------------------------------------- re-parsing stored answers


def test_reparse_updates_only_rows_whose_reading_changes(tmp_path):
    conn = store.connect(tmp_path / "t.sqlite")
    base = {"mode": "forced", "variant": "main", "testset": "us_index_return/v1", "prompt_version": "p1",
            "run_id": "r", "messages": "[]", "cost": 0.0, "model": "google/gemini-2.5-flash"}
    old = {"value": 500.0, "raw": "x", "refused": False, "outcome": "answer"}
    store.insert_call(conn, {**base, "method": "us_index_return", "item_id": "ur-1", "response": RETURN_NONE[0],
                             "parsed": json.dumps(old), "refused": 0})
    good = index_return.parse('{"change_percent": 1.5}')
    store.insert_call(conn, {**base, "method": "us_index_return", "item_id": "ur-2",
                             "response": '{"change_percent": 1.5}', "parsed": json.dumps(good), "refused": 0})
    stamp = store.db_stamp(conn)
    changes = reparse(conn, ["us_index_return"], apply=False)
    assert len(changes) == 1 and changes[0]["item_id"] == "ur-1"
    assert json.loads(conn.execute("SELECT parsed FROM calls WHERE item_id='ur-1'").fetchone()[0])["value"] == 500.0
    reparse(conn, ["us_index_return"], apply=True)
    row = conn.execute("SELECT parsed, refused FROM calls WHERE item_id='ur-1'").fetchone()
    assert json.loads(row[0])["value"] is None and row[1] == 1
    assert store.db_stamp(conn) != stamp           # the app's answer cache sees the change
    assert reparse(conn, ["us_index_return"], apply=False) == []


# ---------------------------------------------------------------- the chart


def _domains(spec):
    out = []
    if isinstance(spec, dict):
        dom = spec.get("scale", {}).get("domain") if isinstance(spec.get("scale"), dict) else None
        if isinstance(dom, list) and len(dom) == 2 and all(isinstance(v, (int, float)) for v in dom):
            out.append(dom)
        for v in spec.values():
            out += _domains(v)
    elif isinstance(spec, list):
        for v in spec:
            out += _domains(v)
    return out


def test_path_chart_scale_ignores_one_wild_answer_and_marks_it():
    from evalbench import ui
    months = [str(m) for m in pd.period_range("2016-01", periods=24, freq="M")]
    truth = pd.DataFrame({"month": months, "value": np.linspace(-8, 9, 24)})
    answers = pd.DataFrame({"model": "openai/gpt-4.1-mini", "month": months, "answer": np.linspace(-6, 7, 24),
                            "truth": np.linspace(-8, 9, 24)})
    answers.loc[5, "answer"] = 500.0
    spec = ui.market_path_chart(truth, answers, ["openai/gpt-4.1-mini"], [], [], "% change", kind="change").to_dict()
    doms = _domains(spec)
    assert doms and max(d[1] for d in doms) < 30
    assert "triangle-up" in json.dumps(spec)


# ---------------------------------------------------------------- speed: cached chart specs


def test_chart_specs_are_built_once_per_input(monkeypatch):
    import altair as alt

    from evalbench import ui
    built = []

    def fake_chart(df, label):
        built.append(label)
        return alt.Chart(df).mark_line().encode(x="a:Q", y="b:Q")

    monkeypatch.setattr(ui, "fake_chart", fake_chart, raising=False)
    ui._chart_spec.clear()
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    s1 = ui._chart_spec("fake_chart", "light", None, (1,), (df, "x"))
    s2 = ui._chart_spec("fake_chart", "light", None, (1,), (df, "x"))
    s3 = ui._chart_spec("fake_chart", "light", None, (1,), (df, "y"))
    assert built == ["x", "y"] and s1 == s2 and s1["mark"]["type"] == "line" and s3 == s1   # same data: same spec


def test_answered_testsets_matches_the_calls():
    from evalbench import ui
    calls = pd.DataFrame({"model": ["mock/oracle@2024-06-30", "openai/gpt-4.1-mini", "openai/gpt-4.1-mini"],
                          "testset": ["a/v1", "b/v1", "b/v1"]})
    answered, real = ui.answered_sets(calls)
    assert answered == ("a/v1", "b/v1") and real == ("b/v1",)


def _marks(spec, out=None):
    out = [] if out is None else out
    if isinstance(spec, dict):
        if "mark" in spec:
            m = spec["mark"]
            out.append(m if isinstance(m, dict) else {"type": m})
        for v in spec.values():
            _marks(v, out)
    elif isinstance(spec, list):
        for v in spec:
            _marks(v, out)
    return out


def test_path_chart_real_path_has_a_halo_and_answers_are_dots_only():
    from evalbench import ui
    months = [str(m) for m in pd.period_range("2016-01", periods=12, freq="M")]
    truth = pd.DataFrame({"month": months, "value": np.linspace(1400, 1600, 12)})
    answers = pd.DataFrame({"model": "openai/gpt-oss-20b", "month": months, "answer": [4200.0, 1200.0] * 6,
                            "truth": np.linspace(1400, 1600, 12)})
    marks = _marks(ui.market_path_chart(truth, answers, ["openai/gpt-oss-20b"], [], [], "index level",
                                        kind="level").to_dict())
    lines = [m for m in marks if m["type"] == "line"]
    p = ui.palette()
    assert [m.get("color") for m in lines] == [p["surface"], p["ink"]]            # halo first, the path on top
    assert lines[0]["strokeWidth"] > lines[1]["strokeWidth"]


def test_path_chart_key_names_every_mark():
    from evalbench import ui
    html = ui.path_chart_key()
    for words in ("real index", "model's answer", "off the scale", "cutoff", "release"):
        assert words in html


# ---------------------------------------------------------------- one database connection per thread


def test_each_thread_gets_its_own_connection(tmp_path, monkeypatch):
    """Streamlit runs every session, and a page left mid-run, in its own thread; one shared SQLite connection used
    by two threads at once returned None from fetchone() ('NoneType' object is not iterable in db_stamp)."""
    import threading

    from evalbench import ui
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "t.sqlite")     # ui.conn() opens store.DB_PATH
    seen, errors = {}, []

    def work(k):
        try:
            c = ui.conn()
            assert c is ui.conn()                      # the same connection within a thread
            seen[k] = id(c)
            for _ in range(200):
                store.db_stamp(c)
        except Exception as e:                         # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=work, args=(k,)) for k in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert errors == [] and len(set(seen.values())) == 4


def test_db_stamp_is_cheap_and_sees_every_change(tmp_path):
    conn = store.connect(tmp_path / "s.sqlite")
    rec = {"mode": "forced", "variant": "main", "testset": "index_return/v1", "prompt_version": "p1", "run_id": "r",
           "messages": "[]", "cost": 0.0, "model": "m", "method": "index_return", "item_id": "ir-1",
           "response": '{"change_percent": 1}', "parsed": json.dumps(index_return.parse('{"change_percent": 1}')),
           "refused": 0}
    s0 = store.db_stamp(conn)
    store.insert_call(conn, rec)
    s1 = store.db_stamp(conn)
    store.insert_call(conn, {**rec, "response": '{"change_percent": 2}'})      # replaces the newest answer
    s2 = store.db_stamp(conn)
    assert len({s0, s1, s2}) == 3
    t0 = time.perf_counter()
    for _ in range(200):
        store.db_stamp(conn)
    assert (time.perf_counter() - t0) / 200 < 0.002


# ---------------------------------------------------------------- keys and gaps on the rolling charts


def test_gap_spans_mark_left_out_windows_inside_a_curve_only():
    from evalbench import ui
    months = ["2024-06", "2024-07", "2024-08", "2025-01", "2025-02", "2025-06"]
    spans = ui.gap_spans(pd.DataFrame({"month": months}))
    assert spans[["start", "end"]].values.tolist() == [["2024-09-01", "2025-01-01"], ["2025-03-01", "2025-06-01"]]
    assert ui.gap_spans(pd.DataFrame({"month": ["2024-06", "2024-07"]})).empty


def test_chart_key_has_a_swatch_and_label_for_every_entry():
    from evalbench import ui
    kinds = ["line", "dashed", "band", "dots", "triangles", "rule-dashed", "rule-solid", "gap"]
    html = ui.chart_key([(k, "#123456", f"label {k}") for k in kinds])
    for k in kinds:
        assert f"label {k}" in html
    assert html.count("#123456") >= len(kinds) - 1          # every swatch but the triangles' text glyph uses the colour


def test_market_keys_name_the_blind_guesser_the_band_and_the_gaps():
    from evalbench import ui
    rolling = ui.rolling_chart_key("correlation")
    for words in ("blind guesser", "95%", "too few answers", "cutoff", "release"):
        assert words in rolling
    compare = ui.market_compare_key()
    assert "Nordic" in compare and "US" in compare and "too few answers" in compare


def test_rolling_chart_shades_gaps_and_names_the_value():
    from evalbench import ui
    months = ["2024-06", "2024-07", "2025-01", "2025-02"]
    frames = pd.DataFrame({"model": "openai/gpt-4.1-mini", "month": months, "mean": [0.5, 0.6, 0.1, 0.0],
                           "lo": [0.3, 0.4, -0.1, -0.2], "hi": [0.7, 0.8, 0.3, 0.2], "n": 30})
    spec = json.dumps(ui.rolling_chart(frames, ["openai/gpt-4.1-mini"], [], [], value_label="correlation").to_dict())
    assert '"rect"' in spec and "too few answers" in spec and '"correlation"' in spec


# ---------------------------------------------------------------- stock answers and the Nordic-vs-US table


def test_repeat_share_finds_the_stock_answer_in_each_window():
    months = list(pd.period_range("2020-01", periods=24, freq="M"))
    rows = []
    for k, m in enumerate(months):
        for s in range(4):
            a = -1.5 if k >= 12 else float(k * 4 + s)        # varied answers, then one stock answer
            rows.append({"month": m, "answer": a, "truth": 0.0})
    out = market.repeat_share(pd.DataFrame(rows), window=12, min_n=20)
    early = out[out["month"] == "2020-06"].iloc[0]
    late = out[out["month"] == "2021-06"].iloc[0]
    assert early["share"] < 0.1 and early["distinct"] > 20
    assert late["share"] == pytest.approx(1.0) and late["value"] == -1.5 and late["distinct"] == 1


def test_stock_answer_chart_builds_with_a_percent_axis():
    from evalbench import ui
    df = pd.DataFrame({"model": "openai/gpt-4.1-mini", "month": ["2020-01", "2020-02"], "share": [0.1, 0.9],
                       "value": [2.0, 0.0], "distinct": [30, 2], "n": [40, 40]})
    spec = json.dumps(ui.stock_answer_chart(df, ["openai/gpt-4.1-mini"], [], []).to_dict())
    assert '"share"' in spec and "%" in spec


def test_market_comparison_rows_match_the_report_functions():
    from evalbench import index_report as ir
    from evalbench import ui
    months = list(pd.period_range("2016-01", periods=60, freq="M"))
    rng = np.random.default_rng(0)
    move = pd.Series({m: rng.normal(0, 4) for m in months})
    def frame(noise):
        return pd.DataFrame([{"month": m, "truth": 0.8 * move[m] + rng.normal(0, 2), "answer": 0.0}
                             for m in months for _ in range(4)]).assign(
            answer=lambda d: d["truth"] + rng.normal(0, noise, len(d)))
    us, nordic = frame(0.5), frame(3.0)
    row = ui.market_comparison_row(us, nordic, move)
    assert row["diff"] == pytest.approx(ir.paired_corr_diff(us, nordic)["diff"])
    assert row["beyond"] == pytest.approx(ir.partial_test(nordic, move)["r"])
    assert row["beyond_lo"] == pytest.approx(ir.partial_test(nordic, move)["lo"])
