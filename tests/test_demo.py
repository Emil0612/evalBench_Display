import pandas as pd

from evalbench import demo
from evalbench.data import load_data
from evalbench.methods import split_clock


def test_demo_data_has_the_extract_schema(tmp_path):
    demo.write_data(tmp_path)
    assert demo.is_demo(tmp_path)
    d = load_data(tmp_path)
    assert len(d.universe) == demo.N_COMPANIES and d.universe["is_primary"].all()
    assert {"insref", "date", "close", "adj_close"} <= set(d.prices.columns)
    assert d.news["lang_group"].isin(["sv", "en"]).all() and d.news["text"].str.len().min() >= 500
    idx = pd.read_parquet(tmp_path / "index_daily.parquet")
    assert set(idx["symbol"]) == {"OMXS30", "OMXSPI", "OMXN40", "OMXHPI"}
    assert "text" not in pd.read_parquet(tmp_path / "news_index.parquet").columns
    assert split_clock.adjustment_events(d.prices)      # the synthetic splits are found


def test_demo_refuses_real_data(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "universe.parquet").write_bytes(b"real")
    assert demo.check_safe(tmp_path / "data", tmp_path / "testsets")
    (tmp_path / "data" / demo.MARKER).write_text("x")
    assert demo.check_safe(tmp_path / "data", tmp_path / "testsets") is None


def test_demo_refuses_built_testsets(tmp_path):
    (tmp_path / "testsets" / "m").mkdir(parents=True)
    (tmp_path / "testsets" / "m" / "v1.jsonl").write_text("{}")
    assert demo.check_safe(tmp_path / "data", tmp_path / "testsets")


def test_demo_uses_free_controls_only():
    from evalbench import runner
    assert not any(runner.is_paid(m) for m in demo.CONTROLS)


def test_no_splits_gives_no_events():
    """A price table without any split must give no events (an empty frame once took the full index: NaN rows)."""
    px = pd.DataFrame({"insref": [1, 1, 2, 2], "date": pd.to_datetime(["2020-01-01", "2020-01-02"] * 2),
                       "close": [10.0, 11.0, 5.0, 5.5], "adj_close": [10.0, 11.0, 5.0, 5.5]})
    assert split_clock.adjustment_events(px) == {}
