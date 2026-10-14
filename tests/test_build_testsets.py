from evalbench import build_testsets as bt
from evalbench.items import load_manifest, load_testset
from evalbench.methods import decop

META = {"p/model": {"pricing": {"prompt": "0.000001", "completion": "0.000002"}, "supported_parameters": ["temperature"]}}


def test_paraphrase_cost_estimate_scales_with_n(data):
    small = bt.estimate_paraphrase_cost(data, 5, 1, "p/model", META)
    big = bt.estimate_paraphrase_cost(data, 20, 1, "p/model", META)
    assert 0 < small < big


def test_generate_decop_never_overwrites(data, tmp_path):
    path1, items1 = bt.generate_decop(data, 6, 1, decop.mock_paraphrases, "mock-paraphraser", root=tmp_path)
    path2, _ = bt.generate_decop(data, 6, 1, decop.mock_paraphrases, "mock-paraphraser", root=tmp_path)
    assert path1.name == "v1.jsonl" and path2.name == "v2.jsonl"
    assert load_testset("decop", "v1", root=tmp_path) == items1
    assert load_manifest("decop", "v1", root=tmp_path)["paraphraser"] == "mock-paraphraser"


def _patch_cli(monkeypatch, data, answer, calls):
    monkeypatch.setattr(bt, "load_data", lambda: data)
    monkeypatch.setattr(bt, "source_md5", lambda: {})
    monkeypatch.setattr(bt, "fetch_models", lambda: [{"id": "p/model", **META["p/model"]}])
    monkeypatch.setattr("builtins.input", lambda prompt="": calls.append(prompt) or answer)

    def fake_paraphraser(model, meta_by_id=None, on_progress=None):
        calls.append("PAID")
        return decop.mock_paraphrases

    monkeypatch.setattr(bt, "make_llm_paraphraser", fake_paraphraser)
    monkeypatch.setattr(bt, "generate_decop", lambda data, n, seed, fn, model, **kw: (calls.append("SAVED"), (bt.TESTSETS / "x.jsonl", []))[1])


def test_cli_paid_generation_needs_confirmation(data, monkeypatch, capsys):
    calls = []
    _patch_cli(monkeypatch, data, "n", calls)
    bt.main(["--methods", "decop", "--paraphraser", "p/model", "--n", "5", "--unfreeze"])
    out = capsys.readouterr().out
    assert "upper bound" in out.lower() and "PAID" not in calls and "SAVED" not in calls
    assert any("[y/N]" in c for c in calls)


def test_cli_paid_generation_runs_after_yes(data, monkeypatch):
    calls = []
    _patch_cli(monkeypatch, data, "y", calls)
    bt.main(["--methods", "decop", "--paraphraser", "p/model", "--n", "5", "--unfreeze"])
    assert "PAID" in calls and "SAVED" in calls


def test_cli_leaves_existing_versions_alone(data, monkeypatch, capsys):
    monkeypatch.setattr(bt, "load_data", lambda: data)
    monkeypatch.setattr(bt, "source_md5", lambda: {})

    def exists(*a, **k):
        raise FileExistsError("testsets/price_recall/v1.jsonl")

    monkeypatch.setattr(bt, "save_testset", exists)
    bt.main(["--methods", "price_recall", "--n", "5"])
    assert "already exists" in capsys.readouterr().out


def test_v2_paraphrase_prompt_asks_for_the_originals_form():
    text = bt.paraphrase_messages("Omsättningen ökade till 507,8 MSEK.", "sv")[0]["content"]
    assert "exactly as written" in text and "punctuation" in text and bt.PARAPHRASE_PROMPT_VERSION == "v2"


def test_generated_manifest_records_the_paraphrase_prompt(data, tmp_path):
    bt.generate_decop(data, 4, 1, decop.mock_paraphrases, "mock-paraphraser", root=tmp_path)
    assert load_manifest("decop", "v1", root=tmp_path)["paraphrase_prompt"] == "v2"


def test_cli_builds_index_sets_and_skips_what_it_cannot_build(data, monkeypatch, capsys, tmp_path):
    """A fresh clone runs the default command (every method) to the end: the index methods are built from the daily
    index closes, the multiple-choice sets are skipped (they come from another set), and missing private US data is
    a skip with the fetch hint, not a crash."""
    import pandas as pd

    days = pd.bdate_range("2016-01-01", "2026-08-31")
    daily = pd.DataFrame({"symbol": "OMXS30", "date": days, "close": 1000.0 + pd.Series(range(len(days)))})
    monkeypatch.setattr(bt, "load_data", lambda: data)
    monkeypatch.setattr(bt, "source_md5", lambda: {})
    monkeypatch.setitem(bt.INDEX_SOURCES, "index_level", lambda: daily)
    monkeypatch.setitem(bt.INDEX_SOURCES, "index_return", lambda: daily)
    missing = tmp_path / "private" / "us_index_daily.parquet"   # not fetched: run fetch_us_indices first
    monkeypatch.setitem(bt.INDEX_SOURCES, "us_index_level", lambda: bt.index_level.load_index(missing))
    saved = []
    monkeypatch.setattr(bt, "save_testset", lambda items, method, version, manifest: saved.append(method) or method)
    bt.main(["--methods", "index_level", "index_return", "price_choice", "us_index_level", "--n", "5"])
    out = capsys.readouterr().out
    assert saved[:2] == ["index_level", "index_return"]
    assert "price_choice: skipped - built from another test set" in out
    assert "us_index_level" not in saved and "us_index_level: skipped" in out and "fetch_us_indices" in out
