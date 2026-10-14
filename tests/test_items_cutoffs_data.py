from datetime import date

from evalbench.cutoffs import group_for, resolve_cutoff, save_override
from evalbench.data import eligible_news
from evalbench.items import Item, load_current_testsets, load_manifest, load_testset, save_testset


def _item(i, d="2022-05-01"):
    return Item(item_id=f"x-{i}", method="demo", date=d, cluster=f"c{i}", payload={"q": "å"}, truth={"a": i},
                strata={"size": "large"})


def test_testset_roundtrip_and_manifest(tmp_path):
    items = [_item(1), _item(2, "2025-01-01")]
    path = save_testset(items, "demo", "v1", {"seed": 7}, root=tmp_path)
    assert path == tmp_path / "demo" / "v1.jsonl"
    assert load_testset("demo", "v1", root=tmp_path) == items
    man = load_manifest("demo", "v1", root=tmp_path)
    assert man["n_items"] == 2 and man["seed"] == 7 and man["items_per_year"] == {"2022": 1, "2025": 1}


def test_current_testsets_prefers_v1(tmp_path):
    save_testset([_item(1)], "demo", "mock", {}, root=tmp_path)
    assert load_current_testsets(tmp_path)["demo"][0] == "demo/mock"
    save_testset([_item(1)], "demo", "v1", {}, root=tmp_path)
    name, items = load_current_testsets(tmp_path)["demo"]
    assert name == "demo/v1" and len(items) == 1


def test_group_for():
    c = date(2024, 6, 30)
    assert group_for("2024-01-01", c) == 1
    assert group_for("2024-05-01", c) == -1
    assert group_for("2024-12-01", c) == 0
    assert group_for("2024-01-01", None) == -1


def test_resolve_cutoff(tmp_path):
    cfg = tmp_path / "cutoffs.yaml"
    meta = {"a/model": {"knowledge_cutoff": "2025-03-01"}, "b/model": {"knowledge_cutoff": None}}
    assert resolve_cutoff("mock/oracle@2024-06-30", path=cfg) == date(2024, 6, 30)
    assert resolve_cutoff("a/model", meta, path=cfg) == date(2025, 3, 1)
    assert resolve_cutoff("b/model", meta, path=cfg) is None
    save_override("a/model", "2024-10-01", path=cfg)
    assert resolve_cutoff("a/model", meta, path=cfg) == date(2024, 10, 1)
    save_override("a/model", None, path=cfg)
    assert resolve_cutoff("a/model", meta, path=cfg) == date(2025, 3, 1)


def test_eligible_news_filters_templates_and_adds_company(data):
    news = eligible_news(data.news, data.universe)
    assert "tpl" not in set(news.newsid)
    row = news[news.newsid == "n3"].iloc[0]
    assert row.company == "Company 4 B" and row.cluster == "c40"


def test_invalid_override_falls_back(tmp_path):
    cfg = tmp_path / "cutoffs.yaml"
    cfg.write_text("a/model: not-a-date\n", encoding="utf-8")
    assert resolve_cutoff("a/model", {"a/model": {"knowledge_cutoff": "2025-03-01"}}, path=cfg) == date(2025, 3, 1)


def test_save_edits_only_writes_changes(tmp_path):
    from evalbench.cutoffs import load_overrides, save_edits
    cfg = tmp_path / "cutoffs.yaml"
    current = {"a/model": date(2025, 3, 1), "b/model": date(2024, 1, 1)}
    changed = save_edits({"a/model": date(2025, 3, 1), "b/model": date(2023, 6, 30)}, current, path=cfg)
    assert changed == ["b/model"] and load_overrides(cfg) == {"b/model": "2023-06-30"}


def test_testsets_fingerprint_changes_on_rebuild(tmp_path):
    from evalbench.items import testsets_fingerprint
    save_testset([_item(1)], "demo", "mock", {}, root=tmp_path)     # frozen versions can't be rebuilt; mock sets can
    before = testsets_fingerprint(tmp_path)
    save_testset([_item(1), _item(2)], "demo", "mock", {}, root=tmp_path)
    assert testsets_fingerprint(tmp_path) != before


def test_current_testsets_prefers_highest_version(tmp_path):
    from evalbench.items import next_version
    save_testset([_item(1)], "demo", "mock", {}, root=tmp_path)
    assert next_version("demo", root=tmp_path) == "v1"
    save_testset([_item(1)], "demo", "v1", {}, root=tmp_path)
    save_testset([_item(1), _item(2)], "demo", "v2", {}, root=tmp_path)
    assert load_current_testsets(tmp_path)["demo"][0] == "demo/v2"
    assert next_version("demo", root=tmp_path) == "v3"


def test_company_size_map_uses_market_cap_terciles(data):
    from evalbench.data import company_size_map
    sizes = company_size_map(data.universe)
    assert sizes["c10"] == "small" and sizes["c90"] == "large" and set(sizes.values()) == {"small", "mid", "large"}


def test_a_frozen_testset_version_is_never_overwritten(tmp_path):
    import pytest
    save_testset([_item(1)], "demo", "v1", {}, root=tmp_path)
    with pytest.raises(FileExistsError):
        save_testset([_item(2)], "demo", "v1", {}, root=tmp_path)
    assert [it.item_id for it in load_testset("demo", "v1", root=tmp_path)] == [_item(1).item_id]
    save_testset([_item(1)], "demo", "mock", {}, root=tmp_path)       # mock sets are pipeline fixtures, not frozen
    save_testset([_item(2)], "demo", "mock", {}, root=tmp_path)


def test_analysis_uses_the_newest_version_that_has_answers(tmp_path):
    from evalbench.items import analysis_versions
    for v in ("v1", "v2"):
        save_testset([_item(1)], "demo", v, {}, root=tmp_path)
    save_testset([_item(1)], "other", "v1", {}, root=tmp_path)
    assert analysis_versions({"demo/v1"}, tmp_path) == {"demo": "v1", "other": "v1"}   # v2 has no answers yet
    assert analysis_versions({"demo/v1", "demo/v2"}, tmp_path)["demo"] == "v2"
    assert analysis_versions(set(), tmp_path)["demo"] == "v2"                          # nothing answered: newest


def test_analysis_prefers_the_newest_version_real_models_answered(tmp_path):
    from evalbench.items import analysis_versions
    for v in ("v1", "v2"):
        save_testset([_item(1)], "demo", v, {}, root=tmp_path)
    # controls have answered v2, real models only v1: pages must keep showing v1
    assert analysis_versions({"demo/v1", "demo/v2"}, tmp_path, preferred={"demo/v1"})["demo"] == "v1"
    assert analysis_versions({"demo/v1", "demo/v2"}, tmp_path, preferred={"demo/v1", "demo/v2"})["demo"] == "v2"
    assert analysis_versions({"demo/v2"}, tmp_path, preferred=set())["demo"] == "v2"    # only controls anywhere
