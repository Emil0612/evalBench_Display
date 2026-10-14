import pytest

from evalbench import selections as sel


def test_builtins_present_without_file(tmp_path):
    saved = sel.load(tmp_path / "s.yaml")
    assert "Controls only (free)" in saved and saved["Controls only (free)"]["builtin"]


def test_save_load_delete_roundtrip(tmp_path):
    path = tmp_path / "s.yaml"
    sel.save("My pilot", ["a/m", "b/m"], ["price_recall"], 100, "forced", path=path)
    got = sel.load(path)["My pilot"]
    assert got == {"models": ["a/m", "b/m"], "methods": ["price_recall"], "n": 100, "mode": "forced",
                   "temperature": 0.0, "reasoning": "auto", "samples": 1, "seed": None, "privacy": "standard",
                   "builtin": False}
    sel.delete("My pilot", path=path)
    assert "My pilot" not in sel.load(path)


def test_builtin_names_are_protected(tmp_path):
    with pytest.raises(ValueError):
        sel.save("Controls only (free)", ["x"], ["decop"], 5, "forced", path=tmp_path / "s.yaml")
    with pytest.raises(ValueError):
        sel.delete("Controls only (free)", path=tmp_path / "s.yaml")


def test_empty_name_rejected(tmp_path):
    with pytest.raises(ValueError):
        sel.save("  ", ["x"], ["decop"], 5, "forced", path=tmp_path / "s.yaml")


def test_malformed_entries_are_skipped(tmp_path):
    path = tmp_path / "s.yaml"
    path.write_text("broken: 3\nok:\n  models: [a/m]\n  methods: [decop]\n  n: 5\n  mode: abstain\n", encoding="utf-8")
    saved = sel.load(path)
    assert "broken" not in saved and saved["ok"]["mode"] == "abstain"
