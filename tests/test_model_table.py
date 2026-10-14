from datetime import date, datetime, timezone

import pytest

from evalbench import model_table as mt


def _ts(d: str) -> int:
    return int(datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp())


def _config(tmp_path, cutoffs: str, sources: str, releases: str):
    paths = []
    for name, body in (("cutoffs", cutoffs), ("sources", sources), ("releases", releases)):
        p = tmp_path / f"{name}.yaml"
        p.write_text(body, encoding="utf-8")
        paths.append(p)
    return paths


def test_sources_and_dates(tmp_path, monkeypatch):
    monkeypatch.setattr(mt, "GROUPS", [("main", ["a/listed", "a/frozen", "a/card", "a/est", "a/bare", "a/none",
                                                 "a/unfrozen"])])
    meta = {"a/listed": {"knowledge_cutoff": "2024-06-30", "created": _ts("2025-01-10")},
            "a/frozen": {"knowledge_cutoff": "2020-01-31", "created": _ts("2023-05-28")},
            "a/card": {"created": _ts("2026-04-02")},
            "a/est": {"created": _ts("2026-04-27")},
            "a/unfrozen": {"knowledge_cutoff": "2023-03-31", "created": _ts("2024-01-01")}}
    paths = _config(tmp_path,
                    "a/frozen: 2021-09-30\na/card: 2025-01-31\na/est: 2024-09-30\na/bare: 2022-03-31\n",
                    "a/frozen: listing\na/card: model card\na/est: estimate\na/unfrozen: listing\n",
                    "a/frozen: 2024-01-25\n")
    rows = {r["model"]: r for r in mt.rows(meta, *paths)}
    assert rows["a/listed"]["cutoff"] == date(2024, 6, 30)
    assert rows["a/listed"]["cutoff_source"] == "OpenRouter listing"
    assert rows["a/listed"]["release"] == date(2025, 1, 10)
    assert rows["a/listed"]["release_source"] == "OpenRouter listing"
    # config wins over the listing, and the table says where the date came from
    assert rows["a/frozen"]["cutoff"] == date(2021, 9, 30)
    assert rows["a/frozen"]["cutoff_source"] == "OpenRouter listing, frozen in config"
    assert rows["a/frozen"]["release"] == date(2024, 1, 25)
    assert rows["a/frozen"]["release_source"] == "set in config (OpenRouter lists 2023-05-28)"
    assert rows["a/card"]["cutoff_source"] == "model card"
    assert rows["a/est"]["cutoff_source"] == "our estimate"
    assert rows["a/bare"]["cutoff_source"] == "set in config"
    # marked 'listing' but no date frozen in config: not called frozen
    assert rows["a/unfrozen"]["cutoff_source"] == "OpenRouter listing"
    assert rows["a/none"]["cutoff"] is None and rows["a/none"]["cutoff_source"] == "unknown"
    assert rows["a/none"]["release"] is None and rows["a/none"]["release_source"] == "unknown"
    table = mt.render(list(rows.values()))
    assert table.splitlines()[0].startswith("| Model (OpenRouter id) | Group | Stated cutoff |")
    assert "| `a/none` | main | unknown | unknown | unknown | unknown |" in table


def test_update_readme_replaces_only_the_block():
    text = f"intro\n{mt.START}\nold table\n{mt.END}\noutro\n"
    once = mt.update_readme(text, "| new |")
    assert once == f"intro\n{mt.START}\n| new |\n{mt.END}\noutro\n"
    assert mt.update_readme(once, "| new |") == once


@pytest.mark.parametrize("text", ["no markers", f"{mt.END}\n{mt.START}", f"{mt.START}{mt.START}{mt.END}"])
def test_update_readme_needs_one_block(text):
    with pytest.raises(ValueError):
        mt.update_readme(text, "| t |")


def test_readme_table_lists_every_model():
    """The committed README carries the generated table with every model the reports cover (regenerate it with
    `python -m evalbench.model_table` after adding a model)."""
    text = mt.README.read_text(encoding="utf-8")
    block = text.split(mt.START)[1].split(mt.END)[0]
    for _, models in mt.GROUPS:
        for m in models:
            assert f"| `{m}` |" in block, m
