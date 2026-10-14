"""The model table in the README: every model the reports cover, its stated cutoff and where that date comes from, and
its release. Built from config/model_cutoffs.yaml, config/model_cutoff_sources.yaml, config/model_releases.yaml and
OpenRouter's public model list (for the dates config does not set). No model calls, no cost.

uv run python -m evalbench.model_table      (rewrites the table between the markers in README.md)
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from . import cutoff_estimate, headline_report, ladder_extend
from .cutoffs import CONFIG, RELEASES, SOURCES, load_overrides, resolve_cutoff, resolve_release
from .items import ROOT
from .llm.openrouter import fetch_models, models_by_id
from .pilot_report import _table

README = ROOT / "README.md"
START, END = "<!-- model-table:start -->", "<!-- model-table:end -->"
GROUPS = [("main", headline_report.MODELS),
          ("ladder extension (index questions only)", ladder_extend.CANDIDATES),
          ("added 2026-10-07", cutoff_estimate.NEW)]
# model_cutoff_sources.yaml value -> how the table names it
SOURCE_LABELS = {"listing": "OpenRouter listing, frozen in config", "model card": "model card",
                 "estimate": "our estimate"}


def cutoff_source(model: str, meta: dict, overrides: dict, sources: dict) -> str:
    if model in sources and (model in overrides or sources[model] != "listing"):
        return SOURCE_LABELS.get(str(sources[model]), str(sources[model]))
    if model in overrides:
        return "set in config"
    if (meta.get(model) or {}).get("knowledge_cutoff"):
        return "OpenRouter listing"
    return "unknown"


def release_source(model: str, meta: dict, releases: dict) -> str:
    if model in releases:
        created = (meta.get(model) or {}).get("created")
        listed = datetime.fromtimestamp(created, tz=timezone.utc).date() if created else None
        return f"set in config (OpenRouter lists {listed})" if listed else "set in config"
    return "OpenRouter listing" if (meta.get(model) or {}).get("created") else "unknown"


def rows(meta: dict, cutoffs_path: Path = CONFIG, sources_path: Path = SOURCES,
         releases_path: Path = RELEASES) -> list[dict]:
    overrides, sources = load_overrides(cutoffs_path), load_overrides(sources_path)
    releases = load_overrides(releases_path)
    out = []
    for group, models in GROUPS:
        for m in models:
            out.append({"model": m, "group": group,
                        "cutoff": resolve_cutoff(m, meta, path=cutoffs_path),
                        "cutoff_source": cutoff_source(m, meta, overrides, sources),
                        "release": resolve_release(m, meta, path=releases_path),
                        "release_source": release_source(m, meta, releases)})
    return out


def render(table_rows: list[dict]) -> str:
    return _table(["Model (OpenRouter id)", "Group", "Stated cutoff", "Cutoff from", "Release", "Release from"],
                  [[f"`{r['model']}`", r["group"], r["cutoff"] or "unknown", r["cutoff_source"],
                    r["release"] or "unknown", r["release_source"]] for r in table_rows])


def update_readme(text: str, table: str) -> str:
    """Replace what lies between the markers with the table; the rest of the README is left as it is."""
    if text.count(START) != 1 or text.count(END) != 1 or text.index(START) > text.index(END):
        raise ValueError(f"README needs exactly one {START} ... {END} block")
    head, rest = text.split(START)
    return head + START + "\n" + table + "\n" + END + rest.split(END, 1)[1]


def main() -> None:
    table = render(rows(models_by_id(fetch_models())))
    text = README.read_text(encoding="utf-8")
    README.write_text(update_readme(text, table), encoding="utf-8", newline="\n")
    print(table)


if __name__ == "__main__":
    main()
