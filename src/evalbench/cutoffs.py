"""Knowledge cutoffs: where they come from and how they split items into PRE / POST / excluded."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

from .items import ROOT

CONFIG = ROOT / "config" / "model_cutoffs.yaml"
RELEASES = ROOT / "config" / "model_releases.yaml"
SOURCES = ROOT / "config" / "model_cutoff_sources.yaml"
BUFFER_DAYS = 91
HEADER = "# Manual knowledge-cutoff overrides: <openrouter model id>: YYYY-MM-DD. Wins over OpenRouter's value.\n"


def load_overrides(path: Path = CONFIG) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_override(model_id: str, cutoff: str | None, path: Path = CONFIG) -> None:
    path = Path(path)
    overrides = load_overrides(path)
    if cutoff:
        overrides[model_id] = str(cutoff)
    else:
        overrides.pop(model_id, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(overrides, sort_keys=True, allow_unicode=True) if overrides else ""
    path.write_text(HEADER + body, encoding="utf-8", newline="\n")


def resolve_cutoff(model_id: str, meta_by_id: dict | None = None, path: Path = CONFIG) -> date | None:
    """Local models carry it after '@'; else a manual override; else OpenRouter's knowledge_cutoff."""
    if "@" in model_id and model_id.split("@")[0] in ("mock/oracle", "mock/random", "baseline/blind"):
        return date.fromisoformat(model_id.split("@")[1])
    override = _parse_date(load_overrides(path).get(model_id))
    if override:
        return override
    return _parse_date(((meta_by_id or {}).get(model_id) or {}).get("knowledge_cutoff"))


def resolve_release(model_id: str, meta_by_id: dict | None = None, path: Path = RELEASES) -> date | None:
    """A manual override (config/model_releases.yaml), else OpenRouter's listing date; None when neither exists.
    An override is for a listing that predates the snapshot it serves (post-training can reach the snapshot's
    release)."""
    override = _parse_date(load_overrides(path).get(model_id))
    if override:
        return override
    created = ((meta_by_id or {}).get(model_id) or {}).get("created")
    return datetime.fromtimestamp(created, tz=timezone.utc).date() if created else None


def _parse_date(value) -> date | None:
    """A YYYY-MM-DD value, or None when missing or malformed (a bad hand edit must not break the app)."""
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def save_edits(edited: dict, current: dict, path: Path = CONFIG) -> list[str]:
    """Persist only the cutoffs the user actually changed; returns the changed model ids."""
    changed = [m for m, v in edited.items() if v != current.get(m)]
    for m in changed:
        save_override(m, edited[m].isoformat() if edited[m] else None, path=path)
    return changed


def group_for(item_date: str, cutoff: date | None, buffer_days: int = BUFFER_DAYS) -> int:
    """1 = PRE, 0 = POST, -1 = within the buffer or no cutoff known."""
    if cutoff is None:
        return -1
    d = date.fromisoformat(item_date)
    if d < cutoff - timedelta(days=buffer_days):
        return 1
    if d > cutoff + timedelta(days=buffer_days):
        return 0
    return -1
