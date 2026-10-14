"""Test-set items and their on-disk format: frozen JSONL plus a committed manifest."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TESTSETS = ROOT / "testsets"


@dataclass
class Item:
    item_id: str
    method: str
    date: str        # ISO date the fact/article belongs to; decides pre/post per model
    cluster: str     # bootstrap cluster: company ("c<key>") or article ("n<newsid>")
    payload: dict    # what the prompt needs
    truth: dict      # ground truth for scoring and for the UI
    strata: dict = field(default_factory=dict)
    fabricated: bool = False


@dataclass
class ItemScore:
    score: float
    refused: bool
    extra: dict = field(default_factory=dict)


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def testset_path(method: str, version: str, root: Path = TESTSETS) -> Path:
    return Path(root) / method / f"{version}.jsonl"


def save_testset(items: list[Item], method: str, version: str, manifest: dict, root: Path = TESTSETS) -> Path:
    path = testset_path(method, version, root)
    if path.exists() and version != "mock":
        # stored answers and pre-registered plans refer to a version by its items: never change a frozen set
        raise FileExistsError(f"{path} already exists; build a new version instead")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
    full = {
        "method": method,
        "version": version,
        "n_items": len(items),
        "n_fabricated": sum(it.fabricated for it in items),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "items_md5": file_md5(path),
        "items_per_year": dict(sorted(Counter(it.date[:4] for it in items).items())),
        "strata_counts": dict(Counter(json.dumps(it.strata, sort_keys=True, ensure_ascii=False) for it in items)),
        **manifest,
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(full, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_testset(method: str, version: str, root: Path = TESTSETS) -> list[Item]:
    with testset_path(method, version, root).open(encoding="utf-8") as f:
        return [Item(**json.loads(line)) for line in f if line.strip()]


def load_manifest(method: str, version: str, root: Path = TESTSETS) -> dict:
    path = testset_path(method, version, root).with_suffix(".manifest.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def testsets_fingerprint(root: Path = TESTSETS) -> tuple:
    """Changes whenever a test set file is written: lets a cache pick up rebuilt test sets."""
    root = Path(root)
    if not root.exists():
        return ()
    return tuple(sorted((str(p.relative_to(root)), p.stat().st_mtime_ns, p.stat().st_size)
                        for p in root.glob("*/*.jsonl")))


def _version_number(version: str) -> int | None:
    return int(version[1:]) if version.startswith("v") and version[1:].isdigit() else None


def next_version(method: str, root: Path = TESTSETS) -> str:
    """The next unused vN for a method: generating a test set never overwrites one results may refer to."""
    folder = Path(root) / method
    numbers = [n for p in folder.glob("*.jsonl") if (n := _version_number(p.name[: -len(".jsonl")])) is not None]
    return f"v{max(numbers, default=0) + 1}"


def _pick(versions: list[str]) -> str:
    numbered = sorted((n, v) for v in versions if (n := _version_number(v)) is not None)
    return numbered[-1][1] if numbered else "mock" if "mock" in versions else sorted(versions)[-1]


# Methods whose analysis pages stay on a registered version although a newer one was answered: M18's app views and
# registered test are on v1 (nine models); v2 is the confirmation set of 2026-10-06, read by its own report
ANALYSIS_PINS = {"headline_date": "v1"}


def analysis_versions(answered: set[str], root: Path = TESTSETS, preferred: set[str] | None = None) -> dict[str, str]:
    """Per method, the version the analysis pages show: the newest one that real models answered (preferred), else
    the newest one with any answers (e.g. only the controls), else the newest one. So results stay visible after a
    newer version is built and only the free controls have run on it. New runs always use the newest version."""
    out: dict[str, str] = {}
    root = Path(root)
    if not root.exists():
        return out
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        versions = [p.name[: -len(".jsonl")] for p in folder.glob("*.jsonl")]
        if versions:
            real = [v for v in versions if f"{folder.name}/{v}" in (preferred or set())]
            used = [v for v in versions if f"{folder.name}/{v}" in answered]
            out[folder.name] = _pick(real or used or versions)
            pin = ANALYSIS_PINS.get(folder.name)
            if pin in versions and f"{folder.name}/{pin}" in answered:
                out[folder.name] = pin
    return out


def load_current_testsets(root: Path = TESTSETS) -> dict[str, tuple[str, list[Item]]]:
    """For each method folder: the highest vN test set if any, else the mock one."""
    out: dict[str, tuple[str, list[Item]]] = {}
    root = Path(root)
    if not root.exists():
        return out
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        versions = sorted(p.name[: -len(".jsonl")] for p in folder.glob("*.jsonl"))
        if not versions:
            continue
        numbered = sorted((n, v) for v in versions if (n := _version_number(v)) is not None)
        version = numbered[-1][1] if numbered else "mock" if "mock" in versions else versions[-1]
        out[folder.name] = (f"{folder.name}/{version}", load_testset(folder.name, version, root))
    return out
