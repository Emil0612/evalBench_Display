"""Named run selections (models, methods, items per method, answer mode) saved from the Run page."""
from __future__ import annotations

from pathlib import Path

import yaml

from .items import ROOT
from .llm.openrouter import EFFORTS, PRIVACY

PATH = ROOT / "config" / "saved_selections.yaml"
HEADER = ("# Saved run selections from the Run page: name -> models, methods, n (items per method), mode, "
          "temperature.\n")
MODES = ("forced", "abstain")
MAX_TEMPERATURE = 2.0
MAX_SAMPLES = 10
REASONING = ("auto", *EFFORTS)
DEFAULTS = {"temperature": 0.0, "reasoning": "auto", "samples": 1, "seed": None, "privacy": "standard"}
BUILTIN = {
    "Controls only (free)": {
        "models": ["mock/oracle@2024-06-30", "mock/random@2024-06-30", "baseline/blind@2024-06-30"],
        "methods": ["price_recall", "number_cloze", "decop", "completion", "price_level", "annual_return",
                    "known_return", "split_clock", "known_level", "monthly_return", "price_choice", "known_choice"],
        "n": 40, "mode": "forced", **DEFAULTS,
    },
}


def _read(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _valid(entry) -> bool:
    return (isinstance(entry, dict)
            and isinstance(entry.get("models"), list) and all(isinstance(m, str) for m in entry["models"])
            and isinstance(entry.get("methods"), list) and all(isinstance(m, str) for m in entry["methods"])
            and isinstance(entry.get("n"), int) and entry["n"] > 0 and entry.get("mode") in MODES
            and isinstance(entry.get("temperature", 0.0), (int, float))
            and 0 <= entry.get("temperature", 0.0) <= MAX_TEMPERATURE
            and entry.get("reasoning", "auto") in REASONING
            and isinstance(entry.get("samples", 1), int) and 1 <= entry.get("samples", 1) <= MAX_SAMPLES
            and (entry.get("seed") is None or (isinstance(entry["seed"], int) and entry["seed"] >= 0))
            and entry.get("privacy", "standard") in PRIVACY)


def load(path: Path = PATH) -> dict[str, dict]:
    """Built-in selections first, then the saved ones; malformed saved entries are skipped."""
    out = {name: {**entry, "builtin": True} for name, entry in BUILTIN.items()}
    for name, entry in _read(path).items():
        if _valid(entry) and name not in BUILTIN:
            out[str(name)] = {"models": entry["models"], "methods": entry["methods"], "n": entry["n"],
                              "mode": entry["mode"], "temperature": float(entry.get("temperature", 0.0)),
                              "reasoning": entry.get("reasoning", "auto"), "samples": entry.get("samples", 1),
                              "seed": entry.get("seed"), "privacy": entry.get("privacy", "standard"), "builtin": False}
    return out


def _write(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(data, sort_keys=True, allow_unicode=True) if data else ""
    path.write_text(HEADER + body, encoding="utf-8", newline="\n")


def save(name: str, models: list[str], methods: list[str], n: int, mode: str, temperature: float = 0.0,
         reasoning: str = "auto", samples: int = 1, seed: int | None = None, privacy: str = "standard",
         path: Path = PATH) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("a selection needs a name")
    if name in BUILTIN:
        raise ValueError(f"'{name}' is a built-in selection")
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if not 0 <= temperature <= MAX_TEMPERATURE:
        raise ValueError(f"temperature must be between 0 and {MAX_TEMPERATURE}")
    if reasoning not in REASONING:
        raise ValueError(f"unknown reasoning effort {reasoning!r}")
    if not 1 <= int(samples) <= MAX_SAMPLES:
        raise ValueError(f"samples must be between 1 and {MAX_SAMPLES}")
    if seed is not None and int(seed) < 0:
        raise ValueError("seed must be 0 or larger")
    if privacy not in PRIVACY:
        raise ValueError(f"unknown privacy level {privacy!r}")
    data = _read(path)
    data[name] = {"models": list(models), "methods": list(methods), "n": int(n), "mode": mode,
                  "temperature": round(float(temperature), 2), "reasoning": reasoning, "samples": int(samples),
                  "seed": None if seed is None else int(seed), "privacy": privacy}
    _write(data, path)
    return name


def delete(name: str, path: Path = PATH) -> None:
    if name in BUILTIN:
        raise ValueError(f"'{name}' is a built-in selection")
    data = _read(path)
    data.pop(name, None)
    _write(data, path)
