"""Self-hosted models: open weights served on your own hardware by any OpenAI-compatible server
(vLLM, llama.cpp's llama-server, Ollama, LM Studio, TGI). Listed in config/self_hosted.yaml as `self/<name>`.

They take part in runs like any other model, but free and without OpenRouter: the request goes straight to the
entry's base_url, and no prompt leaves the machine or network the server runs on. What was run is recorded in the
entry (weights, quantization, revision, checksum), because "the same model" on other hardware or at another
precision is not the same configuration."""
from __future__ import annotations

import os
import re
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import yaml

from ..items import ROOT

CONFIG_PATH = ROOT / "config" / "self_hosted.yaml"
PREFIX = "self/"
# what a server may accept besides model, messages and max_tokens
PARAMETERS = ("temperature", "seed")
FIELDS = ("base_url", "model", "api_key_env", "cutoff", "released", "quantization", "revision", "weights_sha256",
          "hardware", "parameters", "extra_tokens", "extra_body", "concurrency", "timeout_s", "notes")
_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ENV = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def is_self_hosted(model_id: str) -> bool:
    return model_id.startswith(PREFIX)


def stamp(path: Path = CONFIG_PATH) -> int:
    """Changes whenever the config file does (0 when there is none): a cache key."""
    return path.stat().st_mtime_ns if path.exists() else 0


def _read(path: Path) -> dict:
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("models") or {}


def load(path: Path = CONFIG_PATH) -> dict[str, dict]:
    """{"self/<name>": entry} for every configured model; {} when the file does not exist."""
    return {PREFIX + name: dict(entry or {}) for name, entry in _read(path).items()}


def _iso(value) -> str | None:
    return value.isoformat() if isinstance(value, date) else (str(value)[:10] if value else None)


def to_meta(entries: dict[str, dict]) -> dict[str, dict]:
    """The same shape as OpenRouter's model metadata, so the picker, cutoffs, cost bound and labels treat a
    self-hosted model like any other: price 0, the entry kept under "self_hosted"."""
    out = {}
    for model_id, e in entries.items():
        released = _iso(e.get("released"))
        created = (int(datetime.fromisoformat(released).replace(tzinfo=timezone.utc).timestamp())
                   if released else None)
        out[model_id] = {
            "id": model_id,
            "name": f"Self-hosted: {model_id[len(PREFIX):]}",
            "pricing": {"prompt": "0", "completion": "0"},
            "knowledge_cutoff": _iso(e.get("cutoff")),
            "created": created,
            "supported_parameters": ["max_tokens"] + [p for p in (e.get("parameters") or []) if p in PARAMETERS],
            "self_hosted": {**e, "cutoff": _iso(e.get("cutoff")), "released": released},
        }
    return out


def validate(name: str, entry: dict) -> dict:
    """The entry as it will be saved, or ValueError saying what is wrong."""
    if not _NAME.match(name or ""):
        raise ValueError("The name must be lowercase letters, digits, '.', '_' or '-' (e.g. qwen3-8b-q4), "
                         "at most 64 characters.")
    e = {k: v for k, v in entry.items() if k in FIELDS and v not in (None, "", [], {})}
    if not re.match(r"^https?://", e.get("base_url", "")):
        raise ValueError("The server address must start with http:// or https:// (e.g. http://localhost:8000/v1).")
    e["base_url"] = e["base_url"].rstrip("/")
    if not str(e.get("model", "")).strip():
        raise ValueError("The model name the server expects is required (what its /models list shows).")
    if "api_key_env" in e and not _ENV.match(str(e["api_key_env"])):
        raise ValueError("api_key_env must be the name of an environment variable (e.g. VLLM_API_KEY), never the "
                         "key itself.")
    unknown = [p for p in e.get("parameters") or [] if p not in PARAMETERS]
    if unknown:
        raise ValueError(f"Unknown parameters {unknown}; allowed: {list(PARAMETERS)}.")
    for key in ("cutoff", "released"):
        if key in e:
            e[key] = date.fromisoformat(_iso(e[key]))
    return e


def _write(models: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ("# Self-hosted models (OpenAI-compatible servers). Edited by the Run page or by hand.\n"
              "# Keys are never stored here: api_key_env names an environment variable that holds one.\n")
    path.write_text(header + yaml.safe_dump({"models": models}, sort_keys=False, allow_unicode=True), encoding="utf-8")


def save_entry(name: str, entry: dict, path: Path = CONFIG_PATH) -> str:
    """Add or replace one model; the others are kept. Returns its id (self/<name>)."""
    models = _read(path)
    models[name] = validate(name, entry)
    _write(models, path)
    return PREFIX + name


def delete_entry(name: str, path: Path = CONFIG_PATH) -> None:
    models = _read(path)
    models.pop(name, None)
    _write(models, path)


def build_body(meta: dict, messages: list[dict], max_tokens: int, temperature: float = 0.0,
               seed: int | None = None) -> dict:
    """A plain OpenAI chat request: the served model name, only the parameters the server accepts, and the
    entry's extra_body (e.g. {"chat_template_kwargs": {"enable_thinking": false}}). extra_tokens is added to each
    method's answer budget for models that think before they answer."""
    e = meta["self_hosted"]
    supported = set(meta.get("supported_parameters") or [])
    body = {"model": e["model"], "messages": messages, "max_tokens": max_tokens + int(e.get("extra_tokens") or 0)}
    if "temperature" in supported:
        body["temperature"] = temperature
    if seed is not None and seed >= 0 and "seed" in supported:
        body["seed"] = int(seed)
    body.update(e.get("extra_body") or {})
    return body


def strip_thinking(text: str) -> str:
    """Drop <think>…</think> blocks some open models put before the answer."""
    return _THINK.sub("", text or "").strip()


def api_key(entry: dict) -> str | None:
    name = entry.get("api_key_env")
    return os.environ.get(name) if name else None


def check_connection(entry: dict, transport=None, timeout: float = 10.0) -> tuple[bool, str]:
    """GET <base_url>/models: free, sends no prompt. True if the server answers and serves the configured model."""
    headers = {"Authorization": f"Bearer {api_key(entry)}"} if api_key(entry) else {}
    try:
        with httpx.Client(timeout=timeout, transport=transport, headers=headers) as http:
            r = http.get(entry["base_url"].rstrip("/") + "/models")
        r.raise_for_status()
        served = [m.get("id") for m in (r.json().get("data") or [])]
    except (httpx.HTTPError, ValueError) as e:
        return False, f"Could not reach {entry['base_url']}: {e}"
    listed = ", ".join(served[:8]) + (" …" if len(served) > 8 else "")
    if entry["model"] not in served:
        return False, f"The server answers, but {entry['model']} is not served. It serves: {listed or 'nothing'}."
    return True, f"Connected. {entry['model']} is served (the server lists: {listed})."
