"""OpenRouter access: the public model list, request bodies, and an async chat client with retries."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

from ..items import ROOT

BASE_URL = "https://openrouter.ai/api/v1"
MODELS_CACHE = ROOT / "results" / "models_cache.json"
PINS_PATH = ROOT / "config" / "provider_pins.yaml"
# extra output tokens allowed for the reasoning itself, by effort (reasoning tokens count against max_tokens)
REASONING_HEADROOM = {"minimal": 1024, "low": 2048, "medium": 4096, "high": 8192}
EFFORTS = ("none", "minimal", "low", "medium", "high")
# zdr: only zero-data-retention endpoints; no_training: only providers that don't train on prompts
PRIVACY = ("standard", "no_training", "zdr")
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


class MissingKeyError(RuntimeError):
    pass


class OpenRouterError(RuntimeError):
    def __init__(self, status, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


@dataclass
class ChatResult:
    text: str
    cost: float
    prompt_tokens: int | None
    completion_tokens: int | None
    provider: str | None
    model: str | None
    raw: dict = field(default_factory=dict)
    finish_reason: str | None = None
    logprobs: list | None = None     # per output token: token, logprob, top_logprobs (when requested and supported)


def fetch_models(cache: Path = MODELS_CACHE, max_age_hours: float = 24, transport=None) -> list[dict]:
    """Public model list (no key needed), cached on disk."""
    cache = Path(cache)
    if cache.exists() and time.time() - cache.stat().st_mtime < max_age_hours * 3600:
        return json.loads(cache.read_text(encoding="utf-8"))
    with httpx.Client(base_url=BASE_URL, timeout=30, transport=transport) as client:
        r = client.get("/models")
        r.raise_for_status()
        models = r.json()["data"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(models), encoding="utf-8")
    return models


PRICE_FIELDS = ("prompt", "completion", "request")


def load_pins(path: Path = PINS_PATH) -> dict[str, str]:
    """Model id -> the one provider endpoint it is pinned to (e.g. 'novita/bf16'). Missing file: nothing pinned."""
    path = Path(path)
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {str(m): str(r) for m, r in raw.items() if r}


def route_matches(tag: str, route: str) -> bool:
    """OpenRouter slug matching: 'novita/bf16' matches only that endpoint, a base slug 'deepinfra' all of its
    endpoints."""
    return tag == route or tag.startswith(route + "/")


def fetch_endpoints(model_id: str, transport=None) -> list[dict]:
    """Current per-provider endpoints of one model, with their prices (public, no key, never cached)."""
    with httpx.Client(base_url=BASE_URL, timeout=15, transport=transport) as client:
        r = client.get(f"/models/{model_id}/endpoints")
        r.raise_for_status()
        return (r.json().get("data") or {}).get("endpoints") or []


def worst_case_pricing(endpoints: list[dict]) -> dict[str, float]:
    """The highest price per field over every provider: the model list shows one price, but standard routing can
    send a request to any of them."""
    out: dict[str, float] = {}
    for f in PRICE_FIELDS:
        vals = []
        for e in endpoints:
            try:
                vals.append(float((e.get("pricing") or {}).get(f) or 0))
            except (TypeError, ValueError):
                continue
        if vals:
            out[f] = max(vals)
    return out


def with_worst_case_prices(meta_by_id: dict, model_ids, fetch=fetch_endpoints,
                           pins: dict | None = None) -> tuple[dict, dict]:
    """A copy of meta_by_id whose models carry their most expensive provider's current prices, plus per-model info.
    A pinned model is priced at its pinned endpoint only. A model whose price check fails, or whose pin matches no
    current endpoint, keeps its listed price (info[model]["ok"] is False)."""
    priced, info = dict(meta_by_id), {}
    for m in model_ids:
        base = meta_by_id.get(m) or {}
        listed = base.get("pricing") or {}
        route = (pins or {}).get(m, "")
        try:
            endpoints = fetch(m)
            if route:
                endpoints = [e for e in endpoints if route_matches(str(e.get("tag") or ""), route)]
            worst = worst_case_pricing(endpoints)
        except Exception:
            endpoints, worst = [], {}
        if not worst:
            info[m] = {"ok": False, "providers": 0, "listed": listed, "worst": listed, "route": route}
            continue
        pricing = dict(listed)
        for f, v in worst.items():
            try:
                pricing[f] = max(v, float(listed.get(f) or 0))
            except (TypeError, ValueError):
                pricing[f] = v
        priced[m] = {**base, "pricing": pricing}
        info[m] = {"ok": True, "providers": len(endpoints), "listed": listed, "worst": pricing, "route": route}
    return priced, info


def models_by_id(models: list[dict]) -> dict[str, dict]:
    return {m["id"]: m for m in models}


_DATED = re.compile(r"-(?:\d{4}-\d{2}-\d{2}|\d{8})$")


def main_variants(meta_by_id: dict[str, dict]) -> dict[str, dict]:
    """Drop duplicates of models that also exist in their plain form: ':free', ':beta', ':batch' ... variants,
    dated snapshots ('-2024-05-13') and '~' aliases. A model that only exists as a variant is kept."""
    ids = set(meta_by_id)
    out = {}
    for mid, m in meta_by_id.items():
        if mid.startswith("~"):
            continue
        base, sep, _ = mid.partition(":")
        if sep and base in ids:
            continue
        undated = _DATED.sub("", mid)
        if undated != mid and undated in ids:
            continue
        out[mid] = m
    return out


def selectable(meta_by_id: dict[str, dict]) -> dict[str, dict]:
    """Models this benchmark can call: text output, priced, accept max_tokens (require_parameters would
    otherwise reject every request), and no web-search (:online) variants."""
    out = {}
    for mid, m in meta_by_id.items():
        if mid.endswith(":online") or "max_tokens" not in (m.get("supported_parameters") or []):
            continue
        if "text" not in ((m.get("architecture") or {}).get("output_modalities") or ["text"]):
            continue
        if not (m.get("pricing") or {}).get("prompt"):
            continue
        out[mid] = m
    return out


def build_body(meta: dict | None, model_id: str, messages: list[dict], max_tokens: int,
               temperature: float = 0.0, reasoning: str = "auto", seed: int | None = None,
               privacy: str = "standard", route: str | None = None, logprobs: int = 0) -> dict:
    """Only send parameters the model supports: require_parameters would otherwise reject the request."""
    if privacy not in PRIVACY:
        raise ValueError(f"unknown privacy level {privacy!r}")
    meta = meta or {}
    supported = set(meta.get("supported_parameters") or [])
    provider: dict = {"require_parameters": True}
    if privacy == "zdr":
        provider.update(zdr=True, data_collection="deny")
    elif privacy == "no_training":
        provider["data_collection"] = "deny"
    if route:  # pinned: this provider endpoint only, no fallback to others
        provider.update(only=[route], allow_fallbacks=False)
    body = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "usage": {"include": True},
        "provider": provider,
    }
    if "temperature" in supported:
        body["temperature"] = temperature
    if seed is not None and seed >= 0 and "seed" in supported:
        body["seed"] = int(seed)
    if logprobs and {"logprobs", "top_logprobs"} <= supported:   # the top `logprobs` alternatives of every token
        body["logprobs"] = True
        body["top_logprobs"] = int(logprobs)
    if "reasoning" in supported:
        info = meta.get("reasoning") or {}
        effort = choose_effort(reasoning, info.get("supported_efforts") or [], bool(info.get("mandatory")))
        if effort is None:
            # optional reasoning that may be on by default: switch it off explicitly (documented OpenRouter form)
            body["reasoning"] = {"enabled": False}
        else:
            body["reasoning"] = {"effort": effort, "exclude": True}
            if effort != "none":
                body["max_tokens"] = max_tokens + REASONING_HEADROOM.get(effort, REASONING_HEADROOM["low"])
    limit = (meta.get("top_provider") or {}).get("max_completion_tokens")
    if limit:
        body["max_tokens"] = min(body["max_tokens"], int(limit))
    return body


def choose_effort(requested: str, efforts: list[str], mandatory: bool) -> str | None:
    """The reasoning effort to send, or None to leave the parameter out.
    'auto' = as little reasoning as the model allows. A model that must reason never gets 'none'."""
    lowest = next((e for e in ("minimal", "low", "medium", "high") if e in efforts), "low")
    if requested == "auto":
        if "none" in efforts:
            return "none"
        return lowest if mandatory else None
    if requested not in EFFORTS:
        raise ValueError(f"unknown reasoning effort {requested!r}")
    if requested == "none":
        if "none" in efforts:
            return "none"
        return lowest if mandatory else None
    if not efforts or requested in efforts:
        return requested
    rank = EFFORTS.index(requested)  # nearest supported effort, preferring the lower one on a tie
    return min(efforts, key=lambda e: (abs(EFFORTS.index(e) - rank) if e in EFFORTS else 99, EFFORTS.index(e)
                                       if e in EFFORTS else 99))


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, transport=None, concurrency: int = 8, timeout: float = 120.0,
                 sleep=asyncio.sleep, max_attempts: int = 5, base_url: str = BASE_URL, key_required: bool = True):
        """base_url + key_required=False: any OpenAI-compatible server (a self-hosted model), with a key only if
        one is given."""
        if api_key is None and key_required:
            load_dotenv()
            api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key and key_required:
            raise MissingKeyError("OPENROUTER_API_KEY is not set (add it to the env file in the project root)")
        headers = {"X-Title": "evalBench"} | ({"Authorization": f"Bearer {api_key}"} if api_key else {})
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport, headers=headers)
        self._sem = asyncio.Semaphore(concurrency)
        self._sleep = sleep
        self._max_attempts = max_attempts

    async def chat(self, body: dict) -> ChatResult:
        async with self._sem:
            last = "no attempt"
            for attempt in range(self._max_attempts):
                try:
                    r = await self._http.post("/chat/completions", json=body)
                except httpx.TransportError as e:
                    last = f"transport error: {e}"
                    await self._sleep(2 ** attempt)
                    continue
                if r.status_code in RETRY_STATUS:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    await self._sleep(2 ** attempt)
                    continue
                if r.is_error:
                    raise OpenRouterError(r.status_code, r.text[:500])
                data = r.json()
                if "error" in data:
                    err = data["error"] or {}
                    raise OpenRouterError(err.get("code", 0), str(err.get("message"))[:500])
                choices = data.get("choices") or []
                if not choices:
                    raise OpenRouterError(0, f"response without choices: {str(data)[:300]}")
                choice = choices[0]
                if choice.get("error") or choice.get("finish_reason") == "error":
                    err = choice.get("error") or {}
                    raise OpenRouterError(err.get("code", 0), str(err.get("message") or "generation error")[:500])
                message = choice.get("message") or {}
                usage = data.get("usage") or {}
                return ChatResult(
                    text=message.get("content") or "",
                    cost=float(usage.get("cost") or 0.0),
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    provider=data.get("provider"),
                    model=data.get("model"),
                    raw=data,
                    finish_reason=choice.get("finish_reason"),
                    logprobs=(choice.get("logprobs") or {}).get("content"),
                )
            raise OpenRouterError(0, f"gave up after {self._max_attempts} attempts; last: {last}")

    async def aclose(self) -> None:
        await self._http.aclose()
