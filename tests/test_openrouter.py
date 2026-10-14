import asyncio
import json

import httpx
import pytest

from evalbench.llm import openrouter as orr


async def _no_sleep(_):
    return None


def test_build_body_plain_model():
    meta = {"supported_parameters": ["temperature", "max_tokens"]}
    body = orr.build_body(meta, "x/m", [{"role": "user", "content": "hi"}], 60)
    assert body["temperature"] == 0 and body["max_tokens"] == 60
    # privacy defaults to standard routing (see test_run_settings.py)
    assert body["provider"] == {"require_parameters": True}
    assert body["usage"] == {"include": True}
    assert "reasoning" not in body


def test_build_body_reasoning_headroom():
    meta = {"supported_parameters": ["reasoning", "max_tokens"],
            "reasoning": {"mandatory": True, "supported_efforts": ["high", "medium", "low"]}}
    body = orr.build_body(meta, "x/r", [], 60)
    assert "temperature" not in body
    assert body["reasoning"] == {"effort": "low", "exclude": True}
    assert body["max_tokens"] == 60 + orr.REASONING_HEADROOM["low"]


def test_build_body_reasoning_can_be_disabled():
    meta = {"supported_parameters": ["reasoning"], "reasoning": {"supported_efforts": ["none", "low"]}}
    body = orr.build_body(meta, "x/r", [], 60)
    assert body["reasoning"]["effort"] == "none" and body["max_tokens"] == 60


def _ok_response(request):
    assert request.headers["Authorization"] == "Bearer test-key"
    return httpx.Response(200, json={
        "model": "x/m", "provider": "Prov",
        "choices": [{"message": {"content": '{"price": 1}'}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3, "cost": 0.00012},
    })


def test_chat_parses_response():
    client = orr.OpenRouterClient(api_key="test-key", transport=httpx.MockTransport(_ok_response), sleep=_no_sleep)
    res = asyncio.run(client.chat({"model": "x/m", "messages": []}))
    assert res.text == '{"price": 1}' and res.cost == pytest.approx(0.00012) and res.provider == "Prov"


def test_chat_retries_on_429():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(429, text="slow down") if calls["n"] == 1 else _ok_response(request)

    client = orr.OpenRouterClient(api_key="test-key", transport=httpx.MockTransport(handler), sleep=_no_sleep)
    assert asyncio.run(client.chat({"model": "x/m", "messages": []})).provider == "Prov"
    assert calls["n"] == 2


def test_chat_raises_on_402():
    client = orr.OpenRouterClient(api_key="test-key", sleep=_no_sleep,
                                  transport=httpx.MockTransport(lambda r: httpx.Response(402, text="no credits")))
    with pytest.raises(orr.OpenRouterError, match="402"):
        asyncio.run(client.chat({"model": "x/m", "messages": []}))


def test_missing_key(monkeypatch):
    monkeypatch.setattr(orr, "load_dotenv", lambda: None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(orr.MissingKeyError):
        orr.OpenRouterClient()


def test_fetch_models_uses_fresh_cache(tmp_path):
    cache = tmp_path / "models.json"
    cache.write_text(json.dumps([{"id": "a/b"}]))
    boom = httpx.MockTransport(lambda r: httpx.Response(500))
    assert orr.fetch_models(cache=cache, transport=boom) == [{"id": "a/b"}]


def test_fetch_models_downloads(tmp_path):
    cache = tmp_path / "models.json"
    t = httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "c/d"}]}))
    assert orr.fetch_models(cache=cache, transport=t) == [{"id": "c/d"}]
    assert json.loads(cache.read_text()) == [{"id": "c/d"}]


def test_build_body_optional_reasoning_is_left_off():
    meta = {"supported_parameters": ["reasoning", "temperature"],
            "reasoning": {"mandatory": False, "supported_efforts": ["high", "low"]}}
    body = orr.build_body(meta, "x/r", [], 60)
    assert body["reasoning"] == {"enabled": False} and body["max_tokens"] == 60  # explicitly off, no headroom


def _client(payload):
    return orr.OpenRouterClient(api_key="test-key", sleep=_no_sleep,
                                transport=httpx.MockTransport(lambda r: httpx.Response(200, json=payload)))


def test_in_generation_error_raises():
    payload = {"choices": [{"message": {"content": ""}, "finish_reason": "error",
                            "error": {"code": 502, "message": "provider died"}}], "usage": {"cost": 0.0}}
    with pytest.raises(orr.OpenRouterError, match="provider died"):
        asyncio.run(_client(payload).chat({"model": "x/m", "messages": []}))


def test_missing_choices_raises():
    with pytest.raises(orr.OpenRouterError):
        asyncio.run(_client({"usage": {}}).chat({"model": "x/m", "messages": []}))


def test_finish_reason_is_kept():
    payload = {"choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {"cost": 0.001}}
    assert asyncio.run(_client(payload).chat({"model": "x/m", "messages": []})).finish_reason == "length"


def test_selectable_models_filters_unusable():
    models = [
        {"id": "a/ok", "supported_parameters": ["max_tokens", "temperature"], "pricing": {"prompt": "0.000001"},
         "architecture": {"output_modalities": ["text"]}},
        {"id": "b/no-max", "supported_parameters": ["temperature"], "pricing": {"prompt": "0.000001"},
         "architecture": {"output_modalities": ["text"]}},
        {"id": "c/image", "supported_parameters": ["max_tokens"], "pricing": {"prompt": "0.000001"},
         "architecture": {"output_modalities": ["image"]}},
        {"id": "d/web:online", "supported_parameters": ["max_tokens"], "pricing": {"prompt": "0.000001"},
         "architecture": {"output_modalities": ["text"]}},
    ]
    assert list(orr.selectable(orr.models_by_id(models))) == ["a/ok"]


def test_main_variants_hides_duplicates_but_keeps_unique_models():
    ids = ["a/m", "a/m:free", "a/m-2024-05-13", "a/m-20240513", "b/x-2501", "~a/latest", "c/y:beta", "c/y",
           "c/y:batch", "d/z:free"]
    kept = orr.main_variants({i: {} for i in ids})
    assert sorted(kept) == ["a/m", "b/x-2501", "c/y", "d/z:free"]  # d/z exists only as :free, so it stays
