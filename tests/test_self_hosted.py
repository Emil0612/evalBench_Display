import json

import httpx
import pytest

from evalbench import runner, store
from evalbench.llm import selfhosted
from evalbench.methods import price_recall

ENTRY = {"base_url": "http://gpu-box:8000/v1", "model": "Qwen/Qwen3-8B", "cutoff": "2024-12-31",
         "released": "2025-04-29", "quantization": "bf16", "parameters": ["temperature", "seed"]}


def write_config(tmp_path, models: dict):
    path = tmp_path / "self_hosted.yaml"
    for name, entry in models.items():
        selfhosted.save_entry(name, entry, path)
    return path


# ---------------------------------------------------------------- config → metadata
def test_entries_become_free_models_with_their_cutoff_and_release(tmp_path):
    meta = selfhosted.to_meta(selfhosted.load(write_config(tmp_path, {"qwen3-8b": ENTRY})))
    m = meta["self/qwen3-8b"]
    assert m["name"] == "Self-hosted: qwen3-8b"
    assert m["knowledge_cutoff"] == "2024-12-31"
    assert m["pricing"] == {"prompt": "0", "completion": "0"}
    assert m["supported_parameters"] == ["max_tokens", "temperature", "seed"]
    assert m["self_hosted"]["base_url"] == "http://gpu-box:8000/v1" and m["self_hosted"]["quantization"] == "bf16"
    assert m["created"] > 0
    assert selfhosted.is_self_hosted("self/qwen3-8b") and not selfhosted.is_self_hosted("qwen/qwen3-8b")


def test_a_missing_config_file_means_no_self_hosted_models(tmp_path):
    assert selfhosted.load(tmp_path / "absent.yaml") == {}


@pytest.mark.parametrize("name, entry, message", [
    ("Qwen 8B", ENTRY, "name"),
    ("qwen", {**ENTRY, "base_url": "gpu-box:8000"}, "http"),
    ("qwen", {**ENTRY, "model": ""}, "model"),
    ("qwen", {**ENTRY, "api_key_env": "sk-or-v1-abc123"}, "name of an environment variable"),
])
def test_bad_entries_are_refused_with_a_reason(tmp_path, name, entry, message):
    with pytest.raises(ValueError, match=message):
        selfhosted.save_entry(name, entry, tmp_path / "self_hosted.yaml")


def test_saving_keeps_other_entries_and_deleting_removes_one(tmp_path):
    path = write_config(tmp_path, {"a": ENTRY, "b": {**ENTRY, "model": "other"}})
    selfhosted.delete_entry("a", path)
    assert list(selfhosted.load(path)) == ["self/b"]


# ---------------------------------------------------------------- request body
def test_body_uses_the_served_name_and_only_supported_parameters():
    meta = selfhosted.to_meta({"self/q": {**ENTRY, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                                          "extra_tokens": 500}})["self/q"]
    body = selfhosted.build_body(meta, [{"role": "user", "content": "hi"}], max_tokens=40, temperature=0.0, seed=7)
    assert body["model"] == "Qwen/Qwen3-8B"
    assert body["temperature"] == 0.0 and body["seed"] == 7 and body["max_tokens"] == 540
    assert "provider" not in body and "usage" not in body            # OpenRouter-only fields
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    no_params = selfhosted.to_meta({"self/q": {**ENTRY, "parameters": []}})["self/q"]
    assert "temperature" not in selfhosted.build_body(no_params, [], max_tokens=40, temperature=0.0, seed=7)


def test_thinking_blocks_are_removed_from_the_answer():
    assert selfhosted.strip_thinking("<think>Let me recall…</think>\n{\"close\": 101.5}") == "{\"close\": 101.5}"
    assert selfhosted.strip_thinking("{\"close\": 101.5}") == "{\"close\": 101.5}"


# ---------------------------------------------------------------- connection test (free: lists the served models)
def test_connection_check_lists_served_models_and_finds_the_configured_one():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-8B"}, {"id": "other"}]})
    ok, message = selfhosted.check_connection(ENTRY, transport=httpx.MockTransport(handler))
    assert ok and "Qwen/Qwen3-8B" in message


def test_connection_check_reports_a_missing_model_or_an_unreachable_server():
    ok, message = selfhosted.check_connection(
        ENTRY, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": [{"id": "other"}]})))
    assert not ok and "not served" in message

    def down(request):
        raise httpx.ConnectError("connection refused")
    ok, message = selfhosted.check_connection(ENTRY, transport=httpx.MockTransport(down))
    assert not ok and "connection refused" in message


# ---------------------------------------------------------------- runner
def test_a_run_sends_self_hosted_calls_to_their_own_server_without_an_openrouter_key(tmp_path, monkeypatch, data):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("evalbench.llm.openrouter.load_dotenv", lambda *a, **k: None)
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), json.loads(request.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>hm</think>{\"price\": 199.0}"},
                                                      "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 50, "completion_tokens": 9}})
    meta = selfhosted.to_meta({"self/q": ENTRY})
    conn = store.connect(tmp_path / "t.sqlite")
    calls = runner.plan(["self/q"], ["price_recall"], 1, "forced", {"price_recall": ("price_recall/t", price_recall.build_items(data, 1, 1))})
    run_id, state = runner.run(calls, conn, meta, spend_cap=0.0, transport=httpx.MockTransport(handler))
    assert state.done == 1 and not state.errors and state.spent == 0 and not state.stopped
    assert seen[0][0] == "http://gpu-box:8000/v1/chat/completions" and seen[0][1]["model"] == "Qwen/Qwen3-8B"
    row = conn.execute("select response, privacy, provider, cost from calls").fetchone()
    assert row == ("{\"price\": 199.0}", "self-hosted", "self-hosted · http://gpu-box:8000/v1", 0.0)


def test_only_openrouter_models_are_paid():
    assert runner.is_paid("openai/gpt-4.1-mini")
    assert not runner.is_paid("self/q") and not runner.is_paid("mock/oracle@2024-06-30")


def test_self_hosted_models_are_offered_in_the_picker_next_to_openrouter_models():
    from evalbench.llm.openrouter import main_variants, selectable
    meta = {"openai/gpt-4.1-mini": {"pricing": {"prompt": "0.0000004"}, "supported_parameters": ["max_tokens"]},
            **selfhosted.to_meta({"self/q": ENTRY})}
    assert set(main_variants(selectable(meta))) == {"openai/gpt-4.1-mini", "self/q"}
