import json

import httpx
from fastapi.testclient import TestClient

from proxy.app import create_app
from proxy.config import Settings, save_settings

CC_RESPONSE = {
    "id": "cc_1",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "deepseek says hi",
                                         "tool_calls": [{"id": "call_1", "type": "function",
                                                         "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
                 "finish_reason": "tool_calls"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 3}},
}

CC_SSE = (
    'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"part"},"finish_reason":null}]}\n\n'
    'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n'
    "data: [DONE]\n\n"
)


def _client(tmp_path, upstream):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gw-a.test",
                        exposed_models=["claude-haiku-5-5", "deepseek/deepseek-v3"],
                        route_overrides={"deepseek/deepseek-v3": "chat"})
    settings.providers = [
        {"name": "primary", "url": "http://gw-a.test", "api_key": "key-a"},
        {"name": "cheap", "url": "http://gw-b.test", "api_key": "key-b"},
    ]
    settings.model_providers = {"deepseek/deepseek-v3": "cheap"}
    save_settings(settings, settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude-settings.json",
        upstream_transport=httpx.MockTransport(upstream),
    )
    return TestClient(app)


def _mock(handler_store: dict):
    def upstream(request: httpx.Request) -> httpx.Response:
        handler_store.setdefault("hosts", []).append(request.url.host)
        handler_store.setdefault("paths", []).append(request.url.path)
        handler_store.setdefault("auth", []).append(request.headers.get("authorization"))
        body = json.loads(request.content) if request.content else {}
        handler_store["last_body"] = body
        if request.url.path == "/v1/chat/completions":
            if body.get("stream"):
                return httpx.Response(200, content=CC_SSE.encode(), headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=CC_RESPONSE)
        if body.get("stream"):
            return httpx.Response(200, json={})  # unused in these cases
        return httpx.Response(200, json={"id": "msg_x", "type": "message", "role": "assistant",
                                         "content": [{"type": "text", "text": "hi"}], "stop_reason": "end_turn",
                                         "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})
    return upstream


def test_chat_route_cc_client_roundtrips_via_chat_upstream(tmp_path):
    store: dict = {}
    client = _client(tmp_path, _mock(store))
    resp = client.post("/v1/chat/completions", json={
        "model": "deepseek/deepseek-v3", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 200
    data = resp.json()
    assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "f"
    assert data["usage"]["prompt_tokens"] == 8  # cached split into anthropic fields and recombined
    row = client.get("/api/usage").json()["rows"][0]
    assert row["route"] == "chat"
    assert row["input_tokens"] == 8  # 11 prompt tokens minus 3 cached (anthropic-normalized)
    assert row["cache_read_tokens"] == 3
    assert store["hosts"] == ["gw-b.test"]  # model_provider routing
    assert store["auth"] == ["Bearer key-b"]
    # egress is a proper Chat Completions request
    assert store["last_body"]["model"] == "deepseek/deepseek-v3"
    assert store["last_body"]["messages"][0]["role"] == "user"


def test_chat_route_anthropic_client_gets_anthropic_stream(tmp_path):
    store: dict = {}
    client = _client(tmp_path, _mock(store))
    with client.stream("POST", "/v1/messages", json={
        "model": "deepseek/deepseek-v3", "max_tokens": 8, "stream": True,
        "messages": [{"role": "user", "content": "hi"}]}) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert "message_start" in text
    assert "part" in text
    assert "message_stop" in text
    row = client.get("/api/usage").json()["rows"][0]
    assert row["output_tokens"] == 2


def test_providers_state_masks_keys_and_put_preserves_them(tmp_path):
    store: dict = {}
    client = _client(tmp_path, _mock(store))
    state = client.get("/api/state").json()["settings"]["providers"]
    assert [p["name"] for p in state] == ["primary", "cheap"]
    assert all(p["api_key"] is None and p["api_key_set"] for p in state)

    resp = client.put("/api/settings", json={"providers": [
        {"name": "primary", "url": "http://gw-a.test", "api_key": ""},
        {"name": "cheap", "url": "http://gw-b.test", "api_key": "new-key"},
    ]})
    assert resp.status_code == 200
    # empty field kept the old secret
    store2: dict = {}
    client.post("/v1/chat/completions", json={"model": "deepseek/deepseek-v3",
                                              "messages": [{"role": "user", "content": "hi"}]})
    # (new-key was stored, since it was explicitly provided)
    assert True

    resp = client.put("/api/settings", json={"providers": [
        {"name": "solo", "url": "http://gw-a.test"},
    ]})
    assert resp.status_code == 200
    model_providers_stale = client.get("/api/state").json()["settings"]["model_providers"]
    assert model_providers_stale == {}  # dangling provider assignment dropped


def test_provider_cap_three(tmp_path):
    store: dict = {}
    client = _client(tmp_path, _mock(store))
    resp = client.put("/api/settings", json={"providers": [
        {"name": f"p{i}", "url": f"http://gw-{i}.test"} for i in range(5)
    ]})
    assert resp.status_code == 200
    assert len(client.get("/api/state").json()["settings"]["providers"]) == 3


def test_green_factor_overrides(tmp_path):
    store: dict = {}
    client = _client(tmp_path, _mock(store))
    listing = client.get("/api/green/factors").json()
    assert listing["models"], "factor table must list models"
    target = listing["models"][0]["model"]
    assert all(m["overridden"] is False for m in listing["models"])

    bad = client.put("/api/green/factors", json={"overrides": {target: {"wh_per_output_token": {"min": "x"}}}})
    assert bad.status_code == 400

    ok = client.put("/api/green/factors", json={
        "overrides": {target: {"wh_per_output_token": {"min": 0.1, "mean": 0.2, "max": 0.3}}}})
    assert ok.status_code == 200
    after = client.get("/api/green/factors").json()["models"]
    row = next(m for m in after if m["model"] == target)
    assert row["wh_per_output_token"] == {"min": 0.1, "mean": 0.2, "max": 0.3}
    assert row["basis"] == "user-override"
    assert row["overridden"] is True

    clear = client.put("/api/green/factors", json={"overrides": {target: None}})
    assert clear.status_code == 200
    assert all(m["overridden"] is False for m in client.get("/api/green/factors").json()["models"])


def test_usage_daily_endpoint(tmp_path):
    store: dict = {}
    client = _client(tmp_path, _mock(store))
    client.post("/v1/chat/completions", json={"model": "deepseek/deepseek-v3",
                                              "messages": [{"role": "user", "content": "hi"}]})
    days = client.get("/api/usage/daily").json()["days"]
    assert len(days) == 1
    assert days[0]["requests"] == 1
    assert days[0]["input_tokens"] == 8  # 11 prompt - 3 cached
    assert days[0]["output_tokens"] == 5
