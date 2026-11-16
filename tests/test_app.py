import json

import httpx
from fastapi.testclient import TestClient

from proxy.app import create_app
from proxy.config import Settings, save_settings

ANTHROPIC_MESSAGE = {
    "id": "msg_up",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "Hi there"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 3, "cache_read_input_tokens": 6,
              "cache_creation_input_tokens": 0},
}

RESPONSES_MESSAGE = {
    "id": "resp_1",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi!"}]}],
    "usage": {"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 4}},
}

SSE_RESPONSES = (
    'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_2"}}\n\n'
    'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Yo"}\n\n'
    'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_2",'
    '"usage":{"input_tokens":5,"output_tokens":1}}}\n\n'
)


def _upstream(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content) if request.content else {}
    if request.url.path == "/v1/messages":
        assert body["model"] == "claude-haiku-5-5"
        return httpx.Response(200, json=ANTHROPIC_MESSAGE)
    if request.url.path == "/v1/responses":
        assert body["store"] is False
        if body.get("stream"):
            return httpx.Response(200, content=SSE_RESPONSES.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=RESPONSES_MESSAGE)
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "gpt-6-luna", "display_name": "GPT-6 Luna"}]})
    return httpx.Response(404, json={"error": {"message": "not found"}})


def _client(tmp_path, exposed):
    settings_path = tmp_path / "config.json"
    save_settings(Settings(upstream_url="http://gateway.test", exposed_models=exposed), settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude-settings.json",
        upstream_transport=httpx.MockTransport(_upstream),
    )
    return TestClient(app)


def test_anthropic_passthrough_strips_context_suffix_and_logs_usage(tmp_path):
    client = _client(tmp_path, ["claude-haiku-5-5"])
    resp = client.post(
        "/v1/messages",
        json={"model": "claude-haiku-5-5[1m]", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
        headers={"x-api-key": "k"},
    )
    assert resp.status_code == 200
    assert resp.json()["content"][0]["text"] == "Hi there"

    rows = client.get("/api/usage").json()["rows"]
    assert rows[0]["route"] == "anthropic"
    assert rows[0]["cache_read_tokens"] == 6


def test_gpt_model_is_translated_to_responses(tmp_path):
    client = _client(tmp_path, ["gpt-6-luna"])
    resp = client.post(
        "/v1/messages",
        json={"model": "gpt-6-luna", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
    )
    body = resp.json()
    assert resp.status_code == 200
    assert body["model"] == "gpt-6-luna"
    assert body["content"] == [{"type": "text", "text": "Hi!"}]
    assert body["usage"]["input_tokens"] == 6
    assert body["usage"]["cache_read_input_tokens"] == 4


def test_gpt_streaming_is_translated(tmp_path):
    client = _client(tmp_path, ["gpt-6-luna"])
    with client.stream(
        "POST",
        "/v1/messages",
        json={"model": "gpt-6-luna", "stream": True, "max_tokens": 10,
              "messages": [{"role": "user", "content": "hi"}]},
    ) as resp:
        text = "".join(resp.iter_text())
    assert resp.status_code == 200
    assert "event: message_start" in text
    assert '"text": "Yo"' in text
    assert "event: message_stop" in text


def test_unexposed_model_is_rejected(tmp_path):
    client = _client(tmp_path, ["claude-haiku-5-5"])
    resp = client.post(
        "/v1/messages",
        json={"model": "gpt-6-luna", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_models_endpoint_lists_only_exposed(tmp_path):
    client = _client(tmp_path, ["gpt-6-luna"])
    data = client.get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == ["gpt-6-luna"]


def test_models_endpoint_serves_gateway_display_names(tmp_path):
    """Claude Code's model picker shows display names, not raw ids."""
    client = _client(tmp_path, ["gpt-6-luna"])
    data = client.get("/v1/models").json()["data"]
    assert data[0]["type"] == "model"
    assert data[0]["id"] == "gpt-6-luna"
    assert data[0]["display_name"] == "GPT-6 Luna"


def test_models_endpoint_pretty_prints_unknown_ids(tmp_path):
    """No metadata from the gateway -> readable fallback name from the id."""
    client = _client(tmp_path, ["totally-unknown-model:free"])
    data = client.get("/v1/models").json()["data"]
    assert data[0]["display_name"] == "Totally Unknown Model (free)"


def test_models_endpoint_supports_pagination(tmp_path):
    """Gateway discovery pages with limit/after_id; the proxy answers them."""
    client = _client(tmp_path, ["a-one", "b-two", "c-three"])
    page1 = client.get("/v1/models", params={"limit": 2}).json()
    assert [m["id"] for m in page1["data"]] == ["a-one", "b-two"]
    assert page1["last_id"] == "b-two"
    page2 = client.get("/v1/models", params={"after_id": "b-two", "limit": 2}).json()
    assert [m["id"] for m in page2["data"]] == ["c-three"]


def test_claude_apply_writes_bearer_env_with_gateway_key(tmp_path):
    """/api/claude/apply must wire ANTHROPIC_AUTH_TOKEN (bearer), blank the
    x-api-key credential and enable gateway model discovery."""
    settings_path = tmp_path / "config.json"
    save_settings(Settings(upstream_url="http://gateway.test", exposed_models=["claude-haiku-5-5"],
                           api_key="sk-gateway-secret"), settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude-settings.json",
        upstream_transport=httpx.MockTransport(_upstream),
    )
    with TestClient(app) as client:
        assert client.post("/api/claude/apply").status_code == 200
    data = json.loads((tmp_path / "claude-settings.json").read_text(encoding="utf-8"))
    env = data["env"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:18090"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-gateway-secret"
    assert env["ANTHROPIC_API_KEY"] == ""
    assert env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] == "1"
    assert env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "claude-haiku-5-5"
    assert data["availableModels"] == ["claude-haiku-5-5"]


def test_settings_rejects_proxy_as_its_own_upstream(tmp_path):
    client = _client(tmp_path, [])
    resp = client.put("/api/settings", json={"upstream_url": "http://127.0.0.1:18090"})
    assert resp.status_code == 400


def test_dashboard_serves_gui_with_sessions_and_features(tmp_path):
    client = _client(tmp_path, [])
    resp = client.get("/")
    assert resp.status_code == 200
    assert "features-form" in resp.text
    assert "Sessions" in resp.text
    assert "Green metrics" in resp.text
    assert "EcoLogits" in resp.text       # attribution/license footer is mandatory
    assert "green-form" in resp.text       # green settings group renders
    assert "green-attribution" in resp.text
    assert "session-timeline" in resp.text or "sessions-listwrap" in resp.text


def test_malformed_json_body_is_an_anthropic_400(tmp_path):
    client = _client(tmp_path, ["claude-haiku-5-5"])
    resp = client.post("/v1/messages", content=b"{not json",
                       headers={"content-type": "application/json", "x-api-key": "k"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_non_object_body_is_an_anthropic_400(tmp_path):
    client = _client(tmp_path, ["claude-haiku-5-5"])
    resp = client.post("/v1/messages", json=[1, 2, 3], headers={"x-api-key": "k"})
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_gateway_key_replaces_client_credentials_upstream(tmp_path):
    """The upstream must see the gateway's key, never the client's dummy token."""
    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["authorization"] = request.headers.get("authorization")
        seen["x-api-key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json=ANTHROPIC_MESSAGE)

    settings_path = tmp_path / "config.json"
    save_settings(Settings(upstream_url="http://gateway.test", exposed_models=["claude-haiku-5-5"],
                           api_key="sk-gateway-secret"), settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude-settings.json",
        upstream_transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        client.post("/v1/messages", json={"model": "claude-haiku-5-5", "max_tokens": 10,
                                          "messages": [{"role": "user", "content": "hi"}]},
                    headers={"content-type": "application/json", "x-api-key": "sk-client-dummy"})
    assert seen["authorization"] == "Bearer sk-gateway-secret"
    assert seen["x-api-key"] is None


def test_without_gateway_key_client_credentials_flow_through(tmp_path):
    """Transparent mode: with no configured key the proxy forwards client credentials."""
    seen = {}

    def upstream(request: httpx.Request) -> httpx.Response:
        seen["x-api-key"] = request.headers.get("x-api-key")
        seen["authorization"] = request.headers.get("authorization")
        return httpx.Response(200, json=ANTHROPIC_MESSAGE)

    settings_path = tmp_path / "config.json"
    save_settings(Settings(upstream_url="http://gateway.test", exposed_models=["claude-haiku-5-5"]), settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude-settings.json",
        upstream_transport=httpx.MockTransport(upstream),
    )
    with TestClient(app) as client:
        client.post("/v1/messages", json={"model": "claude-haiku-5-5", "max_tokens": 10,
                                          "messages": [{"role": "user", "content": "hi"}]},
                    headers={"content-type": "application/json", "x-api-key": "sk-real-company-key"})
    assert seen["x-api-key"] == "sk-real-company-key"
