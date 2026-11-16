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
    "content": [{"type": "text", "text": "Hi from upstream"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 3},
}

RESPONSES_MESSAGE = {
    "id": "resp_1",
    "output": [{"type": "message", "content": [{"type": "output_text", "text": "Responses here"}]}],
    "usage": {"input_tokens": 10, "output_tokens": 2},
}

ANTHROPIC_SSE = (
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_s1","usage":{"input_tokens":5}}}\n\n'
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"streamed"}}\n\n'
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
    'event: message_stop\ndata: {"type":"message_stop"}\n\n'
)


RESPONSES_SSE = (
    'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_2"}}\n\n'
    'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"Responses here"}\n\n'
    'event: response.completed\ndata: {"type":"response.completed","response":{"id":"resp_2",'
    '"usage":{"input_tokens":10,"output_tokens":2}}}\n\n'
)


def _upstream(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content) if request.content else {}
    if request.url.path == "/v1/messages":
        if body.get("stream"):
            return httpx.Response(200, content=ANTHROPIC_SSE.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=ANTHROPIC_MESSAGE)
    if request.url.path == "/v1/responses":
        if body.get("stream"):
            return httpx.Response(200, content=RESPONSES_SSE.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=RESPONSES_MESSAGE)
    return httpx.Response(404, json={"error": {"message": "not found"}})


def _client(tmp_path, exposed):
    settings_path = tmp_path / "config.json"
    save_settings(Settings(upstream_url="http://gateway.test", exposed_models=exposed), settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude-settings.json",
        desktop_app_dir=tmp_path / "Claude-3p",
        desktop_backup_dir=tmp_path / "backup",
        upstream_transport=httpx.MockTransport(_upstream),
    )
    return TestClient(app)


def test_chat_completions_nonstream_claude_model(tmp_path):
    client = _client(tmp_path, ["claude-haiku-5-5"])
    resp = client.post("/v1/chat/completions", json={
        "model": "claude-haiku-5-5",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        "max_tokens": 32,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Hi from upstream"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    row = client.get("/api/usage").json()["rows"][0]
    assert row["model"] == "claude-haiku-5-5"


def test_chat_completions_streams_chat_chunks(tmp_path):
    client = _client(tmp_path, ["claude-haiku-5-5"])
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "claude-haiku-5-5",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    lines = [line for line in text.splitlines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    events = [json.loads(line[6:]) for line in lines[:-1]]
    assert events[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in events[1:]) == "streamed"


def test_chat_completions_not_exposed_error_is_openai_shaped(tmp_path):
    client = _client(tmp_path, ["claude-haiku-5-5"])
    resp = client.post("/v1/chat/completions", json={
        "model": "secret-model", "messages": [{"role": "user", "content": "hi"}]})
    assert resp.status_code == 404
    assert resp.json()["error"]["type"] == "not_found_error"


def test_responses_ingress_nonstream_routes_to_upstream_messages(tmp_path):
    client = _client(tmp_path, ["claude-haiku-5-5"])
    resp = client.post("/v1/responses", json={
        "model": "claude-haiku-5-5",
        "instructions": "be brief",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hey"}]}],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"].startswith("resp_")
    assert data["output"][0]["content"][0]["text"] == "Hi from upstream"
    assert data["usage"]["input_tokens"] == 12


def test_responses_ingress_gpt_model_uses_responses_upstream(tmp_path):
    client = _client(tmp_path, ["gpt-6-luna"])
    resp = client.post("/v1/responses", json={
        "model": "gpt-6-luna", "input": "hello", "stream": False})
    assert resp.status_code == 200
    data = resp.json()
    assert data["output"][0]["content"][0]["text"] == "Responses here"


def test_responses_ingress_gpt_model_streams(tmp_path):
    client = _client(tmp_path, ["gpt-6-luna"])
    with client.stream("POST", "/v1/responses", json={
        "model": "gpt-6-luna", "input": "hello", "stream": True}) as resp:
        assert resp.status_code == 200
        text = "".join(resp.iter_text())
    assert "response.created" in text
    assert "response.output_text.delta" in text
    assert "response.completed" in text
