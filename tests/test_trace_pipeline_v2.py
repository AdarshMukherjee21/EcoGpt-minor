"""Live-round regression for ledger v2 traces through the FastAPI app.

Covers: t0_offset_ms on every event kind, request-anchored waterfall ordering,
stream marks (ttft/chunk_count/response route), compression mode/byte counts,
and router dispatch_route/tier_model_alias plumbing.
"""

import json

import httpx
from fastapi.testclient import TestClient

from proxy import features as fs
from proxy.app import create_app
from proxy.config import Settings, save_settings


UPSTREAM_OK = {
    "id": "msg_up", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1,
              "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2},
}

CLASSIFIER_OK = {
    "answers": {"difficulty": {"choice": "tier1", "confidence": 0.9,
                               "probabilities": {"tier1": 0.9}}},
}

HEADROOM_OK = {
    "body": {"model": "claude-haiku-5-5", "system": "sys",
             "messages": [{"role": "user", "content": "short"}]},
    "turn_id": "turn-1", "obligations": [],
    "tokens_before": 50, "tokens_after": 10, "tokens_saved": 40,
    "transforms_applied": ["ml_squeeze"],
    "session": {"cached_prefix_replayed": False},
}


def _app(tmp_path, headroom_handler, upstream_seen):
    settings_path = tmp_path / "config.json"
    settings = Settings(
        upstream_url="http://gateway.test",
        exposed_models=["claude-haiku-5-5"],
        headroom={"sidecar_enabled": True, "retries": 0, "backoff_ms": 0},
    )
    settings.features = fs.normalize({"tracing": True, "compression": True,
                                      "green_metrics": False})
    save_settings(settings, settings_path)
    return create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(
            lambda r: (upstream_seen.append(dict(r.headers)), httpx.Response(200, json=UPSTREAM_OK))[1]),
        sidecar_transport=httpx.MockTransport(headroom_handler),
        manage_sidecar=False,
    )


def _entries(tmp_path):
    sessions = list((tmp_path / "sessions").glob("*.jsonl"))
    assert len(sessions) == 1
    return [json.loads(l) for l in sessions[0].read_text(encoding="utf-8").splitlines()]


def test_nonstream_roundtrip_v2_marks(tmp_path):
    app = _app(tmp_path, lambda r: httpx.Response(200, json=HEADROOM_OK), [])
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
        assert resp.status_code == 200

    entries = _entries(tmp_path)
    kinds = [e["type"] for e in entries]
    assert kinds[0] == "request" and kinds[-1] == "response"

    request, response = entries[0], entries[-1]
    # envelope v2 on every entry: schema_version + monotonic t0 offsets
    offsets = [e["t0_offset_ms"] for e in entries]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    assert all(e["schema_version"] == 2 for e in entries)
    assert len({e["span_id"] for e in entries}) == len(entries)

    # response observations: served model + effective route
    assert response["model_served"] == "claude-haiku-5-5"
    assert response["route"] == "anthropic"

    # compression stage provenance: mode derived from transforms + byte counts
    comp = next(e for e in entries if e["type"] == "stage")
    assert comp["provenance"]["mode"] == "ml"  # "ml_squeeze" transform name
    assert comp["provenance"]["bytes_before"] >= comp["provenance"]["bytes_after"] > 0
    assert comp["provenance"]["tokens_saved"] == 40


def test_streaming_marks_chunk_count_and_ttft(tmp_path):
    sse = (
        'event: message_start\ndata: {"type":"message_start","message":{"model":"claude-haiku-5-5",'
        '"usage":{"input_tokens":5,"output_tokens":0}}}\n\n'
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"text_delta","text":"ok"}}\n\n'
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":1}}\n\n'
    )

    def sidecar(request):
        return httpx.Response(200, json=HEADROOM_OK)

    settings_path = tmp_path / "config.json"
    settings = Settings(
        upstream_url="http://gateway.test",
        exposed_models=["claude-haiku-5-5"],
        headroom={"sidecar_enabled": False},
    )
    settings.features = fs.normalize({"tracing": True, "compression": False,
                                      "green_metrics": False})
    save_settings(settings, settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})),
        sidecar_transport=httpx.MockTransport(sidecar),
        manage_sidecar=False,
    )
    with TestClient(app) as client:
        body = {**_request_body(), "stream": True}
        resp = client.post("/v1/messages", json=body, headers=CLAUDE_HEADERS)
        assert resp.status_code == 200
        "".join(chunk for chunk in resp.iter_text())

    entries = _entries(tmp_path)
    response = entries[-1]
    assert response["type"] == "response"
    assert response["chunk_count"] >= 1
    assert response["ttft_ms"] is not None
    assert response["route"] == "anthropic"
    # offsets still monotonic across the whole round trip
    offsets = [e.get("t0_offset_ms") for e in entries]
    assert offsets == sorted(offsets)


def _request_body():
    return {"model": "claude-haiku-5-5", "max_tokens": 10, "system": "sys",
            "messages": [{"role": "user", "content": "a fairly long user prompt"}]}


CLAUDE_HEADERS = {"x-claude-code-session-id": "trace-sess-1", "x-api-key": "k"}


def test_router_dispatch_fields_recorded(tmp_path):
    """router stage provenance carries dispatch_route + tier_model_alias."""
    settings_path = tmp_path / "config.json"
    tier_model = "claude-haiku-5-5"
    router_values = {"enabled": True, "auto_model_id": "eco-auto"}
    settings = Settings(
        upstream_url="http://gateway.test",
        exposed_models=["claude-haiku-5-5", "eco-auto"],
        headroom={"sidecar_enabled": False},
        router=router_values,
        router_tiers=[{"id": "easy", "description": "easy work", "model": tier_model},
                      {"id": "hard", "description": "hard work", "model": "upstream-big"}],
    )
    settings.features = fs.normalize({"tracing": True, "compression": False,
                                      "green_metrics": False})
    save_settings(settings, settings_path)

    def classifier(request):
        return httpx.Response(200, json={
            "answers": {"difficulty": {"choice": "easy", "confidence": 0.9,
                                       "probabilities": {"easy": 0.9, "hard": 0.1}}},
        })

    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(lambda r: httpx.Response(200, json=UPSTREAM_OK)),
        router_transport=httpx.MockTransport(classifier),
        manage_sidecar=False,
    )
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json={**_request_body(), "model": "eco-auto"},
                           headers=CLAUDE_HEADERS)
        assert resp.status_code == 200

    entries = _entries(tmp_path)
    router = next(e for e in entries if e["type"] == "stage" and e.get("feature") == "router")
    provenance = router["provenance"]
    assert provenance["final_tier"] == "easy"
    assert provenance["dispatch_route"]  # effective route is stamped after rules apply
    assert provenance["tier_model_alias"] is False  # tier model used verbatim
    assert provenance["target_model"] == tier_model
