"""End-to-end trace regression: app round trip -> timeline endpoint -> detail.

Owners' goal 1-2 live here: the round trip must be reconstructable from
ledger v2 entries (offsets, stage spans, served model) through the read API.
"""

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


def _app(tmp_path):
    settings_path = tmp_path / "config.json"
    settings = Settings(
        upstream_url="http://gateway.test",
        exposed_models=["claude-haiku-5-5"],
    )
    settings.features = fs.normalize({"tracing": True, "run_label": "e2e"})
    save_settings(settings, settings_path)
    return create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(lambda r: httpx.Response(200, json=UPSTREAM_OK)),
        manage_sidecar=False,
    )


def _request_body():
    return {"model": "claude-haiku-5-5", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hello there"}]}


def test_round_trip_trace_end_to_end(tmp_path):
    app = _app(tmp_path)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_request_body(),
                           headers={"x-api-key": "k"})
        assert resp.status_code == 200

        sessions = client.get("/api/sessions").json()["sessions"]
        assert len(sessions) == 1
        key = sessions[0]["key"]

        timeline = client.get(f"/api/sessions/{key}/timeline").json()
        assert timeline["run_label"] == "e2e"
        assert len(timeline["steps"]) == 1
        step = timeline["steps"][0]
        assert step["status"] == 200
        assert step["model_served"] == "claude-haiku-5-5"
        assert step["usage"]["output"] == 1
        # the round trip must be a drawable timeline: request starts at 0,
        # the compression + upstream spans close before the response completes
        spans = {s["stage"]: s for s in step["spans"]}
        assert spans["request"]["start_ms"] == 0
        for name in ("compression", "upstream"):
            assert spans[name]["start_ms"] is not None
            assert spans[name]["end_ms"] is not None
            assert spans[name]["end_ms"] >= spans[name]["start_ms"]
        # compression ran (disabled path still records a closed zero-ish span)

        detail = client.get(f"/api/trace/calls/{key}/{step['request_id']}").json()
        assert detail["request_id"] == step["request_id"]
        assert detail["ledger_entries"][0]["type"] == "request"

        # summary rollups still in place (ledger v2 additions)
        summary = client.get(f"/api/sessions/{key}/summary").json()
        assert summary["requests"] == 1
        assert summary["timing"]["responses_total"] == 1
