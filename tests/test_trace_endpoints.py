"""Trace read-model and /api/trace endpoints (plan T1-T4)."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import ledger
from proxy.app import create_app
from proxy.config import Settings, save_settings


UPSTREAM_OK = {
    "id": "msg_up", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1,
              "cache_read_input_tokens": 3, "cache_creation_input_tokens": 2},
}


def _app(tmp_path, feature_values=None):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test",
                        exposed_models=["claude-haiku-5-5"])
    settings.features = feature_values or {"tracing": True}
    save_settings(settings, settings_path)
    return create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(lambda r: httpx.Response(200, json=UPSTREAM_OK)),
        manage_sidecar=False,
    )


def _fixture_ledger(tmp_path):
    """Two round trips with v2 offsets so the timeline is drawable."""
    common = dict(session_key="claude:fx", session_source="claude", run_label="fx",
                  ledger_dir=tmp_path)
    ledger.request_event(request_id="r1", requested_model="m", model="m",
                         route="anthropic", stream=False,
                         body={"messages": [{"role": "user", "content": "trace-fx-payload"}]}, **common)
    ledger.stage_event(feature="compression", request_id="r1",
                       provenance={"tokens_before": 10, "tokens_after": 5,
                                   "tokens_saved": 5, "duration_ms": 4, "mode": "ml",
                                   "roundtrip_ms": 0.7}, t0_offset_ms=2.0, **common)
    ledger.response_event(request_id="r1", status=200, duration_ms=120.0,
                          ttft_ms=40.0, t0_offset_ms=130.0, **common)
    ledger.request_event(request_id="r2", requested_model="m", model="m",
                         route="anthropic", stream=False,
                         body={"messages": [{"role": "user", "content": "trace-fx-payload"}]}, **common)
    ledger.response_event(request_id="r2", status=500, duration_ms=10.0,
                          t0_offset_ms=150.0, **common)


def test_timeline_endpoint_gated_on_tracing(tmp_path):
    app = _app(tmp_path)
    with TestClient(app) as client:
        # nothing traced yet
        assert client.get("/api/trace/calls").json()["total"] == 0


def test_timeline_endpoint_lists_round_trips(tmp_path):
    ledger_dir = tmp_path / "sessions"
    common = dict(session_key="claude:fx", session_source="claude", run_label="fx",
                  ledger_dir=ledger_dir)
    ledger.request_event(request_id="r1", requested_model="m", model="m",
                         route="anthropic", stream=False,
                         body={"messages": [{"role": "user", "content": "trace-fx-payload"}]}, **common)
    ledger.response_event(request_id="r1", status=200, duration_ms=120.0,
                          t0_offset_ms=130.0, **common)
    app = _app(tmp_path)
    with TestClient(app) as client:
        calls = client.get("/api/trace/calls").json()
        assert calls["total"] == 1
        row = calls["calls"][0]
        assert row["request_id"] == "r1"
        assert row["session"] == ledger.session_path(ledger_dir, "claude:fx").stem
        detail = client.get(f"/api/trace/calls/{row['session']}/r1").json()
        assert detail["request_id"] == "r1"
        assert detail["route"] == "anthropic"
        assert client.get("/api/trace/calls/nope/x").status_code == 404


def test_timeline_endpoint_shapes(tmp_path):
    ledger_dir = tmp_path / "sessions"
    common = dict(session_key="claude:fx", session_source="claude", run_label="fx",
                  ledger_dir=ledger_dir)
    ledger.request_event(request_id="r1", requested_model="m", model="m",
                         route="anthropic", stream=False,
                         body={"messages": [{"role": "user", "content": "trace-fx-payload"}]}, **common)
    app = _app(tmp_path)
    with TestClient(app) as client:
        key = ledger.session_path(ledger_dir, "claude:fx").stem
        timeline = client.get(f"/api/sessions/{key}/timeline")
        assert timeline.status_code == 200
        body = timeline.json()
        assert body["key"].endswith(".jsonl") is False
        assert isinstance(body["steps"], list) and len(body["steps"]) == 1
        assert client.get("/api/sessions/missing-session-totally/timeline").status_code == 404
