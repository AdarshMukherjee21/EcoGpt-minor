"""Session summary + request drill-down (Phase 1 tracing Work Order B)."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import features as fs
from proxy import green
from proxy import ledger
from proxy.app import create_app
from proxy.config import Settings, save_settings
from proxy.headroom_settings import normalize as headroom_normalize

UPSTREAM_OK = {
    "id": "msg_up", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 3,
              "cache_creation_input_tokens": 2},
}
CLAUDE_HEADERS = {"x-claude-code-session-id": "view-sess-1", "x-api-key": "k"}


def _app(tmp_path, *, feature_values=None):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test", exposed_models=["claude-haiku-5-5"])
    settings.headroom = headroom_normalize({"retries": 0, "backoff_ms": 0})
    settings.features = fs.normalize(feature_values or {"tracing": True})
    save_settings(settings, settings_path)
    return create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(lambda r: httpx.Response(200, json=UPSTREAM_OK)),
        sidecar_transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})),
        manage_sidecar=False,
    )


def _request_body():
    return {"model": "claude-haiku-5-5", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hello"}]}


# ---- fixture ledger ----------------------------------------------------------

SESSION_KEY = "claude:view-sess-1:fixture"
RUN_LABEL = "view-fixture"


def _write_fixture(ledger_dir):
    """Two round trips (different models, one compressed), one error event."""
    common = dict(session_key=SESSION_KEY, session_source="claude",
                  run_label=RUN_LABEL, ledger_dir=ledger_dir)

    rid_1, rid_2 = "req-fixture-1", "req-fixture-2"
    ledger.request_event(requested_model="claude-haiku-5-5", model="claude-haiku-5-5",
                         route="anthropic_passthrough", stream=False, request_id=rid_1,
                         body={"messages": [{"role": "user", "content": "user-zzz-one"}],
                               "system": "sys-zzz", "tools": [{"name": "t1"}]}, **common)
    ledger.stage_event(feature="compression", request_id=rid_1,
                       provenance={"tokens_before": 80, "tokens_after": 20,
                                   "tokens_saved": 60, "fallback": None}, **common)
    ledger.response_event(request_id=rid_1, status=200, duration_ms=120.0,
                          usage={"input_tokens": 5, "output_tokens": 1,
                                 "cache_read_input_tokens": 3,
                                 "cache_creation_input_tokens": 2},
                          stop_reason="end_turn",
                          content=[{"type": "text", "text": "reply-zzz-one"}],
                          impacts={"energy_kwh": {"min": 1.0, "mean": 2.0, "max": 4.0},
                                   "gwp_kgco2e": {"min": 0.5, "mean": 1.0, "max": 2.0},
                                   "model_basis": "estimated"}, **common)
    ledger.request_event(requested_model="claude-haiku-5-5", model="claude-haiku-5-5",
                         route="anthropic_passthrough", stream=False, request_id=rid_2,
                         body={"messages": [{"role": "user", "content": "user-zzz-two"}]}, **common)
    ledger.stage_event(feature="compression", request_id=rid_2,
                       provenance={"tokens_before": 100, "tokens_after": 40,
                                   "tokens_saved": 60, "fallback": "compressor_error"}, **common)
    ledger.error_event(request_id=rid_2, stage="compression", message="boom", **common)
    ledger.response_event(request_id=rid_2, status=200, duration_ms=340.0,
                          usage={"input_tokens": 10, "output_tokens": 4,
                                 "cache_read_input_tokens": 20,
                                 "cache_creation_input_tokens": 6},
                          stop_reason="end_turn",
                          content=[{"type": "text", "text": "reply-zzz-two"}], **common)
    return ledger.session_path(ledger_dir, SESSION_KEY).stem


# ---- tests -------------------------------------------------------------------


def test_summary_aggregates_fixture_ledger(tmp_path):
    ledger_dir = tmp_path / "sessions"
    key = _write_fixture(ledger_dir)
    app = _app(tmp_path)
    with TestClient(app) as client:
        summary = client.get(f"/api/sessions/{key}/summary").json()

    assert summary["run_label"] == RUN_LABEL
    assert summary["requests"] == 2
    assert summary["errors"] == 1
    model = summary["models"]["claude-haiku-5-5"]
    assert model == {"input": 15, "output": 5, "cache_read": 23, "cache_write": 8}
    assert summary["compression"] == {"tokens_before": 180, "tokens_after": 60,
                                      "tokens_saved": 120}
    # only the first response carried impacts: its bounds are the session totals
    assert summary["green"]["energy_kwh"] == pytest.approx({"min": 1.0, "mean": 2.0, "max": 4.0})
    assert summary["green"]["gwp_kgco2e"] == pytest.approx({"min": 0.5, "mean": 1.0, "max": 2.0})
    assert isinstance(summary["wall_clock_seconds"], float)
    assert summary["session_key"] == SESSION_KEY


def test_detail_groups_entries_by_request_id(tmp_path):
    ledger_dir = tmp_path / "sessions"
    key = _write_fixture(ledger_dir)
    app = _app(tmp_path)
    with TestClient(app) as client:
        detail = client.get(f"/api/sessions/{key}").json()

    trips = detail["round_trips"]
    assert [t["request_id"] for t in trips] == ["req-fixture-1", "req-fixture-2"]
    first, second = trips

    assert first["request"]["model"] == "claude-haiku-5-5"
    assert first["request"]["tools"]["names"] == ["t1"]
    assert first["stages"][0]["feature"] == "compression"
    assert first["stages"][0]["provenance"]["tokens_saved"] == 60
    assert first["response"]["usage"]["cache_read"] == 3
    assert first["errors"] == []
    assert "impacts" in first["response"]

    assert second["errors"][0]["stage"] == "compression"
    assert second["stages"][0]["provenance"]["fallback"] == "compressor_error"
    assert "impacts" not in second["response"]
    # grouping must cover every entry exactly once
    grouped = sum(
        (1 if t["request"] else 0) + len(t["stages"])
        + (1 if t["response"] else 0) + len(t["errors"])
        for t in trips
    )
    assert grouped == len(detail["entries"])
    # no payloads anywhere: only metadata, counts and hashes
    raw = json.dumps(detail)
    assert "zzz" not in raw  # payloads never stored, only chars + hashes


def test_unknown_session_and_unsafe_keys_return_404(tmp_path):
    app = _app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/sessions/no-such-session/summary").status_code == 404
        assert client.get("/api/sessions/no-such-session").status_code == 404
        assert client.get("/api/sessions/..").status_code == 404
        assert client.get("/api/sessions/bad%2Fkey/summary").status_code == 404


def test_tracing_off_leaves_no_session_view(tmp_path):
    app = _app(tmp_path, feature_values={"tracing": False})
    with TestClient(app) as client:
        assert client.post("/v1/messages", json=_request_body(),
                           headers=CLAUDE_HEADERS).status_code == 200
        assert client.get("/api/sessions").json()["sessions"] == []
    # no ledger directory contents at all
    assert not list((tmp_path / "sessions").glob("*.jsonl"))


def test_live_request_produces_well_formed_summary(tmp_path):
    app = _app(tmp_path, feature_values={"tracing": True, "green_metrics": True,
                                         "run_label": "view-live"})
    with TestClient(app) as client:
        client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
        key = client.get("/api/sessions").json()["sessions"][0]["key"]
        summary = client.get(f"/api/sessions/{key}/summary").json()
        detail = client.get(f"/api/sessions/{key}").json()

    assert summary["run_label"] == "view-live"
    assert summary["requests"] == 1
    assert summary["models"]["claude-haiku-5-5"]["input"] == 5
    impacts = detail["round_trips"][0]["response"]["impacts"]
    expected = green.compute("claude-haiku-5-5", UPSTREAM_OK["usage"])
    assert impacts["energy_kwh"]["mean"] == pytest.approx(expected["energy_kwh"]["mean"])
    assert impacts["model_basis"] == "estimated"
    assert summary["green"]["energy_kwh"]["mean"] == pytest.approx(expected["energy_kwh"]["mean"])
