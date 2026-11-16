"""Model aliasing (model_overrides): client-facing name -> upstream model id."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import green
from proxy import ledger
from proxy import usage as usage_log
from proxy.app import create_app
from proxy.config import Settings, save_settings
from proxy.headroom_settings import normalize as headroom_normalize

ALIAS = "liquid-lfm-3-1-7b"
UPSTREAM_ID = "liquid/lfm-3-1.7b:free"

UPSTREAM_OK = {
    "id": "msg_up", "type": "message", "role": "assistant", "model": UPSTREAM_ID,
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 100, "output_tokens": 40, "cache_read_input_tokens": 50,
              "cache_creation_input_tokens": 10},
}


def _app(tmp_path, *, overrides=None, exposed=("liquid-lfm-3-1-7b",), upstream=None):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test", exposed_models=list(exposed))
    settings.headroom = headroom_normalize({"retries": 0, "backoff_ms": 0})
    settings.features = {"compression": False, "cost_control": False, "shunt": False,
                         "green_metrics": True, "tracing": True, "run_label": "alias-tests"}
    settings.model_overrides = dict(overrides if overrides is not None else
                                    {ALIAS: UPSTREAM_ID})
    save_settings(settings, settings_path)

    def default_upstream(request):
        return httpx.Response(200, json=UPSTREAM_OK)

    return create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(upstream or default_upstream),
        sidecar_transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})),
        manage_sidecar=False,
    )


def _body(model=ALIAS):
    return {"model": model, "max_tokens": 10,
            "messages": [{"role": "user", "content": "hello"}]}


def _read_only_session(tmp_path):
    files = list((tmp_path / "sessions").glob("*.jsonl"))
    assert len(files) == 1
    return [json.loads(line) for line in
            files[0].read_text(encoding="utf-8").splitlines() if line.strip()]


def test_alias_rewrites_model_for_upstream_but_not_the_ledger(tmp_path):
    seen = {}
    upstream_seen = []

    def upstream(request):
        seen["model"] = json.loads(request.content).get("model")
        upstream_seen.append(request.url.path)
        return httpx.Response(200, json=UPSTREAM_OK)

    app = _app(tmp_path, upstream=upstream)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_body(),
                           headers={"x-claude-code-session-id": "alias-1", "x-api-key": "k"})
        assert resp.status_code == 200

    assert upstream_seen[-1] == "/v1/messages"
    assert seen["model"] == UPSTREAM_ID  # upstream sees the real id
    assert resp.json()["model"] == UPSTREAM_ID  # passthrough is verbatim

    entries = _read_only_session(tmp_path)
    request_event = next(e for e in entries if e["type"] == "request")
    assert request_event["model"] == ALIAS          # client-facing name in the ledger
    assert request_event["upstream_model"] == UPSTREAM_ID
    row = usage_log.recent(tmp_path / "usage.db", 1)[0]
    assert row["model"] == ALIAS


def test_without_alias_behaviour_is_unchanged(tmp_path):
    seen = {}

    def upstream(request):
        seen["model"] = json.loads(request.content).get("model")
        return httpx.Response(200, json=UPSTREAM_OK)

    app = _app(tmp_path, overrides={}, exposed=[UPSTREAM_ID], upstream=upstream)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_body(model=UPSTREAM_ID),
                           headers={"x-claude-code-session-id": "alias-2", "x-api-key": "k"})
        assert resp.status_code == 200
    assert seen["model"] == UPSTREAM_ID
    files = list((tmp_path / "sessions").glob("*.jsonl"))
    assert files, "tracing is on by default here; a session file must exist"


def test_green_impacts_use_the_effective_upstream_model(tmp_path):
    cases = ((UPSTREAM_ID, "estimated"), ("glm-5-2", "official"))
    for i, (upstream_id, expected_basis) in enumerate(cases):
        case_dir = tmp_path / f"case{i}"
        case_dir.mkdir()
        seen = {}

        def upstream(request, _id=upstream_id):
            seen["model"] = json.loads(request.content).get("model")
            body = {**UPSTREAM_OK, "model": _id}
            return httpx.Response(200, json=body)

        app = _app(case_dir, overrides={ALIAS: upstream_id}, upstream=upstream)
        with TestClient(app) as client:
            client.post("/v1/messages", json=_body(),
                        headers={"x-claude-code-session-id": "alias-green", "x-api-key": "k"})
        assert seen["model"] == upstream_id

        entries = _read_only_session(case_dir)
        response_event = next(e for e in entries if e["type"] == "response")
        impacts = response_event["impacts"]
        assert impacts["model_basis"] == expected_basis
        assert impacts["energy_kwh"]["mean"] > 0
        expected = green.compute(upstream_id, UPSTREAM_OK["usage"])
        assert impacts["energy_kwh"]["mean"] == pytest.approx(expected["energy_kwh"]["mean"])


def test_alias_works_on_the_responses_route(tmp_path):
    seen = {}

    def upstream(request):
        if request.url.path == "/v1/responses":
            seen["model"] = json.loads(request.content).get("model")
            return httpx.Response(200, json={
                "id": "resp_1",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}],
                "usage": {"input_tokens": 10, "output_tokens": 2},
            })
        return httpx.Response(200, json=UPSTREAM_OK)

    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test", exposed_models=[ALIAS],
                        route_overrides={ALIAS: "responses"},
                        model_overrides={ALIAS: UPSTREAM_ID})
    settings.headroom = headroom_normalize({"retries": 0, "backoff_ms": 0})
    save_settings(settings, settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(upstream),
        sidecar_transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})),
        manage_sidecar=False,
    )
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_body(),
                           headers={"x-claude-code-session-id": "alias-resp", "x-api-key": "k"})
        assert resp.status_code == 200
    assert seen["model"] == UPSTREAM_ID
    # the translated message keeps the client-facing name
    assert resp.json()["model"] == ALIAS


def test_settings_api_manages_model_overrides(tmp_path):
    app = _app(tmp_path, overrides={})
    with TestClient(app) as client:
        assert client.get("/api/state").json()["settings"]["model_overrides"] == {}
        ok = client.put("/api/settings", json={"model_overrides": {ALIAS: UPSTREAM_ID}})
        assert ok.status_code == 200
        view = client.get("/api/state").json()["settings"]["model_overrides"]
        assert view == {ALIAS: UPSTREAM_ID}
        bad = client.put("/api/settings", json={"model_overrides": {ALIAS: "  "}})
        assert bad.status_code == 400
        # persists to disk
        stored = json.load(open(tmp_path / "config.json", encoding="utf-8"))
        assert stored["model_overrides"] == {ALIAS: UPSTREAM_ID}
