"""App-level router integration: auto models through the whole pipeline.

Router classification uses a mocked SystemOne endpoint (httpx.MockTransport);
everything downstream (tier resolution, provider pinning, ledger, usage DB,
one-click inclusion) runs for real.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import router_settings as router_settings_module
from proxy import router_settings as rs
from proxy.app import create_app
from proxy.config import Settings, save_settings
from proxy.router_engine import Decision

CLASSIFIER_ANSWER = {
    "model": "tev1:4b",
    "answers": {
        "difficulty": {
            "choice": "goldfish",
            "probabilities": {"goldfish": 0.92, "elephant": 0.08},
            "confidence": 0.92,
        }
    },
    "usage": {"input_tokens": 37, "output_tokens": 1},
}


def _classifier_handler(request: httpx.Request) -> httpx.Response:
    """Stand-in SystemOne endpoint: records the envelope, answers the question."""
    payload = json.loads(request.content) if request.content else {}
    assert payload["questions"]["difficulty"]["type"] == "choice"
    return httpx.Response(200, json=CLASSIFIER_ANSWER)


def _router_client(tmp_path, *, exposed="claude-haiku-5-5", providers=("primary",), fail_classify=False,
                   allow_oversized=False, tiers_set_up=True, **router_over):
    settings_path = tmp_path / "config.json"
    values = router_settings_module.defaults()
    values.update(router_over)
    values["allow_oversized_context"] = allow_oversized
    if tiers_set_up:
        tiers = router_settings_module.normalize_tiers([
            {"id": "goldfish", "description": "execution work", "model": "claude-haiku-5-5"},
            {"id": "elephant", "description": "planning work", "model": "claude-sonnet-5-5"},
        ])
    else:
        # Not set up: tier models unassigned (the eco-auto is then hidden).
        tiers = rs.default_tiers()
    settings = Settings(
        upstream_url="http://gateway.test",
        exposed_models=[exposed],
        providers=[{"name": name, "url": "http://gateway.test", "api_key": "k"} for name in providers],
        router=values,
        router_tiers=tiers,
    )
    save_settings(settings, settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude-settings.json",
        upstream_transport=httpx.MockTransport(_upstream_ok),
        router_transport=httpx.MockTransport(_fail_handler if fail_classify else _classifier_handler),
        manage_sidecar=False,
    )
    return TestClient(app)


def _fail_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(500, json={"error": {"message": "classifier is down"}})


def _upstream_ok(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/messages":
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "done"}],
            "usage": {"input_tokens": 12, "output_tokens": 3,
                      "cache_read_input_tokens": 6, "cache_creation_input_tokens": 0},
            "stop_reason": "end_turn",
        })
    return httpx.Response(404, json={"error": {"message": "not found"}})


def _router_body(run_label="", extra=None):
    body: dict[str, Any] = {
        "model": "eco-auto",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello through the router"}],
        "stream": False,
    }
    if run_label:
        body["metadata"] = {"run_label": run_label}
    if extra:
        body.update(extra)
    return body


# --- exposure gating -----------------------------------------------------------

def test_auto_model_hidden_until_setup(tmp_path):
    client = _router_client(tmp_path, tiers_set_up=False)  # no tier models assigned yet
    listings = client.get("/v1/models").json()["data"]
    assert "eco-auto" not in {row["id"] for row in listings}


def test_auto_model_listed_when_valid(tmp_path):
    client = _router_client(tmp_path,
                            router_over={"tier_models_assigned": True})
    listings = client.get("/v1/models").json()["data"]
    auto = next(row for row in listings if row["id"] == "eco-auto")
    assert auto["max_input_tokens"] == 256_000
    assert "Difficulty-tier routed" in auto["description"]


# --- /v1/messages through the router --------------------------------------------

def test_messages_routed_to_tier_model(tmp_path):
    client = _router_client(tmp_path)
    resp = client.post("/v1/messages", json=_router_body())
    assert resp.status_code == 200
    data = resp.json()
    assert data["content"][0]["text"] == "done"
    row = client.get("/api/usage").json()["rows"][0]
    assert row["model"] == "eco-auto"
    assert row["router_tier"] == "goldfish"
    assert row["router_target_model"] == "claude-haiku-5-5"
    assert row["router_confidence"] == pytest.approx(0.92)


def test_messages_classify_failure_uses_planner_fallback(tmp_path):
    from proxy.router_engine import ClassifyError
    client = _router_client(tmp_path, fail_classify=True)
    resp = client.post("/v1/messages", json=_router_body())
    assert resp.status_code == 200
    row = client.get("/api/usage").json()["rows"][0]
    assert row["router_fallback"] == 1
    assert row["router_reason"] == "classify_failed"


def test_count_tokens_for_auto_is_local(tmp_path):
    client = _router_client(tmp_path)
    resp = client.post("/v1/messages/count_tokens", json=_router_body())
    assert resp.status_code == 200
    estimate = resp.json()["input_tokens"]
    exact_sum = row = client.get("/api/usage").json()
    assert estimate >= 1


def test_oversized_context_rejected_unless_flag(tmp_path):
    client = _router_client(tmp_path)
    big_tokens = 300_000
    body = _router_body()
    body["messages"][0]["content"] = "word " * big_tokens  # >256k estimated tokens
    resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 400
    assert "256k" in resp.json()["error"]["message"]

    client = _router_client(tmp_path, allow_oversized=True)
    resp = client.post("/v1/messages", json=body)
    assert resp.status_code == 200


# --- admin API regression: classifier rebuild before first use -------------------

def test_router_put_values_rebuilds_classifier_on_cold_app(tmp_path):
    """Changing the endpoint/config must not 500 when no request has run yet.

    Regression: PUT /api/router closed state["router_classifier"] before that
    lazy key existed (KeyError -> HTTP 500) whenever a save changed the config
    fingerprint before the first classification.
    """
    client = _router_client(tmp_path)
    resp = client.put("/api/router", json={"values": {"endpoint_url": "http://127.0.0.1:9/v1/systemone"}})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_router_put_accepts_tiers_and_values_together(tmp_path):
    """Dashboard save submits edited scalar values AND the tier ladder in one call."""
    client = _router_client(tmp_path)
    resp = client.put("/api/router", json={
        "values": {"endpoint_url": "http://127.0.0.1:9/v1/systemone"},
        "tiers": [
            {"id": "goldfish", "description": "execution work", "model": "claude-haiku-5-5"},
            {"id": "elephant", "description": "planning work", "model": "claude-sonnet-5-5"},
        ]})
    assert resp.status_code == 200
    assert resp.json()["validity"]["valid"] is True
    tier_ids = [t["id"] for t in client.get("/api/router").json()["tiers"]]
    assert tier_ids == ["goldfish", "elephant"]
