"""Wiring tests for the ecoGPT chat-UI integration in proxy/app.py.

Covers (the held items, now landed):
- GET /models alias: the chat-UI model listing route returns the same payload
  as /v1/models (Open WebUI reads the OpenAI listing from the gateway).
- model_row enrichment: additive Anthropic-listing fields every chat-UI
  decoder needs — input_modalities, api_capabilities.anthropic_messages,
  max_output_tokens — while Claude-Code/Desktop-consumed fields are untouched.
- create_app wiring: the /chat router mounts only when manage_chat_ui=True AND
  settings.chat_ui == "openwebui"; the sidecar lifecycle runs only then too.
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import APIRouter as fastapi_APIRouter, Request as _FastAPIRequest, Response
from fastapi.testclient import TestClient

from proxy.app import create_app
from proxy.chat_ui import ChatUISidecar
from proxy.config import Settings, save_settings


def _upstream(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"data": []})


def _client(tmp_path, exposed, **kw) -> TestClient:
    settings_path = tmp_path / "config.json"
    save_settings(Settings(upstream_url="http://gateway.test", exposed_models=exposed), settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude-settings.json",
        desktop_app_dir=tmp_path / "Claude-3p",
        desktop_backup_dir=tmp_path / "backup",
        ledger_dir=tmp_path / "ledger",
        upstream_transport=httpx.MockTransport(_upstream),
        **kw,
    )
    return TestClient(app)


def _rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return payload["data"]


def test_models_alias_returns_same_payload_as_v1_models(tmp_path) -> None:
    client = _client(tmp_path, ["openrouter/free", "baidu/ernie-5.5-flash-preview:free"])
    via_v1 = client.get("/v1/models")
    via_alias = client.get("/models")
    assert via_v1.status_code == 200 and via_alias.status_code == 200
    assert via_alias.json() == via_v1.json()  # the harness must see the same listing


def test_model_row_enrichment_is_additive(tmp_path) -> None:
    client = _client(tmp_path, ["openrouter/free"])
    rows = _rows(client.get("/v1/models").json())
    assert rows, "expected the exposed model row"
    row = rows[0]
    # classic consumers (Claude Code / Claude Desktop) keep their fields:
    assert row["type"] == "model"
    assert row["id"] == "openrouter/free"
    assert row["display_name"]
    assert row["created_at"] == "1970-01-01T00:00:00Z"
    # additive harness-listing fields:
    assert row["input_modalities"] == ["text"]
    assert row["api_capabilities"] == {"anthropic_messages": {}}
    assert isinstance(row["max_output_tokens"], int) and row["max_output_tokens"] > 0


def test_auto_model_row_also_enriched_when_router_offer(tmp_path) -> None:
    client = _client(tmp_path, ["openrouter/free"])
    payload = client.get("/v1/models").json()
    by_id = {r["id"]: r for r in _rows(payload)}
    if "eco-auto" in by_id:  # only when this deployment's router is armed
        row = by_id["eco-auto"]
        assert row["api_capabilities"] == {"anthropic_messages": {}}
        assert row["max_input_tokens"] > 0


def test_chat_router_mounted_and_reachable(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The /chat router is mounted on the daemon app (manage_chat_ui=True and
    settings.chat_ui="openwebui") and answers requests.

    Hermetic: a stub router replaces the live one so the assertion is about
    the WIRING (the real backend behavior has its own tests)."""
    async def stub_route(_request: _FastAPIRequest) -> Response:
        return Response("stub-chat", status_code=200)

    def _stub_router() -> fastapi_APIRouter:
        router = fastapi_APIRouter()
        router.add_api_route("/chat", stub_route, methods=["GET"])
        return router

    monkeypatch.setattr("proxy.app.create_chat_router", lambda *a, **k: _stub_router())

    client = _client(tmp_path, ["openrouter/free"], manage_chat_ui=True)
    response = client.get("/chat", follow_redirects=False)
    assert response.status_code == 200, f"chat route body: {response.text[:300]}"
    assert b"stub-chat" in response.content


def test_chat_router_not_mounted_when_chat_ui_off(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """settings.chat_ui="off" disables the /chat mount even with manage_chat_ui=True."""
    settings_path = tmp_path / "config.json"
    save_settings(Settings(upstream_url="http://gateway.test", exposed_models=["m"], chat_ui="off"),
                  settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude-settings.json",
        desktop_app_dir=tmp_path / "Claude-3p",
        desktop_backup_dir=tmp_path / "backup",
        ledger_dir=tmp_path / "ledger",
        upstream_transport=httpx.MockTransport(_upstream),
        manage_chat_ui=True,
    )
    client = TestClient(app)
    assert client.get("/chat", follow_redirects=False).status_code == 404


def test_sidecar_lifecycle_tied_to_manage_chat_ui(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[bool] = []
    stopped: list[bool] = []

    async def fake_start(self) -> None:
        started.append(True)

    async def fake_stop(self) -> None:
        stopped.append(True)

    monkeypatch.setattr(ChatUISidecar, "start", fake_start)
    monkeypatch.setattr(ChatUISidecar, "stop", fake_stop)

    with _client(tmp_path, ["openrouter/free"], manage_chat_ui=True) as client:
        assert client.get("/v1/models").status_code == 200
    assert started == [True] and stopped == [True]
    baseline_stops = len(stopped)

    with _client(tmp_path, ["openrouter/free"]) as client:  # default: no sidecar
        assert client.get("/v1/models").status_code == 200
    assert len(stopped) == baseline_stops  # no extra lifecycle activity
