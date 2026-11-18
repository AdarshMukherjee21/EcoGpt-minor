"""App-level admin API tests for the Headroom /api/headroom surface.

Regression coverage for the audited failures:
- GET must never echo the sidecar shared token (masked like every other
  credential in this API) while still telling the dashboard it is set.
- PUT with a blank token keeps the stored secret (the dashboard shows a
  "(saved — leave blank to keep)" placeholder); a typed value replaces it.
- PUT with blank required numeric fields snaps to schema defaults instead
  of persisting null (which used to brick the daemon boot).
"""
from __future__ import annotations

import json

import httpx
from fastapi.testclient import TestClient

from proxy.app import create_app
from proxy.config import Settings, save_settings


def _client(tmp_path, headroom_overrides=None):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test",
                        exposed_models=["claude-haiku-5-5"])
    headroom = settings.headroom
    headroom.update(headroom_overrides or {})
    save_settings(settings, settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude-settings.json",
        upstream_transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})),
        manage_sidecar=False,
    )
    return TestClient(app)


def test_headroom_view_masks_sidecar_token(tmp_path) -> None:
    client = _client(tmp_path, {"HEADROOM_PROXY_TOKEN": "top-secret"})
    data = client.get("/api/headroom").json()
    assert data["values"]["HEADROOM_PROXY_TOKEN"] is None  # never echoed
    assert data["proxy_token_set"] is True
    assert "(saved" not in json.dumps(data)


def test_headroom_view_reports_token_absent(tmp_path) -> None:
    client = _client(tmp_path)
    data = client.get("/api/headroom").json()
    assert data["values"]["HEADROOM_PROXY_TOKEN"] is None
    assert data["proxy_token_set"] is False


def test_headroom_put_blank_token_keeps_stored_secret(tmp_path) -> None:
    client = _client(tmp_path, {"HEADROOM_PROXY_TOKEN": "top-secret"})
    resp = client.put("/api/headroom", json={"values": {"HEADROOM_PROXY_TOKEN": ""}})
    assert resp.status_code == 200
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["headroom"]["HEADROOM_PROXY_TOKEN"] == "top-secret"


def test_headroom_put_typed_token_replaces_secret(tmp_path) -> None:
    client = _client(tmp_path, {"HEADROOM_PROXY_TOKEN": "old"})
    resp = client.put("/api/headroom", json={"values": {"HEADROOM_PROXY_TOKEN": "new-secret"}})
    assert resp.status_code == 200
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config["headroom"]["HEADROOM_PROXY_TOKEN"] == "new-secret"
    assert client.get("/api/headroom").json()["proxy_token_set"] is True


def test_headroom_put_blank_numeric_fields_snap_to_defaults(tmp_path) -> None:
    """Regression: blank numerics persisted null and crashed the next boot."""
    client = _client(tmp_path)
    values = {k: "" for k in ("sidecar_port", "retries", "request_timeout_ms",
                              "backoff_ms", "token_budget", "protect_recent")}
    resp = client.put("/api/headroom", json={"values": values})
    assert resp.status_code == 200
    config = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    headroom = config["headroom"]
    assert headroom["sidecar_port"] == 8787
    assert headroom["retries"] == 2
    assert headroom["request_timeout_ms"] == 10000
    assert headroom["backoff_ms"] == 200
    assert headroom["token_budget"] == 1000000
    assert headroom["protect_recent"] == 0
