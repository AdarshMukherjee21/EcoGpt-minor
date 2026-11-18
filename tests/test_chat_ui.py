"""Tests for proxy.chat_ui — the Open WebUI chat-UI sidecar wiring.

Covers:
- ChatUIConfig defaults (private port 3181, /chat prefix, pin constant)
- build_command()/build_env(): venv open-webui entrypoint, serve flags, and
  the loopback env (WEBUI_AUTH=False, gateway as OPENAI_API_BASE_URL, …)
- /chat prefix-stripping HTTP proxy: raw-query forwarding, Host forwarding,
  content-encoding stripping, 503 cold start
- WebSocket tunnel under /chat (socket.io path) and at root /ws (fallback)
- root fallback router: daemon routes keep precedence, WebUI namespaces forward
- lifecycle: start (mocked spawn) -> stop()
"""

from __future__ import annotations

import asyncio
from typing import Any, List

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

import proxy.chat_ui as chat_ui
from proxy.chat_ui import ChatUIConfig, ChatUISidecar, create_chat_router, create_chat_ui_fallback


def _app(router) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def _patched(handler) -> None:
    """Route all chat_ui httpx clients through a MockTransport handler."""
    original = chat_ui.httpx.AsyncClient

    class _Patched(httpx.AsyncClient):
        def __init__(self, *a: Any, **kw: Any) -> None:
            kw.setdefault("transport", httpx.MockTransport(handler))
            super().__init__(*a, **kw)

    chat_ui.httpx.AsyncClient = _Patched
    return original


def test_config_defaults() -> None:
    cfg = ChatUIConfig()
    assert cfg.upstream_origin == "http://127.0.0.1:3181"
    assert cfg.public_origin == "http://127.0.0.1:18090"
    assert cfg.chat_prefix == "/chat"
    assert chat_ui.OPENWEBUI_PIN == "0.13.4"
    assert "openwebui" in cfg.resolved_data_dir()


def test_build_command_uses_venv_entrypoint_and_serve_flags() -> None:
    command = ChatUISidecar(ChatUIConfig()).build_command()
    assert "serve" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "3181"
    head = command[0]
    assert head.endswith("open-webui.exe") or head.endswith("python.exe")


def test_build_env_points_only_at_the_gateway() -> None:
    env = ChatUISidecar(ChatUIConfig()).build_env()
    assert env["WEBUI_AUTH"] == "False"
    assert env["ENABLE_SIGNUP"] == "False"
    assert env["OPENAI_API_BASE_URL"] == "http://127.0.0.1:18090/v1"
    assert env["OPENAI_API_KEY"] == "ecogpt-managed"  # dummy; daemon substitutes its own
    assert env["OLLAMA_BASE_URL"] == "http://127.0.0.1:9"  # dead port: no local ollama
    assert env["WEBUI_NAME"] == "ecoGPT"
    assert env["DATA_DIR"].endswith("openwebui")
    assert env["DO_NOT_TRACK"] == "True"


def test_proxy_strips_prefix_forwards_raw_query_and_host() -> None:
    seen: List[str] = []
    seen_host: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        seen_host.append(request.headers.get("host", ""))
        return httpx.Response(200, json={"path": request.url.path, "url": str(request.url)})

    original = _patched(handler)
    try:
        sidecar = ChatUISidecar(ChatUIConfig())
        client = TestClient(_app(create_chat_router(sidecar, ChatUIConfig())))
        response = client.get("/chat/api/config?x=1", follow_redirects=False)
        assert response.status_code == 200
        assert response.json()["path"] == "/api/config"  # prefix stripped
        assert seen_host == ["testserver"]  # browser Host forwarded verbatim

        # raw query survives verbatim (combo/asset keys contain literal '?')
        response = client.get("/chat/static/??a.js&rev=1", follow_redirects=False)
        assert "/static/??a.js&rev=1" in seen[-1], seen[-1]
    finally:
        chat_ui.httpx.AsyncClient = original


def test_buffered_proxy_response_strips_content_encoding() -> None:
    import gzip

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=gzip.compress(b"plain-body"),
                              headers={"content-encoding": "gzip", "content-type": "text/html"})

    original = _patched(handler)
    try:
        sidecar = ChatUISidecar(ChatUIConfig())
        client = TestClient(_app(create_chat_router(sidecar, ChatUIConfig())))
        response = client.get("/chat/", follow_redirects=False)
        assert response.status_code == 200
        assert response.content == b"plain-body"
        assert "content-encoding" not in response.headers
        assert int(response.headers["content-length"]) == len(response.content)
    finally:
        chat_ui.httpx.AsyncClient = original


def test_cold_sidecar_returns_503() -> None:
    sidecar = ChatUISidecar(ChatUIConfig())
    client = TestClient(_app(create_chat_router(sidecar, ChatUIConfig(
        upstream_origin="http://127.0.0.1:1"))))  # closed port
    response = client.get("/chat", follow_redirects=False)
    assert response.status_code == 503
    assert "starting" in response.text


def test_ws_tunnel_under_chat_prefix() -> None:
    received: List[str] = []

    class _FakeUpstream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            while not received:
                await asyncio.sleep(0.01)
            return "echo:" + received.pop(0)

        async def send(self, message: str) -> None:
            received.append(message)

    async def fake_connect(url: str) -> _FakeUpstream:
        assert url == "http://127.0.0.1:3181/ws/socket.io"
        return _FakeUpstream()

    sidecar = ChatUISidecar(ChatUIConfig())
    client = TestClient(_app(create_chat_router(sidecar, ChatUIConfig(), ws_connect=fake_connect)))
    with client.websocket_connect("/chat/ws/socket.io") as ws:
        ws.send_text("ping")
        assert ws.receive_text() == "echo:ping"


def test_fallback_forwards_webui_namespaces_but_daemon_routes_win() -> None:
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"upstream": request.url.path})

    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    app = FastAPI()

    @app.get("/api/state")
    async def daemon_state() -> dict:
        return {"ours": True}

    app.include_router(create_chat_ui_fallback(sidecar, cfg))

    original = _patched(handler)
    try:
        client = TestClient(app)
        assert client.get("/api/state").json() == {"ours": True}
        assert calls == []  # daemon route never touches the sidecar

        assert client.get("/api/config").json() == {"upstream": "/api/config"}
        assert client.get("/static/loader.js").json() == {"upstream": "/static/loader.js"}
        assert client.get("/manifest.json").json() == {"upstream": "/manifest.json"}
        assert calls == ["/api/config", "/static/loader.js", "/manifest.json"]
    finally:
        chat_ui.httpx.AsyncClient = original


def test_fallback_tunnels_root_ws_socket_io() -> None:
    received: List[str] = []

    class _FakeUpstream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            while not received:
                await asyncio.sleep(0.01)
            return "echo:" + received.pop(0)

        async def send(self, message: str) -> None:
            received.append(message)

    async def fake_connect(url: str) -> _FakeUpstream:
        assert url == "http://127.0.0.1:3181/ws/socket.io"
        return _FakeUpstream()

    sidecar = ChatUISidecar(ChatUIConfig())
    app = FastAPI()
    app.include_router(create_chat_ui_fallback(sidecar, ChatUIConfig(), ws_connect=fake_connect))
    client = TestClient(app)
    with client.websocket_connect("/ws/socket.io?EIO=4&transport=websocket") as ws:
        ws.send_text("0probe")
        assert ws.receive_text() == "echo:0probe"


def test_lifecycle_start_stop_with_mocked_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Proc:
        stdout = None
        returncode = None

        async def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

    async def fake_exec(*a: Any, **kw: Any) -> _Proc:
        captured["cmd"] = a
        captured["env"] = kw.get("env")
        return _Proc()

    async def noop_read(self: ChatUISidecar) -> None:
        return None

    monkeypatch.setattr(chat_ui.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(ChatUISidecar, "_read_output", noop_read)

    async def scenario() -> None:
        sidecar = ChatUISidecar(ChatUIConfig())
        await sidecar.start()
        assert sidecar._proc is not None
        assert captured["env"]["WEBUI_AUTH"] == "False"
        assert captured["env"]["OPENAI_API_BASE_URL"] == "http://127.0.0.1:18090/v1"
        await sidecar.stop()
        assert sidecar._proc is None

    asyncio.run(scenario())
    assert "serve" in captured["cmd"]
    assert any("open-webui" in str(part) or part == "-c" for part in captured["cmd"])
