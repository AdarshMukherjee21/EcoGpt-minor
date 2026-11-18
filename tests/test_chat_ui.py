"""Tests for proxy.chat_ui — the Open WebUI chat-UI sidecar wiring.

Covers:
- ChatUIConfig defaults (own loopback port 18091, pin constant, chat_url
  mapping for the /chat redirect)
- build_command()/build_env(): venv open-webui entrypoint, serve flags, and
  the loopback env (WEBUI_AUTH=False, gateway as OPENAI_API_BASE_URL, …)
- /chat link router: 307 redirect to the sidecar origin when up (deep paths
  map to the sidecar root), 503 "starting" page while it is down
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
from proxy.chat_ui import ChatUIConfig, ChatUISidecar, create_chat_link_router


def _app(router) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_config_defaults() -> None:
    cfg = ChatUIConfig()
    assert cfg.webui_port == 18091
    assert cfg.webui_url() == "http://127.0.0.1:18091"
    assert chat_ui.OPENWEBUI_PIN == "0.13.4"
    assert "openwebui" in cfg.resolved_data_dir()


@pytest.mark.parametrize("path,target", [
    ("", "http://127.0.0.1:18091/"),
    ("c/abc", "http://127.0.0.1:18091/c/abc"),
])
def test_chat_url_maps_daemon_paths_to_sidecar_root(path: str, target: str) -> None:
    assert ChatUIConfig().chat_url(path) == target


def test_build_command_uses_venv_entrypoint_and_serve_flags() -> None:
    command = ChatUISidecar(ChatUIConfig()).build_command()
    assert "serve" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "18091"
    head = command[0]
    assert head.endswith("open-webui.exe") or head.endswith("python.exe")


def test_build_command_honors_configured_port() -> None:
    command = ChatUISidecar(ChatUIConfig(webui_port=19099)).build_command()
    assert command[command.index("--port") + 1] == "19099"


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


def _patch_async_client(handler) -> Any:
    """Route all chat_ui httpx clients through a MockTransport handler."""
    original = chat_ui.httpx.AsyncClient

    class _Patched(httpx.AsyncClient):
        def __init__(self, *a: Any, **kw: Any) -> None:
            kw.setdefault("transport", httpx.MockTransport(handler))
            super().__init__(*a, **kw)

    chat_ui.httpx.AsyncClient = _Patched
    return original


def test_chat_redirects_to_sidecar_when_up() -> None:
    """The real readiness path (httpx probe) then a 307 to the sidecar origin."""
    seen: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, html="<html>webui</html>")

    original = _patch_async_client(handler)
    try:
        sidecar = ChatUISidecar(ChatUIConfig())
        client = TestClient(_app(create_chat_link_router(sidecar, ChatUIConfig())))
        response = client.get("/chat", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "http://127.0.0.1:18091/"
        assert seen[-1].startswith("http://127.0.0.1:18091/")  # probe hit the sidecar

        # deep daemon path maps to the sidecar root
        response = client.get("/chat/c/abc?x=1", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "http://127.0.0.1:18091/c/abc"
    finally:
        chat_ui.httpx.AsyncClient = original


def test_cold_sidecar_serves_starting_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("cold", request=request)

    original = _patch_async_client(handler)
    try:
        sidecar = ChatUISidecar(ChatUIConfig())
        client = TestClient(_app(create_chat_link_router(sidecar, ChatUIConfig())))
        response = client.get("/chat", follow_redirects=False)
        assert response.status_code == 503
        assert "starting" in response.text
        assert "refresh" in response.text  # auto-retry while the sidecar boots
    finally:
        chat_ui.httpx.AsyncClient = original


def test_injected_probe_drives_ready_state() -> None:
    """The router accepts a hermetic readiness probe (used by app-level tests)."""
    calls: List[ChatUIConfig] = []

    async def probe(cfg: ChatUIConfig) -> bool:
        calls.append(cfg)
        return True

    sidecar = ChatUISidecar(ChatUIConfig())
    client = TestClient(_app(create_chat_link_router(sidecar, ChatUIConfig(), probe=probe)))
    response = client.get("/chat", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "http://127.0.0.1:18091/"
    assert calls and calls[0].webui_port == 18091


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
    assert "18091" in captured["cmd"]
    assert any("open-webui" in str(part) or part == "-c" for part in captured["cmd"])
