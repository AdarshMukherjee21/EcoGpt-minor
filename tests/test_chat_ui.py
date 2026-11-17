"""Tests for proxy.chat_ui — the ecoGPT chat-UI sidecar wiring.

Covers:
- ChatUIConfig defaults match the approved plan constants (pin, private port, origins)
- build_command(): pinned npm distribution, private port, --no-open, --profile web
- entry-URL discovery from the sidecar's stdout stream
- /chat redirect + 503 cold-start + prefix-stripping HTTP proxy through the router
- WebSocket tunnel smoke against an in-process fake upstream
- lifecycle: start (mocked subprocess spawn, dummy key env) -> stop()
"""
from __future__ import annotations

import asyncio
from typing import Any, List

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

import proxy.chat_ui as chat_ui
from proxy.chat_ui import ChatUIConfig, ChatUISidecar, create_chat_api_fallback, create_chat_router


def _app(router) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    return app


def test_config_defaults_match_approved_plan() -> None:
    cfg = ChatUIConfig()
    assert cfg.upstream_origin == "http://127.0.0.1:3180"
    assert cfg.public_origin == "http://127.0.0.1:18090"
    assert cfg.chat_prefix == "/chat"
    assert chat_ui.DSH_NPM_PIN  # version documented, never floating
    assert "@" not in chat_ui.DSH_NPM_PIN


def test_build_command_pins_npm_distribution() -> None:
    command = ChatUISidecar(ChatUIConfig()).build_command()
    joined = " ".join(command)
    assert "--profile" in command and "web" in command
    assert "--no-open" in command
    assert f"@deepseek-ai/dsh@{chat_ui.DSH_NPM_PIN}" in joined
    assert "--port" in command
    assert command[command.index("--port") + 1] == "3180"


def test_entry_url_harvested_from_stdout() -> None:
    banner = "dsh web: http://127.0.0.1:3180/?token=abcDEF123-_x"
    match = chat_ui._ENTRY_URL_RE.search(banner)
    assert match is not None
    assert match.group(1) == "http://127.0.0.1:3180/?token=abcDEF123-_x"


async def _read_output_mock_lines(sidecar: ChatUISidecar, lines: List[bytes]) -> None:
    """Feed the harvest regex through the real reader logic path."""

    class _FakeStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            if not sidecar._feed:
                raise StopAsyncIteration
            return sidecar._feed.pop(0)

    sidecar._proc = _FakeProcShim(_FakeStream())
    await sidecar._read_output()


class _FakeProcShim:
    def __init__(self, stream) -> None:
        self.stdout = stream
        self.returncode = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


def test_real_reader_harvests_entry_url() -> None:
    sidecar = ChatUISidecar(ChatUIConfig())

    class _FakeStream:
        def __init__(self, lines: List[bytes]) -> None:
            self._lines = list(lines)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._lines:
                raise StopAsyncIteration
            return self._lines.pop(0)

    class _FakeProc:
        stdout = _FakeStream([b"", b"dsh web: http://127.0.0.1:3180/?token=tX9-_a", b"ready\n"])

    # feed through the real reader path: point the proc at the fake stream
    sidecar._proc = _FakeProc()
    asyncio.run(sidecar._read_output())
    assert sidecar.entry_url == "http://127.0.0.1:3180/?token=tX9-_a"


def test_router_redirects_first_visit_to_entry_url() -> None:
    sidecar = ChatUISidecar(ChatUIConfig())
    sidecar.entry_url = "http://127.0.0.1:3180/?token=tok"
    client = TestClient(_app(create_chat_router(sidecar, ChatUIConfig())))
    response = client.get("/chat", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == sidecar.entry_url


def test_router_serves_503_when_sidecar_cold() -> None:
    cold = ChatUISidecar(ChatUIConfig())
    cold.entry_url = None
    # a closed-port upstream proves the 503 path regardless of local daemons
    client = TestClient(_app(create_chat_router(cold, ChatUIConfig(upstream_origin="http://127.0.0.1:1"))))
    response = client.get("/chat", follow_redirects=False)
    assert response.status_code == 503


def test_proxy_strips_prefix_and_forwards() -> None:
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"ok": True, "path": request.url.path})

    sidecar = ChatUISidecar(ChatUIConfig())
    sidecar.entry_url = "http://127.0.0.1:3180/?token=t"

    class _Patched(httpx.AsyncClient):
        def __init__(self, *a: Any, **kw: Any) -> None:
            kw["transport"] = httpx.MockTransport(handler)
            super().__init__(*a, **kw)

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = _Patched
    try:
        client = TestClient(_app(create_chat_router(sidecar, ChatUIConfig())))
        response = client.get("/chat/api/state", follow_redirects=False)
        assert response.status_code == 200
        assert response.json()["path"] == "/api/state"  # PREFIX STRIPPED before upstream
        assert calls == ["/api/state"]
        # buffered response carries its own recomputed length (upstream stale length stripped)
        assert int(response.headers["content-length"]) == len(response.content)
    finally:
        chat_ui.httpx.AsyncClient = original


def test_proxy_forwards_ws_tunnel_smoke() -> None:
    """WS tunnel: browser text frames reach upstream, upstream replies come back."""
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
        assert url == "http://127.0.0.1:3180/api/remote.mux"
        return _FakeUpstream()

    sidecar = ChatUISidecar(ChatUIConfig())
    sidecar.entry_url = "http://127.0.0.1:3180/?token=t"
    cfg = ChatUIConfig(upstream_origin="http://127.0.0.1:3180")
    client = TestClient(_app(create_chat_router(sidecar, cfg, ws_connect=fake_connect)))
    with client.websocket_connect("/chat/api/remote.mux") as ws:
        ws.send_text("hello")
        reply = ws.receive_text()
        assert reply == "echo:hello"


def test_lifecycle_start_stop_with_mocked_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _Proc:
        stdout = None
        returncode = None

        async def wait(self) -> int:
            return 0

    async def fake_exec(*a: Any, **kw: Any) -> _Proc:
        captured["cmd"] = a
        captured["env"] = kw.get("env")
        return _Proc()

    class _ReadableProc(_Proc):
        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = -9

    async def fake_exec(*a: Any, **kw: Any) -> _ReadableProc:
        captured["cmd"] = a
        captured["env"] = kw.get("env")
        proc = _ReadableProc()
        return proc

    async def noop_read(self: ChatUISidecar) -> None:
        return None

    monkeypatch.setattr(chat_ui.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(ChatUISidecar, "_read_output", noop_read)

    async def scenario() -> None:
        sidecar = ChatUISidecar(ChatUIConfig())
        await sidecar.start()
        assert sidecar._proc is not None
        # the harness child must carry the dummy key (the gateway substitutes its own)
        assert captured["env"]["DEEPSEEK_API_KEY"] == ChatUIConfig().dummy_api_key
        await sidecar.stop()
        assert sidecar._proc is None

    asyncio.run(scenario())
    assert any("dsh@" in str(part) for part in captured["cmd"])


# ---------------------------------------------------------------------------
# Session bootstrap: cold-load 401 -> auto token exchange -> now-works path.
# The harness (vendor/dsh-harness authentication-token plugin) authorizes with
# a signed cookie named dsh-auth-<base64url(sha256(host-authority))>, minted
# only by its root exchange `GET /?token=...` -> 303 + Set-Cookie.
# ---------------------------------------------------------------------------


def _harness_cookie(cfg: ChatUIConfig) -> str:
    return chat_ui.harness_cookie_name(cfg)


def _session_cookie_header(cfg: ChatUIConfig) -> str:
    name = _harness_cookie(cfg)
    return f"{name}=v1.c2Vzc2lvbg.sig; Max-Age=2592000; Path=/; HttpOnly; SameSite=Strict"


def _mock_harness(valid_token: str, exchange_up: bool = True):
    """Mock upstream that behaves like the duck-typed harness webserver.

    Mirrors the vendored contract verified live:
    - GET /?token=<valid> with Host = public authority -> 303 + Set-Cookie
      signed for that authority; any other Host or token -> 401
    - GET <other>          -> 200 for a VALID session cookie, raw 401 without
                             (the real harness verifies the cookie's HMAC)
    - <other>+Origin       -> authenticated but 403 when Origin differs from
                             the Host fence authority (public origin required)
    """

    def handler(request: httpx.Request) -> httpx.Response:
        expected = _session_cookie_header(ChatUIConfig()).split(";")[0]
        cookies = request.headers.get("cookie", "")
        session_present = expected in cookies
        query = request.url.params.get("token") if request.url.path == "/" else None
        if exchange_up and query is not None:
            if query == valid_token and request.headers["host"] == "127.0.0.1:18090":
                return httpx.Response(
                    303, headers={"location": "./", "set-cookie": _session_cookie_header(ChatUIConfig())})
            return httpx.Response(401, text="dsh web authentication required")
        if session_present:
            origin = request.headers.get("origin")
            if origin is not None and origin != "http://127.0.0.1:18090":
                return httpx.Response(403)  # trust fence: cross-origin refused
            return httpx.Response(200, json={"ok": True, "path": request.url.path,
                                             "host": request.headers["host"]})
        return httpx.Response(401, text="dsh web authentication required")

    return handler


class _PatchedClient(httpx.AsyncClient):
    def __init__(self, handler, *a: Any, **kw: Any) -> None:
        kw.setdefault("transport", httpx.MockTransport(handler))
        super().__init__(*a, **kw)


def test_harness_cookie_name_matches_vendor_contract() -> None:
    import base64 as b64
    import hashlib as hs

    expected = "dsh-auth-" + b64.urlsafe_b64encode(
        hs.sha256(b"127.0.0.1:18090").digest()).rstrip(b"=").decode()
    assert chat_ui.harness_cookie_name(ChatUIConfig()) == expected


def test_first_cold_load_exchanges_token_mints_cookie_then_works() -> None:
    """Repro of the reported bug: first page-load of /chat now works.

    Before: a cookieless (or dashboard-cookie-only) visitor got the harness's
    raw 401 relayed through. Now: the proxy exchanges the launch token
    server-side (with Host = the PUBLIC authority), relays the minted session
    cookie, and the next hit serves the SPA index.
    """
    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    # --public-url flavor: the entry URL re-enters the daemon origin.
    sidecar.entry_url = "http://127.0.0.1:18090/chat/?token=tok123"

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(_mock_harness("tok123"), *a, **kw)
    try:
        client = TestClient(_app(create_chat_router(sidecar, cfg)))
        first = client.get("/chat", follow_redirects=False)
        assert first.status_code == 303
        assert first.headers["location"] == "/chat/"
        assert _harness_cookie(cfg) in first.headers["set-cookie"]

        # Follow-up load (cookie now set): straight through to the SPA index,
        # with the browser's public Host forwarded to the upstream.
        second = client.get("/chat", follow_redirects=False)
        assert second.status_code == 200
        assert second.json()["path"] == "/"
        assert second.json()["host"] == "testserver"  # public Host forwarded verbatim
    finally:
        chat_ui.httpx.AsyncClient = original


def test_interstitial_button_targets_entry_route_not_the_gate() -> None:
    """No dead loop: the button must hit /chat/entry (forces the exchange),
    never a path that bounces back to the same interstitial."""
    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    sidecar.entry_url = "http://127.0.0.1:18090/chat/?token=stale"

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(_mock_harness("other-token"), *a, **kw)
    try:
        client = TestClient(_app(create_chat_router(sidecar, cfg)))
        response = client.get("/chat", follow_redirects=False)
        assert response.status_code == 401
        assert f'href="{cfg.chat_prefix}/entry"' in response.text
    finally:
        chat_ui.httpx.AsyncClient = original


def test_entry_route_forces_exchange_and_returns_to_chat() -> None:
    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    sidecar.entry_url = "http://127.0.0.1:18090/chat/?token=tok123"

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(_mock_harness("tok123"), *a, **kw)
    try:
        client = TestClient(_app(create_chat_router(sidecar, cfg)))
        response = client.get("/chat/entry", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/chat/"
        assert _harness_cookie(cfg) in response.headers["set-cookie"]
    finally:
        chat_ui.httpx.AsyncClient = original


def test_entry_route_without_token_renders_helpful_restart_page() -> None:
    """Token never harvested: a helpful message page (per docs/chat-ui.md),
    with no redirect at all — clicking can never loop."""
    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    sidecar.entry_url = None

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(_mock_harness("t"), *a, **kw)
    try:
        client = TestClient(_app(create_chat_router(sidecar, cfg)))
        response = client.get("/chat/entry", follow_redirects=False)
        assert response.status_code == 503
        assert "text/html" in response.headers["content-type"]
        assert "python.exe -m proxy" in response.text  # restart instruction
    finally:
        chat_ui.httpx.AsyncClient = original


def test_cold_deep_link_401_intercepts_and_retries_with_cookie() -> None:
    """A 401 on a deep path (SPA XHR before the session exists) gets the same
    exchange relay: 303 back to the same path with the session cookie."""
    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    sidecar.entry_url = "http://127.0.0.1:18090/chat/?token=tok123"

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(_mock_harness("tok123"), *a, **kw)
    try:
        client = TestClient(_app(create_chat_router(sidecar, cfg)))
        first = client.get("/chat/api/state", follow_redirects=False)
        assert first.status_code == 303
        assert first.headers["location"] == "/chat/api/state"
        assert _harness_cookie(cfg) in first.headers["set-cookie"]

        second = client.get("/chat/api/state", follow_redirects=False)
        assert second.status_code == 200
        assert second.json()["path"] == "/api/state"
    finally:
        chat_ui.httpx.AsyncClient = original


def test_interstitial_replaces_raw_401_when_exchange_fails() -> None:
    """Token no longer valid (exchange 401s): a friendly one-click page, never
    the raw 'dsh web authentication required' body."""
    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    sidecar.entry_url = "http://127.0.0.1:18090/chat/?token=stale"

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(_mock_harness("tok123"), *a, **kw)
    try:
        client = TestClient(_app(create_chat_router(sidecar, cfg)))
        response = client.get("/chat", follow_redirects=False)
        assert response.status_code == 401
        assert "text/html" in response.headers["content-type"]
        assert "Continue as" in response.text and "admin" in response.text
        assert "dsh web authentication required" not in response.text
    finally:
        chat_ui.httpx.AsyncClient = original


def test_interstitial_shown_when_sidecar_upstream_401s_without_entry_url() -> None:
    """Harness up but the entry URL (token) has not been harvested yet: still
    no raw 401 — the interstitial doubles as a still-booting page."""
    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    sidecar.entry_url = None

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(_mock_harness("t"), *a, **kw)
    try:
        client = TestClient(_app(create_chat_router(sidecar, cfg)))
        response = client.get("/chat/api/state", follow_redirects=False)
        assert response.status_code == 401
        assert "text/html" in response.headers["content-type"]
        assert "Continue as" in response.text
    finally:
        chat_ui.httpx.AsyncClient = original


def test_cookied_visitor_still_sees_genuine_upstream_401() -> None:
    """A visitor WITH the session cookie must not get the exchange redirect
    (would loop on genuinely revoked cookies); the upstream 401 relays."""
    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    sidecar.entry_url = "http://127.0.0.1:3180/?token=tok123"

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(_mock_harness("tok123"), *a, **kw)
    try:
        client = TestClient(_app(create_chat_router(sidecar, cfg)))
        client.cookies.set(_harness_cookie(cfg), "v1.revoked.badsig")
        response = client.get("/chat/api/state", follow_redirects=False)
        assert response.status_code == 401
        assert "dsh web authentication required" in response.text
    finally:
        chat_ui.httpx.AsyncClient = original


def test_buffered_proxy_response_strips_content_encoding() -> None:
    """Regression: ERR_CONTENT_DECODING_FAILED in the browser.

    The harness serves gzip; httpx decodes the buffered body via aread(), but
    the relay used to forward the upstream content-encoding header, so the
    browser tried to gunzip plain bytes. The proxied response must carry the
    decoded body with NO content-encoding header.
    """
    import gzip as _gzip

    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    sidecar.entry_url = "http://127.0.0.1:18090/chat/?token=tok123"
    plain = b"<html>ecoGPT chat</html>"
    packed = _gzip.compress(plain)

    def handler(request: httpx.Request) -> httpx.Response:
        # session cookie present -> harness serves the asset
        return httpx.Response(
            200, content=packed,
            headers={"content-type": "text/html", "content-encoding": "gzip",
                     "content-length": str(len(packed))})

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(handler, *a, **kw)
    try:
        client = TestClient(_app(create_chat_router(sidecar, cfg)))
        cookie = _harness_cookie(cfg)
        resp = client.get("/chat/", headers={"cookie": cookie}, follow_redirects=False)
        assert resp.status_code == 200
        assert "content-encoding" not in {k.lower() for k in resp.headers.keys()}
        assert resp.content == plain  # decoded, browser-decodable as-is
    finally:
        chat_ui.httpx.AsyncClient = original


# ---------------------------------------------------------------------------
# /plugins combo URLs keep their RAW query (??id1/client.js,id2/client.js&rev=)
# and the root-absolute /api catch-all forwards to the sidecar only when no
# daemon route matched.
# ---------------------------------------------------------------------------


def test_proxy_forwards_raw_query_verbatim_for_plugin_combos() -> None:
    """The harness combo URL is /plugins/??<ids>&rev=... — the module list is
    the RAW query; re-encoding parsed params would %3F-encode the second '?'
    and every plugin bundle request would 404 upstream."""
    seen: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"path": request.url.path, "url": str(request.url)})

    sidecar = ChatUISidecar(ChatUIConfig())
    sidecar.entry_url = "http://127.0.0.1:3180/?token=t"

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(handler, *a, **kw)
    try:
        client = TestClient(_app(create_chat_router(sidecar, ChatUIConfig())))
        response = client.get("/chat/plugins/??@a/client.js,@b/client.js&rev=abc",
                              follow_redirects=False)
        assert response.status_code == 200
        # upstream must see the literal '??' combo key, not %3F
        assert "/plugins/??" in seen[0], seen
        assert response.json()["path"] == "/plugins/"
    finally:
        chat_ui.httpx.AsyncClient = original


def test_api_fallback_forwards_unmatched_api_but_daemon_routes_win() -> None:
    """Root /api catch-all (harness API_PATH is root-absolute '/api'):
    daemon-registered /api routes take precedence; only unmatched paths are
    proxied to the chat sidecar."""
    calls: List[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"upstream": request.url.path})

    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    sidecar.entry_url = "http://127.0.0.1:18090/chat/?token=t"

    app = FastAPI()

    @app.get("/api/state")
    async def daemon_state() -> dict:
        return {"ours": True}

    app.include_router(create_chat_api_fallback(sidecar, cfg))

    original = chat_ui.httpx.AsyncClient
    chat_ui.httpx.AsyncClient = lambda *a, **kw: _PatchedClient(handler, *a, **kw)
    try:
        client = TestClient(app)
        ours = client.get("/api/state")
        assert ours.status_code == 200 and ours.json() == {"ours": True}
        assert calls == []  # never touched the sidecar

        forwarded = client.get("/api/harness-thing")
        assert forwarded.status_code == 200
        assert forwarded.json() == {"upstream": "/api/harness-thing"}  # no prefix strip at root
        assert calls == ["/api/harness-thing"]
    finally:
        chat_ui.httpx.AsyncClient = original


def test_api_fallback_tunnels_ws_mux_at_root() -> None:
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
        assert url == "http://127.0.0.1:3180/api/remote.mux"
        return _FakeUpstream()

    cfg = ChatUIConfig()
    sidecar = ChatUISidecar(cfg)
    sidecar.entry_url = "http://127.0.0.1:18090/chat/?token=t"
    app = FastAPI()
    app.include_router(create_chat_api_fallback(sidecar, cfg, ws_connect=fake_connect))
    client = TestClient(app)
    with client.websocket_connect("/api/remote.mux") as ws:
        ws.send_text("ping")
        assert ws.receive_text() == "echo:ping"
