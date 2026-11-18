"""ecoGPT chat-UI sidecar — Open WebUI under /chat on the daemon's origin.

Design:

- The chat UI is a pinned Open WebUI (0.13.x) served by its own uvicorn host on
  a private loopback port (default 3181). This module manages that subprocess
  and reverse-proxies HTTP + WebSocket traffic under ``/chat`` on the daemon's
  origin, so the browser sees one origin (18090).
- Model traffic does NOT come from here: Open WebUI is pointed at the ecoGPT
  gateway via ``OPENAI_API_BASE_URL=<daemon public origin>/v1`` and a dummy
  ``OPENAI_API_KEY``; the daemon performs credential substitution upstream.
- ``WEBUI_AUTH=False`` (loopback-only deployment, see the tradeoff note in
  ``ChatUISidecar.start``): no sign-in, so a cold ``/chat`` load just proxies.
- Open WebUI's frontend references root-absolute paths (``/static/...``,
  ``/_app/...``, ``/manifest.json``, ``/api/...``, ``/ws/socket.io``), so
  besides the ``/chat`` prefix proxy the daemon registers a LAST-resort
  fallback router (``create_chat_ui_fallback``) for those namespaces;
  Starlette registration order keeps every daemon route ahead of it.
- Telemetry: ``DO_NOT_TRACK=True`` is set for the sidecar.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine
from urllib.parse import urlsplit

import httpx
from fastapi import APIRouter, Request, Response, WebSocket
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

log = logging.getLogger("proxy.chat_ui")

# Pinned Open WebUI distribution (requirements.txt). Update procedure:
# docs/chat-ui.md ("Updating the pinned Open WebUI").
OPENWEBUI_PIN = "0.13.4"

# The private port the Open WebUI server binds (loopback only; never advertised).
WEBUI_PORT = 3181

# Daemon-side prefix of the chat UI (same-origin with the dashboard).
CHAT_PREFIX = "/chat"

# Root-absolute URL namespaces the Open WebUI frontend uses. The fallback
# router forwards these to the sidecar (registered last on the daemon, so
# daemon routes always win). /ws is WebSocket-only.
WEBUI_HTTP_PREFIXES = ("/api/", "/static/", "/_app/", "/assets/", "/audio/",
                       "/ollama/", "/openai/", "/retrieval/", "/ws/")
WEBUI_HTTP_EXACT = ("/manifest.json", "/robots.txt", "/svelte.config.js")
WEBUI_WS_PREFIXES = ("/ws/",)

# Hop-by-hop headers that must never be forwarded in either direction.
_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade", "host",
                "content-length"}

# ANSI/OSC escape sequences (piped stdout interleaves OSC title sets from the
# npm-style wrappers); stripped per line so logs stay readable.
_ANSI_RE = re.compile(r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-\\])")


@dataclass
class ChatUIConfig:
    """Where the Open WebUI server lives and how the daemon advertises it."""

    upstream_origin: str = f"http://127.0.0.1:{WEBUI_PORT}"
    public_origin: str = "http://127.0.0.1:18090"
    chat_prefix: str = CHAT_PREFIX
    # Branding + loopback-only auth posture of the sidecar.
    webui_name: str = "ecoGPT"
    webui_secret_key: str = "ecogpt-local-webui-secret"  # fixed local string; loopback only
    # No local Ollama: point at a dead port so only the OpenAI (ecoGPT) path is used.
    ollama_base_url: str = "http://127.0.0.1:9"
    # Open WebUI data dir (SQLite etc.); default lives under the repo data dir.
    data_dir: str | None = None
    # Command used to boot the sidecar (overridable for tests).
    command: list[str] | None = None
    # Environment overrides always applied to the sidecar process.
    env: dict[str, str] = field(default_factory=dict)
    # Seconds to wait for the sidecar to answer its first request (cold import
    # + DB migrations are slow on first boot).
    ready_timeout_s: float = 180.0

    def resolved_data_dir(self) -> str:
        if self.data_dir:
            return self.data_dir
        repo_data = Path(__file__).resolve().parents[1] / "data"
        return str(repo_data / "openwebui")


class ChatUISidecar:
    """Owns the Open WebUI subprocess.

    Not a general process manager: exactly one sidecar, restartable, with a
    readiness gate.
    """

    def __init__(self, config: ChatUIConfig | None = None) -> None:
        self.cfg = config or ChatUIConfig()
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None

    def build_command(self) -> list[str]:
        """The sidecar launch line: the venv's ``open-webui`` entrypoint.

        The CLI ignores PORT/HOST env vars, so host/port go as ``serve``
        flags. ``python -m open_webui`` has no ``__main__``, so prefer the
        console-script exe next to sys.executable and fall back to invoking
        the entry point via ``-c``.
        """
        if self.cfg.command is not None:
            return list(self.cfg.command)
        exe = Path(sys.executable).with_name("open-webui.exe")
        if exe.exists():
            launcher: list[str] = [str(exe)]
        else:
            launcher = [sys.executable, "-c", "from open_webui import app; app()"]
        port = urlsplit(self.cfg.upstream_origin).port or WEBUI_PORT
        return [*launcher, "serve", "--host", "127.0.0.1", "--port", str(port)]

    def build_env(self) -> dict[str, str]:
        """The sidecar environment: ecoGPT gateway as the only model backend.

        Security tradeoff note: WEBUI_AUTH=False disables Open WebUI's sign-in.
        This is acceptable because both the daemon and the sidecar bind
        loopback only, so only local processes can reach them; the sidecar
        holds no upstream credentials (OPENAI_API_KEY is a dummy — the daemon
        substitutes its own).
        """
        env = dict(os.environ)
        env.update({
            "WEBUI_AUTH": "False",
            "ENABLE_SIGNUP": "False",
            "OPENAI_API_BASE_URL": self.cfg.public_origin + "/v1",
            "OPENAI_API_KEY": "ecogpt-managed",
            "OLLAMA_BASE_URL": self.cfg.ollama_base_url,
            "DATA_DIR": self.cfg.resolved_data_dir(),
            "WEBUI_NAME": self.cfg.webui_name,
            "WEBUI_SECRET_KEY": self.cfg.webui_secret_key,
            "DO_NOT_TRACK": "True",
        })
        env.update(self.cfg.env)
        return env

    async def start(self) -> None:
        """Spawn the sidecar and start its output reader."""
        command = self.build_command()
        log.info("chat-UI sidecar starting: %s", " ".join(command))
        self._proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self.build_env(),
        )
        self._reader = asyncio.create_task(self._read_output(), name="chat-ui-output-reader")

    async def _read_output(self) -> None:
        """Consume sidecar stdout (log relay only; no token harvest anymore)."""
        assert self._proc is not None and self._proc.stdout is not None
        async for raw in self._proc.stdout:
            line = _ANSI_RE.sub("", raw.decode("utf-8", errors="replace")).rstrip()
            if line:
                log.info("chat-ui: %s", line)

    async def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait until the sidecar answers ANY HTTP response (200 counts)."""
        assert self._proc is not None
        deadline = asyncio.get_event_loop().time() + (timeout or self.cfg.ready_timeout_s)
        async with httpx.AsyncClient(base_url=self.cfg.upstream_origin) as probe:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    response = await probe.get("/", timeout=2.0)
                    if response.status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        return False

    async def stop(self) -> None:
        """Terminate the sidecar."""
        if self._reader:
            self._reader.cancel()
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        self._proc = None


def create_chat_router(
    sidecar: ChatUISidecar,
    config: ChatUIConfig | None = None,
    *,
    # Dependency-injected upstream connector for the WS tunnel (tests).
    ws_connect: Callable[[str], Coroutine[Any, Any, Any]] | None = None,
) -> APIRouter:
    """Build the daemon-side /chat router: a prefix-stripping reverse proxy.

    - /chat/{…} HTTP  → prefix-stripped proxy to the Open WebUI origin.
    - /chat/{…} WS    → WebSocket tunnel (same path upstream), e.g.
                        /chat/ws/socket.io.
    - Cold sidecar    → 503 "starting" page, never an auth wall.
    """
    cfg = config or sidecar.cfg
    router = APIRouter()

    @router.api_route(cfg.chat_prefix + "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(path: str, request: Request) -> Response:
        return await _proxy_request_shared(cfg, request, path)

    @router.get(cfg.chat_prefix)
    async def chat_root(request: Request) -> Response:
        return await _proxy_request_shared(cfg, request, path="")

    @router.websocket(cfg.chat_prefix + "/{path:path}")
    async def ws_tunnel(path: str, websocket: WebSocket) -> None:
        await _tunnel_ws(websocket, cfg, "/" + path, ws_connect)

    return router


def create_chat_ui_fallback(
    sidecar: ChatUISidecar,
    config: ChatUIConfig | None = None,
    *,
    ws_connect: Callable[[str], Coroutine[Any, Any, Any]] | None = None,
) -> APIRouter:
    """Root-absolute fallback for the Open WebUI frontend (registered LAST).

    Open WebUI's built SPA references root-absolute paths (``/static/...``,
    ``/api/...``, ``/ws/socket.io``, …), so while it is served under ``/chat``
    the browser still hits the daemon origin at the root. This router forwards
    exactly those namespaces to the sidecar. It MUST be included after every
    daemon route: Starlette matches in registration order, so all daemon
    routes (``/api/state``, ``/api/usage``, ``/v1/...``, the dashboard, …)
    keep precedence and only unmatched WebUI paths fall through.
    """
    cfg = config or sidecar.cfg
    router = APIRouter()

    for prefix in WEBUI_HTTP_PREFIXES:
        @router.api_route(prefix + "{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
        async def http_fallback(rest: str, request: Request, _prefix: str = prefix) -> Response:
            return await _proxy_request_shared(cfg, request, _prefix.lstrip("/") + rest)

    for exact in WEBUI_HTTP_EXACT:
        @router.api_route(exact, methods=["GET", "HEAD", "OPTIONS"])
        async def http_exact(request: Request, _exact: str = exact) -> Response:
            return await _proxy_request_shared(cfg, request, _exact.lstrip("/"))

    for prefix in WEBUI_WS_PREFIXES:
        @router.websocket(prefix + "{rest:path}")
        async def ws_fallback(rest: str, websocket: WebSocket, _prefix: str = prefix) -> None:
            await _tunnel_ws(websocket, cfg, "/" + _prefix.lstrip("/") + rest, ws_connect)

    return router


async def _proxy_request_shared(cfg: ChatUIConfig, request: Request, path: str) -> Response:
    """Proxy core shared by the /chat router and the root fallback.

    - forwards the browser's Host header (upstream vhosts/fences key off it),
    - forwards the RAW query string (combo/asset URLs must survive verbatim),
    - strips content-encoding on buffered bodies (httpx already decoded them),
    - cold sidecar → 503 "starting" page.
    """
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
    headers["host"] = request.headers["host"]
    client = httpx.AsyncClient(base_url=cfg.upstream_origin, timeout=httpx.Timeout(90.0))
    body = await request.body() if request.method not in ("GET", "HEAD") else None
    url = "/" + path
    if request.url.query:
        url += "?" + request.url.query
    upstream_request = client.build_request(request.method, url, headers=headers, content=body)
    try:
        upstream_response = await client.send(upstream_request, stream=True)
    except httpx.HTTPError:
        await client.aclose()
        return Response("ecoGPT chat is starting; retry in a moment.", status_code=503)
    content_type = upstream_response.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        # Genuinely incremental: relay raw bytes as they arrive.
        return StreamingResponse(
            upstream_response.aiter_raw(),
            status_code=upstream_response.status_code,
            headers={k: v for k, v in upstream_response.headers.items() if k.lower() not in _HOP_HEADERS},
            background=BackgroundTask(client.aclose),
        )
    # Bufferable JSON/HTML/etc: read fully, then close upstream deterministically.
    data = await upstream_response.aread()
    await upstream_response.aclose()
    await client.aclose()
    return Response(data, status_code=upstream_response.status_code,
                    headers={k: v for k, v in upstream_response.headers.items()
                             # Response() recomputes length; content-encoding must go too:
                             # httpx already decoded the body (aread), forwarding the
                             # gzip/br header makes browsers fail with
                             # ERR_CONTENT_DECODING_FAILED
                             if k.lower() not in _HOP_HEADERS | {"content-length", "content-encoding"}})


async def _tunnel_ws(websocket: WebSocket, cfg: ChatUIConfig, upstream_path: str,
                     ws_connect: Callable[[str], Coroutine[Any, Any, Any]] | None) -> None:
    """WebSocket tunnel core: relays frames between browser and sidecar."""
    connector = ws_connect or _default_ws_connect
    await websocket.accept()
    upstream = await connector(cfg.upstream_origin + upstream_path)
    pump_down = asyncio.create_task(_pump_ws(upstream, websocket))
    pump_up = asyncio.create_task(_pump_ws_reverse(websocket, upstream))
    done, _pending = await asyncio.wait({pump_down, pump_up}, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        if task.exception() is not None:
            log.warning("chat-UI WS tunnel ended: %s", task.exception())
    for task in (pump_down, pump_up):
        task.cancel()
    await websocket.close()


def _default_ws_connect(url: str) -> Coroutine[Any, Any, Any]:
    """Default upstream WS connector (real protocol client)."""
    import websockets  # imported late: only needed when a WS client attaches

    return websockets.connect(url)


async def _pump_ws(source: Any, target: WebSocket) -> None:
    """Rebyte-relay: upstream WS -> browser WS."""
    async for message in source:
        if isinstance(message, bytes):
            await target.send_bytes(message)
        else:
            await target.send_text(message)


async def _pump_ws_reverse(source: WebSocket, target: Any) -> None:
    """Browser WS -> upstream WS (text frames only, per the socket.io relay)."""
    while True:
        message = await source.receive_text()
        await target.send(message)
