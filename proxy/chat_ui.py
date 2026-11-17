"""ecoGPT chat-UI sidecar (Phase-2 wiring for the vendored dsh-harness).

STATUS (this commit): NOT yet imported by proxy/app.py — the parent wiring
(reviewed + approved separately) will:

    from .chat_ui import ChatUIConfig, ChatUISidecar, create_chat_router
    app.include_router(create_chat_router(ChatUIConfig()))

Design (ADR summary, mirrored from the approved plan):

- The vendored dsh `web` profile keeps serving its own Node host on a private
  port (default 3180, loopback only). This module manages that subprocess and
  reverse-proxies HTTP + WebSocket traffic under ``/chat`` on the daemon's
  origin, so the browser sees one origin (18090).
- Model traffic does NOT come here: the harness's llm-deepseek adapter is
  config-pinned to ``http://127.0.0.1:18090/v1/messages`` (the ecoGPT
  pipeline) via the dsh user profile / --patch overlay. See
  v1/docs/chat-ui.md.
- This module holds no secrets: the dsh client key is a fixed dummy value
  (the daemon substitutes its own upstream credentials).
- Telemetry rows (otel / session-telemetry-otel / desktop-product-telemetry)
  are disabled at the composition layer, not here.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - only used for a bounded taskkill tree-kill on win32
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

import httpx
from fastapi import APIRouter, Request, Response, WebSocket
from starlette.background import BackgroundTask
from starlette.responses import RedirectResponse, StreamingResponse

log = logging.getLogger("proxy.chat_ui")

# Pinned npm distribution of the vendored harness. Update procedure:
# v1/docs/chat-ui.md ("Updating the pinned harness").
DSH_NPM_PIN = "0.2.2-rc.3"

# The private port the dsh webserver binds (loopback only; never advertised).
DSH_PORT = 3180

# Daemon-side prefix of the chat UI (same-origin with the dashboard).
CHAT_PREFIX = "/chat"

# dsh prints its tokenized entry URL on its stdout; this is what the browser
# is redirected to on first visit. Token exchange then sets the signed
# session cookie (httpOnly; the token is one-time per boot).
_ENTRY_URL_RE = re.compile(r"(https?://\S*token=[A-Za-z0-9_-]+)")

# Hop-by-hop headers that must never be forwarded in either direction.
_HOP_HEADERS = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                "te", "trailer", "transfer-encoding", "upgrade", "host",
                "content-length"}


@dataclass
class ChatUIConfig:
    """Where the dsh webserver lives and how the daemon advertises it."""

    upstream_origin: str = f"http://127.0.0.1:{DSH_PORT}"
    public_origin: str = "http://127.0.0.1:18090"
    chat_prefix: str = CHAT_PREFIX
    dsh_npm_pin: str = DSH_NPM_PIN
    dummy_api_key: str = "ecogpt-local-dummy"
    # Command used to boot the harness (overridable for tests). Default uses
    # the pinned npm distribution: `npx -y @deepseek-ai/dsh@<pin> --profile web --no-open`
    command: list[str] | None = None
    # Environment overrides always applied to the sidecar process.
    env: dict[str, str] = field(default_factory=dict)
    # Seconds to wait for the sidecar to answer its first request.
    ready_timeout_s: float = 60.0


class ChatUISidecar:
    """Owns the vendored harness subprocess and its discovered entry URL.

    Not a general process manager: exactly one sidecar, restartable, with a
    readiness gate and a captured entry URL for the daemon-side redirect.
    """

    def __init__(self, config: ChatUIConfig | None = None) -> None:
        self.cfg = config or ChatUIConfig()
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self.entry_url: str | None = None

    def build_command(self) -> list[str]:
        """The sidecar launch line.

        Resolution order mirrors the spike: npx resolves the pinned npm
        distribution (`@deepseek-ai/dsh@<pin>`) so deployment needs no source
        checkout; ``--no-open`` suppresses the harness's own browser handoff
        (the daemon owns the redirect) and the port is our private 3180.
        """
        npx = shutil.which("npx") or shutil.which("npx.cmd") or "npx"
        command = self.cfg.command
        if command is None:
            command = ["npx", "-y", f"@deepseek-ai/dsh@{self.cfg.dsh_npm_pin}",
                       "--profile", "web", "--no-open", "--port", str(DSH_PORT)]
        if command and command[0] == "npx":
            command = [npx, *command[1:]]
        return command

    async def start(self) -> None:
        """Spawn the sidecar and start its output reader."""
        env = dict(os.environ)
        env["DEEPSEEK_API_KEY"] = self.cfg.dummy_api_key  # proxy substitutes its own
        env.update(self.cfg.env)
        command = self.build_command()
        log.info("chat-UI sidecar starting: %s", " ".join(command))
        self._proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
        )
        self._reader = asyncio.create_task(self._read_output(), name="chat-ui-output-reader")

    async def _read_output(self) -> None:
        """Consume sidecar stdout, harvesting the tokenized entry URL."""
        assert self._proc is not None and self._proc.stdout is not None
        async for raw in self._proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            log.debug("chat-ui: %s", line)
            if self.entry_url is None:
                match = _ENTRY_URL_RE.search(line)
                if match:
                    self.entry_url = match.group(1)
                    log.info("chat-UI entry URL discovered: %s", self.entry_url)

    async def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait until the sidecar answers ANY HTTP response (401/303 count)."""
        assert self._proc is not None
        deadline = asyncio.get_event_loop().time() + (timeout or self.cfg.ready_timeout_s)
        async with httpx.AsyncClient(base_url=self.cfg.upstream_origin) as probe:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    response = await probe.get("/", timeout=2.0)
                    # dsh answers 401 (auth needed) / 303 (token exchange) / 200 once up.
                    if response.status_code >= 300 or response.status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        return False

    async def stop(self) -> None:
        """Terminate the sidecar (and its process tree on Windows)."""
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

    - GET /chat           → 401-until-cookie (redirects to the tokenized entry
                            URL when the sidecar has published one).
    - /chat/api/<ws>      → WebSocket tunnel to the private origin.
    - everything else     → HTTP streaming proxy, prefix-stripped.
    """
    cfg = config or ChatUIConfig()
    router = APIRouter()

    @router.get(cfg.chat_prefix)
    async def chat_root() -> Response:
        # First visit: send the browser to the token exchange.
        if sidecar.entry_url:
            return RedirectResponse(sidecar.entry_url, status_code=307)
        return Response("ecoGPT chat is starting; retry in a moment.", status_code=503)

    @router.websocket(cfg.chat_prefix + "/api/remote.mux")
    async def remote_mux(websocket: WebSocket) -> None:
        connector = ws_connect or _default_ws_connect
        await websocket.accept()
        upstream = await connector(cfg.upstream_origin + "/api/remote.mux")
        pump_down = asyncio.create_task(_pump_ws(upstream, websocket))
        pump_up = asyncio.create_task(_pump_ws_reverse(websocket, upstream))
        done, _pending = await asyncio.wait({pump_down, pump_up}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task.exception() is not None:
                log.warning("chat-UI WS tunnel ended: %s", task.exception())
        for task in (pump_down, pump_up):
            task.cancel()
        await websocket.close()

    @router.api_route(cfg.chat_prefix + "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(path: str, request: Request) -> Response:
        target = cfg.upstream_origin + "/" + path
        if request.url.query:
            target += "?" + request.url.query
        headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_HEADERS}
        client = httpx.AsyncClient(base_url=cfg.upstream_origin, timeout=httpx.Timeout(90.0))
        body = await request.body() if request.method not in ("GET", "HEAD") else None
        upstream_request = client.build_request(request.method, "/" + path, headers=headers, content=body,
                                                params=request.query_params)
        upstream_response = await client.send(upstream_request, stream=True)
        content_type = upstream_response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            # Genuinely incremental: relay raw bytes as they arrive.
            return StreamingResponse(
                upstream_response.aiter_raw(),
                status_code=upstream_response.status_code,
                headers={k: v for k, v in upstream_response.headers.items() if k.lower() not in _HOP_HEADERS},
                background=BackgroundTask(client.aclose),
            )
        # Bufferable JSON/etc: read fully, then close upstream deterministically.
        data = await upstream_response.aread()
        await upstream_response.aclose()
        await client.aclose()
        return Response(data, status_code=upstream_response.status_code,
                        headers={k: v for k, v in upstream_response.headers.items()
                                 if k.lower() not in _HOP_HEADERS | {"content-length"}})  # Response() recomputes length

    return router


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
    """Browser WS -> upstream WS (text frames only, per the harness protocol)."""
    while True:
        message = await source.receive_text()
        await target.send(message)
