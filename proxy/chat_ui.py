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
- Session bootstrap is zero-touch: the sidecar's harvested launch token is
  exchanged server-side against the harness (`GET /?token=…` → 303 +
  Set-Cookie), and the minted harness session cookie is relayed to the
  browser, so the first /chat load works without the raw-401 token dance.
  When no exchange is possible yet, a one-click "Continue as admin"
  interstitial replaces the harness's raw 401.
- Telemetry rows (otel / session-telemetry-otel / desktop-product-telemetry)
  are disabled at the composition layer, not here.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - only used for a bounded taskkill tree-kill on win32
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine
from urllib.parse import parse_qs, urlsplit, urlparse

import httpx
from fastapi import APIRouter, Request, Response, WebSocket
from fastapi.responses import HTMLResponse
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
#
# Harvest hardening: through a piped (non-TTY) stdout the npm/npx wrapper on
# Windows interleaves ANSI escape sequences into the output (OSC title sets
# like `\x1b]0;C:\\WINDOWS\\system32\\cmd.exe \x1b\\` can land mid-line, even
# between the banner text and the URL), so escape sequences are stripped from
# each line BEFORE the URL regex runs. The token itself cannot be pinned: the
# CLI exposes no token flag and the plugin derives it per process from
# randomBytes (browser-auth.ts processLaunchToken), so stdout scraping — with
# the success/failure logs below — is the only bootstrap source.
_ENTRY_URL_RE = re.compile(r"(https?://\S*token=[A-Za-z0-9_-]+)")

# ANSI/OSC escape-sequence stripper (CSI, OSC with BEL or ST terminator, and
# two-byte remains); applied per line before the entry-URL regex.
_ANSI_RE = re.compile(r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-\\])")

# Session-cookie mechanics of the harness's authentication-token plugin
# (vendor/dsh-harness packages/host/authentication-token/src/browser-auth.ts):
# requests are authorized by a signed browser session cookie named
# dsh-auth-<base64url(sha256(host-authority))>; the only way to obtain one is
# the harness's own root exchange `GET /?token=<launch-token>` -> 303 +
# Set-Cookie, because the cookie's HMAC signing key stays inside the harness
# process. The proxy therefore relays that exchange instead of forging a
# cookie.
DSH_COOKIE_PREFIX = "dsh-auth-"
DSH_TOKEN_QUERY = "token"

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
    # Display name on the "Continue as …" login interstitial.
    admin_name: str = "admin"
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
        ``--public-url`` advertises the daemon origin + /chat so the printed
        tokenized entry route goes through the proxy, and ``--trusted-host``
        admits the daemon's authority for the harness's /api trust fence.
        """
        npx = shutil.which("npx") or shutil.which("npx.cmd") or "npx"
        command = self.cfg.command
        if command is None:
            authority = urlparse(self.cfg.public_origin).netloc
            command = ["npx", "-y", f"@deepseek-ai/dsh@{self.cfg.dsh_npm_pin}",
                       "--profile", "web", "--no-open", "--port", str(DSH_PORT),
                       "--public-url", self.cfg.public_origin + self.cfg.chat_prefix + "/",
                       "--trusted-host", authority]
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
        self._harvest_watch = asyncio.create_task(
            self._log_harvest_failure_once(), name="chat-ui-harvest-watch")

    async def _read_output(self) -> None:
        """Consume sidecar stdout, harvesting the tokenized entry URL."""
        assert self._proc is not None and self._proc.stdout is not None
        async for raw in self._proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            line = _ANSI_RE.sub("", line)
            log.debug("chat-ui: %s", line)
            if self.entry_url is None:
                match = _ENTRY_URL_RE.search(line)
                if match:
                    self.entry_url = match.group(1)
                    log.info("chat-UI entry URL discovered: %s", self.entry_url)

    async def _log_harvest_failure_once(self) -> None:
        """After the readiness window, say so when the token never arrived."""
        await asyncio.sleep(self.cfg.ready_timeout_s)
        if self.entry_url is None and self._proc is not None and self._proc.returncode is None:
            log.info(
                "chat-UI: no entry URL with token= found in sidecar output; "
                "first-load sign-in unavailable")

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
        if self._harvest_watch:
            self._harvest_watch.cancel()
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

    - GET /chat           → token-exchange relay until the harness session
                            cookie exists (auto-mints the cookie on first
                            load; falls back to the tokenized entry URL
                            redirect, then the login interstitial — never the
                            raw 401).
    - /chat/api/<ws>      → WebSocket tunnel to the private origin.
    - everything else     → HTTP streaming proxy, prefix-stripped, with the
                            same 401 interception.
    """
    cfg = config or ChatUIConfig()
    router = APIRouter()

    @router.get(cfg.chat_prefix)
    async def chat_root(request: Request) -> Response:
        # First visit (no harness session cookie yet): exchange the sidecar's
        # one-time launch token server-side and relay the minted session
        # cookie to the browser, so the very first page load works with zero
        # manual setup. When no cookie can be minted yet, degrade to the
        # tokenized entry-URL redirect (via --public-url this re-enters the
        # proxy) or, as a last resort, the friendly interstitial instead of
        # the harness's raw 401. A cookied visitor passes straight through to
        # the SPA index.
        if sidecar.entry_url and not _has_harness_session(request, cfg):
            automatic = await _auto_exchange(cfg, sidecar, return_to=cfg.chat_prefix + "/")
            if automatic is not None:
                return automatic
            if sidecar.entry_url.startswith(cfg.public_origin):
                return _login_interstitial(cfg)
            # Entry URL points at some other origin (dev spike default): the
            # legacy handoff still beats a raw 401.
            return RedirectResponse(sidecar.entry_url, status_code=307)
        return await _proxy_request_shared(cfg, sidecar, request, path="")

    # Dedicated login route: ALWAYS forces the token exchange and returns to
    # the chat index, never back into the gate the button is escaping from.
    # It is the interstitial button's target, so a stale/absent token here
    # renders the helpful restart message below instead of redirecting.
    @router.get(cfg.chat_prefix + "/entry")
    async def chat_entry(request: Request) -> Response:
        token = _launch_token(sidecar.entry_url)
        if sidecar.entry_url and token is not None:
            automatic = await _auto_exchange(cfg, sidecar, return_to=cfg.chat_prefix + "/")
            if automatic is not None:
                return automatic
            return _entry_unavailable(cfg, "the harness rejected the launch-token exchange; "
                                           "the token is stale or the sidecar was restarted")
        return _entry_unavailable(cfg, "the sidecar has not published its tokenized entry URL yet")

    @router.websocket(cfg.chat_prefix + "/api/remote.mux")
    async def remote_mux(websocket: WebSocket) -> None:
        await _tunnel_ws(websocket, cfg, ws_connect)

    @router.api_route(cfg.chat_prefix + "/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def proxy(path: str, request: Request) -> Response:
        return await _proxy_request_shared(cfg, sidecar, request, path)

    return router


def create_chat_api_fallback(
    sidecar: ChatUISidecar,
    config: ChatUIConfig | None = None,
    *,
    ws_connect: Callable[[str], Coroutine[Any, Any, Any]] | None = None,
) -> APIRouter:
    """Root-absolute /api catch-all for the chat SPA (registered LAST).

    The harness web client's compile-time API_PATH is '/api' (root-absolute,
    packages/client/connection/src/api-path.ts), so the SPA served at /chat/
    calls http://<daemon>/api/... and wss://<daemon>/api/remote.mux — NOT
    /chat/api/.... The daemon must therefore forward unmatched /api paths to
    the chat sidecar. This router is meant to be included after every daemon
    route: Starlette matches in registration order, so all of the daemon's own
    /api/* endpoints keep precedence and only the remainder falls through.
    """
    cfg = config or sidecar.cfg
    router = APIRouter()

    @router.websocket("/api/remote.mux")
    async def api_remote_mux(websocket: WebSocket) -> None:
        await _tunnel_ws(websocket, cfg, ws_connect)

    @router.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
    async def api_fallback(rest: str, request: Request) -> Response:
        # No prefix strip: the harness owns /api at its root.
        return await _proxy_request_shared(cfg, sidecar, request, "api/" + rest)

    return router


async def _proxy_request_shared(cfg: ChatUIConfig, sidecar: ChatUISidecar,
                                request: Request, path: str) -> Response:
    """Proxy core shared by the /chat router and the root /api fallback.

    - forwards the browser's public Host (cookie audience + Origin fence),
    - forwards the RAW query string (the /plugins/??combo keys must survive),
    - intercepts a cold 401 with the launch-token exchange / interstitial.
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
    # Bufferable JSON/etc: read fully, then close upstream deterministically.
    data = await upstream_response.aread()
    await upstream_response.aclose()
    await client.aclose()
    if upstream_response.status_code == 401 and not _has_harness_session(request, cfg):
        # Auth wall hit (cold deep-link / SPA XHR before the session exists):
        # rerun the launch-token exchange and hand the minted cookie back with
        # a redirect to the same path — the browser retries with the cookie in
        # hand. When no exchange is possible, show the friendly interstitial;
        # the harness's raw 401 body never reaches an uncookied browser. A
        # browser that already sent the session cookie still gets the relayed
        # upstream 401 (re-running the exchange for it could redirect-loop on
        # revoked cookies).
        if sidecar.entry_url:
            automatic = await _auto_exchange(cfg, sidecar, return_to=request.url.path)
            if automatic is not None:
                return automatic
        return _login_interstitial(cfg)
    return Response(data, status_code=upstream_response.status_code,
                    headers={k: v for k, v in upstream_response.headers.items()
                             # Response() recomputes length; content-encoding must go too:
                             # httpx already decoded the body (aread), forwarding the
                             # gzip/br header makes browsers fail with
                             # ERR_CONTENT_DECODING_FAILED
                             if k.lower() not in _HOP_HEADERS | {"content-length", "content-encoding"}})


async def _tunnel_ws(websocket: WebSocket, cfg: ChatUIConfig,
                     ws_connect: Callable[[str], Coroutine[Any, Any, Any]] | None) -> None:
    """WebSocket tunnel core shared by /chat/api/remote.mux and /api/remote.mux."""
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


def public_authority(cfg: ChatUIConfig) -> str:
    """The authority requests must carry as `Host` when they reach the harness.

    The vendored authentication-token plugin validates every request against
    the literal Host header on the wire (browser-auth.ts requestAudience);
    `--public-url` advertises a URL only and configures no authentication, and
    the client-connection trust fence compares an attached browser Origin
    against that same Host. The daemon therefore forwards the browser's public
    Host (127.0.0.1:18090) on every proxied request — and mints the session
    cookie under that same public authority in the exchange — while
    `--trusted-host` covers the fence's non-loopback allowances.
    """
    return urlsplit(cfg.public_origin).netloc


def harness_cookie_name(cfg: ChatUIConfig) -> str:
    """The harness session-cookie name expected on proxied requests.

    Mirrors the authentication-token plugin's cookieName(): a host-authority
    SHA-256 digest, base64url (unpadded) after the fixed prefix. The authority
    is the one the harness sees on the wire — the PUBLIC origin's netloc,
    because the proxy forwards the browser's Host header (see
    public_authority()) — so this name must match the minted side.
    """
    digest = hashlib.sha256(public_authority(cfg).encode("utf-8")).digest()
    return DSH_COOKIE_PREFIX + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _launch_token(entry_url: str | None) -> str | None:
    """The harness launch token harvested from the sidecar's entry URL."""
    if not entry_url:
        return None
    tokens = parse_qs(urlsplit(entry_url).query).get(DSH_TOKEN_QUERY)
    return tokens[0] if tokens else None


def _has_harness_session(request: Request, cfg: ChatUIConfig) -> bool:
    """True when the browser carries the harness's signed session cookie.

    The proxy cannot verify the cookie's HMAC signature (that secret stays in
    the harness process); checking the exact name and v1/v2 prefix is enough
    to route cold first-loads to the token exchange while passing real
    sessions straight through — the harness re-verifies every request anyway.
    """
    value = request.cookies.get(harness_cookie_name(cfg))
    return value is not None and value.startswith(("v1.", "v2."))


async def _exchange_session_cookie(cfg: ChatUIConfig, launch_token: str) -> str | None:
    """Perform the harness's root token exchange server-side.

    The harness mints its session cookie only in response to its own exchange
    (``GET /?token=<launch-token>`` → 303 + Set-Cookie, signed with a secret
    that stays in the harness process), so the proxy performs that logged-in
    request on the browser's behalf and returns the harness-supplied
    Set-Cookie verbatim. Returns None when the upstream does not answer with
    the exchange handshake.
    """
    async with httpx.AsyncClient(base_url=cfg.upstream_origin,
                                 timeout=httpx.Timeout(10.0),
                                 follow_redirects=False) as client:
        try:
            # Host carries the PUBLIC authority so the minted cookie's signed
            # audience (and name) matches the proxied request stream, which
            # forwards the same public Host (see public_authority()).
            response = await client.get("/", params={DSH_TOKEN_QUERY: launch_token},
                                        headers={"host": public_authority(cfg)})
        except httpx.HTTPError:
            return None
    if response.status_code != 303:
        return None
    wanted = harness_cookie_name(cfg) + "="
    for cookie in response.headers.get_list("set-cookie"):
        if cookie.startswith(wanted):
            return cookie
    return None


async def _auto_exchange(cfg: ChatUIConfig, sidecar: ChatUISidecar,
                         *, return_to: str) -> Response | None:
    """Relay the launch-token exchange and hand the minted cookie to the browser.

    Auto-approves the localhost login without user interaction. Tradeoff note:
    the harness is loopback-only (127.0.0.1:3180) and this router serves it on
    the daemon's loopback origin, so anyone able to reach the daemon could
    mint the private session themselves; the launch token is never forwarded
    anywhere except the loopback upstream. Returns None when no token was
    harvested or the upstream did not complete the exchange.
    """
    token = _launch_token(sidecar.entry_url)
    if token is None:
        return None
    cookie = await _exchange_session_cookie(cfg, token)
    if cookie is None:
        return None
    return RedirectResponse(return_to, status_code=303,
                            headers={"set-cookie": cookie, "cache-control": "no-store"})


def _login_interstitial(cfg: ChatUIConfig) -> Response:
    """The one-click login page shown instead of the harness's raw 401.

    Styled like the rest of the ecoGPT UI (dark/neutral dashboard palette);
    the button targets the dedicated /chat/entry route, which always forces
    the token exchange with a return to the chat index — never back into a
    gate that bounces to this same page.
    """
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ecoGPT chat — sign in</title>
<style>
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center;
          background: #0b0d10; color: #f1f5f9; font: 15px/1.6 system-ui, sans-serif; }}
  .card {{ width: min(26rem, 90vw); background: #12151c; border: 1px solid #1d222d;
           border-radius: 14px; padding: 2rem 2.25rem; text-align: center; }}
  h1 {{ font-size: 1.15rem; margin: 0.2rem 0 0.4rem; }}
  p {{ color: #94a3b8; margin: 0 0 1.4rem; }}
  .leaf {{ font-size: 1.5rem; color: #3cffd0; }}
  a.button {{ display: inline-block; background: #06120e; color: #3cffd0;
              border: 1px solid #1d4428; padding: 0.55rem 1.4rem; border-radius: 999px;
              text-decoration: none; font-weight: 600; }}
  a.button:hover {{ background: #1d4428; }}
</style>
</head>
<body>
  <div class="card">
    <div class="leaf">&#127807;</div>
    <h1>ecoGPT chat</h1>
    <p>Finish signing in to the local chat session. Nothing leaves this machine.</p>
    <a class="button" href="{cfg.chat_prefix}/entry">Continue as {cfg.admin_name}</a>
  </div>
</body>
</html>
"""
    return HTMLResponse(html, status_code=401, headers={"cache-control": "no-store"})


def _entry_unavailable(cfg: ChatUIConfig, reason: str) -> Response:
    """Helpful page for a login exchange that cannot complete right now.

    Matches the `_login_interstitial` styling; per docs/chat-ui.md, a stale or
    never-harvested entry URL is fixed by restarting the daemon (which boots a
    fresh sidecar and prints a fresh token). No redirects: the retry button
    re-enters /chat/entry only if the user clicks it.
    """
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ecoGPT chat — sign-in unavailable</title>
<style>
  body {{ margin: 0; min-height: 100vh; display: grid; place-items: center;
          background: #0b0d10; color: #f1f5f9; font: 15px/1.6 system-ui, sans-serif; }}
  .card {{ width: min(30rem, 90vw); background: #12151c; border: 1px solid #1d222d;
           border-radius: 14px; padding: 2rem 2.25rem; text-align: center; }}
  h1 {{ font-size: 1.15rem; margin: 0.2rem 0 0.4rem; }}
  p {{ color: #94a3b8; margin: 0 0 1rem; }}
  code {{ background: #0e1116; border: 1px solid #1d222d; border-radius: 6px;
          padding: 0.15rem 0.5rem; color: #3cffd0; font-size: 0.85rem; }}
  .leaf {{ font-size: 1.5rem; color: #3cffd0; }}
  a.button {{ display: inline-block; background: #06120e; color: #3cffd0;
              border: 1px solid #1d4428; padding: 0.55rem 1.4rem; border-radius: 999px;
              text-decoration: none; font-weight: 600; }}
  a.button:hover {{ background: #1d4428; }}
</style>
</head>
<body>
  <div class="card">
    <div class="leaf">&#127807;</div>
    <h1>Sign-in unavailable right now</h1>
    <p>{reason}.</p>
    <p>Restart the daemon to boot a fresh chat sidecar:<br><code>./.venv/Scripts/python.exe -m proxy</code><br>(docs/chat-ui.md)</p>
    <a class="button" href="{cfg.chat_prefix}/entry">Try again</a>
  </div>
</body>
</html>
"""
    return HTMLResponse(html, status_code=503, headers={"cache-control": "no-store"})


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
