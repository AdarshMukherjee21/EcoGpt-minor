"""ecoGPT chat-UI sidecar — Open WebUI on its own local port.

Design:

- The chat UI is a pinned Open WebUI (0.13.x) served UNMODIFIED on its own
  loopback port (default 18091) by its own uvicorn host. Reverse-proxying it
  under a daemon prefix was tried and abandoned: Open WebUI's SvelteKit SPA
  bakes a root base into its HTML and hardcodes root-absolute full-page
  jumps, and its socket.io/WebSocket layer is fragile behind a tunnel.
  Serving it natively on a dedicated origin removes all of those seams.
- This module owns that subprocess and a tiny daemon router:
  GET /chat redirects to the Open WebUI origin when it is up, and serves a
  "starting" page (auto-retry) while it is not. The dashboard links to it.
- Model traffic does NOT come from here: Open WebUI is pointed at the ecoGPT
  gateway via ``OPENAI_API_BASE_URL=<daemon origin>/v1`` and a dummy
  ``OPENAI_API_KEY``; the daemon performs credential substitution upstream.
- ``WEBUI_AUTH=False`` (loopback-only deployment, see the tradeoff note in
  ``ChatUISidecar.build_env``): no sign-in; the UI auto-provisions
  ``admin@localhost`` on first use.
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

import httpx
from fastapi import APIRouter, Response
from starlette.responses import HTMLResponse, RedirectResponse

log = logging.getLogger("proxy.chat_ui")

# Pinned Open WebUI distribution (requirements.txt). Update procedure:
# docs/chat-ui.md ("Updating the pinned Open WebUI").
OPENWEBUI_PIN = "0.13.4"

# Default port the Open WebUI server binds (loopback only). Adjacent to the
# daemon's 18090 so the pair reads as one product; overridable in settings.
WEBUI_PORT = 18091

# Daemon-side path that links/redirects to the chat UI (dashboard target).
CHAT_PREFIX = "/chat"

# ANSI/OSC escape sequences (piped stdout interleaves OSC title sets from the
# npm-style wrappers); stripped per line so logs stay readable.
_ANSI_RE = re.compile(r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-\\])")

# "Chat is starting" page served while the sidecar cold-boots (first boot
# imports the whole Open WebUI backend and runs DB migrations). Auto-reloads
# so the browser lands on the redirect as soon as the sidecar is up.
_STARTING_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ecoGPT chat is starting</title>
<meta http-equiv="refresh" content="3">
<style>
  body { margin: 0; min-height: 100vh; display: grid; place-items: center;
         background: #0e1116; color: #e6e8eb;
         font: 14px/1.6 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif; }
  .card { text-align: center; padding: 32px 40px; border-radius: 12px;
          background: #151a21; border: 1px solid #232a33; }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 6px; }
  p { margin: 0.2em 0; color: #9aa4b0; }
</style>
</head>
<body>
  <div class="card">
    <h1>ecoGPT chat is starting</h1>
    <p>Open WebUI takes a moment on first boot.</p>
    <p>This page reloads automatically&hellip;</p>
  </div>
</body>
</html>
"""


@dataclass
class ChatUIConfig:
    """Where the Open WebUI server lives and how the daemon links to it."""

    # Loopback port the Open WebUI server binds (its own origin, not proxied).
    webui_port: int = WEBUI_PORT
    # Daemon origin: used for OPENAI_API_BASE_URL (server-side traffic) so
    # all chat requests flow through the ecoGPT pipeline.
    public_origin: str = "http://127.0.0.1:18090"
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

    def webui_url(self) -> str:
        return f"http://127.0.0.1:{self.webui_port}"

    def chat_url(self, path: str = "") -> str:
        """Where the daemon sends the browser: the Open WebUI origin root.

        ``/chat/<rest>`` daemon paths map to ``/<rest>`` on the sidecar (the
        SPA lives at its own origin root).
        """
        rest = (path or "").strip("/")
        return self.webui_url() + ("/" + rest if rest else "/")

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
        # The console script lands as open-webui.exe (Windows Scripts/) or
        # open-webui (POSIX bin/) next to the running interpreter.
        exe = Path(sys.executable).with_name("open-webui.exe")
        if not exe.exists():
            exe = Path(sys.executable).with_name("open-webui")
        if exe.exists():
            launcher: list[str] = [str(exe)]
        else:
            launcher = [sys.executable, "-c", "from open_webui import app; app()"]
        return [*launcher, "serve", "--host", "127.0.0.1", "--port", str(self.cfg.webui_port)]

    def build_env(self) -> dict[str, str]:
        """The sidecar environment: ecoGPT gateway as the only model backend.

        Security tradeoff note: WEBUI_AUTH=False disables Open WebUI's
        sign-in. This is acceptable because both the daemon and the sidecar
        bind loopback only, so only local processes can reach them; the
        sidecar holds no upstream credentials (OPENAI_API_KEY is a dummy —
        the daemon substitutes its own).
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
        log.info("chat-UI sidecar starting on port %s: %s", self.cfg.webui_port, " ".join(command))
        self._proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=self.build_env(),
        )
        self._reader = asyncio.create_task(self._read_output(), name="chat-ui-output-reader")

    async def _read_output(self) -> None:
        """Consume sidecar stdout (log relay only)."""
        assert self._proc is not None and self._proc.stdout is not None
        async for raw in self._proc.stdout:
            line = _ANSI_RE.sub("", raw.decode("utf-8", errors="replace")).rstrip()
            if line:
                log.info("chat-ui: %s", line)

    async def _probe(self, timeout: float = 2.0) -> bool:
        """True when the sidecar answers ANY HTTP response (200 counts)."""
        try:
            async with httpx.AsyncClient(base_url=self.cfg.webui_url()) as probe:
                response = await probe.get("/", timeout=timeout)
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def wait_ready(self, timeout: float | None = None) -> bool:
        """Wait until the sidecar answers its first HTTP response."""
        deadline = asyncio.get_event_loop().time() + (timeout or self.cfg.ready_timeout_s)
        while asyncio.get_event_loop().time() < deadline:
            if await self._probe():
                return True
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


def create_chat_link_router(
    sidecar: ChatUISidecar,
    config: ChatUIConfig | None = None,
    *,
    # Dependency-injected readiness probe (tests).
    probe: Callable[[ChatUIConfig], Coroutine[Any, Any, bool]] | None = None,
) -> APIRouter:
    """Build the daemon-side /chat router: a pointer, not a proxy.

    - sidecar up   → 307 redirect to the Open WebUI origin (own port).
    - sidecar down → 503 "starting" HTML page that auto-retries.
    """
    cfg = config or sidecar.cfg
    router = APIRouter()

    async def _ready() -> bool:
        if probe is not None:
            return await probe(cfg)
        return await sidecar._probe()

    async def link(path: str = "") -> Response:
        if await _ready():
            return RedirectResponse(cfg.chat_url(path), status_code=307)
        return HTMLResponse(_STARTING_PAGE, status_code=503)

    @router.get(CHAT_PREFIX)
    @router.head(CHAT_PREFIX)
    async def chat_root() -> Response:
        return await link()

    @router.get(CHAT_PREFIX + "/{path:path}")
    async def chat_deep(path: str) -> Response:
        return await link(path)

    return router
