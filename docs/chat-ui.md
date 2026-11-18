# ecoGPT Chat UI (Open WebUI) — runbook

Status: wired in `proxy/chat_ui.py` and mounted by `proxy/app.py` when
`settings.chat_ui == "openwebui"` (default) and the daemon boots with
`manage_chat_ui=True` (`python -m proxy`).

## What this is

The user-facing chat UI is a pinned **Open WebUI** (`open-webui==0.13.4`,
see `requirements.txt`):

- Served NATIVELY by Open WebUI's own uvicorn host on its own loopback port
  (**18091** by default). No reverse proxying: Open WebUI's SPA bakes a root
  base into its HTML, hardcodes root-absolute full-page jumps
  (`/auth?...`), and carries a socket.io WebSocket layer — all of which are
  fragile behind a prefix-stripping proxy. Its own origin is the zero-seam
  deployment.
- The daemon owns the sidecar lifecycle and a tiny **`/chat` link router**:
  `GET /chat` 307-redirects to the Open WebUI origin when the sidecar is up,
  and serves a "chat is starting" page (auto-retry) while it cold-boots.
  Deep paths map to the sidecar root (`/chat/c/abc` → `18091/c/abc`).
- The dashboard sidebar has an **Open Chat** button pointing at `/chat`
  (hidden when `chat_ui` is `"off"`).
- All model traffic is pinned to the ecoGPT gateway: the sidecar gets
  `OPENAI_API_BASE_URL=http://127.0.0.1:18090/v1` and a dummy
  `OPENAI_API_KEY=ecogpt-managed`; the daemon performs credential
  substitution upstream, so the sidecar holds no real credentials.

## Sidecar env (set in `ChatUISidecar.build_env`)

| Var | Value | Why |
|---|---|---|
| `WEBUI_AUTH` | `False` | loopback-only; see security note below |
| `ENABLE_SIGNUP` | `False` | no accounts on a local sidecar |
| `OPENAI_API_BASE_URL` | `<daemon origin>/v1` | all chat traffic flows through the ecoGPT pipeline |
| `OPENAI_API_KEY` | `ecogpt-managed` | dummy; the gateway substitutes its own key |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:9` | dead port — no local Ollama, OpenAI path only |
| `DATA_DIR` | `<repo>/data/openwebui` | SQLite + uploads live under the repo |
| `WEBUI_NAME` | `ecoGPT` | branding |
| `WEBUI_SECRET_KEY` | fixed local string | session signing; loopback only |
| `DO_NOT_TRACK` | `True` | telemetry off |

The launch line is the venv entrypoint (the CLI ignores PORT/HOST env, so
flags are mandatory):

```sh
.venv/Scripts/open-webui.exe serve --host 127.0.0.1 --port 18091
```

(`python -m open_webui` has no `__main__`; `proxy/chat_ui.py` falls back to
`python -c "from open_webui import app; app()"` when the exe is absent.
Python ≥3.13 needs `audioop-lts` for pydub — both are pinned in
`requirements.txt`.)

## Settings gates

- `settings.chat_ui` (`data/config.json`): `"openwebui"` (default) or
  `"off"`; editable via `PUT /api/settings` (`{"chat_ui": "off"}`). `"off"`
  unmounts `/chat` and skips spawning the sidecar.
- `settings.chat_ui_port`: the sidecar's loopback port (default **18091**);
  editable via `PUT /api/settings` (`{"chat_ui_port": 18092}`), validated to
  stay in 1024–65535 and off the daemon port. Effect after a daemon restart.
- `/api/state` reports `chat: {enabled, ui, url}` for the dashboard button.

A daemon restart is needed for any of these to take effect.

## Security note

`WEBUI_AUTH=False` disables Open WebUI's sign-in (the UI auto-provisions
`admin@localhost`). This is acceptable because both the daemon (18090) and
the sidecar (18091) bind loopback only, so only local processes can reach
them. Do not expose either port beyond loopback.

## Sharing with other people

- **Everyone runs their own copy** (the pip/release packaging): nothing
  changes, the loopback posture above is correct as-is.
- **You host one central instance**: bind the daemon to the LAN/public
  interface, put both ports behind one reverse proxy (Caddy/nginx; one vhost
  with the dashboard root and a `chat` location or subdomain), flip the
  sidecar to real accounts (`WEBUI_AUTH=True` + `ENABLE_SIGNUP=True`), and
  gate the daemon's admin APIs before exposing them. The sidecar's
  `OPENAI_API_BASE_URL` points at the daemon origin and is used server-side,
  so it can stay loopback forever.

## Version pin + update procedure

- `open-webui==0.13.4` in `requirements.txt` and `OPENWEBUI_PIN` in
  `proxy/chat_ui.py` carry the pin; bump both together.
- Update procedure:
  1. Check the Open WebUI release notes (breaking changes are common).
  2. `pip index versions open-webui` (or the release page) to pick the next pin.
  3. Reinstall the venv (`scripts/setup`), boot the daemon, and re-verify the
     live checklist: `GET /chat` → 307 to the sidecar origin, the Open WebUI
     root loads the SPA without sign-in, and a chat completion flows through
     `/v1/...` into the dashboard ledger.
  4. Update the pin in both places, run `./.venv/Scripts/python.exe -m pytest -q`.

## License caveat

Open WebUI is **not** OSI-licensed (its "Open WebUI Modified License" restricts
some commercial/SaaS reuse). ecoGPT bundles it as a local, loopback-only
sidecar; if you redistribute or offer ecoGPT as a hosted service, review that
license first.
