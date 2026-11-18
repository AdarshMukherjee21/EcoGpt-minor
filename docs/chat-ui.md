# ecoGPT Chat UI (Open WebUI) — runbook

Status: wired in `proxy/chat_ui.py` and mounted by `proxy/app.py` when
`settings.chat_ui == "openwebui"` (default) and the daemon boots with
`manage_chat_ui=True` (`python -m proxy`).

## What this is

The user-facing chat UI is a pinned **Open WebUI** (`open-webui==0.13.4`,
see `requirements.txt`):

- Served by Open WebUI's own uvicorn host on the private loopback port **3181**.
- The daemon reverse-proxies it at **/chat** (prefix-stripping HTTP + WS), so
  the browser sees one origin (127.0.0.1:18090).
- Open WebUI's frontend also calls root-absolute paths (`/static/...`,
  `/_app/...`, `/manifest.json`, `/api/...`, `wss /ws/socket.io`). Those
  namespaces are forwarded by a fallback router registered **last** in
  `create_app`, so every daemon route (`/`, dashboard, `/api/state`,
  `/api/usage`, `/v1/...`, …) keeps precedence; only unmatched WebUI
  namespaces reach the sidecar.
- All model traffic is pinned to the ecoGPT gateway: the sidecar gets
  `OPENAI_API_BASE_URL=http://127.0.0.1:18090/v1` and a dummy
  `OPENAI_API_KEY=ecogpt-managed`; the daemon performs credential
  substitution upstream, so the sidecar holds no real credentials.

## Sidecar env (set in `ChatUISidecar.build_env`)

| Var | Value | Why |
|---|---|---|
| `WEBUI_AUTH` | `False` | loopback-only; see security note below |
| `ENABLE_SIGNUP` | `False` | no accounts on a local sidecar |
| `OPENAI_API_BASE_URL` | `<daemon public origin>/v1` | all chat traffic flows through the ecoGPT pipeline |
| `OPENAI_API_KEY` | `ecogpt-managed` | dummy; the gateway substitutes its own key |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:9` | dead port — no local Ollama, OpenAI path only |
| `DATA_DIR` | `<repo>/data/openwebui` | SQLite + uploads live under the repo |
| `WEBUI_NAME` | `ecoGPT` | branding |
| `WEBUI_SECRET_KEY` | fixed local string | session signing; loopback only |
| `DO_NOT_TRACK` | `True` | telemetry off |

The launch line is the venv entrypoint (the CLI ignores PORT/HOST env, so
flags are mandatory):

```sh
.venv/Scripts/open-webui.exe serve --host 127.0.0.1 --port 3181
```

(`python -m open_webui` has no `__main__`; `proxy/chat_ui.py` falls back to
`python -c "from open_webui import app; app()"` when the exe is absent.
Python ≥3.13 needs `audioop-lts` for pydub — both are pinned in
`requirements.txt`.)

## Security note

`WEBUI_AUTH=False` disables Open WebUI's sign-in. This is acceptable because
both the daemon (18090) and the sidecar (3181) bind loopback only, so only
local processes can reach them. Do not expose either port beyond loopback.

## Settings gate

`settings.chat_ui` (`data/config.json`) is `"openwebui"` (default) or `"off"`;
editable via `PUT /api/settings` (`{"chat_ui": "off"}`). `"off"` unmounts
`/chat` and the fallback namespaces and skips spawning the sidecar. A daemon
restart is needed for the change to take effect.

## Version pin + update procedure

- `open-webui==0.13.4` in `requirements.txt` and `OPENWEBUI_PIN` in
  `proxy/chat_ui.py` carry the pin; bump both together.
- Update procedure:
  1. Check the Open WebUI release notes (breaking changes are common).
  2. `pip index versions open-webui` (or the release page) to pick the next pin.
  3. Reinstall the venv (`scripts/setup`), boot the daemon, and re-verify the
     live checklist: `GET /chat/` → 200 SPA HTML (no sign-in), an SPA asset
     through `/chat`, `GET /chat/api/config` → proxied 200, the socket.io WS
     path answers, and a chat completion flows through `/v1/...` into the
     dashboard ledger.
  4. Update the pin in both places, run `./.venv/Scripts/python.exe -m pytest -q`.

## License caveat

Open WebUI is **not** OSI-licensed (its "Open WebUI Modified License" restricts
some commercial/SaaS reuse). ecoGPT bundles it as a local, loopback-only
sidecar; if you redistribute or offer ecoGPT as a hosted service, review that
license first.
