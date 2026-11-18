# ecoGPT

A local AI-gateway daemon that sits between your model clients and your model
providers. Claude Code, Claude Desktop, OpenAI-compatible apps and the built-in
ecoGPT chat UI all speak to one local proxy — and every model request flows
through a single observable pipeline:

**identify → ledger → compress → route → measure → relay**

Along the way ecoGPT tracks what it did and what it cost the planet: token
usage, session timelines, per-call traces, energy and environmental impact
(energy, CO₂e, water, abiotic resource use) — with zero dependence on what
your upstream provider reports.

- One gateway, three dialects: Anthropic Messages, OpenAI Chat Completions and
  OpenAI Responses — any combination of client and provider format works,
  including SSE streaming.
- Prompt compression (Headroom sidecar), applied adaptively per request.
- Optional agentic tier router (System One): a classifier picks the right
  model for each turn of the auto model — cheap executors for easy asks,
  capable planners for hard ones — with warm-cache-friendly routing.
- Green metrics on every request, via EcoLogits compute core: energy (kWh),
  GWP (kgCO₂e), water (L), ADPe (kgSbeq), primary energy (MJ).
- An all-in-one local dashboard: configuration, routing, live trace, usage and
  environmental rollups. Plus a bundled chat UI (Open WebUI) at `/chat`.
- Local-first security posture: the daemon binds loopback only; your provider
  credentials never leave `data/config.json`.

## Quickstart

Requires Python **3.13+** (uv recommended, not required).

```sh
git clone <this-repo> ecoGPT && cd ecoGPT
scripts/setup          # creates .venv and installs pinned requirements
.venv/Scripts/python.exe -m proxy    # Windows
# .venv/bin/python -m proxy          # macOS / Linux
```

Windows note: run the commands from Git Bash (the `scripts/setup` and run docs
assume it). The venv layout is auto-detected on POSIX.

When the daemon starts:

| Surface | URL |
| --- | --- |
| Dashboard | `http://127.0.0.1:18090/` |
| Chat UI (Open WebUI) | `http://127.0.0.1:18090/chat` → redirects to `http://127.0.0.1:18091` |
| Anthropic Messages | `POST http://127.0.0.1:18090/v1/messages` |
| OpenAI Chat Completions | `POST http://127.0.0.1:18090/v1/chat/completions` |
| OpenAI Responses | `POST http://127.0.0.1:18090/v1/responses` |
| Models listing | `GET http://127.0.0.1:18090/v1/models` |
| Admin API | `/api/state`, `/api/settings`, `/api/usage`, `/api/trace/...`, ... |

First thing to do: open the dashboard → **Setup & Providers**, point it at your
gateway (up to three providers: name, base URL, API key) and expose the models
you want clients to see. Claude Code and Claude Desktop can be pointed at
ecoGPT with one click each (with automatic backup/restore of their existing
config).

The chat UI is a pinned Open WebUI spawned and managed by the daemon on its
own loopback port (18091 by default, `chat_ui_port` setting). It uses the
gateway as its only model backend — everything you chat with it flows through
the same pipeline, ledger and green accounting. Sign-in is disabled
(`WEBUI_AUTH=False`); on first use it provisions a local `admin@localhost`
user. See `docs/chat-ui.md`.

Requests are routed per model id by default (`claude-*` → Anthropic-style
upstream, `gpt-*`/others → Responses/Chat-style upstream), with per-model
`route_overrides` and aliasing `model_overrides` configurable in the dashboard.

## Screenshots

All from the dashboard at `http://127.0.0.1:18090` unless noted.

### Overview — traffic, savings, footprint

![Dashboard overview: stat cards and environmental footprint](<screenshots/Screenshot (192).png>)

The Overview page leads with what matters at a glance: proxied request count,
tokens in, tokens **saved by compression** (Headroom), cost, and the
environmental footprint cards — energy (Wh), carbon (g CO₂e), water (mL),
provider cache hits and compression fallbacks — with an honest coverage note
(requests without an estimate are excluded, never counted as zero).

![Recent requests table and daily charts](<screenshots/Screenshot (193).png>)

Scrolling down: the recent-requests ledger — model, route, input/output/cache
tokens, tokens saved, latency, CO₂e, water, impact basis and HTTP status per
call — plus tokens-per-day and cost-per-day charts and an export-to-CSV button.

![Energy by model, per-model basis table, methodology attribution](<screenshots/Screenshot (194).png>)

The green detail: energy per model (bar = mean, whiskers = min–max), a per-model
table with each row's basis (here `USER-OVERRIDE`), provenance for every number
(EcoLogits 0.10.2 catalogue, dataset revision, region scenario, PUE/WUE source),
attribution/license strings for the methodology, and the recent sessions list
that deep-links into Live Trace.

### Live Trace — per-session trajectory

![Live Trace: session waterfall and per-step forensics](<screenshots/Screenshot (195).png>)

The multi-lane trajectory inspector: a duration waterfall across pipeline lanes
(input / router / model / tools) for each session, the step list with routing
badges, and a step-forensics panel — requested model vs. served model
(`claude-eco-1` → `baidu/ernie-5.5-flash-preview:free`), chosen tier (`goldfish`),
route, status, duration, TTFT, and the span waterfall (request → compression →
router → upstream) showing where the 3.15 s actually went.

### Setup & Providers — clients, gateways, models, features

![One-click Claude Code / Desktop setup and upstream providers](<screenshots/Screenshot (196).png>)

One-click client integration: Claude Code and Claude Desktop each get an
Apply/Restore-original card (byte-for-byte backup of their existing settings),
plus the upstream provider list — up to 3 gateways, masked API keys
("leave blank to keep"), first provider as default.

![Model exposure with per-model route and provider pinning, feature flags](<screenshots/Screenshot (197).png>)

Model exposure (14 models here): checkbox per model, the wire format each model
speaks (`chat` / `responses` / `anthropic`), and per-model provider pinning —
followed by the proxy feature toggles (compression, green metrics, session
tracing, run label).

### Compression — Headroom sidecar

![Compression settings and Headroom sidecar controls](<screenshots/Screenshot (198).png>)

The Headroom compression sidecar's lifecycle (running, healthy, endpoint,
start/stop/restart, optional Kompress ML model download) and its prompt posture:
mode (`cache` keeps provider prompt caches intact), savings profile, context
limit, failure behavior, and per-turn usage posting.

### System One Router — the tier ladder

![System One Router tier ladder and classifier endpoint](<screenshots/Screenshot (199).png>)

The auto model (`eco-auto`) gets classified per turn by a System One decision
endpoint and dispatched down a tier ladder — here `goldfish` (routine execution
→ `openrouter/free`) and `elephant` (planning → `baidu/ernie-…`). Classifier
endpoint presets (local Ollama, OpenRouter) with credentials stored server-side.

### Green Metrics — region & calibration

![Green metrics region and calibration settings](<screenshots/Screenshot (200).png>)

Region scenario (country, grid-intensity bounds, PUE bounds) and calibration
knobs (cached-input fraction, input/output energy ratio, embodied share) that
feed the EcoLogits impact math; below, the per-model expert profiles.

### Chat UI — Open WebUI via the gateway

![ecoGPT chat (Open WebUI on port 18091) with the auto model selected](<screenshots/Screenshot (201).png>)

The built-in chat UI — pinned Open WebUI on its own port (18091), reached via
the dashboard's **Open Chat** button (`/chat`). The model picker shows the
exposed gateway models including `eco-auto`; every message flows back through
the daemon's pipeline, so chats appear in the ledger, usage and green metrics
like any other client.

## Configuration

Everything user-configurable lives in `data/config.json` and is edited live
from the dashboard (no daemon restart needed, except for a few noted cases):

- Providers and exposed models, per-model provider mapping/stickiness.
- Router ladder and classifier endpoint (see below).
- Compression (Headroom) settings; the compression sidecar (port 8787) is a
  managed child process — no separate install or launch step.
- Green-metrics scenarios and per-model profiles.
- Desktop picker aliases for Claude Desktop, feature flags, chat UI on/off
  and its port.

The `data/` directory also holds the SQLite usage DB, the JSONL session
ledger (`data/sessions/`), green factors/overrides, the vendored EcoLogits
catalogue (`data/ecologits/`, committed for reproducible impact math), IKP
calibration data (`data/ikp/`) and the Open WebUI data dir (`data/openwebui/`).
Deleting `data/` resets everything to defaults on next boot.

## Pipeline, in order

1. **Identify** — the request's session identity (client, session key, request
   id) is pinned for tracing.
2. **Ledger** — an entry is opened in the JSONL session ledger
   (schema v2) and the usage DB (SQLite).
3. **Compress** — prompts run through the local Headroom sidecar
   (`headroom-ai[proxy]`); structural-only by default, ML (Kompress) lossless
   compression when its model cache is present.
4. **Route** — either straight to the model's provider, or through the System
   One tier router when the client selected the auto model (`eco-auto`).
5. **Measure** — tokens, durations, TPS/TTFT are observed from the actual
   stream; environmental impact is computed per request and snapshotted.
6. **Relay** — responses (including SSE) are re-encoded into the client's
   requested dialect.

## System One tier router

Clients can select the auto model (default id `eco-auto`) instead of a
specific model. ecoGPT then asks a classifier decision model — **any**
well-formed System One endpoint (Ollama and OpenRouter presets ship built-in)
which tier the prompt needs, and dispatches to that tier's model:

- A configurable ordered ladder (2–26 tiers, easiest → most capable), each
  tier with a prompt-kind description and exactly one model
  (default ladder: `goldfish` executor / `elephant` planner).
- Guards: confidence-floor escalation, long-context floor, a warmth hold
  (no downgrades within 120 s) and cache-stickiness (borderline flips within
  300 s stay put to keep provider prompt caches hot). All toggleable.
- Full traceability: tier, target model, confidence and routing reason are
  recorded per turn in the ledger and usage DB, and visible in the dashboard's
  Live Trace.

Classifier credentials are stored server-side in `data/config.json`, masked in
every dashboard/API view, and never travel in request bodies.

## Green metrics

When the `green_metrics` feature flag is on, every request gets an
environmental impact record computed by EcoLogits (`ecologits==0.10.2`
compute core, catalogue vendored in-repo):

- Five categories: energy (kWh), GWP (kgCO₂e), water (L), ADPe (kgSbeq),
  primary energy (MJ) — from the effective upstream model's parameters, the
  observed request duration, and TPS/TTFT when streamed.
- Model resolution is explicit: per-model expert profile → factor-table row →
  exact catalogue match → `unconfigured_model` (fail closed; no fuzzy
  matches, no invented numbers).
- Expert profiles per model: active/total parameters, average TPS/TTFT, and a
  datacenter scenario (PUE, on-site WUE) backed by EcoLogits electricity
  datasets with a World fallback; every estimate carries a provenance string.
- Unknown models can be sized with a resumable, free-tier-friendly IKP
  estimation run (runs from Settings → Green metrics, persists progress,
  applied only on explicit acceptance).
- Historical records are immutable snapshots: changing settings never
  re-scores past requests. The dashboard shows coverage (share of requests
  with real numbers) alongside totals.

## Observability

- **Live Trace** in the dashboard: per-session timeline of every call —
  what compressed, the router's decision (tier, model, confidence), the
  upstream provider, streaming shape, and impact.
- **Sessions**: request-level ledgers grouped by client session, DeepSeek-style
  JSONL under `data/sessions/`.
- **Usage**: SQLite at `data/usage.db` — tokens, models, providers, router
  columns, impact columns; CSV export from the dashboard.
- **Green rollups**: daily and per-model rollups surfaced in the dashboard.

`tools/capture_probe.py` is a standalone logging forwarder for probing client
behavior if you want to inspect what a specific client sends.

## Tests

```sh
.venv/Scripts/python.exe -m pytest -q     # or .venv/bin/python -m pytest
```

The pytest suite (290+ tests) covers the translation layers (all three
dialects × streaming), routing, router guards, green metrics arithmetic and
provenance, ledger schemas, settings round-trips, sidecar wiring and the
chat-UI integration.

## Project layout

```
proxy/                    FastAPI daemon package
  app.py                  endpoints, settings, telemetry surfaces
  chat_ui.py              Open WebUI sidecar lifecycle + /chat link
  config.py               settings dataclass + data/config.json round-trip
  router_engine.py        System One classifier + tier dispatch
  green_impact.py         EcoLogits impact core + provenance
  green_estimate.py       resumable IKP estimation runs
  ledger.py               JSONL session ledger (schema v2)
  translate_*.py          Anthropic <-> Chat <-> Responses translation
  sidecar*.py              Headroom sidecar spawn/client
  claude_config.py         Claude Code one-click config
  claude_desktop.py        Claude Desktop apply/backup/restore
dashboard/index.html      single-file dashboard served at /
data/                     config, usage DB, ledgers, vendored datasets
docs/                     chat UI runbook, session-probe findings
scripts/setup             venv + pinned install
tests/                    pytest suite
pyproject.toml            pip-installable package (console script: ecogpt)
```

## Security model

EcoGPT is a **local, loopback-only** deployment, by design:

- The daemon, the Headroom sidecar and the Open WebUI sidecar all bind
  `127.0.0.1` only.
- Provider and classifier API keys live in `data/config.json` and are masked in
  every dashboard and API view.
- The chat UI runs with sign-in disabled (single local-user posture).
- Do not expose the daemon port to a network without adding auth/HTTPS at a
  reverse proxy — the dashboard and admin API are not authentication-gated.
  Multi-user hosting is planned for the packaging phase
  (`docs/chat-ui.md` has notes).

## Status

Working: the full pipeline, all three ingress dialects, Claude Code/Desktop
one-click, green metrics core + expert profiles + IKP estimation, tier router,
trace/sessions, dashboard, chat UI.

Planned (in flight): packaging for external users (single-command setup /
GitHub release / pip), formal cost model (headers-scraped today), multi-user
hosting posture.

## Licenses

- ecoGPT's own code: MIT
- Open WebUI (bundled as the chat UI): "Open WebUI Modified License" — not
  OSI-licensed; keep the deployment local/review the license before hosted
  commercial use (details in `docs/chat-ui.md`).
- EcoLogits (compute core + datasets) and `headroom-ai` are dependencies
  installed from PyPI under their own licenses.
