# ecoGPT

A local daemon that sits between AI clients (Claude Code, Claude Desktop, the
ecoGPT chat harness) and model gateways. Every model request flows through one
pipeline: **identify → ledger → compress → route → observe → relay**, so the
dashboard sees green metrics, token usage, cost and full session traceability.

Adopted from the mdao_hackathon Token Proxy codebase; this repo is the
integrated v1.

## Endpoints

Ingress (all served on one proxy):

| Endpoint | Format |
|---|---|
| `POST /v1/messages` (+ `/count_tokens`) | Anthropic Messages |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | OpenAI Responses |
| `GET /v1/models` | Anthropic shape |

Requests are routed per model: `claude-*` → gateway `/v1/messages` (passthrough),
`gpt-*` → gateway `/v1/responses` (translated from Anthropic Messages), with
per-model `route_overrides` and `model_overrides` (aliasing) in the dashboard.
Ingress in any format is normalized to Anthropic Messages, runs through the
pipeline, and is re-encoded back to the caller's format — streaming (SSE)
included.

## Components

- `proxy/` — FastAPI app on `http://127.0.0.1:18090`, SQLite usage ledger with
  energy/GWP estimates (EcoLogits methodology, factor table in
  `data/green_factors.json`), JSONL session ledger, Headroom compression
  sidecar (managed child process on port 8787, `headroom-ai[proxy]`).
- `dashboard/index.html` — single-file zero-dependency dashboard served at `/`:
  gateway + key, model exposure and routing, one-click **Claude Code** and
  **Claude Desktop** configuration (with byte-for-byte backup/restore),
  feature flags, green metrics rollups, sidecar controls, usage and session
  timelines.
- `tests/` — pytest suite; `tools/capture_probe.py` — standalone logging
  forwarder for client-behavior probing.

## Run

```sh
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m proxy
```

Dashboard at `http://127.0.0.1:18090/`. Tests: `.venv/Scripts/python.exe -m pytest`.

Config lives in `data/config.json` (editable live from the dashboard; the
ablation harness may rewrite `features`/`green` between requests — dashboard
saves respect on-disk changes).

## Status / next phases

- Done: proxy pipeline, all three ingress formats, Claude Code + Desktop
  one-click config, green metrics, session ledger, dashboard.
- TODO: formal cost model (pricing tables — cost is currently scraped from
  gateway headers), router/compression config polish, green factor editor +
  location/grid detection, ecoGPT chat harness integration.
