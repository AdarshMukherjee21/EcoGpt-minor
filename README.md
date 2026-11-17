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
  impact estimates (pinned `ecologits==0.10.2` compute core for energy/GWP/water/
  ADPe/primary energy; the `wh_per_output_token` override table in
  `data/green_factors.json` keeps the factor arithmetic for overridden rows),
  JSONL session ledger, Headroom compression sidecar (managed child process on
  port 8787, `headroom-ai[proxy]`).
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

## Routing: System One tier router

When a client selects an **auto model** (default id `eco-auto`), the request is
classified by a **System One decision model** (Jev API: `POST <endpoint_url>`
with `{model, state, questions}` returning `{answers: {difficulty: {choice,
probabilities, confidence}}}`) and dispatched to the matching tier's model.

- **Tier ladder** (user-configurable count 2–26, ordered easiest → most
  capable): each tier has a `id`, a **prompt-kind description** (classifier
  criteria text) and exactly **one model**. The shipped default ladder is
  goldfish (executor) / elephant (planner) with descriptions from the chat
  harness proposal; tiers, descriptions and models are editable in the Router
  settings tab.
- **Rules**: confidence-floor escalation (+1 tier when classification
  confidence < 0.45), long-context floor (prompts over the 256k advertised
  window jump to the long-context floor tier), **warmth hold** (no tier
  downgrade within 120 s) and **cache stickiness** (borderline flips within
  300 s stay on the previous tier so provider KV/prompt caches stay hot).
  All toggles/Windows live in Router settings; warmth and stickiness are ON
  by default.
- **Generic endpoints**: the classifier is any well-formed SystemOne URL
  (Ollama `/v1/systemone` and OpenRouter `/api/alpha/decisions` ship as
  presets); credentials (bearer/x-api-key) are stored server-side in
  `data/config.json` and never travel in request bodies or dashboard payloads.
  Sending prompts to a **non-loopback** endpoint shows a persistent egress
  notice in the dashboard.
- **Traceability**: every routing decision is appended to the per-session JSONL
  ledger (`data/sessions/`, DeepSeek-style) — requested auto model, chosen
  tier, target model, confidence, routing reason, warmth/stickiness flags —
  and to the usage DB (`router_tier`, `router_reason`, `router_target_model`,
  `router_confidence`, ...), so dashboards and exports can show why each turn
  was routed the way it was.

## Green metrics (Phase 2)

Defaults are explicit scenarios, never inferred data. When the `green_metrics`
feature flag is on, `proxy/green_impact.py` computes five impact categories per
request — energy (kWh, min/mean/max), GWP (kgCO₂e, min/mean/max), water (L),
ADPe (kgSbeq) and primary energy (MJ) — from the active/total parameters of the
*effective upstream model*, the observed request duration and, when streaming
observed them, TPS and time-to-first-token, via EcoLogits'
`compute_llm_impacts()`.

- **Model parameters** resolve in order: per-model override (dashboard →
  `data/green_overrides.json`, merged on mtime change) → factor-table row →
  exact match in the pinned EcoLogits catalogue (`data/ecologits/`, committed
  for reproducibility) → `unconfigured_model`. No fuzzy matches: a provider
  serving a model need not own it.
- **Per-model expert profiles** (Settings → Green metrics): an EcoLogits
  Calculator expert-mode card per exposed model — Active/Total parameters (B),
  average TPS and TTFT (used only when a stream did not observe them), plus the
  datacenter scenario (PUE, on-site WUE, location, and grid-mix GHG/abiotic/
  primary/water factors). Blank datacenter fields fall back to the region
  settings and the EcoLogits electricity dataset (displayed with their source).
- **Estimation for unknown models**: a stratified IKP probe run (7 tiers,
  public calibration from `data/ikp/`, judge model of your choice) estimates a
  model's knowledge-equivalent size — free-tier friendly (adjustable pacing
  between calls, exponential retry with Retry-After support for HTTP 429) and
  fully resumable: per-probe progress persists to `data/green_estimation.json`,
  a stop/failure/crash leaves a resumable state (resume via the manual
  **Continue** button, never a rerun from scratch), and an estimate is applied
  only when explicitly accepted (**Apply estimate** records an explicit
  `ikp_estimate` scenario — knowledge-equivalent capacity is not a verified
  weight count).
- **Dual path**: a row with a `wh_per_output_token` override keeps the original
  factor arithmetic (`proxy/green.py`) — overrides keep their meaning. Rows
  with parameter counts use the EcoLogits core. Input and cache-read tokens are
  folded into the effective output count on both paths; `embodied_share` applies
  to the factor path only (the library models embedded impacts itself).
- **Fail closed**: unknown models, missing usage and calculation errors are
  recorded with an `impact_status` (`unconfigured_model`, `missing_usage`,
  `_calculation_error_`) and never contribute numbers to totals. The dashboard
  shows the environmental coverage ratio.
- **Provenance on every estimate**: the impact record and dashboard always pair
  water/ADPe/primary-energy numbers with an `impact_provenance` string (model
  basis + electricity dataset revision + "PUE/WUE: region scenario"). Region
  grid/PUE come from the dashboard's Green settings; electricity-mix water,
  ADPe and PE come from the vendored EcoLogits dataset with a World fallback.
- The usage DB and the JSONL ledger store historical estimates as fixed
  snapshots; changing settings never re-scores past requests.

## Status / next phases

- Done: proxy pipeline, all three ingress formats, Claude Code + Desktop
  one-click config, green metrics (EcoLogits core, per-model expert profiles,
  IKP estimation), session ledger, dashboard.
- TODO: formal cost model (pricing tables — cost is currently scraped from
  gateway headers), router/compression config polish, ecoGPT chat harness
  integration.
