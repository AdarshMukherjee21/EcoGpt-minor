# ecoGPT — Midterm Demo

A small Flask app that demonstrates **three ways to make LLM usage more environmentally efficient**, and **quantifies the energy / carbon savings** of each.

> **Status:** Backend working end-to-end. Frontend is a placeholder. See [Current state](#current-state) and [Next steps](#next-steps).

---

## What this project is about

ecoGPT is a teaching demo built around the question:

> _"What if we routed easy prompts to a small model, compressed every prompt before sending it, and actually measured how much energy / carbon we saved?"_

The app combines:

1. **Smart routing** — a learned router (Microsoft RouteLLM) decides whether a prompt is _easy_ (route to a small model) or _hard_ (route to a large model). Both models are reached over **OpenAI-compatible HTTP endpoints** — the demo is provider-agnostic. The MF router uses a **local MiniLM embedding model** (via `sentence-transformers`) for win-rate prediction — no OpenAI API key needed for routing.
2. **Prompt compression** — [LLMLingua-2](https://github.com/microsoft/LLMLingua) shrinks prompts before inference, reducing input tokens. Device (`cpu` / `cuda` / `auto`) is configurable.
3. **Green-metric estimation** — a black-box Monte Carlo estimator returns per-request energy (Wh), carbon (g CO₂e), and water (L), with a full P5–P95 percentile distribution rather than a single misleading point estimate.

The headline UI story:

- **Compression tab** — send one prompt, get a side-by-side comparison of _without_ vs _with_ compression, plus the green-metric deltas.
- **Router tab** — send one prompt, see which model was chosen. If the small model was chosen, the large model's output is also shown for fidelity comparison, and savings are highlighted. If the large model was chosen, only the large model is shown. Green metrics are computed for every message.

---

## Architecture

```
midterm_demo/
├── app.py                 # Flask server, 4 endpoints
├── router.py              # RouteLLM wrapper (mf / bert router)
├── prompt_compression.py  # LLMLingua-2 wrapper + tiktoken counter
├── prompt_compresstion.py # DEPRECATED back-compat shim for the misspelled name
├── green_metric.py        # BlackBoxLLMEstimator (energy + carbon)
├── raw_model_runner.py    # OpenAI-compatible HTTP client
├── run.py                 # CLI smoke test of the full pipeline
├── index.html             # Placeholder tester UI (deferred)
├── question_to_ask.md     # Empty placeholder
├── README.md              # ← you are here
├── config.py              # Singleton config loader
├── config.yaml            # Non-secret config
├── .env                   # API keys (gitignored)
├── .env.example           # Template for .env
├── .gitignore
└── requirements.txt
```

### One-line file purposes

| File                                               | Purpose                                                                                                                                                                                                                                                                        |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [`app.py`](app.py)                                 | Flask app on port 5000. Exposes `/api/router`, `/api/raw-runner`, `/api/compress`, `/api/green-metric`. All model names, endpoints, and API keys come from `config.settings`.                                                                                                  |
| [`router.py`](router.py)                           | Wraps Microsoft RouteLLM `Controller`. Uses the **mf** router by default (`threshold=0.1159`). Reads model names, endpoints, API keys, and threshold from config.                                                                                                              |
| [`prompt_compression.py`](prompt_compression.py)   | Loads `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`. Device is resolved through `torch.device(...)` with an `auto` / `cpu` / `cuda` override from config. Exposes `compress_prompt(prompt, target_rate)` and `count_tokens(text)` (tiktoken `cl100k_base`). |
| [`prompt_compresstion.py`](prompt_compresstion.py) | **Deprecated** back-compat shim. Re-exports `PromptCompressorClient` and `tokenizer` from `prompt_compression`. Emits a `DeprecationWarning`. New code should import from `prompt_compression`.                                                                                |
| [`green_metric.py`](green_metric.py)               | `BlackBoxLLMEstimator`. Maps a model tag → HHAI size band (or `per_model_overrides`) → GPU count → reference TPS → Monte Carlo energy (10k samples) → carbon + water. Returns P5–P95 distributions for all three metrics.                                                      |
| [`raw_model_runner.py`](raw_model_runner.py)       | Posts to **any OpenAI-compatible `/v1/chat/completions` endpoint**. Provider-agnostic — works with OpenAI, Azure, Together, Groq, vLLM, llama.cpp's server, LM Studio, etc.                                                                                                    |
| [`config.py`](config.py)                           | Singleton config loader. Reads `config.yaml` + `.env` once, exposes a typed `Settings` object with sections: `project`, `models`, `router`, `compression`, `raw_runner`, `green_metric`.                                                                                       |
| [`config.yaml`](config.yaml)                       | Non-secret configuration: model names, endpoints, router tunables, compression rate, raw-runner timeout, full green-metric tunables, optional `per_model_overrides`.                                                                                                           |
| [`.env`](.env) / [`.env.example`](.env.example)    | API keys (gitignored) and the template.                                                                                                                                                                                                                                        |
| [`run.py`](run.py)                                 | End-to-end CLI script: compress a prompt, route both versions, print a savings table. Useful as a smoke test without the HTTP layer.                                                                                                                                           |
| [`index.html`](index.html)                         | Single-page tester with two screens and two quick-test fieldsets. Calls the four Flask endpoints. **Will be replaced.**                                                                                                                                                        |

### Endpoints (current)

| Endpoint            | Method | Body                                                              | Behavior                                                                                              |
| ------------------- | ------ | ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `/api/router`       | POST   | `{ "prompt": "..." }`                                             | Routes the prompt. If small model chosen, also runs the large model and returns a savings comparison. |
| `/api/raw-runner`   | POST   | `{ "model": "...", "prompt": "..." }`                             | Bare call to an OpenAI-compatible endpoint, no compression / routing / metrics.                       |
| `/api/compress`     | POST   | `{ "prompt": "...", "model": "...", "target_rate": 0.7 }`         | Compresses, runs pre + post, returns pre/post green metrics + savings.                                |
| `/api/green-metric` | POST   | `{ "model": "...", "input_prompt": "...", "output_text": "..." }` | Standalone energy/carbon calculator.                                                                  |

---

## Provider-agnostic by design

The project does **not** assume Ollama anywhere. Both the small and large models are reached via plain OpenAI-compatible HTTP (`/v1/chat/completions`), so any of the following will work:

- OpenAI (`https://api.openai.com/v1`)
- Azure OpenAI
- Together AI, Groq, Fireworks, OpenRouter
- Self-hosted **vLLM** or **llama.cpp server** (which both expose OpenAI-compatible APIs)
- LM Studio's local server
- Any other drop-in replacement

In `config.yaml` each model has its own `api_base` and `api_key_env` (the name of the environment variable to read from `.env`). See [Config system](#config-system) below.

> **Why we dropped Ollama from the contract:** Ollama's native API (`/api/generate`) is _not_ OpenAI-compatible, which means every new provider would require a different client. The OpenAI `/v1/chat/completions` shape is the de-facto standard and is what RouteLLM's controller speaks natively. This also makes the demo portable to free hosted providers for live classroom use.

> **Why the mock endpoint in `router.py` is going away:** RouteLLM's `Controller` needs a single OpenAI-compatible base URL for both strong and weak models. The current code hard-codes `api_base="http://localhost:8000/v1"` and points `OPENAI_API_BASE` at it. That mock is jank — it's a leftover from an earlier two-process setup. The clean replacement is to read both endpoints (small + large) from config and let RouteLLM call them directly.

---

## How the green-metric estimator works

A six-stage pipeline, every assumption labeled:

1. **Classify the model** from its tag — explicit param count (`8b → 8`) when present, else HHAI-style naming heuristic (`nano → Small`, `mini → Medium`, `tiny/micro → Micro`, else `Large`).
2. **Assign GPUs** by HHAI band — Nano 1, Micro 1, Small 2, Medium 4, Large 8.
3. **Reference throughput** — Nano/Micro/Small use a hardcoded external table (Llama 3.2 3B, Llama 3.1 8B, etc.). Medium/Large query the [Joule TPS database](https://github.com/microsoft/aienergy_simulation), preferring H100 rows closest in input/output length (log-distance), taking the median TPS of the top 5.
4. **Hardware power** — linear scale from a 10.2 kW / 8-GPU H100 anchor.
5. **Monte Carlo energy (10k samples)** — `energy_wh = PUE × Power × (output_tokens / TPS) × 1000/3600`, where Power and PUE are lognormal samples derived from P5/P95 bounds.
6. **Carbon** — `energy_wh × 0.3528 kg/kWh` (U.S. grid average, configurable).

The output includes `classification_method`, `reference_source`, and `reference_assumption` strings so the UI can show _why_ a number was chosen, not just the number.

---

## Current state

✅ **Working end-to-end:**

- All four Flask endpoints respond correctly.
- Compression tab: pre/post inference + green metrics in a single response.
- Router tab: routes via the **mf** router; when the small model is selected, the large model is also run for comparison and savings are computed.
- Both small and large models are reached over OpenAI-compatible HTTP. Any OpenAI-compatible provider works.
- LLMLingua-2 loads once at server start and is reused.
- The `BlackBoxLLMEstimator` returns a full P5–P95 distribution; `app.py` surfaces `P50` as the headline metric.

⚠️ **Known limitations / things to be aware of:**

- **Estimates are scenario-based**, not measured. The docstring is explicit about this. Assumptions are labeled in the output but the numbers are illustrative.
- **Water is on-site cooling only.** The WUE × energy model captures facility cooling water; it does NOT include thermoelectric or embodied water. Acceptable for a teaching demo; a more rigorous model is a future extension.
- **RouteLLM is a local sibling repo** at `C:\Users\art vandelay\Desktop\ecoGPT\midterm_demo\RouteLLM` and is appended to `sys.path` at import time. If the folder is missing or the path is wrong, `router.py` will fail.
- **If strong + weak models live on different bases**, RouteLLM's controller only sees one `OPENAI_API_BASE`. `app.py` uses a separate `RawModelRunner` call for the large-model comparison to work around this; verify your setup if you split providers.
- **Frontend is a placeholder** — `index.html` works as a tester but doesn't match the spec's two-tab layout. Deferred.

---

## Config system

Config is split between two files:

- **`config.yaml`** — non-secret settings. Model names, endpoints, router type/threshold, LLMLingua device and rate, estimator tunables, optional per-model overrides.
- **`.env`** — API keys only. Gitignored. Modeled after the project's `api_key_env` field in `config.yaml`.

`.env` is loaded **inside `config.py`** at import time, so every module that imports `config` gets the same env-loaded state regardless of import order.

### Planned `config.yaml` shape

```yaml
project:
  name: ecoGPT
  port: 5000
  host: 0.0.0.0

models:
  small:
    name: qwen2.5:3b # model identifier the provider expects
    api_base: http://localhost:11434/v1
    api_key_env: SMALL_MODEL_API_KEY
  large:
    name: gpt-oss:120b-cloud
    api_base: http://localhost:11434/v1
    api_key_env: LARGE_MODEL_API_KEY

router:
  type: mf
  threshold: 0.1159
  # Embedding model for the MF router. "sentence-transformers" runs locally
  # (no API key needed); "openai" uses the OPENAI_API_KEY env var.
  embedding:
    provider: sentence-transformers
    model_name: sentence-transformers/all-MiniLM-L6-v2
    device: auto # auto | cpu | cuda
  # Per-model overrides: if the routed model matches a key, use these values
  # instead of falling through to HHAI bands.
  per_model_overrides: {}

compression:
  model_name: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
  device: auto # auto | cpu | cuda
  target_rate: 0.7

raw_runner:
  timeout_seconds: 120
  # Optional max_tokens, temperature, etc. to pass to providers
  request_defaults: {}

green_metric:
  carbon_intensity_kg_per_kwh: 0.3528
  monte_carlo_samples: 10000
  random_seed: 42
  h100_8gpu_max_power_kw: 10.2
  power_p5_fraction: 0.40
  power_p95_fraction: 0.90
  pue_p5: 1.05
  pue_p95: 1.40
  # HHAI bands stay as the fallback classifier for unknown model tags
  class_gpu_count:
    Nano: 1
    Micro: 1
    Small: 2
    Medium: 4
    Large: 8
```

### Planned `.env.example`

```ini
# Replace with real keys for non-local providers.
# Both models can share one key if they live on the same provider.
SMALL_MODEL_API_KEY=sk-replace-me
LARGE_MODEL_API_KEY=sk-replace-me
```

### `compression.device` semantics

- `auto` — try `cuda` if `torch.cuda.is_available()`, else fall back to `cpu`. Logged at startup.
- `cpu` — force CPU regardless of GPU availability.
- `cuda` — force CUDA; fail loudly at startup if it's not available.

The actual `torch.device(...)` is built once inside `PromptCompressorClient.__init__` and reused for the BERT model.

---

## Next steps

### Phase 1 — Configuration system ✅ done

Create the config-driven setup so model names, endpoints, thresholds, device, and estimator tunables aren't scattered across files.

| New file           | Purpose                                                                                                                                                |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `config.py`        | Singleton loader. Reads `config.yaml` + `.env` once, exposes a typed `settings` object to all modules. Loads `.env` at import.                         |
| `config.yaml`      | Non-secret config (shape shown above).                                                                                                                 |
| `.env`             | API keys. Gitignored.                                                                                                                                  |
| `.env.example`     | Template documenting expected env vars.                                                                                                                |
| `.gitignore`       | Ignore `.env`, `__pycache__`, etc.                                                                                                                     |
| `requirements.txt` | Add `pyyaml`, `python-dotenv`; pin existing deps (`flask`, `flask-cors`, `pandas`, `numpy`, `requests`, `tiktoken`, `llmlingua`, `torch`, `routellm`). |

Quick smoke test:

```powershell
python config.py
# Project:        ecoGPT on 0.0.0.0:5000
# Small model:    qwen2.5:3b @ http://localhost:11434/v1
#   api_key_env:  SMALL_MODEL_API_KEY -> set
# Large model:    gpt-oss:120b-cloud @ http://localhost:11434/v1
#   api_key_env:  LARGE_MODEL_API_KEY -> set
# Router:         mf @ threshold=0.1159
# Compression:    microsoft/llmlingua-2-... on device=auto rate=0.7
# ...
```

### Phase 2 — Refactor existing modules to use config ✅ done

| File                    | Change                                                                                                                                                                                                                                       |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`                | Read model names from `settings.models`. Initialize `LLMRouterClient` and `RawModelRunner` from config.                                                                                                                                      |
| `router.py`             | Read `strong_model`, `weak_model`, `router_type`, `threshold` from config. **Drop the mock OpenAI base URL** — RouteLLM's controller now talks directly to each model's real endpoint. API key resolved via `api_key_env` lookup at runtime. |
| `prompt_compression.py` | Read `model_name`, `device`, `target_rate` from config. Resolve `device` through `torch.device(...)` honoring `auto` / `cpu` / `cuda`. Old `prompt_compresstion.py` filename kept as a back-compat shim.                                     |
| `green_metric.py`       | Read all tunables from config. Honor `per_model_overrides` for known model tags. **Keep HHAI bands as fallback** for unknown tags.                                                                                                           |
| `raw_model_runner.py`   | Read `api_base` per model from config. Switch from Ollama's `/api/generate` to OpenAI-compatible `/v1/chat/completions`. Read `timeout_seconds` and `request_defaults` from config.                                                          |

### Phase 3 — Add water metric ✅ done

`green_metric.py` now returns `energy_wh`, `carbon_gco2e`, and `water_liters` — all with the same P5–P95 percentile distribution.

- **Source:** on-site cooling water via WUE (Water Usage Effectiveness), e.g. `1.2 L/kWh` U.S. average. Configurable via `green_metric.wue_l_per_kwh` in `config.yaml`.
- **Calculation:** `water_L = energy_Wh × (WUE_L_per_kWh / 1000)`.
- **Wiring:** `water_liters` (and `wue_l_per_kwh`) are surfaced in `/api/router`, `/api/compress`, and `/api/green-metric` payloads, with `water_saved_pct` added to the savings deltas.

### Phase 4 — Cleanup pass ✅ done

- `app.py` — dropped `pd.options.mode.chained_assignment` and the `FutureWarning` filter.
- `router.py` — cleaned import order; `bert` is still accepted as input but no longer advertised in the default.
- `green_metric.py` — every fallback path returns a value; error messages tightened.
- `prompt_compresstion.py` (typo) — implementation moved to correctly-spelled `prompt_compression.py`; old file kept as a `DeprecationWarning`-emitting shim so older import paths keep working.
- All modules — consistent import order, dead code removed, type hints on public functions.

### Phase 5 — Frontend (deferred)

User will provide a two-tab UI. Backend contracts above are the source of truth; once the new UI is pasted in, wire it to the existing endpoints (no contract changes expected).

---

## How to run

```powershell
# 1. Install deps (includes sentence-transformers for the MF router)
pip install -r requirements.txt

# 2. Copy the env template and edit
cp .env.example .env

# 3. Edit config.yaml to point at your provider endpoints
#    (the MF router uses local MiniLM embeddings — no OpenAI key needed)

# 4. Start the Flask server
python app.py

# 5. (Optional) run the CLI smoke test
python run.py
```

Endpoints live on `http://localhost:5000`. The placeholder UI is at `http://localhost:5000/` (Flask static — currently `index.html`).

> The previous "make sure Ollama is running" steps are **no longer required**. Any OpenAI-compatible endpoint will work — local (vLLM, llama.cpp, LM Studio) or hosted (OpenAI, Together, Groq, etc.).

---

## Open questions / gotchas for a fresh agent

- **RouteLLM is a local repo**, not a pip package. It must sit next to `app.py` and is imported via `sys.path` mutation in `router.py`. Do not try to `pip install routellm`.
- **The estimator is a college-project estimate**, not a measurement. Don't add claims of "real" energy use anywhere in the UI; the demo's value is in the _comparison_ between configurations, not the absolute numbers.
- **HHAI bands stay as a fallback.** If a model tag isn't in the config's `per_model_overrides`, fall through to the existing classification code in `green_metric.py`. Don't try to remove the HHAI logic entirely.
- **Provider-agnostic by contract.** Do not import or reference `ollama` anywhere in the codebase. All HTTP goes to OpenAI-compatible `/v1/chat/completions`. If a provider doesn't support that shape, write an adapter — don't change the contract.
- **`prompt_compresstion.py` is a back-compat shim.** The implementation lives in `prompt_compression.py`. Old import paths still work but emit a `DeprecationWarning` — update them when convenient.
- **`question_to_ask.md` is empty** and currently unused at runtime.
- **The `mf` router uses local MiniLM embeddings** (via `sentence-transformers`), not OpenAI's embedding API. No real OpenAI key is needed for routing. The embedding model is configurable in `config.yaml` under `router.embedding`.
- **Device resolution for LLMLingua** happens once at startup. `auto` will silently fall back to CPU if CUDA isn't available — log it so the user knows.

---

## Quick file map (for fresh sessions)

| If you want to…                          | Look in…                                                  |
| ---------------------------------------- | --------------------------------------------------------- |
| Change a model name                      | `config.yaml` → `models.small.name` / `models.large.name` |
| Add a new endpoint                       | `app.py`                                                  |
| Tweak compression rate                   | `config.yaml` → `compression.target_rate`                 |
| Set LLMLingua device (cpu / cuda / auto) | `config.yaml` → `compression.device`                      |
| Change router threshold                  | `config.yaml` → `router.threshold`                        |
| Change MF embedding model                | `config.yaml` → `router.embedding.model_name`             |
| Adjust estimator assumptions             | `config.yaml` → `green_metric.*`                          |
| Modify the UI                            | `index.html` (placeholder) → new frontend (deferred)      |
| Run the pipeline without the server      | `run.py`                                                  |
