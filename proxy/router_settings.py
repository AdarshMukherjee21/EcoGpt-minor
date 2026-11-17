"""Dashboard schema + validation for the System One tier router.

Same declarative pattern as ``features.py`` / ``headroom_settings.py``: a schema
list rendered by the dashboard, persisted under the ``router`` key in
config.json, validated by :func:`normalize` (unknown keys raise, missing keys
take defaults) and read from disk at request time via :func:`load_router` —
config edits take effect without a proxy restart, and any read/validation
failure fails open to defaults.

Backend model (owner decision: fully generic SystemOne)
    Any SystemOne-compatible decision endpoint reachable with a plain POST of
    the Jev envelope ``{"model", "state", "questions": {...}}``. Nothing is
    hardcoded to Ollama or OpenRouter:

    - ``endpoint_url`` is the full POST target (``http://127.0.0.1:11434/v1/systemone``,
      ``https://openrouter.ai/api/alpha/decisions``, a self-hosted vLLM
      SystemOne server, anything). Validation only requires a well-formed URL.
    - ``auth_scheme`` selects the credential header: ``none`` (no header),
      ``bearer`` (``Authorization: Bearer <key>``) or ``x-api-key``
      (``x-api-key: <key>``). The key lives server-side in config.json only and
      never travels in request bodies or dashboard payloads (masked as
      ``api_key_set``).
    - ``ollama`` / ``openrouter`` ship as *presets* (convenience fills in the
      URL + scheme), never as hardcoded backends.

Tiers are a user-configurable list of ``{"id", "description", "model"}``
ordered from easiest/cheapest to most capable. ``description`` doubles as the
classifier ``criteria`` text for that tier. System One choice questions accept
2–26 options (Ollama hard-rejects more), so tier count is validated to
[2, 26].

A tier ladder is *valid* (auto model becomes available, one-click flows
include it) only when enabled, and every tier has non-empty id/model/description.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

# Advertised context window of auto models in /v1/models (owner constant).
# Contexts beyond this are supported but experimental and gated behind
# ``allow_oversized_context`` (off by default).
ADVERTISED_CONTEXT_TOKENS = 256_000

AUTO_MODEL_ID = "eco-auto"

DEFAULT_ROUTING_QUESTION = (
    "Which difficulty tier best matches the computational, coding, or "
    "reasoning complexity required to answer this request?"
)

SCHEMA: list[dict[str, Any]] = [
    # --- Core ----------------------------------------------------------------
    {"key": "enabled", "group": "Router", "kind": "router", "type": "bool", "default": True,
     "label": "Enable router",
     "help": "Routes requests made to the auto model through the System One classifier "
             "and dispatches them to the selected tier's model. Off resolves the auto "
             "model statically to the top (most capable) tier."},
    {"key": "auto_model_id", "group": "Router", "kind": "router", "type": "str", "default": AUTO_MODEL_ID,
     "label": "Auto model id",
     "help": "Client-facing model id for routed traffic (lowercase letters, digits, dashes). "
             "Shown in /v1/models and the coding-agent pickers once every tier has a model."},
    # --- SystemOne endpoint (generic) -----------------------------------------
    {"key": "endpoint_url", "group": "Classifier endpoint", "kind": "router", "type": "str",
     "default": "http://127.0.0.1:11434/v1/systemone",
     "label": "Endpoint URL",
     "help": "Full POST target of any SystemOne (Jev-API) decision endpoint. Ollama >= 0.35 "
             "serves /v1/systemone locally; OpenRouter serves /api/alpha/decisions."},
    {"key": "auth_scheme", "group": "Classifier endpoint", "kind": "router", "type": "choice",
     "choices": ["none", "bearer", "x-api-key"], "default": "none",
     "label": "Auth scheme",
     "help": "Credential header for the decision endpoint: none (local Ollama), bearer "
             "(Authorization: Bearer) or x-api-key. The key is stored server-side and never leaves this machine via the dashboard."},
    {"key": "api_key", "group": "Classifier endpoint", "kind": "router", "type": "str", "default": "",
     "label": "Classifier API key",
     "help": "Sent per the auth scheme when set. Stored in config.json only; the dashboard "
             "shows whether a key is set, never the value."},
    {"key": "router_model", "group": "Classifier endpoint", "kind": "router", "type": "str",
     "default": "tev1:4b",
     "label": "Classifier model",
     "help": "Decision model id served by the endpoint (e.g. tev1:4b or nimble on Ollama; "
             "on OpenRouter pin a dated snapshot such as typesafe/jev-1.13 — model ids churn)."},
    {"key": "classify_timeout_ms", "group": "Classifier endpoint", "kind": "router", "type": "int",
     "default": 8000, "min": 500, "max": 120000,
     "label": "Classification timeout (ms)",
     "help": "How long one classification call may take. The first call to a freshly "
             "pulled local model can be slow, so one timeout is retried once."},
    # --- Classification tuning ------------------------------------------------
    {"key": "routing_question", "group": "Classification", "kind": "router", "type": "str",
     "default": DEFAULT_ROUTING_QUESTION,
     "label": "Routing question",
     "help": "The decision instruction sent to the classifier. Tier descriptions become "
             "the answer options (criteria)."},
    {"key": "confidence_floor", "group": "Classification", "kind": "router", "type": "float",
     "default": 0.45, "min": 0.0, "max": 1.0,
     "label": "Confidence floor",
     "help": "Below this classifier confidence the request escalates one tier as a safety "
             "margin (never past the top tier)."},
    {"key": "long_context_token_threshold", "group": "Classification", "kind": "router", "type": "int",
     "default": ADVERTISED_CONTEXT_TOKENS, "min": 1000,
     "label": "Long-context threshold (tokens)",
     "help": "Prompts estimated above this token count are promoted to at least the tier "
             "configured as the long-context floor."},
    {"key": "min_tier_for_long_context", "group": "Classification", "kind": "router", "type": "int",
     "default": 1, "min": 0,
     "label": "Long-context floor tier",
     "help": "Zero-based tier index (list position) enforced for over-threshold prompts."},
    {"key": "state_token_budget", "group": "Classification", "kind": "router", "type": "int",
     "default": 1900, "min": 200,
     "label": "Classifier state budget (tokens)",
     "help": "Small local decision models (tev1 family) accept roughly 2K input tokens; "
             "the composed classification text is truncated to this budget. Cloud-type "
             "endpoints with large windows can raise it."},
    {"key": "allow_oversized_context", "group": "Classification", "kind": "router", "type": "bool",
     "default": False,
     "label": "Allow oversized contexts (experimental)",
     "help": "Off: requests over the advertised 256k tokens to the auto model are rejected "
             "with a clear error. On: they are classified and forwarded regardless."},
    # --- Warmth and cache stickiness (on by default per owner) ------------------
    {"key": "warmth_enabled", "group": "Warmth and stickiness", "kind": "router", "type": "bool",
     "default": True,
     "label": "Warmth and stickiness",
     "help": "Holds timing-based tier decisions steady so provider prompt caches (KV cache) "
             "stay warm across consecutive turns."},
    {"key": "warmth_window_seconds", "group": "Warmth and stickiness", "kind": "router", "type": "float",
     "default": 120.0, "min": 0.0,
     "label": "Warmth window (s)",
     "help": "Within this window after the previous turn the router never routes below the "
             "tier that turn used (no downgrade while the model is warm)."},
    {"key": "stickiness_window_seconds", "group": "Warmth and stickiness", "kind": "router", "type": "float",
     "default": 300.0, "min": 0.0,
     "label": "Stickiness window (s)",
     "help": "Within this window the router suppresses borderline tier flips (confidence "
             "margin below) so consecutive requests keep reusing the provider cache."},
    {"key": "stickiness_confidence_margin", "group": "Warmth and stickiness", "kind": "router", "type": "float",
     "default": 0.10, "min": 0.0, "max": 0.5,
     "label": "Stickiness margin",
     "help": "A flip away from the previous tier is suppressed when confidence is below "
             "0.5 + this margin (the classifier was not clearly sure)."},
    # --- Failure posture --------------------------------------------------------
    {"key": "on_classify_failure", "group": "Failure", "kind": "router", "type": "choice",
     "choices": ["planner", "goldfish", "passthrough"], "default": "planner",
     "label": "When classification fails",
     "help": "planner: fall back to the top tier (safest for coding agents). goldfish: fall "
             "back to the cheapest tier. passthrough: fall back to the tier's top model "
             "unrouted — same as planner for the default ladder."},
]

_BY_KEY = {entry["key"]: entry for entry in SCHEMA}

# Convenience presets for the endpoint trio (URL + scheme + suggested model).
# These are dashboard fill-in helpers only; any well-formed URL works.
PRESETS: dict[str, dict[str, str]] = {
    "ollama": {
        "endpoint_url": "http://127.0.0.1:11434/v1/systemone",
        "auth_scheme": "none",
        "router_model": "tev1:4b",
    },
    "openrouter": {
        "endpoint_url": "https://openrouter.ai/api/alpha/decisions",
        "auth_scheme": "bearer",
        "router_model": "typesafe/jev-1.13",
    },
}

AUTO_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
TIER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MIN_TIERS = 2   # System One choice questions take 2-26 options
MAX_TIERS = 26


def default_tiers() -> list[dict[str, str]]:
    """Shipped two-tier coding-agent ladder (owner-approved descriptions).

    Tier models ship empty: the Router tab requires the user to pick one model
    per tier before auto models become available ("eco-auto only after setup").
    """
    return [
        {"id": "goldfish",
         "description": (
             "Routine, well-specified execution work: single-file edits, boilerplate, "
             "mechanical refactors, formatting, running prescribed steps, short lookups, "
             "straightforward Q&A. Clear, local intent; no design decisions and no "
             "cross-file reasoning required."),
         "model": ""},
        {"id": "elephant",
         "description": (
             "Work needing judgment or wide context: planning and task decomposition, "
             "architecture decisions, ambiguous or underspecified requirements, complex "
             "debugging across files, long-context synthesis, novel algorithm design, "
             "anything with cascading decisions where a wrong choice is expensive."),
         "model": ""},
    ]


def defaults() -> dict[str, Any]:
    values = {entry["key"]: entry["default"] for entry in SCHEMA}
    return values


def _coerce(entry: dict[str, Any], raw: Any) -> Any:
    key = entry["key"]
    kind = entry["type"]
    if raw is None or (isinstance(raw, str) and raw.strip() == ""):
        # Cleared/empty values snap back to the declared default (a blank
        # routing_question or endpoint URL must never reach the classifier).
        return entry["default"]
    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str) and raw.strip().lower() in ("1", "true", "yes", "on"):
            return True
        if isinstance(raw, str) and raw.strip().lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key}: expected true or false")
    if kind == "int":
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: expected a whole number") from exc
    elif kind == "float":
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: expected a number") from exc
    elif kind == "choice":
        value = str(raw)
        if value not in entry["choices"]:
            raise ValueError(f"{key}: must be one of {', '.join(entry['choices'])}")
        return value
    else:
        return str(raw).strip()
    if "min" in entry and value < entry["min"]:
        raise ValueError(f"{key}: must be at least {entry['min']}")
    if "max" in entry and value > entry["max"]:
        raise ValueError(f"{key}: must be at most {entry['max']}")
    return value


def normalize(values: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce scalar router settings; unknown keys raise, missing keys take defaults."""
    unknown = sorted(set(values) - set(_BY_KEY))
    if unknown:
        raise ValueError(f"unknown router setting: {unknown[0]}")
    out: dict[str, Any] = {}
    for entry in SCHEMA:
        raw = values.get(entry["key"], entry["default"])
        if raw is None:
            out[entry["key"]] = entry["default"]
            continue
        out[entry["key"]] = _coerce(entry, raw)
    return out


def normalize_tiers(raw: Any) -> list[dict[str, str]]:
    """Validate a tier list; returns clean tier dicts or raises ValueError."""
    if not isinstance(raw, list) or not raw:
        raise ValueError("tiers: expected a non-empty list")
    tiers: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for index, tier in enumerate(raw):
        if not isinstance(tier, dict):
            raise ValueError(f"tiers[{index}]: expected an object")
        tier_id = str(tier.get("id") or "").strip()
        description = str(tier.get("description") or "").strip()
        model = str(tier.get("model") or "").strip()
        if not TIER_ID_RE.match(tier_id):
            raise ValueError(f"tiers[{index}].id: must be 1-64 characters of letters, digits, dots, dashes or underscores")
        if tier_id.lower() in seen_ids:
            raise ValueError(f"tiers[{index}].id: '{tier_id}' is used twice")
        seen_ids.add(tier_id.lower())
        if not description:
            raise ValueError(f"tiers[{index}] ('{tier_id}'): description is required — it is the classifier's criteria text")
        if len(model) > 0 and len(model) > 200:
            raise ValueError(f"tiers[{index}] ('{tier_id}'): model id too long")
        tiers.append({"id": tier_id, "description": description, "model": model})
    if len(tiers) < MIN_TIERS:
        raise ValueError(f"tiers: at least {MIN_TIERS} tiers are required (System One choice questions take {MIN_TIERS}-{MAX_TIERS} options)")
    if len(tiers) > MAX_TIERS:
        raise ValueError(f"tiers: at most {MAX_TIERS} tiers are supported (System One choice-question limit)")
    return tiers


def validate_tiers(tiers: list[dict[str, str]]) -> list[str]:
    """Problems that keep the ladder from being *active* (models not yet assigned).

    Unlike :func:`normalize_tiers` (which raises on malformed input), this
    reports incomplete-but-well-formed ladders so the dashboard can demand
    setup without rejecting saves in progress.
    """
    problems: list[str] = []
    for index, tier in enumerate(tiers):
        if not str(tier.get("model") or "").strip():
            problems.append(f"tier '{tier.get('id') or index + 1}' has no model assigned")
    return problems


def is_auto_model(values: dict[str, Any], model: str) -> bool:
    """True when `model` is the configured auto model id."""
    return bool(model) and model == str(values.get("auto_model_id") or AUTO_MODEL_ID)


def block_is_valid(values: dict[str, Any], tiers: list[dict[str, str]]) -> list[str]:
    """Empty problem list == the router can serve the auto model."""
    problems: list[str] = []
    auto_id = str(values.get("auto_model_id") or AUTO_MODEL_ID)
    if not AUTO_ID_RE.match(auto_id):
        problems.append(f"auto model id '{auto_id}' is invalid (lowercase letters, digits, dashes)")
    endpoint = str(values.get("endpoint_url") or "").strip()
    parts = urlsplit(endpoint)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        problems.append("endpoint URL must be a well-formed http(s) URL")
    scheme = str(values.get("auth_scheme") or "none")
    if scheme != "none" and not str(values.get("api_key") or "").strip():
        problems.append(f"auth scheme '{scheme}' selected but no API key is set")
    floor = int(values.get("min_tier_for_long_context") or 0)
    if not (0 <= floor < len(tiers)):
        problems.append("long-context floor tier is outside the tier list")
    problems.extend(validate_tiers(tiers))
    return problems


def is_local_endpoint(endpoint_url: str) -> bool:
    """True when the decision endpoint is loopback: no prompt text leaves the machine."""
    parts = urlsplit((endpoint_url or "").strip())
    return parts.hostname in ("127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1")


def load_router(config_path: Path) -> dict[str, Any]:
    """Read the ``router`` block from config.json — the request-time source of truth.

    Called per request, so Router-tab edits take effect without a restart.
    A missing block, corrupt file or invalid settings fails open to defaults.
    """
    try:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return defaults()
    stored = raw.get("router") if isinstance(raw, dict) else None
    if not isinstance(stored, dict):
        return defaults()
    known = {key: value for key, value in stored.items() if key in _BY_KEY}
    try:
        return normalize(known)
    except ValueError:
        return defaults()


def load_tiers(config_path: Path) -> list[dict[str, str]]:
    """Tier list from config.json; falls back to the shipped default ladder."""
    try:
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default_tiers()
    stored = raw.get("router") if isinstance(raw, dict) else None
    stored = stored.get("tiers") if isinstance(stored, dict) else None
    if not isinstance(stored, list) or not stored:
        return default_tiers()
    try:
        return normalize_tiers(stored)
    except ValueError:
        return default_tiers()
