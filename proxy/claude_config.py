"""Points Claude Code at the proxy by editing ~/.claude/settings.json, with a restorable backup.

The env recipe follows the OpenRouter and DeepSeek "Claude Code integration"
cookbooks (https://openrouter.ai/docs/cookbook/coding-agents/claude-code-integration.md,
https://api-docs.deepseek.com/quick_start/agent_integrations/claude_code.md) for
gateways that authenticate with a bearer token, applied on top of this proxy:

- ANTHROPIC_BASE_URL points at the proxy (the proxy sits between Claude Code
  and the gateway, so it is never the raw gateway URL).
- ANTHROPIC_AUTH_TOKEN carries the gateway API key; Claude Code sends it as the
  "Authorization: Bearer" header (the credential variable Anthropic documents
  for bearer-authenticated gateways).
- ANTHROPIC_API_KEY is explicitly blanked to "" so Claude Code never falls back
  to direct-Anthropic x-api-key authentication or a cached login.
- CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1 turns on the gateway model
  picker, which lists every model served by GET /v1/models.
- Role model aliases (ANTHROPIC_MODEL, ANTHROPIC_DEFAULT_OPUS/SONNET/HAIKU/FABLE
  MODEL, ANTHROPIC_SMALL_FAST_MODEL, CLAUDE_CODE_SUBAGENT_MODEL) pin each model
  class to an exposed model so the picker stays usable even when discovery has
  no Anthropic-family names to map roles onto.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

# Env vars Claude Code reads to resolve each model class: class key -> env var.
# Opus/Fable/Sonnet get the first exposed model (general coding); Haiku/small-fast
# and subagents fall back to a second exposed model when one is exposed, else the
# same model. All of them can be overridden per class from the dashboard.
ROLE_ENV = {
    "default": "ANTHROPIC_MODEL",
    "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "fable": "ANTHROPIC_DEFAULT_FABLE_MODEL",
    "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "small_fast": "ANTHROPIC_SMALL_FAST_MODEL",
    "subagent": "CLAUDE_CODE_SUBAGENT_MODEL",
}
PRIMARY_CLASSES = ("default", "opus", "sonnet", "fable")
SECONDARY_CLASSES = ("haiku", "small_fast", "subagent")


def role_model_env(exposed_models: list[str], model_classes: dict[str, str] | None = None) -> dict[str, str]:
    """Map Claude Code's model-class env vars onto exposed (or user-chosen) models."""
    if not exposed_models:
        return {}
    primary = exposed_models[0]
    secondary = exposed_models[1] if len(exposed_models) > 1 else primary
    chosen = model_classes or {}
    values = {}
    for class_key, env_key in ROLE_ENV.items():
        override = str(chosen.get(class_key) or "").strip()
        values[env_key] = override if override and override != "auto" else (
            primary if class_key in PRIMARY_CLASSES else secondary
        )
    return values


def backup_path(settings_path: Path) -> Path:
    return settings_path.with_name(settings_path.stem + ".ecogpt-backup.json")


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return handle.read()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _load_object(text: str | None) -> dict[str, Any]:
    if not text or not text.strip():
        return {}
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Claude Code settings must be a JSON object")
    return data


def status(settings_path: Path) -> dict[str, Any]:
    base_url = None
    auth_token_set = False
    api_key_blank = False
    gateway_discovery = False
    error = None
    try:
        data = _load_object(_read_text(settings_path))
        env = data.get("env") or {}
        if isinstance(env, dict):
            base_url = env.get("ANTHROPIC_BASE_URL")
            auth_token_set = bool(env.get("ANTHROPIC_AUTH_TOKEN"))
            api_key_blank = env.get("ANTHROPIC_API_KEY") == ""
            gateway_discovery = env.get("CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY") == "1"
            model_classes = {
                class_key: str(env.get(env_key) or "")
                for class_key, env_key in ROLE_ENV.items()
                if env.get(env_key)
            }
    except (ValueError, OSError) as exc:
        error = str(exc)
    return {
        "settings_path": str(settings_path),
        "base_url": base_url,
        "auth_token_set": auth_token_set,
        "api_key_blank": api_key_blank,
        "gateway_discovery_enabled": gateway_discovery,
        "model_classes": model_classes if isinstance(model_classes, dict) else {},
        "backup_exists": backup_path(settings_path).exists(),
        "error": error,
    }


def apply(
    settings_path: Path,
    proxy_url: str,
    exposed_models: list[str],
    gateway_key: str | None = None,
    model_classes: dict[str, str] | None = None,
) -> dict[str, Any]:
    backup = backup_path(settings_path)
    if not backup.exists():
        original = _read_text(settings_path)
        backup.write_text(
            json.dumps({"existed": original is not None, "content": original}),
            encoding="utf-8",
        )
    data = _load_object(_read_text(settings_path))
    env = data.get("env")
    if not isinstance(env, dict):
        env = {}
        data["env"] = env
    env["ANTHROPIC_BASE_URL"] = proxy_url
    # Bearer-auth gateway per the integration cookbooks: the token travels as
    # "Authorization: Bearer" and the direct-Anthropic credential is blanked so
    # Claude Code neither prompts for it nor authenticates against Anthropic.
    env["ANTHROPIC_API_KEY"] = ""
    if gateway_key:
        env["ANTHROPIC_AUTH_TOKEN"] = gateway_key
    env["CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"] = "1"
    env.update(role_model_env(list(exposed_models), model_classes))
    if exposed_models:
        data["availableModels"] = list(exposed_models)
    _write_text(settings_path, json.dumps(data, indent=2) + "\n")
    return status(settings_path)


def restore(settings_path: Path) -> dict[str, Any]:
    backup = backup_path(settings_path)
    if not backup.exists():
        raise ValueError("No backup found, so there is nothing to restore")
    saved = json.loads(backup.read_text(encoding="utf-8"))
    if saved.get("existed"):
        _write_text(settings_path, saved.get("content") or "")
    elif settings_path.exists():
        settings_path.unlink()
    backup.unlink()
    return status(settings_path)
