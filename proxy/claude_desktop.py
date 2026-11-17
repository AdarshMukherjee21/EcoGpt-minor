"""Points Claude Desktop at the proxy by writing an inference-gateway profile
into Claude Desktop's 3p config library, with restorable backups.

Claude Desktop (MSIX build, userData under %LOCALAPPDATA%/Claude-3p) selects a
"gateway" profile via configLibrary/_meta.json -> appliedId; the profile JSON
carries inferenceGatewayBaseUrl / inferenceGatewayApiKey. The app re-seals any
plaintext key it finds on next launch (user DPAPI), so the script may write the
key as plaintext and let the app encrypt it itself.

Applying requires quitting Claude Desktop first (it rewrites these files on
exit); restoring swaps the byte-for-byte backups taken on first apply.

Auth findings from the app bundle (app.asar, build 5.11492.5, string
inspection only): the gateway auth scheme is read from
"inferenceGatewayAuthScheme" (enum ["bearer", "x-api-key"], default "bearer");
with the "static" credential kind (inferenceGatewayApiKey) and the "bearer"
scheme the app sends "Authorization: Bearer <key>" to the gateway (an
"x-api-key" scheme would send "x-api-key: <key>" instead) and exposes the key
to embedded Claude Code as ANTHROPIC_AUTH_TOKEN with ANTHROPIC_API_KEY blank.
We pin the scheme explicitly and seed "inferenceModels" with the proxy's
exposed model IDs so the model picker is populated without depending on live
model discovery. Verifying end-to-end against a running Claude Desktop is not
possible in this environment, so the live behavior remains unverified.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .config import DATA_DIR
from .model_names import slugify

# Fixed identity so repeated applies reuse the same entry.
ECOGPT_ENTRY_ID = "ec0cde01-5eed-4a11-ad1e-2a9f40e5d001"

DEFAULT_APP_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local"))) / "Claude-3p"
DEFAULT_BACKUP_DIR = DATA_DIR / "claude-desktop-backup"

PROFILE_FILES = ("developer_settings.json", "claude_desktop_config.json")


def _config_library(app_dir: Path) -> Path:
    return app_dir / "configLibrary"


def _meta_path(app_dir: Path) -> Path:
    return _config_library(app_dir) / "_meta.json"


def _entry_path(app_dir: Path, entry_id: str) -> Path:
    return _config_library(app_dir) / f"{entry_id}.json"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


def _entry_ids(meta: dict[str, Any] | None) -> list[str]:
    if not meta:
        return []
    return [
        str(entry["id"])
        for entry in meta.get("entries") or []
        if isinstance(entry, dict) and entry.get("id")
    ]


def _backup_record(app_dir: Path) -> dict[str, str]:
    """Existing files (relative to app_dir) to preserve on first apply -> backup names."""
    record: dict[str, str] = {}
    for name in PROFILE_FILES:
        if (app_dir / name).exists():
            record[name] = name
    meta = _meta_path(app_dir)
    if meta.exists():
        record["configLibrary/_meta.json"] = "configLibrary__meta.json"
        for entry_id in _entry_ids(_read_json(meta)):
            if _entry_path(app_dir, entry_id).exists():
                record[f"configLibrary/{entry_id}.json"] = f"configLibrary_{entry_id}.json"
    return record


def _take_backups(app_dir: Path, backup_dir: Path) -> None:
    if backup_dir.exists():
        return
    paths = _backup_record(app_dir)
    if not paths:
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    (backup_dir / "index.json").write_text(json.dumps(paths, indent=2), encoding="utf-8")
    for source_rel, backup_name in paths.items():
        shutil.copy2(app_dir / source_rel, backup_dir / backup_name)


def _write_developer_settings(app_dir: Path) -> None:
    path = app_dir / "developer_settings.json"
    data = _read_json(path) or {}
    data["allowDevTools"] = True
    _write_json(path, data)


def desktop_alias_name(display_name: str) -> str:
    """Validation-safe alias id for one exposed model: a slug of its display name.

    Lowercase, only [a-z0-9-]; spaces/symbols/underscores become single dashes;
    leading/trailing dashes are trimmed ("GPT 6 Luna" -> "gpt-6-luna"). This is
    what Claude Desktop sends VERBATIM as the gateway model id, so the proxy
    maps it 1:1 to the real upstream model via model_overrides. Callers keep the
    alias unique (see desktop_alias_map).
    """
    return slugify(display_name)


# Old scheme (claude-eco-1/2/3): kept ONLY to recognize and migrate entries
# written by earlier versions of the generator.
LEGACY_ALIAS_RE = re.compile(r"^claude-eco-\d+$")

_ALIAS_RE = LEGACY_ALIAS_RE  # backwards-compatible name


def is_legacy_alias(name: str) -> bool:
    return bool(LEGACY_ALIAS_RE.match(name or ""))


def is_alias(name: str, alias_map: dict[str, str] | None = None) -> bool:
    """True when `name` is an alias id this proxy created (never a real gateway id).

    Two alias generations are recognized: legacy ``claude-eco-N`` ids (pattern),
    and current display-name slugs, which are only recognizable through the
    persisted alias registry (`alias_map` as stored in Settings.desktop_aliases).
    """
    if is_legacy_alias(name or ""):
        return True
    return bool(alias_map) and name in alias_map


_FALLBACK_ALIASES = ("claude", "sonnet", "opus", "haiku", "fable", "mythos", "anthropic")


def desktop_alias_map(
    real_models: list[str],
    display_names: list[str] | None,
    previous: dict[str, str] | None = None,
) -> dict[str, str]:
    """Unique alias -> real-model mapping for the Desktop model picker.

    The alias for a model is the slug of its display name (raw id slugged when
    no display name resolves). Already-known aliases are reused so repeated
    applies keep the same ids (idempotent). Collisions get -2, -3 suffixes and
    never shadow a real model id. A slug that coincides with the real model id
    itself needs no alias. If slugging produces nothing usable, a positional
    fallback name is used.
    """
    real_set = set(real_models)
    used: set[str] = set(real_models)
    previous = {alias: real for alias, real in dict(previous or {}).items()
                if not is_legacy_alias(alias) and real in real_set and alias not in real_set}
    # alias -> real stored; look reuse up by real model id
    reverse_previous = {real: alias for alias, real in previous.items()}

    def _pick(base: str, allow_self: str = "") -> str:
        if not base or is_legacy_alias(base):
            return ""
        if base == allow_self:
            return base
        if base not in used:
            return base
        for n in range(2, 100):
            candidate = f"{base}-{n}"
            if candidate not in used:
                return candidate
        return ""

    mapping: dict[str, str] = {}
    for index, model in enumerate(real_models):
        display = ""
        if display_names is not None and index < len(display_names):
            display = str(display_names[index] or "").strip()
        chosen = ""
        reused = reverse_previous.get(model) or ""
        if reused:
            chosen = _pick(reused)
        if not chosen:
            base = slugify(display) or slugify(model)
            chosen = _pick(base, allow_self=model if base == model else "")
        if not chosen:
            chosen = _pick(f"ecogpt-model-{index + 1}")
        if not chosen or chosen == model:
            continue  # nothing usable, or the slug is the real id itself
        used.add(chosen)
        mapping[chosen] = model
    return mapping


def desktop_models(
    exposed_models: list[str] | None,
    display_names: list[str] | None = None,
    aliases: list[str] | None = None,
) -> list[dict[str, Any]] | None:
    """inferenceModels entries for the profile: alias name + real display name.

    Tiers make Desktop's sonnet/haiku/opus class slots resolve to our entries;
    the first entry of each tier is that tier's default. ``display_names`` is a
    caller-resolved list aligned with ``exposed_models`` (raw gateway ids are
    not picker-friendly); missing/blank entries fall back to the raw model id.
    ``aliases`` optionally pins the alias names (from desktop_alias_map) so
    repeated applies reuse the same ids; they are generated fresh otherwise.
    """
    if not exposed_models:
        return None
    tier_cycle = ("sonnet", "haiku", "opus", "fable")
    seen_tiers: set[str] = set()
    entries = []
    for index, model in enumerate(exposed_models[:200]):
        tier = tier_cycle[index] if index < len(tier_cycle) else tier_cycle[index % len(tier_cycle)]
        label = model
        if display_names is not None and index < len(display_names):
            label = str(display_names[index] or "").strip() or model
        alias = ""
        if aliases is not None and index < len(aliases):
            alias = str(aliases[index] or "").strip()
        entry: dict[str, Any] = {
            "name": alias or desktop_alias_name(label),
            "labelOverride": label,
            "anthropicFamilyTier": tier,
        }
        if tier not in seen_tiers:
            entry["isFamilyDefault"] = True
            seen_tiers.add(tier)
        entries.append(entry)
    return entries


def _write_entry(app_dir: Path, base_url: str, api_key: str, model_entries: list[dict[str, Any]] | None = None) -> None:
    profile: dict[str, Any] = {
        "authentication": {"disableClaudeAiSignIn": True},
        "unstableDisableModelVerification": True,
        "inferenceGatewayBaseUrl": base_url,
        "inferenceGatewayApiKey": api_key,
        "bootstrapEnabled": False,
        "inferenceProvider": "gateway",
        "inferenceCredentialKind": "static",
        # The app's auth-scheme enum is ["bearer", "x-api-key"] and its default
        # is already "bearer"; pinning it makes the credential path explicit and
        # stable: static keys under the "bearer" scheme are sent as
        # "Authorization: Bearer <inferenceGatewayApiKey>" (and surface as
        # ANTHROPIC_AUTH_TOKEN for embedded Claude Code), never as x-api-key.
        "inferenceGatewayAuthScheme": "bearer",
    }
    # Seed the model list from the proxy's exposed models, aliased to slugged
    # display-name ids (Desktop rejects ids with provider prefixes like
    # 'openrouter/free': they lack an Anthropic allow-token and may carry deny
    # tokens; see desktop_models()). The alias name is what the app sends as
    # the model id, so the proxy aliases it via model_overrides.
    if model_entries:
        profile["inferenceModels"] = model_entries
    _write_json(_entry_path(app_dir, ECOGPT_ENTRY_ID), profile)


def _set_applied(app_dir: Path, entry_id: str, name: str) -> None:
    meta = _read_json(_meta_path(app_dir)) or {}
    entries = [entry for entry in (meta.get("entries") or []) if isinstance(entry, dict)]
    if not any(str(entry.get("id")) == entry_id for entry in entries):
        entries.append({"id": entry_id, "name": name})
    meta["appliedId"] = entry_id
    meta["entries"] = entries
    _write_json(_meta_path(app_dir), meta)


def status(app_dir: Path = DEFAULT_APP_DIR, backup_dir: Path = DEFAULT_BACKUP_DIR) -> dict[str, Any]:
    base_url = None
    applied_id = None
    entry_name = None
    error = None
    try:
        meta = _read_json(_meta_path(app_dir))
        if meta:
            applied_id = meta.get("appliedId")
            for entry in meta.get("entries") or []:
                if isinstance(entry, dict) and entry.get("id") == applied_id:
                    entry_name = entry.get("name")
        if applied_id:
            profile = _read_json(_entry_path(app_dir, str(applied_id)))
            if profile:
                base_url = profile.get("inferenceGatewayBaseUrl")
    except (ValueError, OSError) as exc:
        error = str(exc)
    return {
        "app_dir": str(app_dir),
        "app_dir_exists": app_dir.exists(),
        "applied_id": applied_id,
        "entry_name": entry_name,
        "ecogpt_entry_id": ECOGPT_ENTRY_ID,
        "base_url": base_url,
        "backup_exists": backup_dir.exists(),
        "error": error,
    }


def apply(app_dir: Path, base_url: str, api_key: str, *, name: str = "ecoGPT",
          backup_dir: Path = DEFAULT_BACKUP_DIR, model_entries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    if not app_dir.exists():
        raise ValueError(
            f"Claude Desktop data not found at {app_dir}; start Claude Desktop once before applying"
        )
    _take_backups(app_dir, backup_dir)
    _write_developer_settings(app_dir)
    _write_entry(app_dir, base_url, api_key, model_entries)
    _set_applied(app_dir, ECOGPT_ENTRY_ID, name)
    return status(app_dir, backup_dir)


def restore(app_dir: Path, backup_dir: Path = DEFAULT_BACKUP_DIR) -> dict[str, Any]:
    index = backup_dir / "index.json"
    if not index.exists():
        raise ValueError("No backup found, so there is nothing to restore")
    paths = json.loads(index.read_text(encoding="utf-8"))
    # Remove ecogpt-created config artifacts that never had a backup.
    entry = _entry_path(app_dir, ECOGPT_ENTRY_ID)
    if entry.exists() and f"configLibrary/{ECOGPT_ENTRY_ID}.json" not in paths:
        entry.unlink()
    for source_rel, backup_name in paths.items():
        target = app_dir / source_rel
        backup_file = backup_dir / backup_name
        if backup_file.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup_file, target)
    shutil.rmtree(backup_dir)
    return status(app_dir, backup_dir)
