"""Points Claude Desktop at the proxy by writing an inference-gateway profile
into Claude Desktop's 3p config library, with restorable backups.

Claude Desktop (MSIX build, userData under %LOCALAPPDATA%/Claude-3p) selects a
"gateway" profile via configLibrary/_meta.json -> appliedId; the profile JSON
carries inferenceGatewayBaseUrl / inferenceGatewayApiKey. The app re-seals any
plaintext key it finds on next launch (user DPAPI), so the script may write the
key as plaintext and let the app encrypt it itself.

Applying requires quitting Claude Desktop first (it rewrites these files on
exit); restoring swaps the byte-for-byte backups taken on first apply.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from .config import DATA_DIR

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


def _write_entry(app_dir: Path, base_url: str, api_key: str) -> None:
    profile: dict[str, Any] = {
        "authentication": {"disableClaudeAiSignIn": True},
        "unstableDisableModelVerification": True,
        "inferenceGatewayBaseUrl": base_url,
        "inferenceGatewayApiKey": api_key,
        "bootstrapEnabled": False,
        "inferenceProvider": "gateway",
        "inferenceCredentialKind": "static",
    }
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
          backup_dir: Path = DEFAULT_BACKUP_DIR) -> dict[str, Any]:
    if not app_dir.exists():
        raise ValueError(
            f"Claude Desktop data not found at {app_dir}; start Claude Desktop once before applying"
        )
    _take_backups(app_dir, backup_dir)
    _write_developer_settings(app_dir)
    _write_entry(app_dir, base_url, api_key)
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
