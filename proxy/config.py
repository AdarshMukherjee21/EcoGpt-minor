from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import features as features_settings
from . import green as green_settings
from . import headroom_settings
from . import router_settings
from .routing import upstream_base

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
CONFIG_PATH = DATA_DIR / "config.json"
USAGE_DB_PATH = DATA_DIR / "usage.db"
SIDECAR_LOG_PATH = DATA_DIR / "sidecar.log"
LEDGER_DIR = DATA_DIR / "sessions"
DASHBOARD_DIR = PROJECT_DIR / "dashboard"

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 18090
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:18080"


@dataclass
class Settings:
    upstream_url: str = DEFAULT_UPSTREAM_URL
    api_key: str = ""
    exposed_models: list[str] = field(default_factory=list)
    route_overrides: dict[str, str] = field(default_factory=dict)
    model_overrides: dict[str, str] = field(default_factory=dict)
    # Desktop picker alias ids (slug of display name) -> real upstream model id.
    # Registry of aliases this proxy created, so repeated /api/desktop/apply
    # calls keep the same ids and legacy claude-eco-N ids can be recognized
    # and migrated away.
    desktop_aliases: dict[str, str] = field(default_factory=dict)
    providers: list[dict[str, Any]] = field(default_factory=list)
    model_providers: dict[str, str] = field(default_factory=dict)
    headroom: dict[str, Any] = field(default_factory=headroom_settings.defaults)
    features: dict[str, Any] = field(default_factory=features_settings.defaults)
    green: dict[str, Any] = field(default_factory=green_settings.defaults)
    # System One tier router: scalar settings (endpoint, warmth, guards) and the
    # ordered tier ladder. api_key lives here and is masked in every view.
    router: dict[str, Any] = field(default_factory=router_settings.defaults)
    router_tiers: list[dict[str, Any]] = field(default_factory=router_settings.default_tiers)
    # System One tier router: scalar settings + the ordered tier ladder
    # ({"id", "description", "model"}). Tiers ride along in the same block; the
    # classifier API key never leaves this dict (masked in every view).
    router: dict[str, Any] = field(default_factory=router_settings.defaults)
    router_tiers: list[dict[str, Any]] = field(default_factory=router_settings.default_tiers)

    def __post_init__(self) -> None:
        self.migrate_providers()

    def migrate_providers(self) -> None:
        """Back-compat: a config with only upstream_url/api_key becomes provider[0]."""
        if not self.providers:
            self.providers = [{"name": "primary", "url": self.upstream_url, "api_key": self.api_key}]

    def effective_providers(self) -> list[dict[str, str]]:
        out = []
        for provider in self.providers[:3]:
            url = str(provider.get("url") or "").strip() or self.upstream_url
            name = str(provider.get("name") or "").strip() or "primary"
            out.append({"name": name, "url": upstream_base(url), "api_key": str(provider.get("api_key") or "")})
        return out or [{"name": "primary", "url": self.upstream_url, "api_key": self.api_key}]

    def provider_for(self, model: str) -> dict[str, str]:
        providers = self.effective_providers()
        name = self.model_providers.get(model or "")
        for provider in providers:
            if provider["name"] == name:
                return provider
        return providers[0]

    def default_provider(self) -> dict[str, str]:
        return self.effective_providers()[0]

    def effective_key(self, provider: dict[str, str] | None = None) -> str:
        # Per-provider key wins; the legacy TOKEN_PROXY_UPSTREAM_KEY env still
        # overrides everything (ablation harness expectation).
        env = os.environ.get("TOKEN_PROXY_UPSTREAM_KEY")
        if env:
            return env
        if provider is not None:
            return provider.get("api_key") or ""
        return self.api_key

    def proxy_url(self) -> str:
        return f"http://{LISTEN_HOST}:{LISTEN_PORT}"

    def sidecar_url(self) -> str:
        return f"http://{LISTEN_HOST}:{self.headroom.get('sidecar_port', 8787)}"


def load_settings(path: Path = CONFIG_PATH) -> Settings:
    settings = Settings()
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        settings.upstream_url = raw.get("upstream_url", settings.upstream_url)
        settings.api_key = raw.get("api_key", "")
        settings.exposed_models = list(raw.get("exposed_models", []))
        settings.route_overrides = dict(raw.get("route_overrides", {}))
        overrides = raw.get("model_overrides") or {}
        settings.model_overrides = {str(k): str(v) for k, v in dict(overrides).items()}
        settings.desktop_aliases = {str(k): str(v) for k, v in dict(raw.get("desktop_aliases") or {}).items()}
        stored_providers = raw.get("providers")
        if isinstance(stored_providers, list):
            settings.providers = [
                {
                    "name": str(p.get("name") or "").strip() or f"provider-{index + 1}",
                    "url": str(p.get("url") or "").strip(),
                    "api_key": str(p.get("api_key") or ""),
                }
                for index, p in enumerate(stored_providers[:3])
                if isinstance(p, dict)
            ]
        stored_model_providers = raw.get("model_providers")
        if isinstance(stored_model_providers, dict):
            settings.model_providers = {str(k): str(v) for k, v in stored_model_providers.items()}
        stored = {
            key: value for key, value in dict(raw.get("headroom", {})).items()
            if key in {e["key"] for e in headroom_settings.SCHEMA}
        }
        settings.headroom = headroom_settings.normalize(stored)
        stored_features = {
            key: value for key, value in dict(raw.get("features", {})).items()
            if key in {e["key"] for e in features_settings.FEATURES}
        }
        settings.features = features_settings.normalize(stored_features)
        stored_green = {
            key: value for key, value in dict(raw.get("green", {})).items()
            if key in {e["key"] for e in green_settings.GREEN_SCHEMA}
        }
        settings.green = green_settings.normalize(stored_green)
        stored_router = {
            key: value for key, value in dict(raw.get("router", {})).items()
            if key in {e["key"] for e in router_settings.SCHEMA} and key != "tiers"
        }
        try:
            settings.router = router_settings.normalize(stored_router)
        except ValueError:
            settings.router = router_settings.defaults()
        if isinstance(raw.get("router"), dict):
            stored_tiers = raw["router"].get("tiers")
            if isinstance(stored_tiers, list) and stored_tiers:
                try:
                    settings.router_tiers = router_settings.normalize_tiers(stored_tiers)
                except ValueError:
                    settings.router_tiers = router_settings.default_tiers()
    return settings


def save_settings(settings: Settings, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(settings)
    # The tier ladder is stored nested INSIDE the router block (the request-time
    # loader reads data/config.json's "router"."tiers"); asdict emits it as a
    # sibling field, so re-nest before writing.
    router = dict(payload.get("router") or {})
    router["tiers"] = payload.pop("router_tiers", router_settings.default_tiers())
    payload["router"] = router
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
