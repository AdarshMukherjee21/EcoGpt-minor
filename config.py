"""
config.py — Singleton config loader for ecoGPT.

Loads:
  - .env                 (API keys, via python-dotenv)
  - config.yaml          (non-secret settings, via PyYAML)

Exposes:
  - settings             (a `Settings` instance with structured access to every section)

Usage:
    from config import settings

    settings.models.small.name          # e.g. "qwen2.5:3b"
    settings.models.small.api_base      # e.g. "http://localhost:11434/v1"
    settings.models.small.api_key       # resolved from os.environ[<api_key_env>]
    settings.router.threshold           # e.g. 0.1159
    settings.compression.device         # resolved torch.device, "auto" handled here
    settings.green_metric.carbon_intensity_kg_per_kwh
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config.yaml"
ENV_PATH = PROJECT_ROOT / ".env"
ENV_EXAMPLE_PATH = PROJECT_ROOT / ".env.example"


# ---------------------------------------------------------------------------
# Section dataclasses (typed views over the YAML)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    name: str
    api_base: str
    api_key_env: str
    api_key: Optional[str] = None  # populated at load time, not from YAML


@dataclass(frozen=True)
class ModelsConfig:
    small: ModelConfig
    large: ModelConfig


@dataclass(frozen=True)
class RouterConfig:
    type: str
    threshold: float
    per_model_overrides: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional["EmbeddingConfig"] = None


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str  # "sentence-transformers" | "openai"
    model_name: str
    device: str  # "auto" | "cpu" | "cuda"


@dataclass(frozen=True)
class CompressionConfig:
    model_name: str
    device: str  # raw value from YAML: "auto" | "cpu" | "cuda"
    target_rate: float


@dataclass(frozen=True)
class RawRunnerConfig:
    timeout_seconds: int
    request_defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GreenMetricConfig:
    carbon_intensity_kg_per_kwh: float
    wue_l_per_kwh: float
    monte_carlo_samples: int
    random_seed: int
    h100_8gpu_max_power_kw: float
    power_p5_fraction: float
    power_p95_fraction: float
    pue_p5: float
    pue_p95: float
    class_gpu_count: Dict[str, int]


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    host: str
    port: int


@dataclass(frozen=True)
class Settings:
    project: ProjectConfig
    models: ModelsConfig
    router: RouterConfig
    compression: CompressionConfig
    raw_runner: RawRunnerConfig
    green_metric: GreenMetricConfig
    config_path: Path
    env_path: Path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required config key '{key}' in {context}.")
    return mapping[key]


def _build_settings(yaml_data: Mapping[str, Any]) -> Settings:
    # ---- project ----
    proj = _require(yaml_data, "project", "config.yaml")
    project_cfg = ProjectConfig(
        name=str(_require(proj, "name", "project")),
        host=str(_require(proj, "host", "project")),
        port=int(_require(proj, "port", "project")),
    )

    # ---- models ----
    models_raw = _require(yaml_data, "models", "config.yaml")
    small_raw = _require(models_raw, "small", "models")
    large_raw = _require(models_raw, "large", "models")

    small_key_env = str(_require(small_raw, "api_key_env", "models.small"))
    large_key_env = str(_require(large_raw, "api_key_env", "models.large"))

    small_cfg = ModelConfig(
        name=str(_require(small_raw, "name", "models.small")),
        api_base=str(_require(small_raw, "api_base", "models.small")),
        api_key_env=small_key_env,
        api_key=os.environ.get(small_key_env),
    )
    large_cfg = ModelConfig(
        name=str(_require(large_raw, "name", "models.large")),
        api_base=str(_require(large_raw, "api_base", "models.large")),
        api_key_env=large_key_env,
        api_key=os.environ.get(large_key_env),
    )
    models_cfg = ModelsConfig(small=small_cfg, large=large_cfg)

    # ---- router ----
    router_raw = _require(yaml_data, "router", "config.yaml")
    embedding_raw = router_raw.get("embedding")
    embedding_cfg: Optional[EmbeddingConfig] = None
    if embedding_raw:
        embedding_cfg = EmbeddingConfig(
            provider=str(_require(embedding_raw, "provider", "router.embedding")),
            model_name=str(_require(embedding_raw, "model_name", "router.embedding")),
            device=str(_require(embedding_raw, "device", "router.embedding")).lower(),
        )
    router_cfg = RouterConfig(
        type=str(_require(router_raw, "type", "router")),
        threshold=float(_require(router_raw, "threshold", "router")),
        per_model_overrides=dict(router_raw.get("per_model_overrides", {}) or {}),
        embedding=embedding_cfg,
    )

    # ---- compression ----
    comp_raw = _require(yaml_data, "compression", "config.yaml")
    comp_cfg = CompressionConfig(
        model_name=str(_require(comp_raw, "model_name", "compression")),
        device=str(_require(comp_raw, "device", "compression")).lower(),
        target_rate=float(_require(comp_raw, "target_rate", "compression")),
    )

    # ---- raw_runner ----
    rr_raw = _require(yaml_data, "raw_runner", "config.yaml")
    rr_cfg = RawRunnerConfig(
        timeout_seconds=int(_require(rr_raw, "timeout_seconds", "raw_runner")),
        request_defaults=dict(rr_raw.get("request_defaults", {}) or {}),
    )

    # ---- green_metric ----
    gm_raw = _require(yaml_data, "green_metric", "config.yaml")
    class_gpu_count_raw = _require(gm_raw, "class_gpu_count", "green_metric")
    gm_cfg = GreenMetricConfig(
        carbon_intensity_kg_per_kwh=float(
            _require(gm_raw, "carbon_intensity_kg_per_kwh", "green_metric")
        ),
        wue_l_per_kwh=float(gm_raw.get("wue_l_per_kwh", 1.2)),
        monte_carlo_samples=int(
            _require(gm_raw, "monte_carlo_samples", "green_metric")
        ),
        random_seed=int(_require(gm_raw, "random_seed", "green_metric")),
        h100_8gpu_max_power_kw=float(
            _require(gm_raw, "h100_8gpu_max_power_kw", "green_metric")
        ),
        power_p5_fraction=float(_require(gm_raw, "power_p5_fraction", "green_metric")),
        power_p95_fraction=float(
            _require(gm_raw, "power_p95_fraction", "green_metric")
        ),
        pue_p5=float(_require(gm_raw, "pue_p5", "green_metric")),
        pue_p95=float(_require(gm_raw, "pue_p95", "green_metric")),
        class_gpu_count={k: int(v) for k, v in class_gpu_count_raw.items()},
    )

    return Settings(
        project=project_cfg,
        models=models_cfg,
        router=router_cfg,
        compression=comp_cfg,
        raw_runner=rr_cfg,
        green_metric=gm_cfg,
        config_path=CONFIG_YAML_PATH,
        env_path=ENV_PATH,
    )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_settings: Optional[Settings] = None
_settings_lock = threading.Lock()


def get_settings() -> Settings:
    """Return the cached `Settings`, building it on first access."""
    global _settings
    if _settings is not None:
        return _settings

    with _settings_lock:
        if _settings is not None:
            return _settings

        # 1. Load .env (does nothing if file is missing)
        load_dotenv(dotenv_path=ENV_PATH, override=False)

        # 2. Load config.yaml
        if not CONFIG_YAML_PATH.exists():
            raise FileNotFoundError(
                f"config.yaml not found at {CONFIG_YAML_PATH}. "
                "Copy the template or run from the project root."
            )
        with CONFIG_YAML_PATH.open("r", encoding="utf-8") as fh:
            yaml_data = yaml.safe_load(fh) or {}

        _settings = _build_settings(yaml_data)
        return _settings


# Eagerly expose `settings` for convenience. Tests / scripts can call
# `get_settings()` to force a rebuild after editing the YAML.
settings: Settings = get_settings()


if __name__ == "__main__":
    # Quick smoke test: `python config.py`
    s = get_settings()
    print(f"Project:        {s.project.name} on {s.project.host}:{s.project.port}")
    print(f"Small model:    {s.models.small.name} @ {s.models.small.api_base}")
    print(
        f"  api_key_env:  {s.models.small.api_key_env} -> {'set' if s.models.small.api_key else 'MISSING'}"
    )
    print(f"Large model:    {s.models.large.name} @ {s.models.large.api_base}")
    print(
        f"  api_key_env:  {s.models.large.api_key_env} -> {'set' if s.models.large.api_key else 'MISSING'}"
    )
    print(f"Router:         {s.router.type} @ threshold={s.router.threshold}")
    if s.router.embedding:
        emb = s.router.embedding
        print(
            f"  Embedding:    {emb.provider} / {emb.model_name} on device={emb.device}"
        )
    print(
        f"Compression:    {s.compression.model_name} on device={s.compression.device} rate={s.compression.target_rate}"
    )
    print(f"Raw runner:     timeout={s.raw_runner.timeout_seconds}s")
    print(
        f"Green metric:   carbon={s.green_metric.carbon_intensity_kg_per_kwh} kg/kWh, "
        f"wue={s.green_metric.wue_l_per_kwh} L/kWh, "
        f"mc_samples={s.green_metric.monte_carlo_samples}"
    )
