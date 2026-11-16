"""Green metrics: per-request energy and GWP estimates — pure arithmetic, an observer.

Methodology and licensing
-------------------------
Impacts follow the EcoLogits methodology (CC BY-SA 4.0, mlco2/ecologits) with energy
anchors from The AI Footprint (CC BY 4.0, theaifootprint.com) and official numbers from
the GLM-5.2 / DeepSeek-V4-Pro model cards; per-model rows carry their ``basis``. The
attribution strings in ``data/green_factors.json`` must be shown wherever impacts are
displayed — the licenses require it.

Interval semantics
-----------------
Every factor carries a mean plus min/max set explicitly in the factor table (two
independent estimation methods). Each request is computed three times with the matching
bound of *every* input factor, so uncertainty propagates through the whole chain.

Documented simplifications
-------------------------
- The EcoLogits server/idle-overhead term is omitted; it contributes <0.1% at typical
  request sizes.
- MoE rows multiply the per-GPU cost by the full GPU count of the deployed cluster,
  which is deliberately conservative.
- Region assumes one deployment region (``region.country``) for all traffic.

Model table and settings
------------------------
``models`` lives in data/green_factors.json only (edit the file, not the dashboard).
The region block and calibration knobs are dashboard-editable settings persisted under
the ``green`` key in config.json; the file supplies their defaults. The factors file is
reloaded when its mtime changes; the compute path does no other I/O.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parent.parent
GREEN_FACTORS_PATH = PROJECT_DIR / "data" / "green_factors.json"
GREEN_OVERRIDES_PATH = PROJECT_DIR / "data" / "green_overrides.json"

# Fallback used only when the factors file is missing or unreadable; matches it.
_FALLBACK = {
    "region": {
        "country": "IRL",
        "grid_gco2_kwh": {"mean": 260, "min": 250, "max": 290},
        "pue": {"mean": 1.15, "min": 1.10, "max": 1.20},
    },
    "knobs": {
        "cached_input_fraction": 0.25,
        "input_output_ratio": 1.0,
        "embodied_share": 0.15,
    },
    "attribution": [
        "EcoLogits methodology (CC BY-SA 4.0), mlco2/ecologits",
        "The AI Footprint anchors (CC BY 4.0), theaifootprint.com",
        "GLM-5.2 / DeepSeek-V4-Pro: official model cards",
        "Proprietary rows: operator priors over public estimates (rumor-tier)",
    ],
    "models": {},
}

_DATE_SLUG = re.compile(r"-(?:19|20)\d{6,8}$")

GREEN_SCHEMA: list[dict[str, Any]] = [
    {"key": "region_country", "group": "Green metrics — region", "kind": "str", "type": "str",
     "label": "Country", "help": "ISO code of the deployment region (single region assumed in the PoC)."},
    {"key": "region_grid_gco2_mean", "group": "Green metrics — region", "kind": "float", "type": "float",
     "default": 260, "min": 0, "label": "Grid intensity mean", "help": "gCO2e per kWh of the regional grid."},
    {"key": "region_grid_gco2_min", "group": "Green metrics — region", "kind": "float", "type": "float",
     "default": 250, "min": 0, "label": "Grid intensity min", "help": "Lower bound of the grid factor."},
    {"key": "region_grid_gco2_max", "group": "Green metrics — region", "kind": "float", "type": "float",
     "default": 290, "min": 0, "label": "Grid intensity max", "help": "Upper bound of the grid factor."},
    {"key": "region_pue_mean", "group": "Green metrics — region", "kind": "float", "type": "float",
     "default": 1.15, "min": 1.0, "label": "PUE mean", "help": "Power usage effectiveness of the data centre."},
    {"key": "region_pue_min", "group": "Green metrics — region", "kind": "float", "type": "float",
     "default": 1.10, "min": 1.0, "label": "PUE min", "help": "Lower bound of the PUE factor."},
    {"key": "region_pue_max", "group": "Green metrics — region", "kind": "float", "type": "float",
     "default": 1.20, "min": 1.0, "label": "PUE max", "help": "Upper bound of the PUE factor."},
    {"key": "knob_cached_input_fraction", "group": "Green metrics — knobs", "kind": "float", "type": "float",
     "default": 0.25, "min": 0, "max": 1, "label": "Cached input fraction",
     "help": "Share of a cached input token's energy charged for a cache read."},
    {"key": "knob_input_output_ratio", "group": "Green metrics — knobs", "kind": "float", "type": "float",
     "default": 1.0, "min": 0, "label": "Input/output ratio",
     "help": "How one input token's energy compares to one output token's (r_in)."},
    {"key": "knob_embodied_share", "group": "Green metrics — knobs", "kind": "float", "type": "float",
     "default": 0.15, "min": 0, "max": 1, "label": "Embodied share",
     "help": "Hardware embodied energy added on top of operational energy."},
]

_BY_KEY = {entry["key"]: entry for entry in GREEN_SCHEMA}
_BOUNDS = ("min", "mean", "max")

_factors_cache: dict[str, Any] = {}
_factors_mtime: float | None = None
_overrides_mtime: float | None = None


def load_overrides(path: Path | None = None) -> dict[str, Any]:
    """Per-model factor overrides (user-editable from the dashboard); fails open."""
    path = Path(path or GREEN_OVERRIDES_PATH)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, dict) else {}


def load_factors(path: Path | None = None, overrides_path: Path | None = None) -> dict[str, Any]:
    """The factor table with overrides merged in; cached, reloads on mtime change. Fails open."""
    global _factors_cache, _factors_mtime, _overrides_mtime
    path = Path(path or GREEN_FACTORS_PATH)
    overrides_path = Path(overrides_path or GREEN_OVERRIDES_PATH)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return _FALLBACK
    try:
        overrides_mtime = overrides_path.stat().st_mtime
    except OSError:
        overrides_mtime = None
    if _factors_cache and mtime == _factors_mtime and overrides_mtime == _overrides_mtime:
        return _factors_cache
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("factors file must be a JSON object")
    except (OSError, ValueError):
        _factors_cache = _FALLBACK
        _factors_mtime = mtime
        _overrides_mtime = overrides_mtime
        return _factors_cache
    overrides = load_overrides(overrides_path)
    if overrides:
        models = data.get("models") if isinstance(data.get("models"), dict) else {}
        for key, override in overrides.items():
            base_row = models.get(key)
            if isinstance(override, dict) and isinstance(base_row, dict):
                merged = dict(base_row)
                merged.update(override)
                merged["_overridden"] = True
                models[key] = merged
        data["models"] = models
    _factors_cache = data
    _factors_mtime = mtime
    _overrides_mtime = overrides_mtime
    return data


def defaults() -> dict[str, Any]:
    """Region and knob defaults from the factors file (fallback constants if absent)."""
    factors = load_factors()
    region = factors.get("region") if isinstance(factors.get("region"), dict) else {}
    knobs = factors.get("knobs") if isinstance(factors.get("knobs"), dict) else {}
    grid = region.get("grid_gco2_kwh") if isinstance(region.get("grid_gco2_kwh"), dict) else {}
    pue = region.get("pue") if isinstance(region.get("pue"), dict) else {}
    raw = {
        "region_country": region.get("country", _FALLBACK["region"]["country"]),
        "region_grid_gco2_mean": grid.get("mean", 260),
        "region_grid_gco2_min": grid.get("min", 250),
        "region_grid_gco2_max": grid.get("max", 290),
        "region_pue_mean": pue.get("mean", 1.15),
        "region_pue_min": pue.get("min", 1.10),
        "region_pue_max": pue.get("max", 1.20),
        "knob_cached_input_fraction": knobs.get("cached_input_fraction", 0.25),
        "knob_input_output_ratio": knobs.get("input_output_ratio", 1.0),
        "knob_embodied_share": knobs.get("embodied_share", 0.15),
    }
    return normalize(raw, _trusted=True)


def normalize(values: dict[str, Any], *, _trusted: bool = False) -> dict[str, Any]:
    """Validate and coerce green settings; unknown keys raise, missing keys take defaults."""
    if not _trusted:
        unknown = sorted(set(values) - set(_BY_KEY))
        if unknown:
            raise ValueError(f"unknown green setting: {unknown[0]}")
        values = {**defaults(), **values}
    out: dict[str, Any] = {}
    for entry in GREEN_SCHEMA:
        key = entry["key"]
        raw = values.get(key, entry.get("default"))
        if entry["type"] == "str":
            out[key] = str(raw or "").strip()
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: expected a number") from exc
        if "min" in entry and value < entry["min"]:
            raise ValueError(f"{key}: must be at least {entry['min']}")
        if "max" in entry and value > entry["max"]:
            raise ValueError(f"{key}: must be at most {entry['max']}")
        out[key] = value
    return out


def attribution(path: Path | None = None) -> list[str]:
    """The attribution strings that must accompany any display of impacts."""
    items = load_factors(path).get("attribution")
    return [str(item) for item in items] if isinstance(items, list) else list(_FALLBACK["attribution"])


def model_key(model: str, factors: dict[str, Any] | None = None) -> str | None:
    """Resolve a normalized model id to a factor-table row.

    Exact match first; a dated snapshot slug (e.g. claude-opus-5-5-20260301) resolves
    to the undated row. Anything else is unknown.
    """
    models = (factors if factors is not None else load_factors()).get("models")
    if not isinstance(models, dict):
        return None
    name = (model or "").strip()
    if not name:
        return None
    if name in models:
        return name
    stripped = _DATE_SLUG.sub("", name)
    return stripped if stripped != name and stripped in models else None


def _triple(settings: dict[str, Any], prefix: str) -> dict[str, float]:
    return {bound: float(settings.get(f"{prefix}_{bound}", 0.0)) for bound in _BOUNDS}


def compute(
    model: str,
    usage: dict[str, Any] | None,
    green_settings: dict[str, Any] | None = None,
    path: Path | None = None,
) -> dict[str, Any]:
    """Impacts for one request; token counts come from the observed usage row.

    Anthropic semantics: ``input_tokens`` is the *fresh* portion, cache reads are
    reported separately and charged at ``knob_cached_input_fraction``; cache writes are
    ignored for energy. Unknown models return null impacts with basis "unknown".
    """

    def zero() -> dict[str, float]:
        return {bound: 0.0 for bound in _BOUNDS}

    factors = load_factors(path)
    key = model_key(model, factors)
    if key is None:
        return {"energy_kwh": None, "gwp_kgco2e": None, "model_basis": "unknown"}
    row = factors.get("models", {}).get(key, {})
    wpt = row.get("wh_per_output_token")
    if not isinstance(wpt, dict):
        return {"energy_kwh": None, "gwp_kgco2e": None, "model_basis": "unknown"}

    g = dict(defaults())
    g.update(green_settings or {})
    grid = _triple(g, "region_grid_gco2")
    pue = _triple(g, "region_pue")
    r_in = float(g.get("knob_input_output_ratio", 1.0))
    cached_frac = float(g.get("knob_cached_input_fraction", 0.25))
    embodied = float(g.get("knob_embodied_share", 0.15))

    usage = usage or {}
    fresh = usage.get("input_tokens") or 0
    cached = usage.get("cache_read_input_tokens") or 0
    out = usage.get("output_tokens") or 0
    effective_output = out + r_in * fresh + r_in * cached_frac * cached

    energy_kwh: dict[str, float] = zero()
    gwp_kgco2e: dict[str, float] = zero()
    for bound in _BOUNDS:
        gpu_wh = float(wpt.get(bound, 0.0)) * effective_output
        total_wh = gpu_wh * pue[bound] * (1.0 + embodied)
        energy_kwh[bound] = total_wh / 1000.0
        gwp_kgco2e[bound] = energy_kwh[bound] * grid[bound] / 1000.0
    return {
        "energy_kwh": energy_kwh,
        "gwp_kgco2e": gwp_kgco2e,
        "model_basis": str(row.get("basis", "estimated")),
    }
