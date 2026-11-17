"""EcoLogits-native green compute (demo-style dual path) — pure function, an observer.

Paths
-----
factor      the legacy arithmetic in :mod:`proxy.green` (``wh_per_output_token``
            triples, grid x PUE x embodied, three-bound interval propagation).
            Selected when the resolved model row carries a ``wh_per_output_token``
            override (or an explicit ``mode: "factor"`` on its row).
ecologits   the real ``ecologits.impacts.llm.compute_llm_impacts()`` from the
            pinned library (requirements.txt) with active/total parameters,
            observed request latency and, when streaming observed them, TPS/TTFT.
            Selected by default for rows with parameters (``arch`` in
            data/green_factors.json, or a params override, or an exact match in
            the vendored EcoLogits catalogue under data/ecologits/).

Approved semantics (owner sign-off): input and cache tokens are folded into the
effective output token count on BOTH paths (``input_output_ratio`` and
``cached_input_fraction`` knobs). ``embodied_share`` stays factor-path-only: the
EcoLogits library already models embedded impacts, so applying it again would
double-count.

Model parameter resolution order: per-model override row (data/green_overrides.json,
also merged by proxy/green.load_factors()) > factor-table row via green.model_key()
(dated snapshot slugs collapse) > exact EcoLogits catalogue match (strip ``:free``,
aliases, ownership-aware brand check, no fuzzy match) > unconfigured_model.

Electricity: the existing region settings (region_grid_gco2_*, region_pue_*) map to
the GWP and PUE inputs (gCO2e/kWh / 1000 -> kgCO2e/kWh); ADPe, PE and water come
from the vendored EcoLogits electricity_mixes.json for the region country, with a
World fallback. The factor path derives water/ADPe/PE from the same grid mix
(kgSbeq/kWh, MJ/kWh, L/kWh times the computed energy) — an approximation flagged
in its provenance string.

Interval semantics: the legacy path keeps the three-bound propagation it has
always had. On the ecologits path the request is computed three times, once per
grid/PUE bound, so energy and GWP keep min/mean/max in the usage DB; water, ADPe
and primary energy are recorded from the mean bound and must always be shown with
the provenance caption.

Fail-closed: only ``impact_status == "estimated"`` rows carry numbers. Unknown
models, missing usage and calculation errors return null impacts plus a status —
no fake green estimates. This module never raises; the app-layer caller treats
its return value as data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .green import load_factors, load_overrides, model_key

PROJECT_DIR = Path(__file__).resolve().parent.parent
ECOLOGITS_DIR = PROJECT_DIR / "data" / "ecologits"

# Vendored EcoLogits catalogues (pinned revision; see data/ecologits/PROVENANCE.txt).
# Explicit env overrides win, for reproducible or restricted environments.
MODELS_JSON = Path(os.getenv("ECOLOGITS_MODELS_JSON") or (ECOLOGITS_DIR / "models.json"))
ELECTRICITY_JSON = Path(
    os.getenv("ECOLOGITS_ELECTRICITY_JSON") or (ECOLOGITS_DIR / "electricity_mixes.json")
)
ECOLOGITS_REVISION = "ecologits 0.10.2 catalogue"

# EcoLogits World-average mix, used when the country is absent from the dataset.
FALLBACK_WORLD = {"gwp": 0.458290, "adpe": 1.349e-8, "pe": 2.587, "wue": 3.908}

_BOUNDS = ("min", "mean", "max")

_catalogue_cache: dict[str, tuple[float | None, Any]] = {}


def model_config(model: str) -> dict[str, Any]:
    """Per-model expert configuration from the override file (dashboard-editable).

    Stored in data/green_overrides.json under ``models[model]`` and may carry:
    ``params`` {total_b, active_b}, ``tps`` / ``ttft`` (configured averages, used
    only when the stream did not observe them — same precedence as the reference
    demo's catalogue deployment values), and datacenter scenario fields ``pue``,
    ``wue_onsite``, ``country`` and ``electricity`` {gwp, adpe, pe, wue} which
    override the global region settings for that model. Legacy entries holding
    only ``wh_per_output_token`` keep their factor-arithmetic meaning.
    """
    entry = load_overrides().get(model)
    if not isinstance(entry, dict):
        entry = {}

    def num(key: str) -> float | None:
        try:
            value = float(entry[key])
        except (KeyError, TypeError, ValueError):
            return None
        return value if value == value and value not in (float("inf"), float("-inf")) else None

    params_block = entry.get("params") if isinstance(entry.get("params"), dict) else None
    params = None
    if params_block:
        total, active = _scalar(params_block.get("total_b")), _scalar(params_block.get("active_b"))
        if total and active:
            params = {"active_b": active, "total_b": total,
                      "source": str(entry.get("basis") or "user-override")}
    electricity = entry.get("electricity") if isinstance(entry.get("electricity"), dict) else None
    electricity = ({k: float(electricity[k]) for k in ("gwp", "adpe", "pe", "wue")
                    if isinstance(electricity.get(k), (int, float))} if electricity else None)
    return {
        "entry": entry,
        "params": params,
        "tps": num("tps"),
        "ttft": num("ttft"),
        "pue": num("pue"),
        "wue_onsite": num("wue_onsite"),
        "country": (str(entry.get("country") or "").strip().upper() or None),
        "electricity": electricity,
    }


def _scalar(value: Any) -> float | None:
    """EcoLogits catalogue sizes may be scalar, {'min','max'} or {'total'}."""
    if isinstance(value, dict):
        if value.get("min") is not None and value.get("max") is not None:
            return (float(value["min"]) + float(value["max"])) / 2
        if value.get("total") is not None:
            return _scalar(value["total"])
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _value(x: Any) -> float | None:
    """Coerce an EcoLogits impact (value / Interval / plain number) to a float."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        if x.get("value") is not None:
            return _value(x["value"])
        if x.get("min") is not None and x.get("max") is not None:
            return (_value(x["min"]) + _value(x["max"])) / 2
        return None
    value = getattr(x, "value", None)
    if value is not None:
        return _value(value)
    lo, hi = getattr(x, "min", None), getattr(x, "max", None)
    if lo is not None and hi is not None:
        lo_v, hi_v = _value(lo), _value(hi)
        if lo_v is not None and hi_v is not None:
            return (lo_v + hi_v) / 2
    return None


def catalogue(kind: str) -> Any:
    """Load a vendored catalogue (kind: 'models' or 'electricity'); cached on mtime; fails closed."""
    path = MODELS_JSON if kind == "models" else ELECTRICITY_JSON
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        raise OSError(f"vendored EcoLogits catalogue missing: {path}") from exc
    cached = _catalogue_cache.get(kind)
    if cached and cached[0] == mtime:
        return cached[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    _catalogue_cache[kind] = (mtime, data)
    return data


def catalogue_match(model_id: str) -> dict[str, Any] | None:
    """Exact EcoLogits catalogue match for a model id — no substring, no provider substitution.

    A provider serving the model need not own it, so the brand prefix must be
    compatible with the catalogue provider. Returns parameters plus the metadata
    the app layer shows in provenance and the dashboard overlay.
    """
    try:
        data = catalogue("models")
    except (OSError, ValueError):
        return None
    name = (model_id or "").strip()
    if not name:
        return None
    raw = name.split("/", 1)[-1].removesuffix(":free")
    for alias in data.get("aliases", []):
        if isinstance(alias, dict) and alias.get("name") == raw:
            raw = str(alias.get("alias") or raw)
            break
    brand = name.split("/", 1)[0].replace("-", "_") if "/" in name else ""
    for entry in data.get("models", []):
        if not isinstance(entry, dict) or entry.get("name") != raw:
            continue
        eco_provider = str(entry.get("provider", "")).lower().replace("-", "_")
        compatible = (
            brand == eco_provider
            or brand == ""  # undated bare ids like "gpt-4o": accept the catalogue's own provider
            or (brand in {"google", "google_genai"} and eco_provider == "google_genai")
            or (brand == "meta_llama" and eco_provider == "huggingface_hub")
        )
        if not compatible:
            continue
        arch = entry.get("architecture") or {}
        counts = arch.get("parameters")
        if counts is None:
            continue
        if arch.get("type") == "moe":
            total, active = _scalar(counts.get("total")), _scalar(counts.get("active"))
        else:
            total = active = _scalar(counts)
        if not total or not active:
            continue
        deployment = entry.get("deployment") or {}
        return {
            "active_b": active,
            "total_b": total,
            "architecture": arch.get("type", "unknown"),
            "warnings": entry.get("warnings") or [],
            "sources": entry.get("sources") or [],
            "deployment": deployment,
        }
    return None


def electricity(country: str, override: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    """Electricity mix factors for one ISO3 country (+ World fallback), or a manual override.

    Returns ``({gwp, adpe, pe, wue}, source_string)``; gwp in kgCO2e/kWh,
    adpe in kgSbeq/kWh, pe in MJ/kWh, wue in L/kWh.
    """
    if override:
        try:
            factors = {key: float(override[key]) for key in ("gwp", "adpe", "pe", "wue")}
        except (KeyError, TypeError, ValueError):
            factors = {}
        if len(factors) == 4:
            return factors, "manual electricity factors"
    try:
        rows = catalogue("electricity").get("electricity_mixes", [])
        zone = (country or "").upper().strip() or "WOR"
        for row in rows:
            if isinstance(row, dict) and row.get("name") == zone:
                return ({key: float(row[key]) for key in ("gwp", "adpe", "pe", "wue")},
                        f"{ECOLOGITS_REVISION}: {zone}")
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return FALLBACK_WORLD.copy(), f"{ECOLOGITS_REVISION}: World fallback (country not verified)"


def _effective_output(usage: dict[str, Any] | None, green_settings: dict[str, Any]) -> float:
    """Fold input and cache-read tokens into the effective output count (approved default)."""
    usage = usage or {}
    fresh = usage.get("input_tokens") or 0
    cached = usage.get("cache_read_input_tokens") or 0
    out = usage.get("output_tokens") or 0
    r_in = float(green_settings.get("knob_input_output_ratio", 1.0))
    cached_frac = float(green_settings.get("knob_cached_input_fraction", 0.25))
    return out + r_in * fresh + r_in * cached_frac * cached


def _missing(status: str, provenance: str, basis: str = "unknown") -> dict[str, Any]:
    return {
        "energy_kwh": None,
        "gwp_kgco2e": None,
        "water_l": None,
        "adpe_kg_sbeq": None,
        "primary_mj": None,
        "model_basis": basis,
        "impact_status": status,
        "impact_provenance": provenance,
    }


def _row_mode(row: dict[str, Any]) -> str:
    """Which compute path a resolved factor-table row takes."""
    mode = row.get("mode")
    if mode in ("factor", "ecologits"):
        return str(mode)
    if row.get("_overridden") and isinstance(row.get("wh_per_output_token"), dict):
        return "factor"  # an explicit user factor override keeps factor arithmetic
    arch = row.get("arch")
    if isinstance(arch, dict) and arch.get("total_b") and arch.get("active_b"):
        return "ecologits"
    if isinstance(row.get("wh_per_output_token"), dict):
        return "factor"
    return "ecologits"


def _params_from_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """active_b/total_b from a factor row's arch block or top-level params override."""
    params = row.get("params") if isinstance(row.get("params"), dict) else None
    arch = row.get("arch") if isinstance(row.get("arch"), dict) else None
    source_block = params or arch or {}
    total, active = _scalar(source_block.get("total_b")), _scalar(source_block.get("active_b"))
    if not total or not active:
        return None
    return {
        "active_b": active,
        "total_b": total,
        "source": "user-override" if params else str(row.get("basis", "factor_table")),
    }


def resolve_model(model: str) -> dict[str, Any]:
    """Resolve a model id to compute inputs: params (if any), path mode, basis, provenance.

    Resolution order: per-model override entry (data/green_overrides.json — the
    expert-mode card or an applied IKP estimate; works for ANY model id, exposed
    or not) > factor-table row > EcoLogits catalogue > unconfigured. A per-model
    config also carries datacenter scenario overlays the compute path applies.
    """
    resolved: dict[str, Any] = {
        "params": None,
        "mode": "ecologits",
        "basis": "unknown",
        "param_source": None,
        "catalogue": None,
        "config": {},
        "config_scoped": False,
    }
    config = model_config(model)
    resolved["config"] = config
    factors = load_factors()
    key = model_key(model, factors)

    # (1) explicit per-model parameters win over everything
    if config["params"] is not None:
        resolved["params"] = config["params"]
        resolved["param_source"] = config["params"]["source"]
        resolved["basis"] = config["params"]["source"]
        resolved["mode"] = "ecologits"
        scoped = any(config[field] is not None
                     for field in ("pue", "wue_onsite", "country", "electricity"))
        resolved["config_scoped"] = scoped
        # a legacy wh_per_output_token override on a factor-table row keeps factor arithmetic
        if config["entry"].get("wh_per_output_token") is not None and key is not None:
            resolved["mode"] = "factor"
        return resolved

    if key is not None:
        row = (factors.get("models") or {}).get(key, {})
        resolved["basis"] = str(row.get("basis", "unknown"))
        params = _params_from_row(row)
        if params:
            resolved["params"] = params
            resolved["param_source"] = params["source"]
        mode = _row_mode(row)
        if mode == "factor":
            resolved["mode"] = "factor"
        elif params:
            resolved["mode"] = "ecologits"
            resolved["config_scoped"] = any(config[field] is not None
                                            for field in ("pue", "wue_onsite", "country",
                                                          "electricity"))
        else:
            resolved["mode"] = "unconfigured"
        return resolved

    match = catalogue_match(model)
    if match:
        resolved["params"] = {
            "active_b": match["active_b"],
            "total_b": match["total_b"],
            "source": "ecologits_catalogue",
        }
        resolved["param_source"] = "ecologits_catalogue"
        resolved["basis"] = "ecologits_catalogue"
        resolved["catalogue"] = match
        resolved["mode"] = "ecologits"
        resolved["config_scoped"] = any(config[field] is not None
                                        for field in ("pue", "wue_onsite", "country",
                                                      "electricity"))
    else:
        resolved["mode"] = "unconfigured"
    return resolved


def _overlay_settings(green_settings: dict[str, Any], config: dict[str, Any],
                      resolution: dict[str, Any]) -> dict[str, Any]:
    """Apply the per-model datacenter scenario onto a copy of the green settings."""
    settings = dict(green_settings)
    if config.get("country"):
        settings["region_country"] = config["country"]
    if config.get("pue") is not None:
        # a user-supplied PUE is a fixed scenario for that model (bound propagation
        # comes from the grid triple; the demo treats single PUE values the same way)
        for bound in _BOUNDS:
            settings[f"region_pue_{bound}"] = float(config["pue"])
    return settings


def _ecologits_impacts(
    *,
    params: dict[str, Any],
    green_settings: dict[str, Any],
    effective_output: float,
    duration_ms: float | None,
    ttft_ms: float | None,
    tps: float | None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or {}
    grid = {bound: float(green_settings.get(f"region_grid_gco2_{bound}", 0.0)) / 1000.0 for bound in _BOUNDS}
    pue = {bound: float(green_settings.get(f"region_pue_{bound}", 0.0)) for bound in _BOUNDS}
    mix, mix_source = electricity(str(green_settings.get("region_country", "WOR")),
                                  override=config.get("electricity"))

    latency = (duration_ms or 0) / 1000.0
    kwargs = dict(
        model_active_parameter_count=float(params["active_b"]),
        model_total_parameter_count=float(params["total_b"]),
        output_token_count=float(effective_output),
        request_latency=float(latency),
        if_electricity_mix_adpe=mix["adpe"],
        if_electricity_mix_pe=mix["pe"],
        if_electricity_mix_gwp=grid["mean"],
        if_electricity_mix_wue=mix["wue"],
        datacenter_pue=pue["mean"],
        # On-site WUE: 0 unless the per-model card supplies it (the region settings
        # track grid water via if_electricity_mix_wue).
        datacenter_wue=float(config.get("wue_onsite") or 0.0),
    )
    if tps and tps > 0 and ttft_ms is not None and ttft_ms >= 0:
        kwargs.update(tps=float(tps), ttft=float(ttft_ms) / 1000.0)
    elif config.get("tps") and config.get("ttft") is not None:
        # configured averages (expert card), same precedence as the demo's
        # catalogue deployment values — only when the stream did not measure them
        kwargs.update(tps=float(config["tps"]), ttft=float(config["ttft"]))

    from ecologits.impacts.llm import compute_llm_impacts

    energy: dict[str, float] = {}
    gwp: dict[str, float] = {}
    water = adpe = primary = None
    for bound in _BOUNDS:
        kwargs["if_electricity_mix_gwp"] = grid[bound]
        kwargs["datacenter_pue"] = pue[bound]
        impacts = compute_llm_impacts(**kwargs)
        energy[bound] = _value(impacts.energy) or 0.0
        gwp[bound] = _value(impacts.gwp) or 0.0
        if bound == "mean":
            water = _value(impacts.wcf)
            adpe = _value(impacts.adpe)
            primary = _value(impacts.pe)
    if any(v is None for v in (water, adpe, primary)):
        raise ValueError("EcoLogits produced a missing impact value")
    scenario = "per-model scenario" if (config.get("electricity") or config.get("pue") is not None
                                        or config.get("wue_onsite") is not None) else "region scenario"
    return {
        "energy_kwh": energy,
        "gwp_kgco2e": gwp,
        "water_l": water,
        "adpe_kg_sbeq": adpe,
        "primary_mj": primary,
        "provenance_power": f"{mix_source}; PUE/WUE: {scenario}",
    }


def _factor_impacts(
    model: str,
    usage: dict[str, Any] | None,
    green_settings: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Legacy three-bound arithmetic plus grid-mix water/ADPe/PE side outputs."""
    from .green import compute as legacy_compute

    legacy = legacy_compute(model, usage, green_settings)
    if legacy.get("model_basis") == "unknown" or legacy.get("energy_kwh") is None:
        return _missing("unconfigured_model", "No factor table row with parameters", basis="unknown")
    config = config or {}
    mix, mix_source = electricity(str(green_settings.get("region_country", "WOR")),
                                  override=config.get("electricity"))
    energy_mean = float(legacy["energy_kwh"]["mean"])
    gwp_mean = float(legacy["gwp_kgco2e"]["mean"])
    onsite = float(config.get("wue_onsite") or 0.0)
    scenario = ("per-model scenario" if (config.get("electricity") or config.get("pue") is not None
                or config.get("wue_onsite") is not None or config.get("country")) else "region scenario")
    return {
        "energy_kwh": legacy["energy_kwh"],
        "gwp_kgco2e": legacy["gwp_kgco2e"],
        # Approximation: grid-mix coefficients (+ optional on-site WUE) applied to
        # the computed energy — the legacy factor table has no water/ADPe terms.
        "water_l": energy_mean * (mix["wue"] + onsite),
        "adpe_kg_sbeq": energy_mean * mix["adpe"],
        "primary_mj": gwp_mean / max(float(green_settings.get("region_grid_gco2_mean", 1.0)), 1e-9) * mix["pe"],
        "provenance_power": (f"{mix_source}; PUE/WUE: {scenario}; "
                             "water/ADPe/PE from grid mix (approximation)"),
    }


def compute(
    model: str,
    usage: dict[str, Any] | None,
    green_settings: dict[str, Any] | None = None,
    *,
    duration_ms: float | None = None,
    ttft_ms: float | None = None,
    tps: float | None = None,
) -> dict[str, Any]:
    """Full impact payload for one request; never raises; fails closed.

    Anthropic usage semantics: ``input_tokens`` is the fresh portion, cache reads
    reported separately and discounted by ``cached_input_fraction``; cache writes
    are ignored for energy (unchanged from Phase 1).
    """
    from .green import defaults as green_defaults

    settings = {**green_defaults(), **(green_settings or {})}
    usage = usage or {}
    out_tokens = usage.get("output_tokens") or 0
    if out_tokens <= 0 and not (usage.get("input_tokens") or usage.get("cache_read_input_tokens")):
        return _missing("missing_usage", "No reliable token count reported by upstream")

    resolution = resolve_model(model)
    config = resolution.get("config") or {}
    settings = _overlay_settings(settings, config, resolution)
    if resolution["mode"] == "unconfigured" or resolution["params"] is None:
        return _missing(
            "unconfigured_model",
            f"Parameter counts not established for '{model}' "
            "(no factor-table row, no override, no exact EcoLogits catalogue match)",
        )

    effective = _effective_output(usage, settings)
    if effective <= 0:
        return _missing("missing_usage", "Effective token count is zero", basis=resolution["basis"])
    if not duration_ms or duration_ms <= 0:
        return _missing("missing_usage", "No reliable request latency", basis=resolution["basis"])

    try:
        if resolution["mode"] == "factor":
            result = _factor_impacts(model, usage, settings, config)
            basis = resolution["basis"]
            param_source = resolution["basis"]
        else:
            result = _ecologits_impacts(
                params=resolution["params"],
                green_settings=settings,
                effective_output=effective,
                duration_ms=duration_ms,
                ttft_ms=ttft_ms,
                tps=tps,
                config=config,
            )
            basis = resolution["basis"]
            param_source = resolution["param_source"] or "factor_table"
    except Exception as exc:  # noqa: BLE001 - impacts must never break a response
        import logging

        logging.getLogger("token_proxy").error("green impact computation failed: %s", exc)
        return _missing(
            "calculation_error",
            f"{type(exc).__name__}: {exc}"[:230],
            basis=resolution["basis"],
        )

    result["model_basis"] = basis
    result["impact_status"] = "estimated"
    result["impact_provenance"] = f"{param_source}; {result.pop('provenance_power')}"
    return result
