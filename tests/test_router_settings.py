"""Router settings schema + validation (proxy/router_settings.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from proxy import router_settings as rs


# --- defaults ---------------------------------------------------------------

def test_defaults_two_tier_coding_profile():
    tiers = rs.default_tiers()
    ids = [t["id"] for t in tiers]
    assert ids == ["goldfish", "elephant"]
    # Both shipped descriptions must be the owner-approved prompt-kind texts.
    assert "single-file edits" in tiers[0]["description"]
    assert "no design decisions" in tiers[0]["description"]  # executor tier marker
    assert "architecture decisions" in tiers[1]["description"]
    # Models ship empty: setup required before the auto model is offered.
    assert all(t["model"] == "" for t in tiers)


def test_defaults_keys_and_values():
    values = rs.defaults()
    assert values["enabled"] is True
    assert values["warmth_enabled"] is True                    # ON by default (owner)
    assert values["warmth_window_seconds"] == 120.0
    assert values["stickiness_window_seconds"] == 300.0
    assert values["confidence_floor"] == 0.45
    assert values["long_context_token_threshold"] == 256_000
    assert values["min_tier_for_long_context"] == 1
    assert values["allow_oversized_context"] is False          # experimental, off
    assert values["on_classify_failure"] == "planner"          # owner-confirmed
    assert values["auto_model_id"] == "eco-auto"
    assert values["auth_scheme"] in ("none", "bearer", "x-api-key")
    # Generic SystemOne endpoint: a full URL, not a backend name
    assert "://127.0.0.1:11434/" in values["endpoint_url"]


# --- normalize ---------------------------------------------------------------

def test_normalize_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown router setting"):
        rs.normalize({"nonsense": 1})


def test_normalize_coercion():
    values = rs.normalize({"warmth_window_seconds": "60", "confidence_floor": "0.6",
                           "classify_timeout_ms": 2000, "auth_scheme": "bearer"})
    assert values["warmth_window_seconds"] == 60.0
    assert values["confidence_floor"] == 0.6
    assert values["classify_timeout_ms"] == 2000


def test_normalize_choice_rejection():
    with pytest.raises(ValueError, match="auth_scheme"):
        rs.normalize({"auth_scheme": "cookie"})


def test_normalize_min_max():
    with pytest.raises(ValueError, match="confidence_floor"):
        rs.normalize({"confidence_floor": 1.5})
    with pytest.raises(ValueError):
        rs.normalize({"classify_timeout_ms": 10})


# --- tiers -------------------------------------------------------------------

def test_normalize_tiers_happy_path():
    tiers = rs.normalize_tiers([
        {"id": "goldfish", "description": "execution", "model": "m-small"},
        {"id": "elephant", "description": "planning", "model": "m-big"},
    ])
    assert [t["id"] for t in tiers] == ["goldfish", "elephant"]
    assert tiers[0]["model"] == "m-small"


def test_normalize_tiers_rejects_single_tier():
    with pytest.raises(ValueError, match="at least 2"):
        rs.normalize_tiers([{"id": "solo", "description": "x", "model": "m"}])


def test_normalize_tiers_rejects_over_26():
    tiers = [{"id": f"t{i}", "description": "x", "model": f"m{i}"} for i in range(27)]
    with pytest.raises(ValueError, match="at most 26"):
        rs.normalize_tiers(tiers)


def test_normalize_tiers_rejects_duplicates_and_bad_ids():
    dup = [{"id": "t1", "description": "x", "model": "a"},
           {"id": "T1", "description": "y", "model": "b"}]
    with pytest.raises(ValueError, match="twice"):
        rs.normalize_tiers(dup)
    with pytest.raises(ValueError, match="tiers\\[0\\].id"):
        rs.normalize_tiers([{"id": "bad id!", "description": "x", "model": "a"},
                            {"id": "t2", "description": "y", "model": "b"}])


def test_normalize_tiers_requires_description():
    with pytest.raises(ValueError, match="description is required"):
        rs.normalize_tiers([{"id": "t1", "description": " ", "model": "a"},
                            {"id": "t2", "description": "y", "model": "b"}])


def test_validate_tiers_reports_missing_models_without_raising():
    tiers = rs.default_tiers()
    problems = rs.validate_tiers(tiers)
    assert len(problems) == 2
    assert "'goldfish' has no model" in problems[0]
    tiers[0]["model"] = "m-small"
    problems = rs.validate_tiers(tiers)
    assert len(problems) == 1
    assert "'elephant'" in problems[0]


def test_block_is_valid_complete_ladder():
    values = rs.defaults()
    values["auto_model_id"] = "eco-auto"
    tiers = [{"id": "goldfish", "description": "d", "model": "g-1"},
             {"id": "elephant", "description": "d", "model": "e-1"}]
    assert rs.block_is_valid(values, tiers) == []
    values["auth_scheme"] = "bearer"
    assert "no API key" in rs.block_is_valid(values, tiers)[0]
    values["api_key"] = "sk-test"
    assert rs.block_is_valid(values, tiers) == []
    values["endpoint_url"] = "not a url"
    assert "well-formed" in rs.block_is_valid(values, tiers)[0]


# --- loaders ----------------------------------------------------------------

def test_load_router_missing_file_fails_open(tmp_path: Path):
    assert rs.load_router(tmp_path / "absent.json") == rs.defaults()


def test_load_router_partial_config_merges_defaults(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"router": {"confidence_floor": 0.7}}), encoding="utf-8")
    values = rs.load_router(config)
    assert values["confidence_floor"] == 0.7
    assert values["warmth_enabled"] is True
    assert values["on_classify_failure"] == "planner"


def test_load_router_corrupt_fails_open(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text("{not json", encoding="utf-8")
    assert rs.load_router(config) == rs.defaults()
    assert rs.load_tiers(config) == rs.default_tiers()


def test_load_tiers_round_trip_through_save_shape(tmp_path: Path):
    """Tiers live nested inside the on-disk router block (data/config.json shape)."""
    from proxy.config import Settings, save_settings

    settings = Settings()
    settings.router_tiers = rs.normalize_tiers([
        {"id": "goldfish", "description": "execution", "model": "g-1"},
        {"id": "elephant", "description": "planning", "model": "e-1"},
    ])
    save_settings(settings, tmp_path / "config.json")
    on_disk = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert on_disk["router"]["tiers"][0]["id"] == "goldfish"
    assert "api_key" in on_disk["router"]  # classifier key lives server-side
    assert rs.load_tiers(tmp_path / "config.json")[0]["model"] == "g-1"
    assert rs.load_router(tmp_path / "config.json")["enabled"] is True


def test_load_tiers_invalid_falls_back(tmp_path: Path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"router": {"tiers": [{"id": "solo"}]}}), encoding="utf-8")
    assert rs.load_tiers(config) == rs.default_tiers()


# --- endpoint locality / presets ----------------------------------------------

def test_is_local_endpoint():
    assert rs.is_local_endpoint("http://127.0.0.1:11434/v1/systemone")
    assert rs.is_local_endpoint("http://localhost:8080/x")
    assert not rs.is_local_endpoint("https://openrouter.ai/api/alpha/decisions")
    assert not rs.is_local_endpoint("http://192.168.1.10:8000/v1/systemone")  # LAN: treat as remote


def test_presets_are_fill_in_only():
    assert rs.PRESETS["ollama"]["auth_scheme"] == "none"
    assert rs.PRESETS["openrouter"]["endpoint_url"] == "https://openrouter.ai/api/alpha/decisions"
    # Presets must remain suggested URL schemes, not hardcoded backends: the
    # schema has no backend enum at all.
    assert "backend" not in {e["key"] for e in rs.SCHEMA}


def test_is_auto_model():
    values = rs.defaults()
    assert rs.is_auto_model(values, "eco-auto")
    assert not rs.is_auto_model(values, "claude-sonnet-5-5")
    assert not rs.is_auto_model(values, "")
    values["auto_model_id"] = "route-magic"
    assert rs.is_auto_model(values, "route-magic")
