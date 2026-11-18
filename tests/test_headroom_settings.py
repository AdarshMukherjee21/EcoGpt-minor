from pathlib import Path

import headroom
import pytest

from proxy import headroom_settings as hs


def _installed_source() -> str:
    root = Path(headroom.__file__).parent
    return "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in root.rglob("*.py"))


def test_every_env_name_exists_in_installed_headroom():
    source = _installed_source()
    for entry in hs.SCHEMA:
        if entry["kind"] == "env" and entry["key"] != "HF_HUB_OFFLINE":
            assert entry["key"] in source, f"{entry['key']} not found in headroom-ai"


def test_defaults_round_trip_through_normalize():
    defaults = hs.defaults()
    assert hs.normalize(defaults) == defaults


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        hs.normalize({"retries": -1})
    with pytest.raises(ValueError):
        hs.normalize({"HEADROOM_MODE": "turbo"})
    with pytest.raises(ValueError):
        hs.normalize({"not_a_setting": 1})


def test_sidecar_env_maps_bools_and_skips_empty():
    values = hs.normalize({"HEADROOM_OFFLINE": True, "HEADROOM_TOOL_DESC_MAX_CHARS": 200})
    env = hs.sidecar_env(values)
    assert env["HEADROOM_OFFLINE"] == "1"
    assert env["HEADROOM_TOOL_DESC_MAX_CHARS"] == "200"
    assert env["HEADROOM_BEACON"] == "off"
    assert "HEADROOM_PROTECT_TOOL_RESULTS" not in env


def test_request_config_carries_session_and_mode():
    config = hs.request_config(hs.normalize({"config_mode": "lossless", "target_ratio": 0.5}), "claude:s:f")
    assert config["session_id"] == "claude:s:f"
    assert config["mode"] == "lossless"
    assert config["target_ratio"] == 0.5


def test_unset_mode_is_omitted_from_request():
    config = hs.request_config(hs.normalize({"config_mode": "unset"}), "claude:s:f")
    assert "mode" not in config


def test_gateway_block_never_requests_redrive():
    block = hs.gateway_block(hs.normalize({"can_redrive": True}))
    assert block["can_redrive"] is False


def test_blank_required_numeric_fields_snap_to_defaults():
    """Regression: blanking a required numeric field in the dashboard sent ""

    through config.json as null — daemon then failed to boot
    (int(None) on sidecar_port) and every sessioned request 500'd
    (int(None) on request_timeout_ms). Blank inputs must snap to defaults.
    """
    values = dict(hs.defaults())
    for key in ("sidecar_port", "retries", "request_timeout_ms", "backoff_ms",
                "token_budget", "protect_recent"):
        values[key] = ""
    normalized = hs.normalize(values)
    assert normalized["sidecar_port"] == 8787
    assert normalized["retries"] == 2
    assert normalized["request_timeout_ms"] == 10000
    assert normalized["backoff_ms"] == 200
    assert normalized["token_budget"] == 1000000
    assert normalized["protect_recent"] == 0


def test_blank_optional_numeric_stays_none():
    """Entries whose default IS None (optional) keep null on blank input."""
    normalized = hs.normalize({"HEADROOM_TOOL_DESC_MAX_CHARS": ""})
    assert normalized["HEADROOM_TOOL_DESC_MAX_CHARS"] is None
