import json

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import green
from proxy import usage as usage_log
from proxy.app import create_app
from proxy.config import Settings, save_settings
from proxy.headroom_settings import normalize as headroom_normalize

USAGE = {"input_tokens": 1000, "output_tokens": 200,
         "cache_read_input_tokens": 500, "cache_creation_input_tokens": 999}


def _factors():
    return green.load_factors()


def test_formula_matches_hand_computation_from_config():
    row = _factors()["models"]["glm-5-2"]
    region = _factors()["region"]
    grid = region["grid_gco2_kwh"]["mean"]
    pue = region["pue"]["mean"]
    knobs = _factors()["knobs"]
    # E_gpu = wh_per_output_token x (out + r_in*fresh + r_in*cached_fraction*cached)
    effective = USAGE["output_tokens"] + knobs["input_output_ratio"] * USAGE["input_tokens"] \
        + knobs["input_output_ratio"] * knobs["cached_input_fraction"] * USAGE["cache_read_input_tokens"]
    expected_kwh = row["wh_per_output_token"]["mean"] * effective * pue * (1 + knobs["embodied_share"]) / 1000
    expected_gwp = expected_kwh * grid / 1000

    result = green.compute("glm-5-2", USAGE)
    assert result["model_basis"] == "official"
    assert f"{result['energy_kwh']['mean']:.3g}" == f"{expected_kwh:.3g}"
    assert f"{result['gwp_kgco2e']['mean']:.3g}" == f"{expected_gwp:.3g}"


def test_every_model_mean_reproducible_from_arch_fields():
    for model, row in _factors()["models"].items():
        arch = row["arch"]
        expected = (5.71e-7 * arch["active_b"] + 4.05e-5) * arch["gpus"]
        mean = row["wh_per_output_token"]["mean"]
        assert abs(expected - mean) <= 0.05 * mean, model


def test_interval_bounds_propagate_independently():
    factors = _factors()
    row = factors["models"]["claude-sonnet-5-5"]
    result = green.compute("claude-sonnet-5-5", USAGE)
    for name in ("energy_kwh", "gwp_kgco2e"):
        assert result[name]["min"] < result[name]["mean"] < result[name]["max"], name
    # the max bound uses ONLY max factors: verify by recomputing
    knobs = factors["knobs"]
    effective = USAGE["output_tokens"] + USAGE["input_tokens"] + 0.25 * USAGE["cache_read_input_tokens"]
    energy_max = row["wh_per_output_token"]["max"] * effective \
        * factors["region"]["pue"]["max"] * (1 + knobs["embodied_share"]) / 1000
    assert result["energy_kwh"]["max"] == pytest.approx(energy_max)
    gwp_max = energy_max * factors["region"]["grid_gco2_kwh"]["max"] / 1000
    assert result["gwp_kgco2e"]["max"] == pytest.approx(gwp_max)


def test_dated_snapshot_slugs_resolve_and_unknown_models_do_not_raise():
    assert green.model_key("claude-opus-5-5-20260301") == "claude-opus-5-5"
    assert green.model_key("claude-haiku-5-5") == "claude-haiku-5-5"
    assert green.model_key("mystery-model-9") is None
    assert green.model_key("") is None
    result = green.compute("mystery-model-9", USAGE)
    assert result == {"energy_kwh": None, "gwp_kgco2e": None, "model_basis": "unknown"}


def test_cache_accounting_cached_discounted_and_cache_write_ignored():
    base = {"input_tokens": 1000, "output_tokens": 100, "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0}
    with_cache_write = {**base, "cache_creation_input_tokens": 5000}
    assert green.compute("glm-5-2", with_cache_write) == green.compute("glm-5-2", base)
    # 500 cached tokens charged as 0.25 * 500 = 125 fresh tokens
    cached = {**base, "cache_read_input_tokens": 500}
    shifted = {**base, "input_tokens": 1125}
    assert green.compute("glm-5-2", cached) == green.compute("glm-5-2", shifted)


def test_green_settings_normalize_and_defaults_from_file():
    defaults = green.normalize({})
    assert defaults["region_grid_gco2_mean"] == _factors()["region"]["grid_gco2_kwh"]["mean"]
    assert defaults["knob_cached_input_fraction"] == _factors()["knobs"]["cached_input_fraction"]
    with pytest.raises(ValueError):
        green.normalize({"not_a_green_key": 1})
    with pytest.raises(ValueError):
        green.normalize({"knob_embodied_share": 2.0})  # > max 1
    # a knob change flows into the computation
    tuned = green.normalize({"knob_embodied_share": 0.5})
    default_result = green.compute("glm-5-2", USAGE)
    tuned_result = green.compute("glm-5-2", USAGE, tuned)
    assert tuned_result["energy_kwh"]["mean"] > default_result["energy_kwh"]["mean"]


def test_factors_table_fails_open_when_file_is_missing(tmp_path, monkeypatch):
    result = green.compute("glm-5-2", USAGE, path=tmp_path / "nope.json")
    # fallback has no models -> unknown, never a crash
    assert result["model_basis"] == "unknown"
    assert green.model_key("glm-5-2", green.load_factors(tmp_path / "nope.json")) is None
    assert green.attribution(tmp_path / "nope.json")  # fallback attribution strings exist


def test_record_with_and_without_impacts(tmp_path):
    path = tmp_path / "usage.db"
    usage_log.init_db(path)
    impacts = green.compute("glm-5-2", USAGE)
    usage_log.record(path, model="glm-5-2", route="anthropic", stream=False, status=200,
                     usage=USAGE, cost=None, duration_ms=5, impacts=impacts)
    usage_log.record(path, model="glm-5-2", route="anthropic", stream=False, status=200,
                     usage=USAGE, cost=None, duration_ms=5)  # flag off: no impacts
    usage_log.record(path, model="mystery", route="anthropic", stream=False, status=200,
                     usage=USAGE, cost=None, duration_ms=5,
                     impacts={"energy_kwh": None, "gwp_kgco2e": None, "model_basis": "unknown"})
    rows = usage_log.recent(path, 3)
    assert rows[2]["energy_kwh_mean"] is not None
    assert rows[2]["model_basis"] == "official"
    assert rows[1]["energy_kwh_mean"] is None          # old-style row reads as null
    assert rows[1]["model_basis"] is None
    assert rows[0]["energy_kwh_mean"] is None         # unknown model: nulls, watchable basis
    assert rows[0]["model_basis"] == "unknown"
    panel = usage_log.green_panel(path)
    assert panel["totals"]["requests"] == 1
    assert len(panel["by_model"]) == 1
    assert panel["by_model"][0]["model"] == "glm-5-2"
    assert panel["by_day"][0]["requests"] == 1


# ---- app wiring ------------------------------------------------------------

UPSTREAM_OK = {
    "id": "msg_up", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 3,
              "cache_creation_input_tokens": 2},
}
CLAUDE_HEADERS = {"x-claude-code-session-id": "green-sess-1", "x-api-key": "k"}


def _app(tmp_path, *, feature_values=None, exposed=("claude-haiku-5-5",), upstream=None):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test", exposed_models=list(exposed))
    settings.headroom = headroom_normalize({"retries": 0, "backoff_ms": 0})
    settings.features = dict(settings.features)
    settings.features.update(feature_values or {})
    save_settings(settings, settings_path)

    def default_upstream(request):
        return httpx.Response(200, json=UPSTREAM_OK)

    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(upstream or default_upstream),
        sidecar_transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})),
        manage_sidecar=False,
    )
    return app


def _request_body():
    return {"model": "claude-haiku-5-5", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hello"}]}


def test_app_writes_impacts_on_row_and_ledger_when_flag_on(tmp_path):
    db = tmp_path / "usage.db"
    app = _app(tmp_path, feature_values={"tracing": True, "green_metrics": True, "run_label": "green-A"})
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 200

    row = usage_log.recent(db, 1)[0]
    assert row["energy_kwh_mean"] is not None
    assert row["gwp_kgco2e_mean"] is not None
    assert row["model_basis"] == "estimated"  # compression failed open; usage relayed fine

    ledger_file = list((tmp_path / "sessions").glob("*.jsonl"))[0]
    entries = [json.loads(l) for l in ledger_file.read_text(encoding="utf-8").strip().splitlines()]
    response = [e for e in entries if e["type"] == "response"][0]
    assert response["impacts"]["energy_kwh"]["mean"] == pytest.approx(row["energy_kwh_mean"])
    assert response["impacts"]["model_basis"] == "estimated"
    assert response["run_label"] == "green-A"


def test_app_flag_off_writes_no_impacts_anywhere(tmp_path):
    db = tmp_path / "usage.db"
    app = _app(tmp_path, feature_values={"tracing": True, "green_metrics": False})
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 200
    row = usage_log.recent(db, 1)[0]
    assert row["energy_kwh_mean"] is None and row["model_basis"] is None
    ledger_file = list((tmp_path / "sessions").glob("*.jsonl"))[0]
    raw = ledger_file.read_text(encoding="utf-8")
    assert "impacts" not in raw


def test_unknown_model_gets_null_impacts_and_one_ledger_event_per_session(tmp_path):
    app = _app(tmp_path, feature_values={"tracing": True, "green_metrics": True},
               exposed=["mystery-model-9"])
    with TestClient(app) as client:
        body = {**_request_body(), "model": "mystery-model-9"}
        assert client.post("/v1/messages", json=body, headers=CLAUDE_HEADERS).status_code == 200
        assert client.post("/v1/messages", json=body, headers=CLAUDE_HEADERS).status_code == 200
        # a different session logs its own single event
        assert client.post("/v1/messages", json=body,
                           headers={"x-claude-code-session-id": "green-sess-2", "x-api-key": "k"}).status_code == 200
    events = []
    for f in (tmp_path / "sessions").glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").strip().splitlines():
            entry = json.loads(line)
            if entry.get("type") == "error" and entry.get("stage") == "green_unknown_model":
                events.append(entry)
    assert len(events) == 2  # one per session, deduped across requests
    assert all(e["stage"] == "green_unknown_model" for e in events)
    rows = usage_log.recent(tmp_path / "usage.db", 3)
    assert all(r["model_basis"] == "unknown" and r["energy_kwh_mean"] is None for r in rows)


def test_green_exception_fails_open(tmp_path, monkeypatch):
    app = _app(tmp_path, feature_values={"tracing": True, "green_metrics": True})

    def boom(*args, **kwargs):
        raise RuntimeError("green exploded")

    monkeypatch.setattr("proxy.green.compute", boom)
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 200  # the response proceeds without impacts
    row = usage_log.recent(tmp_path / "usage.db", 1)[0]
    assert row["energy_kwh_mean"] is None
    ledger_file = list((tmp_path / "sessions").glob("*.jsonl"))[0]
    raw = ledger_file.read_text(encoding="utf-8")
    assert "green exploded" in raw  # error event landed on the ledger


def test_green_settings_endpoints(tmp_path):
    db = tmp_path / "usage.db"
    app = _app(tmp_path, feature_values={"tracing": True, "green_metrics": True})
    with TestClient(app) as client:
        view = client.get("/api/green").json()
        assert {e["key"] for e in view["schema"]} == {e["key"] for e in green.GREEN_SCHEMA}
        assert view["values"]["region_grid_gco2_mean"] == green.normalize({})["region_grid_gco2_mean"]
        assert any("EcoLogits" in a for a in view["attribution"])

        put = client.put("/api/green", json={"values": {"region_grid_gco2_mean": 300}})
        assert put.status_code == 200
        assert client.get("/api/green").json()["values"]["region_grid_gco2_mean"] == 300
        bad = client.put("/api/green", json={"values": {"not_a_green_setting": 1}})
        assert bad.status_code == 400
        panel = client.get("/api/green/panel").json()
        assert panel["totals"].get("requests", 0) == 0 and panel["by_model"] == []

        # the updated grid intensity flows into newly computed impacts
        client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
        row = usage_log.recent(db, 1)[0]
        expected = green.compute("claude-haiku-5-5", UPSTREAM_OK["usage"],
                                 {"region_grid_gco2_mean": 300})
        assert row["gwp_kgco2e_mean"] == pytest.approx(expected["gwp_kgco2e"]["mean"])
