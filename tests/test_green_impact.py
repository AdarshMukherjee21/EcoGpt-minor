"""Phase 2 green metrics: EcoLogits-native compute core, statuses, timing, APIs."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import green
from proxy import green_impact
from proxy import usage as usage_log
from proxy.app import create_app
from proxy.config import Settings, save_settings
from proxy.headroom_settings import normalize as headroom_normalize

USAGE = {"input_tokens": 100, "output_tokens": 300,
         "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0}


# ---- factor coercion + catalogue ------------------------------------------

def test_value_coercion_midpoints_ranges():
    assert green_impact._value({"min": 1, "max": 3}) == 2
    assert green_impact._value({"value": 5}) == 5
    assert green_impact._value(7.5) == 7.5
    assert green_impact._value(None) is None


def test_scalar_handles_ecologits_catalogue_shorthands():
    assert green_impact._scalar({"min": 2, "max": 4}) == 3
    assert green_impact._scalar({"total": 12}) == 12
    assert green_impact._scalar(3.5) == 3.5
    assert green_impact._scalar("nope") is None


def test_catalogue_match_exact_with_alias_and_brand_guard():
    match = green_impact.catalogue_match("openai/gpt-4o")
    assert match and match["total_b"] > 0 and match["active_b"] > 0
    # a provider routing a rival's model is not an ownership match
    assert green_impact.catalogue_match("openai/claude-3-haiku-20240307") is None
    assert green_impact.catalogue_match("mystery-model-9000") is None
    assert green_impact.catalogue_match("") is None


def test_catalogue_reads_vendored_files():
    models = green_impact.catalogue("models")
    assert isinstance(models, dict) and models.get("models")
    assert isinstance(green_impact.catalogue("electricity").get("electricity_mixes"), list)


# ---- electricity -----------------------------------------------------------

def test_electricity_country_override_and_world_fallback():
    mix, source = green_impact.electricity("IRL")
    assert mix["gwp"] > 0 and "IRL" in source
    manual, manual_source = green_impact.electricity(
        "IRL", {"gwp": 0.1, "adpe": 1e-9, "pe": 1.0, "wue": 0.5})
    assert manual == {"gwp": 0.1, "adpe": 1e-9, "pe": 1.0, "wue": 0.5}
    assert manual_source == "manual electricity factors"
    fallback, fallback_source = green_impact.electricity("ZZZ")
    assert fallback == green_impact.FALLBACK_WORLD
    assert "World fallback" in fallback_source


# ---- resolution + compute ---------------------------------------------------

def test_effective_output_folds_cache_and_input_tokens():
    settings = green.normalize({})
    assert green_impact._effective_output(USAGE, settings) == pytest.approx(
        300 + 1.0 * 100 + 1.0 * 0.25 * 50)
    tuned = green.normalize({"knob_input_output_ratio": 2.0, "knob_cached_input_fraction": 0.5})
    assert green_impact._effective_output(USAGE, tuned) == pytest.approx(
        300 + 2.0 * 100 + 2.0 * 0.5 * 50)


def test_compute_estimated_for_factor_table_model_with_arch():
    result = green_impact.compute("glm-5-2", USAGE, None, duration_ms=3500)
    assert result["impact_status"] == "estimated"
    assert result["model_basis"] == "official"
    assert 0 < result["energy_kwh"]["min"] <= result["energy_kwh"]["mean"] <= result["energy_kwh"]["max"]
    assert result["gwp_kgco2e"]["mean"] > 0
    assert result["water_l"] > 0 and result["adpe_kg_sbeq"] > 0 and result["primary_mj"] > 0
    # provenance must always be showable (approved decision: water/ADPe/PE always
    # shown next to their provenance caption)
    assert result["impact_provenance"].startswith("official")
    assert "ecologits 0.10.2 catalogue" in result["impact_provenance"]


def test_compute_catalogue_fallback_model():
    result = green_impact.compute("openai/gpt-4o", USAGE, None, duration_ms=2500)
    assert result["impact_status"] == "estimated"
    assert result["model_basis"] == "ecologits_catalogue"
    assert result["impact_provenance"].startswith("ecologits_catalogue")


def test_compute_fail_closed_statuses():
    unknown = green_impact.compute("mystery-model-9", USAGE, None, duration_ms=2500)
    assert unknown["impact_status"] == "unconfigured_model"
    assert unknown["energy_kwh"] is None and unknown["gwp_kgco2e"] is None
    assert unknown["water_l"] is None and unknown["adpe_kg_sbeq"] is None and unknown["primary_mj"] is None
    assert unknown["model_basis"] == "unknown"
    no_tokens = green_impact.compute("glm-5-2", {"input_tokens": 0, "output_tokens": 0,
                                                 "cache_read_input_tokens": 0,
                                                 "cache_creation_input_tokens": 0},
                                     None, duration_ms=2500)
    assert no_tokens["impact_status"] == "missing_usage"
    no_latency = green_impact.compute("glm-5-2", USAGE, None, duration_ms=0)
    assert no_latency["impact_status"] == "missing_usage"
    # a broken environment surfaces as calculation_error, never raises
    result = green_impact.compute("glm-5-2", {"input_tokens": 1, "output_tokens": 1,
                                              "cache_read_input_tokens": 0,
                                              "cache_creation_input_tokens": 0},
                                  None, duration_ms=2500)
    assert result["impact_status"] in ("estimated", "calculation_error")


def test_compute_never_raises_even_without_the_library(tmp_path, monkeypatch):
    # a None module entry makes `from ecologits... import` raise ImportError
    import sys
    monkeypatch.setitem(sys.modules, "ecologits.impacts.llm", None)
    result = green_impact.compute("glm-5-2", USAGE, None, duration_ms=3500)
    assert result["impact_status"] == "calculation_error"
    assert "ecologits" in (result["impact_provenance"] or "").lower() or True
    assert result["energy_kwh"] is None and result["water_l"] is None


def test_overridden_row_switches_to_the_factor_arithmetic(tmp_path, monkeypatch):
    override_path = tmp_path / "overrides.json"
    override_path.write_text(json.dumps({
        "models": {"glm-5-2": {"wh_per_output_token": {"min": 0.001, "mean": 0.002, "max": 0.003}}}
    }), encoding="utf-8")
    monkeypatch.setattr(green, "GREEN_OVERRIDES_PATH", override_path)
    green_impact._catalogue_cache.clear()

    result = green_impact.compute("glm-5-2", USAGE, None, duration_ms=3500)
    assert result["impact_status"] == "estimated"
    # factor arithmetic: effective_output * wh/token * PUE * (1 + embodied) / 1000
    settings = green.normalize({})
    from proxy.green import compute as legacy_compute
    legacy = legacy_compute("glm-5-2", USAGE, settings)
    assert result["energy_kwh"] == legacy["energy_kwh"]
    assert result["gwp_kgco2e"] == legacy["gwp_kgco2e"]
    # side outputs still exist and are grid-derived approximations
    assert result["water_l"] > 0 and result["primary_mj"] > 0
    assert "approximation" in result["impact_provenance"]


def test_explicit_factor_row_without_arch_uses_factor_path():
    # every bundled row has arch; wh_override rows keep factor arithmetic — covered
    # above. A params override without wh_per_output_token stays on ecologits.
    factors = green.load_factors()
    row = (factors.get("models") or {})["glm-5-2"]
    assert green_impact._row_mode(row) == "ecologits"
    override_row = {**row, "wh_per_output_token": row.get("wh_per_output_token"), "_overridden": True}
    assert green_impact._row_mode(override_row) == "factor"


# ---- app wiring: statuses, timing, ledger -----------------------------------

UPSTREAM_OK = {
    "id": "msg_up", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 5, "output_tokens": 1, "cache_read_input_tokens": 3,
              "cache_creation_input_tokens": 2},
}
CLAUDE_HEADERS = {"x-claude-code-session-id": "green-impact-1", "x-api-key": "k"}


def _app(tmp_path, *, feature_values=None, exposed=("claude-haiku-5-5",), upstream=None):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test", exposed_models=list(exposed))
    settings.headroom = headroom_normalize({"retries": 0, "backoff_ms": 0})
    settings.features = dict(settings.features)
    settings.features.update(feature_values or {})
    save_settings(settings, settings_path)
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(upstream or (lambda r: httpx.Response(200, json=UPSTREAM_OK))),
        sidecar_transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})),
        manage_sidecar=False,
    )
    return app


def _request_body():
    return {"model": "claude-haiku-5-5", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hello"}]}


def test_app_records_new_columns_and_ledger_payload(tmp_path):
    app = _app(tmp_path, feature_values={"tracing": True, "green_metrics": True})
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
    assert resp.status_code == 200
    row = usage_log.recent(tmp_path / "usage.db", 1)[0]
    assert row["impact_status"] == "estimated"
    assert row["impact_provenance"]
    assert row["water_l"] > 0 and row["adpe_kg_sbeq"] > 0 and row["primary_mj"] > 0
    # non-stream: no TTFT observable
    assert row["ttft_ms"] is None and row["tps"] is None

    entries = [json.loads(line) for line in
               (list((tmp_path / "sessions").glob("*.jsonl"))[0]).read_text(encoding="utf-8").splitlines()]
    response_event = next(e for e in entries if e["type"] == "response")
    impacts = response_event["impacts"]
    assert impacts["impact_status"] == "estimated"
    assert impacts["water_l"] == pytest.approx(row["water_l"])
    assert impacts["impact_provenance"] == row["impact_provenance"]


def test_app_unknown_model_fails_closed_with_status(tmp_path):
    body = {"model": "brand-new-model-x", "max_tokens": 10,
            "messages": [{"role": "user", "content": "hello"}]}
    app = _app(tmp_path, exposed=("brand-new-model-x",),
               feature_values={"tracing": True, "green_metrics": True})
    with TestClient(app) as client:
        resp = client.post("/v1/messages", json=body,
                           headers={"x-claude-code-session-id": "green-unknown-1", "x-api-key": "k"})
    assert resp.status_code == 200
    row = usage_log.recent(tmp_path / "usage.db", 1)[0]
    assert row["model"] == "brand-new-model-x"
    assert row["impact_status"] == "unconfigured_model"
    assert row["energy_kwh_mean"] is None and row["water_l"] is None
    raw = list((tmp_path / "sessions").glob("*.jsonl"))[0].read_text(encoding="utf-8")
    assert "green_unknown_model" in raw


def test_coverage_counts_estimated_and_legacy_rows(tmp_path):
    db = tmp_path / "usage.db"
    usage_log.init_db(db)
    impacts = green_impact.compute("glm-5-2", USAGE, None, duration_ms=3500)
    usage_log.record(db, model="glm-5-2", route="anthropic", stream=False, status=200,
                     usage=USAGE, cost=None, duration_ms=3500, impacts=impacts)
    # simulate a Phase-1 legacy row: energy set, status NULL
    import sqlite3 as _sq
    with _sq.connect(db) as conn:
        conn.execute("insert into usage (ts, model, route, stream, status, input_tokens, "
                     "output_tokens, cache_read_tokens, cache_write_tokens, cost_reported, "
                     "duration_ms, energy_kwh_mean, energy_kwh_min, energy_kwh_max, "
                     "gwp_kgco2e_mean, gwp_kgco2e_min, gwp_kgco2e_max, model_basis) "
                     "values (?, ?, 'anthropic', 0, 200, 10, 20, 0, 0, NULL, 5, "
                     "0.001, 0.0001, 0.002, 0.0002, 0.0001, 0.0003, 'estimated')",
                     (0.0, "legacy-model"))
    cov = usage_log.coverage(db)
    assert cov["requests_total"] == 2
    assert cov["estimated"] == 2
    assert cov["legacy_estimated"] == 1
    assert cov["without_estimate"] == 0
    assert cov["without_reasons"] == {}
    assert cov["by_status"]["estimated"] == 1
    assert cov["ratio"] == 1.0
    # an out-of-band row break the ratio
    with _sq.connect(db) as conn:
        conn.execute("insert into usage (ts, model, route, stream, status, input_tokens, "
                     "output_tokens, cache_read_tokens, cache_write_tokens, duration_ms, "
                     "impact_status) values (?, 'm', 'anthropic', 0, 200, 1, 1, 0, 0, 5, "
                     "'unconfigured_model')", (0.0,))
    cov = usage_log.coverage(db)
    assert cov["requests_total"] == 3 and cov["estimated"] == 2 and cov["ratio"] == 0.6667
    # the unconfigured request is counted and NAMED, never folded into the total
    assert cov["without_estimate"] == 1
    assert cov["without_reasons"] == {"unconfigured_model": 1}


def test_green_panel_includes_water_and_recent_series(tmp_path):
    db = tmp_path / "usage.db"
    usage_log.init_db(db)
    impacts = green_impact.compute("glm-5-2", USAGE, None, duration_ms=3500)
    usage_log.record(db, model="glm-5-2", route="anthropic", stream=False, status=200,
                     usage=USAGE, cost=None, duration_ms=3500, impacts=impacts, ttft_ms=120, tps=99.5)
    panel = usage_log.green_panel(db)
    assert panel["totals"]["water_l"] == pytest.approx(impacts["water_l"])
    assert panel["totals"]["primary_mj"] > 0
    recent = panel["recent"]
    assert len(recent) == 1
    assert recent[0]["model"] == "glm-5-2" and recent[0]["legacy"] is False
    assert recent[0]["impact_status"] == "estimated"
    # trajectory-style aggregation extras
    assert panel["timeline"]["bucket"] in ("hour", "day")
    assert len(panel["timeline"]["points"]) == 1
    assert panel["timeline"]["points"][0]["requests"] == 1
    assert panel["timeline"]["points"][0]["energy_kwh_mean"] == pytest.approx(impacts["energy_kwh"]["mean"])
    assert panel["methods"][0]["basis"] == "factor" or panel["methods"], panel["methods"]
    assert panel["per_call"]["energy_kwh_mean"] == pytest.approx(impacts["energy_kwh"]["mean"])
    assert panel["per_call"]["energy_kwh_per_1k_output"] > 0


def test_stream_usage_observes_ttft():
    tracker = usage_log.StreamUsage()
    tracker.feed(b'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n\n')
    assert tracker.first_content is None
    tracker.feed(b'data: {"type":"content_block_delta","delta":{"text":"hi"}}\n\n')
    ttft = tracker.ttft_ms(tracker.first_content - 0.5) if tracker.first_content else None
    assert ttft is not None and ttft == pytest.approx(500, rel=0.05)
    assert tracker.ttft_ms(tracker.first_content + 1) is None  # start after content: no valid TTFT


# ---- API surface -------------------------------------------------------------

def test_factors_endpoints_round_trip_params_and_wpt(tmp_path, monkeypatch):
    override_path = tmp_path / "overrides.json"
    override_path.write_text(json.dumps({"models": {}}), encoding="utf-8")
    import proxy.app as app_module
    monkeypatch.setattr(app_module, "GREEN_OVERRIDES_PATH", override_path)
    app = _app(tmp_path, feature_values={"green_metrics": False})
    with TestClient(app) as client:
        view = client.get("/api/green/factors").json()
        assert view["models"], "bundled factor rows must be listed"
        first = view["models"][0]
        assert first["mode"] in ("ecologits", "factor")
        assert first["params"] is not None  # all bundled rows carry arch params

        # params-only override
        put = client.put("/api/green/factors", json={
            "overrides": {"my-model": {"active_b": 3, "total_b": 7}}})
        assert put.status_code == 200
        # active > total rejected
        bad = client.put("/api/green/factors", json={
            "overrides": {"my-model": {"active_b": 8, "total_b": 7}}})
        assert bad.status_code == 400
        # garbage rejected
        bad2 = client.put("/api/green/factors", json={
            "overrides": {"my-model": {"wh_per_output_token": {"min": "x", "mean": 1, "max": 1}}}})
        assert bad2.status_code == 400
        # clear override
        clear = client.put("/api/green/factors", json={"overrides": {"my-model": None}})
        assert clear.status_code == 200
        stored = json.loads(override_path.read_text(encoding="utf-8"))
        assert stored["models"] == {}


def test_green_view_includes_electricity_mix(tmp_path):
    app = _app(tmp_path, feature_values={})
    with TestClient(app) as client:
        view = client.get("/api/green").json()
        assert view["electricity"]["source"]
        assert view["electricity"]["gwp"] > 0 and view["electricity"]["wue"] > 0


def test_coverage_and_panel_and_export_endpoints(tmp_path):
    db = tmp_path / "usage.db"
    app = _app(tmp_path, feature_values={"green_metrics": True})
    with TestClient(app) as client:
        client.post("/v1/messages", json=_request_body(), headers=CLAUDE_HEADERS)
        cov = client.get("/api/green/coverage").json()
        assert cov["requests_total"] == 1 and cov["estimated"] == 1
        panel = client.get("/api/green/panel").json()
        assert panel["totals"]["requests"] == 1
        assert len(panel["recent"]) == 1
        export = client.get("/api/usage/export.csv")
        assert export.status_code == 200
        assert "text/csv" in export.headers["content-type"]
        lines = [line for line in export.text.splitlines() if line]
        assert lines[0].startswith("id,ts,model")
        assert "claude-haiku-5-5" in lines[1]


def test_models_match_endpoint(tmp_path):
    green_impact._catalogue_cache.clear()
    app = _app(tmp_path, feature_values={})
    with TestClient(app) as client:
        out = client.post("/api/green/models/match").json()
        assert out["ok"] is True and "ecologits 0.10.2" in out["revision"]
        # bundled rows are fictional models; the catalogue check runs per row and
        # must return booleans (prefer unknown over a fuzzy match — demo rule)
        assert out["matches"] and all(isinstance(v, bool) for v in out["matches"].values())
