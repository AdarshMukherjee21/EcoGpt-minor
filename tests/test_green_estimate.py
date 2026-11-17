"""Per-model green profiles + IKP estimation lifecycle (start/pause/fail/continue)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from proxy import green as green_metrics
from proxy import green_estimate, green_impact
from proxy import usage as usage_log
from proxy.app import create_app
from proxy.config import Settings, save_settings
from proxy.headroom_settings import normalize as headroom_normalize

TARGET = "brand-new-free-model:free"
JUDGE = "judge-model"


@pytest.fixture(autouse=True)
def _isolate_override_writes(tmp_path, monkeypatch):
    """Every override write AND read lands in this test's tmp file, never repo data.
    Two names must be patched: proxy.green.GREEN_OVERRIDES_PATH (what the compute
    core reads) and the copy proxy.app imported at module import (what it writes)."""
    path = tmp_path / "green_overrides.json"
    path.write_text(json.dumps({"models": {}}), encoding="utf-8")
    import proxy.app as app_module
    monkeypatch.setattr(green_metrics, "GREEN_OVERRIDES_PATH", path)
    monkeypatch.setattr(app_module, "GREEN_OVERRIDES_PATH", path)


def _chat(content: str, model: str) -> bytes:
    return json.dumps({"id": "chatcmpl-1", "object": "chat.completion", "created": 0,
                       "model": model, "choices": [{"index": 0, "finish_reason": "stop",
                       "message": {"role": "assistant", "content": content}}],
                       "usage": {"prompt_tokens": 5, "completion_tokens": 7}}).encode()


class _Population:
    """Stateful mock gateway: fails while broken=True, then answers normally.
    Counts target-probe calls so tests can prove resume never re-asks old probes."""

    def __init__(self, *, fail_probe_from: int | None = None, slow: float = 0.0):
        self.fail_probe_from = fail_probe_from
        self.slow = slow
        self.probe_calls = 0
        self.judge_calls = 0
        self.broken = False
        self.rate_limited_until_call = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model = body.get("model")
        import time as _time
        if self.slow:
            _time.sleep(self.slow)
        if model == JUDGE:
            self.judge_calls += 1
            return httpx.Response(200, content=_chat("CORRECT", model))
        self.probe_calls += 1
        if self.rate_limited_until_call and self.probe_calls <= self.rate_limited_until_call:
            return httpx.Response(429, headers={"retry-after": "0"}, content=b'{"error":{"message":"slow down"}}')
        if self.broken or (self.fail_probe_from is not None and self.probe_calls >= self.fail_probe_from):
            return httpx.Response(500, content=b'{"error":{"message":"upstream exploded"}}')
        return httpx.Response(200, content=_chat("Paris", model))


def _app(tmp_path, *, population=None, exposed=(TARGET, JUDGE)):
    settings_path = tmp_path / "config.json"
    settings = Settings(upstream_url="http://gateway.test", exposed_models=list(exposed))
    settings.headroom = headroom_normalize({"retries": 0, "backoff_ms": 0})
    save_settings(settings, settings_path)
    transport = httpx.AsyncTransport if False else None
    app = create_app(
        settings_path=settings_path,
        db_path=tmp_path / "usage.db",
        claude_settings_path=tmp_path / "claude.json",
        ledger_dir=tmp_path / "sessions",
        upstream_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        sidecar_transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})),
        manage_sidecar=False,
        estimation_path=tmp_path / "green_estimation.json",
        estimation_transport=httpx.MockTransport(population.handler) if population else None,
    )
    return app, settings_path


def _estimate(client, model=TARGET, **kw):
    body = {"judge": JUDGE, "sample_n": 7, "pace_s": 0.0, "max_tokens": 256}
    body.update(kw)
    return client.post(f"/api/green/models/{model}/estimate", json=body)


def _state(client, model=TARGET):
    return client.get(f"/api/green/models/{model}/estimate").json()


def _wait_terminal(client, model=TARGET, timeout_s: float = 20.0):
    import time
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = _state(client, model)
        if (await_status := state.get("status")) in ("done", "failed", "paused"):
            return state
        time.sleep(0.05)
    return _state(client, model)


# ---- happy path: run to done, then apply -------------------------------------

def test_estimate_full_lifecycle(tmp_path):
    population = _Population()
    app, _ = _app(tmp_path, population=population)
    with TestClient(app) as client:
        response = _estimate(client)
        assert response.status_code == 200
        final = _wait_terminal(client)
    assert final["status"] == "done"
    assert final["complete_n"] == final["sample_n"] == 7
    assert final["accuracy"] == pytest.approx(1.0)
    assert final["effective_b"] > 0
    assert final["per_tier"] and all(v == 1.0 for v in final["per_tier"].values())
    assert population.probe_calls == 7 and population.judge_calls == 7


def test_estimate_state_persists_per_probe_for_resume(tmp_path):
    population = _Population()
    app, _ = _app(tmp_path, population=population)
    with TestClient(app) as client:
        _estimate(client)
        state = _wait_terminal(client)
    # the state file holds every row; resume machinery reads from disk, not memory
    stored = green_estimate.load_state(tmp_path / "green_estimation.json")
    record = stored[TARGET]
    assert record["status"] == "done"
    assert len(record["rows"]) == 7
    assert all(r["verdict"] == "CORRECT" for r in record["rows"])


def test_estimate_start_rejected_when_running_or_bad_shape(tmp_path):
    population = _Population(slow=0.05)
    app, _ = _app(tmp_path, population=population)
    with TestClient(app) as client:
        # shape validation fires even when nothing is running
        assert _estimate(client, sample_n=10).status_code == 400
        assert _estimate(client, sample_n=0).status_code == 400
        assert _estimate(client, judge="  ").status_code == 400
        # start a slow run...
        assert _estimate(client).status_code == 200
        # ...second start while running -> 409
        assert _estimate(client).status_code == 409
        # stop the slow run for cleanliness
        stop = client.post(f"/api/green/models/{TARGET}/estimate/stop")
        assert stop.status_code == 200
        assert stop.json()["estimate"]["status"] == "paused"


def test_estimate_continue_never_reruns_finished_probes(tmp_path):
    population = _Population(fail_probe_from=3)
    app, _ = _app(tmp_path, population=population)
    with TestClient(app) as client:
        assert _estimate(client).status_code == 200
        failed = _wait_terminal(client)
        assert failed["status"] == "failed"
        calls_before = population.probe_calls
        assert failed["complete_n"] < 3  # died mid-run

        population.broken = False
        population.fail_probe_from = None
        continue_response = client.post(f"/api/green/models/{TARGET}/estimate/continue")
        assert continue_response.status_code == 200
        final = _wait_terminal(client)

    assert final["status"] == "done"
    assert final["sample_n"] == 7
    # the three probes already graded/asked were not re-run against the gateway
    assert population.probe_calls - calls_before <= 7 - failed["complete_n"]


def test_estimate_stale_running_reset_to_paused_on_restart(tmp_path):
    app, _ = _app(tmp_path)
    state_path = tmp_path / "green_estimation.json"
    green_estimate.save_state(
        {TARGET: {"status": "running", "step": "probing 2/7", "sample_n": 7,
                  "complete_n": 2, "rows": []}}, state_path)
    with TestClient(app):
        state = green_estimate.load_state(state_path)
    assert state[TARGET]["status"] == "paused"
    assert "continue" in state[TARGET]["step"]


def test_estimate_continue_requires_a_run(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        missing = client.post(f"/api/green/models/{TARGET}/estimate/continue")
        assert missing.status_code == 404
        # done runs cannot continue (nothing to resume)
        state_path = tmp_path / "green_estimation.json"
        green_estimate.save_state({TARGET: {"status": "done", "rows": [],
                                            "effective_b": 1.0, "sample_n": 7, "complete_n": 7}}, state_path)
        done = client.post(f"/api/green/models/{TARGET}/estimate/continue")
        assert done.status_code == 400


def test_estimate_rate_limiting_retries_and_completes(tmp_path):
    population = _Population()
    population.rate_limited_until_call = 2  # first two target calls see 429s
    app, _ = _app(tmp_path, population=population)
    with TestClient(app) as client:
        assert _estimate(client, judge=JUDGE).status_code == 200
        final = _wait_terminal(client)
    assert final["status"] == "done"
    assert population.probe_calls >= 7 + 2  # the 429s did not lose probes


def test_estimate_apply_writes_an_explicit_scenario(tmp_path):
    population = _Population()
    app, _ = _app(tmp_path, population=population)
    with TestClient(app) as client:
        # apply before completion -> 400
        assert client.post(f"/api/green/models/{TARGET}/estimate/apply").status_code == 400
        _estimate(client)
        _wait_terminal(client)
        apply_response = client.post(f"/api/green/models/{TARGET}/estimate/apply")
        assert apply_response.status_code == 200
        profile = client.get(f"/api/green/models/{TARGET}").json()
    assert profile["basis"] == "ikp_estimate"
    assert profile["params"]["active_b"] == pytest.approx(profile["params"]["total_b"])
    overrides = green_metrics.load_overrides()
    assert overrides[TARGET]["basis"] == "ikp_estimate"
    assert "IKP" in overrides[TARGET]["evidence"]
    # the estimate flows into the compute core for this untabled model
    result = green_impact.compute(TARGET, {"input_tokens": 10, "output_tokens": 20,
                                           "cache_read_input_tokens": 0,
                                           "cache_creation_input_tokens": 0},
                                  None, duration_ms=2000)
    assert result["impact_status"] == "estimated"
    assert result["model_basis"] == "ikp_estimate"


# ---- per-model expert profile endpoints --------------------------------------

def test_profile_roster_lists_exposed_and_configured_models(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        roster = client.get("/api/green/models").json()
    models = [m["model"] for m in roster["models"]]
    assert TARGET in models and JUDGE in models
    profile = next(m for m in roster["models"] if m["model"] == TARGET)
    assert profile["exposed"] is True
    assert set(profile["datacenter"]) >= {"country", "pue", "wue_onsite", "electricity",
                                          "electricity_source", "per_model_overrides"}
    # regions dataset fallback for a model without per-model country
    assert profile["datacenter"]["electricity"]["gwp"] > 0
    assert profile["estimate"]["status"] in ("idle",)


def test_profile_put_and_clear_round_trip(tmp_path):
    app, _ = _app(tmp_path)
    with TestClient(app) as client:
        put = client.put(f"/api/green/models/{TARGET}", json={
            "params": {"active_b": 3.5, "total_b": 7.0},
            "tps": 42.0, "ttft": 0.4, "pue": 1.3, "wue_onsite": 0.5,
            "country": "irl",
            "electricity": {"gwp": 0.1, "adpe": 1e-9, "pe": 2.0, "water": 0.7},
        })
        assert put.status_code == 200
        profile = client.get(f"/api/green/models/{TARGET}").json()
        assert profile["basis"] == "user-override"
        assert profile["params"] == {"active_b": 3.5, "total_b": 7.0}
        assert profile["tps"] == 42.0 and profile["ttft"] == 0.4
        dc = profile["datacenter"]
        assert dc["pue"] == 1.3 and dc["wue_onsite"] == 0.5 and dc["country"] == "IRL"
        assert dc["per_model_overrides"] == {"pue": True, "wue_onsite": True,
                                             "country": True, "electricity": True}
        # only valid mix keys stored (the bogus 'water' key was dropped)
        stored_profile = green_metrics.load_overrides()[TARGET]
        assert set(stored_profile["electricity"]) == {"gwp", "adpe", "pe"}
        # validation errors
        assert client.put(f"/api/green/models/{TARGET}", json={
            "params": {"active_b": 8, "total_b": 7}}).status_code == 400
        assert client.put(f"/api/green/models/{TARGET}", json={"pue": 0.5}).status_code == 400
        # partial update keeps the rest
        client.put(f"/api/green/models/{TARGET}", json={"tps": 55.0})
        assert client.get(f"/api/green/models/{TARGET}").json()["params"]["total_b"] == 7.0
        stored = green_metrics.load_overrides()[TARGET]
        assert stored["tps"] == 55.0

        # clearing params restores table/catalogue resolution
        clear = client.put(f"/api/green/models/{TARGET}/params")
        assert clear.status_code == 200
        assert green_impact.model_config(TARGET)["params"] is None


def test_profile_manual_params_feed_the_compute_core(tmp_path):
    client = TestClient(create_app(
        settings_path=_write_cfg(tmp_path),
        db_path=tmp_path / "u.db", claude_settings_path=tmp_path / "c.json",
        ledger_dir=tmp_path / "s", manage_sidecar=False,
        upstream_transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
        sidecar_transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})),
        estimation_path=tmp_path / "ge.json"))
    with client:
        assert client.put(f"/api/green/models/{TARGET}", json={
            "params": {"active_b": 2, "total_b": 2}}).status_code == 200
    result = green_impact.compute(TARGET, {"input_tokens": 10, "output_tokens": 20,
                                           "cache_read_input_tokens": 0,
                                           "cache_creation_input_tokens": 0},
                                  None, duration_ms=2000)
    assert result["impact_status"] == "estimated"
    assert result["model_basis"] == "user-override"
    # per-model country override shows up in provenance
    assert client.put(f"/api/green/models/{TARGET}", json={
        "params": {"active_b": 2, "total_b": 2}, "country": "FRA"}).status_code == 200
    result = green_impact.compute(TARGET, {"input_tokens": 10, "output_tokens": 20,
                                           "cache_read_input_tokens": 0,
                                           "cache_creation_input_tokens": 0},
                                  None, duration_ms=2000)
    assert "FRA" in result["impact_provenance"]


def _write_cfg(tmp_path) -> "object":
    """A settings file with the target exposed (for a second app instance)."""
    settings = Settings(upstream_url="http://gateway.test", exposed_models=[TARGET])
    save_settings(settings, tmp_path / "config.json")
    return tmp_path / "config.json"


# ---- asyncio-level unit checks (no app) ---------------------------------------

def test_summarize_requires_all_probes_graded():
    rows = [{"tier": f"T{i}", "verdict": "CORRECT"} for i in range(1, 8)]
    result = green_estimate.summarize(rows)
    assert result and result["accuracy"] == pytest.approx(1.0) and result["effective_b"] > 0
    rows[3]["verdict"] = None
    assert green_estimate.summarize(rows) is None


def test_judge_prompt_contract():
    prompt = green_estimate._judge_prompt("Capital of Canada?", "Ottawa; ottawa", "ottawa city")
    assert "CORRECT, REFUSAL, or WRONG" in prompt and "Ottawa OR ottawa" in prompt
