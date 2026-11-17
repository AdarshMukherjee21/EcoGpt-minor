"""Ledger schema v2: timeline spans, observation marks, rollups.

Trajectory-borrowed honesty rule exercised throughout: a mark that was never
observed is absent or null, never 0. All writers remain fail-open and
hash-only (payloads never stored).
"""

import json

from proxy import ledger


def _write_request(tmp_path, session_key="claude:v2:f1", request_id="req-1", **kw):
    return ledger.request_event(
        tmp_path, session_key=session_key, session_source="claude_header",
        request_id=request_id, run_label="run-v2",
        requested_model="claude-haiku-5-5", model="claude-haiku-5-5",
        route="anthropic", stream=False,
        body={"messages": [{"role": "user", "content": "v2-payload-text"}]},
        **kw,
    )


# ---- envelope additions ------------------------------------------------------


def test_request_event_carries_v2_envelope(tmp_path):
    entry = _write_request(tmp_path)
    assert entry["schema_version"] == ledger.SCHEMA_VERSION == 2
    assert entry["span_id"]
    assert entry["t0_offset_ms"] == 0  # the request is the timeline anchor
    assert len({entry["request_id"]}) == 1
    # span ids are unique per entry
    second = _write_request(tmp_path)
    assert second["span_id"] != entry["span_id"]


def test_stage_and_error_offsets_included_when_measured(tmp_path):
    stage = ledger.stage_event(
        tmp_path, session_key="s", session_source=None, request_id="r",
        run_label="", feature="compression",
        provenance={"duration_ms": 5}, t0_offset_ms=12.6,
    )
    assert stage["t0_offset_ms"] == 12
    assert stage["span_id"]

    error = ledger.error_event(
        tmp_path, session_key="s", session_source=None, request_id="r",
        run_label="", stage="router", message="bad", t0_offset_ms=30.0,
    )
    assert error["t0_offset_ms"] == 30


def test_unmeasured_marks_are_absent_not_zero(tmp_path):
    stage = ledger.stage_event(
        tmp_path, session_key="s", session_source=None, request_id="r",
        run_label="", feature="compression", provenance={},
    )
    assert "t0_offset_ms" not in stage  # never observed -> absent, never 0

    response = ledger.response_event(
        tmp_path, session_key="s", session_source=None, request_id="r",
        run_label="", status=200, duration_ms=10,
    )
    for absent in ("model_served", "route", "ttft_ms", "tps", "chunk_count", "t0_offset_ms"):
        assert absent not in response, absent


def test_response_v2_observations(tmp_path):
    response = ledger.response_event(
        tmp_path, session_key="s", session_source=None, request_id="r",
        run_label="", status=200, duration_ms=100.0,
        usage={"input_tokens": 1, "output_tokens": 2},
        model_served="upstream-a", route="chat",
        ttft_ms=123.4567, tps=9.87654, chunk_count=42, t0_offset_ms=99.9,
    )
    assert response["model_served"] == "upstream-a"
    assert response["route"] == "chat"
    assert response["ttft_ms"] == 123.457
    assert response["tps"] == 9.877
    assert response["chunk_count"] == 42
    assert response["t0_offset_ms"] == 99


def test_v1_files_still_read(tmp_path):
    """Old v1-shaped entries (no schema_version/span/offsets) keep working."""
    path = ledger.session_path(tmp_path, "legacy:s1")
    v1_lines = [
        {"type": "request", "ts": "2026-01-01T00:00:00+00:00", "run_label": "old",
         "session_key": "legacy:s1", "session_source": "claude", "request_id": "r1",
         "requested_model": "m", "model": "m", "route": "anthropic", "stream": False,
         "message_count": 1, "messages": [{"role": "user", "block_counts": {"text": 1},
                                            "chars": 5, "hash": "aa"}],
         "system": None, "tools": {"names": [], "count": 0, "hash": "bb"}},
        {"type": "stage", "ts": "2026-01-01T00:00:01+00:00", "run_label": "old",
         "session_key": "legacy:s1", "session_source": "claude", "request_id": "r1",
         "feature": "compression", "provenance": {"tokens_before": 10, "tokens_after": 5,
                                                  "tokens_saved": 5, "duration_ms": 4}},
        {"type": "response", "ts": "2026-01-01T00:00:02+00:00", "run_label": "old",
         "session_key": "legacy:s1", "session_source": "claude", "request_id": "r1",
         "status": 200, "duration_ms": 900, "usage": {"input": 1, "output": 2,
                                                      "cache_read": 0, "cache_write": 0},
         "stop_reason": "end_turn", "content": None},
    ]
    path.write_text("\n".join(json.dumps(l) for l in v1_lines) + "\n", encoding="utf-8")

    detail = ledger.read_session(tmp_path, path.stem)
    assert len(detail["round_trips"]) == 1
    trip = detail["round_trips"][0]
    assert trip["stages"][0]["provenance"]["tokens_saved"] == 5
    summary = ledger.summarize(detail["entries"])
    assert summary["requests"] == 1
    assert summary["compression"]["tokens_saved"] == 5
    assert summary["timing"] == {"ttft_p50_ms": None, "responses_with_ttft": 0,
                                 "responses_total": 1}


# ---- summarize rollups (v2) --------------------------------------------------


def test_summarize_router_and_timing_rollups(tmp_path):
    common = dict(session_key="s", session_source=None, run_label="run", ledger_dir=tmp_path)
    ledger.request_event(request_id="a", requested_model="m", model="m",
                         route="anthropic", stream=False,
                         body={"messages": []}, **common)
    ledger.stage_event(feature="router", request_id="a",
                       provenance={"final_tier": "tier1", "target_model": "m1",
                                   "routing_reason": "direct_classification",
                                   "confidence": 0.8, "classify_latency_ms": 20.0}, **common)
    ledger.stage_event(feature="router", request_id="a",
                       provenance={"final_tier": "tier1", "target_model": "m1",
                                   "routing_reason": "direct_classification",
                                   "confidence": 0.6, "classify_latency_ms": 40.0}, **common)
    ledger.response_event(request_id="a", status=200, duration_ms=100, usage=None,
                          ttft_ms=250.0, **common)
    ledger.response_event(request_id="a", status=200, duration_ms=100, usage=None,
                          ttft_ms=150.0, **common)
    # a response that never observed TTFT counts as missing, not zero
    ledger.response_event(request_id="a", status=200, duration_ms=100, usage=None, **common)

    detail = ledger.read_session(tmp_path, ledger.session_path(tmp_path, "s").stem)
    summary = ledger.summarize(detail["entries"])

    assert summary["timing"] == {"ttft_p50_ms": 150.0, "responses_with_ttft": 2,
                                 "responses_total": 3}  # nearest-rank p50 of [150, 250]
    router = summary["router"]
    assert router["decisions"] == 2
    assert router["classify_latency_ms_p50"] == 20.0  # nearest-rank p50 of [20, 40]
    assert router["classify_latency_ms_mean"] == 30.0
    assert router["confidence_mean"] == 0.7


def test_wrapping_is_hash_only(tmp_path):
    # new metadata fields never carry payload text
    _write_request(tmp_path)
    ledger.response_event(
        tmp_path, session_key="claude:v2:f1", session_source=None, request_id="req-1",
        run_label="run-v2", status=200, duration_ms=1, model_served="m", ttft_ms=1.0,
    )
    raw = ledger.session_path(tmp_path, "claude:v2:f1").read_text(encoding="utf-8")
    assert "v2-payload-text" not in raw
