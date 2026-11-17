"""Pure trace-view model tests + the read-at-10k-entries benchmark decision."""

import json
import time

import pytest

from proxy import ledger
from proxy import trace_view


def _trip_entries(offsets=(0, 2, 130), request_id="r1"):
    """(request, compression stage, response) with v2 offsets."""
    request = {"type": "request", "ts": "2026-01-01T00:00:00+00:00", "schema_version": 2,
               "run_label": "", "session_key": "s", "session_source": None,
               "request_id": request_id, "span_id": "a", "t0_offset_ms": offsets[0],
               "requested_model": "m", "model": "m", "route": "anthropic",
               "stream": False, "message_count": 1, "messages": [],
               "system": None, "tools": {}}
    stage = {"type": "stage", "ts": "2026-01-01T00:00:00.002+00:00", "schema_version": 2,
             "run_label": "", "session_key": "s", "session_source": None,
             "request_id": request_id, "span_id": "b", "t0_offset_ms": offsets[1],
             "feature": "compression",
             "provenance": {"tokens_before": 10, "tokens_after": 5, "tokens_saved": 5,
                            "duration_ms": 4, "mode": "ml", "roundtrip_ms": 0.7,
                            "fallback": None}}
    response = {"type": "response", "ts": "2026-01-01T00:00:00.13+00:00",
                "schema_version": 2, "run_label": "", "session_key": "s",
                "session_source": None, "request_id": request_id, "span_id": "c",
                "t0_offset_ms": offsets[2], "status": 200, "duration_ms": 120.0,
                "usage": {"input": 5, "output": 1, "cache_read": 0, "cache_write": 0},
                "stop_reason": "end_turn", "content": None,
                "model_served": "upstream-a", "route": "anthropic", "ttft_ms": 40.0}
    return [request, stage, response]


def test_build_spans_orders_and_derives_upstream_span():
    spans = trace_view.build_spans(_trip_entries())
    by_stage = {s["stage"]: s for s in spans}
    assert by_stage["request"]["start_ms"] == 0
    assert by_stage["compression"]["start_ms"] == 2
    assert by_stage["compression"]["end_ms"] == 6
    assert by_stage["compression"]["detail"]["mode"] == "ml"
    # upstream span derived from the response's completion offset and duration
    assert by_stage["upstream"]["start_ms"] == 10
    assert by_stage["upstream"]["end_ms"] == 130
    assert by_stage["upstream"]["detail"]["model_served"] == "upstream-a"


def test_build_spans_v1_entries_without_offsets_stay_flat():
    old = [{"type": "request", "ts": "2026-01-01T00:00:00+00:00", "run_label": "",
            "session_key": "s", "session_source": None, "request_id": "r1",
            "requested_model": "m", "model": "m", "route": "anthropic",
            "stream": False, "message_count": 1, "messages": [], "system": None,
            "tools": {}}]
    spans = trace_view.build_spans(old)
    assert spans[0]["stage"] == "request"
    # request anchor falls back to 0; no invented timings
    assert spans[0]["start_ms"] == 0


def test_session_timeline_gaps_and_steps():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        c = dict(session_key="claude:g1", session_source="claude", run_label="g",
                 ledger_dir=Path(tmp))
        ledger.request_event(request_id="r1", requested_model="m", model="m",
                             route="anthropic", stream=False, body={"messages": []}, **c)
        ledger.response_event(request_id="r1", status=200, duration_ms=100.0,
                              ttft_ms=50.0, t0_offset_ms=110.0, **c)
        ledger.request_event(request_id="r2", requested_model="m", model="m",
                             route="anthropic", stream=False, body={"messages": []}, **c)
        ledger.response_event(request_id="r2", status=200, duration_ms=100.0,
                              t0_offset_ms=400.0, **c)
        key = ledger.session_path(Path(tmp), "claude:g1").stem
        detail = ledger.read_session(Path(tmp), key)
        timeline = trace_view.session_timeline(detail, key)

    assert len(timeline["steps"]) == 2
    first, second = timeline["steps"]
    assert first["router"] is None and first["compression"] is None
    assert second["ttft_ms"] if False else second.get("ttft_ms") is None
    # gap = wall clock between consecutive responses, upper bound, in seconds->ms round
    assert second["gap_since_previous_ms"] is not None
    assert second["gap_since_previous_ms"] >= 0
    assert "spans" in first and "spans" in second


@pytest.mark.parametrize("count", [10_000])
def test_read_session_benchmark_10k_entries(count):
    """Benchmark: full-file read_session at 10k entries.

    Decision gate for the trajectory RecordIndex (byte-offset incremental read):
    if this stays in the low seconds, ecoGPT keeps the simple full-file read.
    """
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        c = dict(session_key="claude:bench", session_source="claude",
                 run_label="bench", ledger_dir=d)
        for i in range(count):
            ledger.request_event(request_id=f"r{i}", requested_model="m", model="m",
                                 route="anthropic", stream=False,
                                 body={"messages": [{"role": "user", "content": f"bench-{i}"}]},
                                 **c)
            ledger.response_event(request_id=f"r{i}", status=200, duration_ms=50.0, **c)
        key = ledger.session_path(d, "claude:bench").stem
        started = time.monotonic()
        detail = ledger.read_session(d, key)
        elapsed = time.monotonic() - started
        assert len(detail["round_trips"]) == count
        assert elapsed < 10.0  # generous CI bound; report the actual number
        print(f"\nread_session({count} entries): {elapsed:.2f}s")
