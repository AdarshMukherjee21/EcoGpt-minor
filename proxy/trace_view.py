"""Read-only trace views built from ledger entries (trajectory-style).

Pure functions over the JSONL ledger: turn entries into timeline spans, session
trajectories and per-call details. Never writes; never mutates its inputs;
payloads are never part of the output (metadata + hashes only).

Timeline model (from ``ledger`` v2 envelope fields):
  * every entry may carry ``t0_offset_ms`` — milliseconds since the round trip's
    request arrival, stamped on a monotonic clock;
  * a stage event carries ``provenance.duration_ms`` — the span width;
  * a response event marks upstream completion, so its ``duration_ms`` gives
    the upstream dispatch span (``[offset - duration, offset]``);
  * a mark that was never observed is ``None`` — the UI degrades to a flat
    list rather than inventing positions.
"""

from __future__ import annotations

from typing import Any

from . import ledger

# Fields promoted from stage provenance into a span's detail block, by feature.
_COMPRESSION_FIELDS = ("compressed", "tokens_before", "tokens_after", "tokens_saved",
                       "fallback", "mode", "transforms", "roundtrip_ms",
                       "bytes_before", "bytes_after", "turn_id", "attempts")
_ROUTER_FIELDS = ("final_tier", "target_model", "raw_choice", "routing_reason",
                  "confidence", "probabilities", "full_prompt_tokens",
                  "state_tokens", "router_input_tokens", "router_output_tokens",
                  "endpoint_host", "router_model",
                  "warmth_applied", "stickiness_applied", "fallback_used")


def _offset(entry: dict[str, Any]) -> float | None:
    value = entry.get("t0_offset_ms")
    return float(value) if isinstance(value, (int, float)) else None


def _pick(mapping: dict[str, Any] | None, fields: tuple[str, ...]) -> dict[str, Any]:
    return {k: mapping[k] for k in fields
            if isinstance(mapping, dict) and mapping.get(k) is not None}


def build_spans(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One round trip's timeline: ordered spans with start/end offsets in ms."""
    spans: list[dict[str, Any]] = []
    for entry in entries:
        kind = entry.get("type")
        offset = _offset(entry)
        if kind == "request":
            start = offset if offset is not None else 0.0
            spans.append({"stage": "request", "start_ms": start, "end_ms": start,
                          "detail": {"route": entry.get("route"),
                                     "stream": entry.get("stream"),
                                     "requested_model": entry.get("requested_model")}})
        elif kind == "stage":
            provenance = entry.get("provenance") or {}
            duration = provenance.get("duration_ms")
            end = (offset + duration) if offset is not None and isinstance(duration, (int, float)) else None
            fields = _COMPRESSION_FIELDS if entry.get("feature") == "compression" else _ROUTER_FIELDS
            spans.append({"stage": str(entry.get("feature")), "start_ms": offset,
                          "end_ms": end, "detail": _pick(provenance, fields)})
        elif kind == "response":
            duration = entry.get("duration_ms")
            start = (offset - duration) if offset is not None and isinstance(duration, (int, float)) else None
            spans.append({"stage": "upstream", "start_ms": start, "end_ms": offset,
                          "detail": {"status": entry.get("status"),
                                     "usage": entry.get("usage"),
                                     "stop_reason": entry.get("stop_reason"),
                                     "model_served": entry.get("model_served"),
                                     "ttft_ms": entry.get("ttft_ms"),
                                     "ups_duration_ms": duration}})
        elif kind == "error":
            spans.append({"stage": str(entry.get("stage")), "start_ms": offset,
                          "end_ms": offset, "detail": {"error": entry.get("message")}})
    # Landmarks first, then by start; entries without offsets keep file order
    # at the end so the UI can still render a flat fallback list.
    return sorted(spans, key=lambda s: (s["start_ms"] is None, s["start_ms"] or 0))


def _step(trip: dict[str, Any], index: int, prev: dict[str, Any] | None) -> dict[str, Any]:
    """One round trip as a trajectory step row (trajectory::sessions._step port)."""
    request = trip.get("request") or {}
    stages = trip.get("stages") or []
    response = trip.get("response") or {}
    errors = trip.get("errors") or []
    trip_entries = ([request] if request else []) + stages + \
                   ([response] if response else []) + errors

    compression = next((s for s in stages if s.get("feature") == "compression"), None)
    router = next((s for s in stages if s.get("feature") == "router"), None)

    gap = None
    if prev is not None and response.get("ts") and (prev.get("response") or {}).get("ts"):
        from datetime import datetime
        try:
            end_prev = datetime.fromisoformat(prev["response"]["ts"])
            start = datetime.fromisoformat(response["ts"])
            gap = round(max(0.0, (start - end_prev).total_seconds()), 3)
        except ValueError:
            gap = None

    usage = response.get("usage") or {}
    step = {
        "index": index,
        "request_id": request.get("request_id"),
        "ts": request.get("ts"),
        "requested_model": request.get("requested_model"),
        "model": request.get("model"),
        "model_served": response.get("model_served"),
        "route": request.get("route"),
        "stream": request.get("stream"),
        "status": response.get("status"),
        "stop_reason": response.get("stop_reason"),
        "duration_ms": response.get("duration_ms"),
        "usage": usage,
        # upper bound: wall clock from one response to the next (tool + agent + network)
        "gap_since_previous_ms": gap,
        "compression": _pick((compression or {}).get("provenance"), _COMPRESSION_FIELDS) if compression else None,
        "router": _pick((router or {}).get("provenance"), _ROUTER_FIELDS) if router else None,
        "green": response.get("impacts"),
        "errors": [(e.get("stage"), e.get("message")) for e in errors],
        "spans": build_spans(trip_entries),
        "ttft_ms": (response or {}).get("ttft_ms"),
        "final_tier": ((next((s for s in stages if s.get("feature") == "router"), None) or {})
                       .get("provenance") or {}).get("final_tier"),
    }
    return step


def session_timeline(detail: dict[str, Any], key: str) -> dict[str, Any]:
    """A session's trajectory: summary + per-round-trip steps with timings."""
    trips = detail.get("round_trips") or []
    steps: list[dict[str, Any]] = []
    prev = None
    for index, trip in enumerate(trips):
        step = _step(trip, index, prev)
        steps.append(step)
        prev = trip
    return {
        "key": key,
        "session_key": detail.get("session_key"),
        "session_source": detail.get("session_source"),
        "run_label": detail.get("run_label"),
        "summary": ledger.summarize(detail.get("entries") or []),
        "steps": steps,
    }


def recent_calls(ledger_dir, limit: int, offset: int,
                 session_filter: str | None = None) -> tuple[list[dict[str, Any]], int]:
    """Compact cross-session call rows (trajectory's "All calls" table)."""
    rows: list[dict[str, Any]] = []
    for session in ledger.list_sessions(ledger_dir):
        if session_filter and session.get("key") != session_filter:
            continue
        key = session["key"]
        detail = ledger.read_session(ledger_dir, key)
        if detail is None:
            continue
        for trip in detail.get("round_trips") or []:
            request = trip.get("request") or {}
            response = trip.get("response") or {}
            router = next((s for s in trip.get("stages") or [] if s.get("feature") == "router"), None)
            compression = next((s for s in trip.get("stages") or [] if s.get("feature") == "compression"), None)
            rows.append({
                "session": key,
                "session_key": session.get("session_key"),
                "request_id": request.get("request_id"),
                "ts": request.get("ts"),
                "run_label": session.get("run_label"),
                "model": request.get("model"),
                "model_served": response.get("model_served"),
                "route": request.get("route"),
                "status": response.get("status"),
                "stop_reason": response.get("stop_reason"),
                "duration_ms": response.get("duration_ms"),
                "usage": response.get("usage"),
                "final_tier": ((router or {}).get("provenance") or {}).get("final_tier"),
                "compression_saved": ((compression or {}).get("provenance") or {}).get("tokens_saved"),
                "error": bool(trip.get("errors")),
            })
    rows.sort(key=lambda r: (r.get("ts") or ""), reverse=True)
    return rows[offset:offset + limit], len(rows)


def call_detail(ledger_dir, key: str, request_id: str) -> dict[str, Any] | None:
    """One round trip, fully joined: step view + spans + raw ledger entries."""
    detail = ledger.read_session(ledger_dir, key)
    if detail is None:
        return None
    trips = detail.get("round_trips") or []
    trip = next((t for t in trips
                 if (t.get("request") or {}).get("request_id") == request_id), None)
    if trip is None:
        return None
    entries = [e for e in detail.get("entries") or []
               if e.get("request_id") == request_id]
    # Reuse the step builder for the metadata block
    step = _step(trip, 0, None)
    return {**step, "session": key, "session_key": detail.get("session_key"),
            "run_label": detail.get("run_label"), "ledger_entries": entries}
