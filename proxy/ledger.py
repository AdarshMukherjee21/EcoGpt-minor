"""Append-only, privacy-first session ledger.

One JSONL file per session key under ``data/sessions/``. File names are the
sanitized key plus a short SHA-256 of the raw key, so no two distinct keys ever
collide and the name is safe on Windows and in URLs.

Entries carry metadata and SHA-256 content hashes only — conversation content,
system prompts, and tool schemas are never stored. The ledger is an observer:
every write failure goes to stderr and never breaks the request.

The ablation harness stamps each entry with the ``run_label`` from the features
block in config.json; ``request_id`` correlates all entries of one round trip.

Schema v2 (trajectory-style timeline support). Additive, backward compatible:
``schema_version: 2`` on every new entry; ``span_id`` (one write, one span) and
``t0_offset_ms`` (milliseconds since the round trip's request arrival, monotonic
clock) appear only when known — a timing that was never observed is absent or
``null``, never 0. Old v1 files read unchanged: readers ignore unknown keys and
fall back to order-based rendering when offsets are missing.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

NO_SESSION_KEY = "_no-session"
SCHEMA_VERSION = 2
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_KEY_OK = re.compile(r"^[A-Za-z0-9._-]+$")


def new_span_id() -> str:
    """Short id for one ledger entry (one write == one span)."""
    return uuid.uuid4().hex[:12]


def session_path(ledger_dir: Path | str, session_key: str | None) -> Path:
    """Sanitized filename + short hash of the raw key, so distinct keys get distinct files."""
    key = session_key or NO_SESSION_KEY
    safe = _UNSAFE_CHARS.sub("_", key)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
    return Path(ledger_dir) / f"{safe}-{digest}.jsonl"


def _canonical(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def chars_of(data: Any) -> int:
    """Public alias: canonical serialization size of a body (metadata, not content)."""
    return len(_canonical(data))


def _hash(data: Any) -> str:
    return hashlib.sha256(_canonical(data).encode("utf-8")).hexdigest()


def _chars(data: Any) -> int:
    """Character size of the canonical serialization."""
    return len(_canonical(data))


def _block_counts(content: Any) -> dict[str, int]:
    if isinstance(content, str):
        return {"text": 1}
    return dict(Counter(block.get("type", "unknown") for block in content or [] if isinstance(block, dict)))


def _content_profile(content: Any) -> dict[str, Any] | None:
    if content is None or content == [] or content == "":
        return None
    return {"chars": _chars(content), "hash": _hash(content), "block_counts": _block_counts(content)}


def _message_profile(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content", "")
    return {
        "role": message.get("role", ""),
        "block_counts": _block_counts(content),
        "chars": _chars(content),
        "hash": _hash(content),
    }


def _envelope(
    kind: str,
    *,
    run_label: str,
    session_key: str | None,
    session_source: str | None,
    request_id: str,
    t0_offset_ms: float | None = None,
    span_id: str | None = None,
) -> dict[str, Any]:
    entry = {
        "type": kind,
        "ts": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "run_label": run_label or "",
        "session_key": session_key,
        "session_source": session_source,
        "request_id": request_id,
    }
    # Trajectory honesty rule: a mark that was never observed is absent, never 0.
    if span_id:
        entry["span_id"] = span_id
    if t0_offset_ms is not None:
        entry["t0_offset_ms"] = int(t0_offset_ms)
    return entry


def _append(ledger_dir: Path | str, session_key: str | None, entry: dict[str, Any]) -> None:
    """Append one entry; a ledger failure logs to stderr and never raises."""
    try:
        path = session_path(ledger_dir, session_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"ledger write failed for session {session_key!r}: {exc}", file=sys.stderr)


def _system_profile(body: dict[str, Any]) -> dict[str, Any] | None:
    system = body.get("system")
    if system is None:
        return None
    return {"chars": _chars(system), "hash": _hash(system)}


def request_event(
    ledger_dir: Path | str,
    *,
    session_key: str | None,
    session_source: str | None,
    request_id: str,
    run_label: str,
    requested_model: str,
    model: str,
    route: str,
    stream: bool,
    body: dict[str, Any],
    upstream_model: str | None = None,
    t0_offset_ms: float | None = 0,
) -> dict[str, Any]:
    """Log the request half: models, route, message/system/tool fingerprints. No payloads.

    The request is the round trip's timeline anchor: ``t0_offset_ms`` is 0 by
    construction (callers may still pass their measured offset).
    """
    messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
    tools = body.get("tools") or []
    entry = _envelope(
        "request", run_label=run_label, session_key=session_key,
        session_source=session_source, request_id=request_id,
        span_id=new_span_id(), t0_offset_ms=t0_offset_ms if t0_offset_ms is not None else 0,
    )
    entry.update({
        "requested_model": requested_model or "",
        "model": model or "",
        "route": route,
        "stream": bool(stream),
        "message_count": len(messages),
        "messages": [_message_profile(m) for m in messages],
        "system": _system_profile(body),
        "tools": {
            "names": [t.get("name", "") for t in tools if isinstance(t, dict)],
            "count": len(tools),
            "hash": _hash(tools),
        },
    })
    if upstream_model and upstream_model != model:
        # Model aliasing: record the effective upstream model id alongside the
        # client-facing name.
        entry["upstream_model"] = upstream_model
    _append(ledger_dir, session_key, entry)
    return entry


def stage_event(
    ledger_dir: Path | str,
    *,
    session_key: str | None,
    session_source: str | None,
    request_id: str,
    run_label: str,
    feature: str,
    provenance: dict[str, Any] | None = None,
    t0_offset_ms: float | None = None,
) -> dict[str, Any]:
    """Log one pipeline stage and its provenance (tokens, fallback reason, duration).

    ``t0_offset_ms`` (when the caller measured it) anchors the stage on the
    round trip's timeline; ``provenance["duration_ms"]`` then gives the span's
    width.
    """
    entry = _envelope(
        "stage", run_label=run_label, session_key=session_key,
        session_source=session_source, request_id=request_id,
        span_id=new_span_id(), t0_offset_ms=t0_offset_ms,
    )
    entry.update({"feature": feature, "provenance": dict(provenance or {})})
    _append(ledger_dir, session_key, entry)
    return entry


def response_event(
    ledger_dir: Path | str,
    *,
    session_key: str | None,
    session_source: str | None,
    request_id: str,
    run_label: str,
    status: int,
    duration_ms: float,
    usage: dict[str, Any] | None = None,
    stop_reason: str | None = None,
    content: Any = None,
    impacts: dict[str, Any] | None = None,
    model_served: str | None = None,
    route: str | None = None,
    ttft_ms: float | None = None,
    tps: float | None = None,
    chunk_count: int | None = None,
    t0_offset_ms: float | None = None,
) -> dict[str, Any]:
    """Log the response half: status, duration, usage split, assistant fingerprint.

    ``impacts`` (green metrics) is attached only when the caller computed it, so
    flag-off responses carry no green field at all.

    Schema v2 observations, included only when actually observed (absent, not
    zero): ``model_served`` (the effective upstream model), ``route``, stream
    timings ``ttft_ms``/``tps``/``chunk_count``, and the timeline anchor
    ``t0_offset_ms``.
    """
    usage = usage or {}
    entry = _envelope(
        "response", run_label=run_label, session_key=session_key,
        session_source=session_source, request_id=request_id,
        span_id=new_span_id(), t0_offset_ms=t0_offset_ms,
    )
    entry.update({
        "status": int(status),
        "duration_ms": int(duration_ms or 0),
        "usage": {
            "input": usage.get("input_tokens") or 0,
            "output": usage.get("output_tokens") or 0,
            "cache_read": usage.get("cache_read_input_tokens") or 0,
            "cache_write": usage.get("cache_creation_input_tokens") or 0,
        },
        "stop_reason": stop_reason,
        "content": _content_profile(content),
    })
    if model_served:
        entry["model_served"] = model_served
    if route:
        entry["route"] = route
    if ttft_ms is not None:
        entry["ttft_ms"] = round(float(ttft_ms), 3)
    if tps is not None:
        entry["tps"] = round(float(tps), 3)
    if chunk_count is not None:
        entry["chunk_count"] = int(chunk_count)
    if impacts is not None:
        entry["impacts"] = impacts
    _append(ledger_dir, session_key, entry)
    return entry


def error_event(
    ledger_dir: Path | str,
    *,
    session_key: str | None,
    session_source: str | None,
    request_id: str,
    run_label: str,
    stage: str,
    message: Any,
    t0_offset_ms: float | None = None,
) -> dict[str, Any]:
    """Log a stage failure: the stage that failed and why."""
    entry = _envelope(
        "error", run_label=run_label, session_key=session_key,
        session_source=session_source, request_id=request_id,
        span_id=new_span_id(), t0_offset_ms=t0_offset_ms,
    )
    entry.update({"stage": stage, "message": str(message)[:1000]})
    _append(ledger_dir, session_key, entry)
    return entry


def list_sessions(ledger_dir: Path | str) -> list[dict[str, Any]]:
    """Summaries of every session file: names, run label, counts, size, first/last ts."""
    out: list[dict[str, Any]] = []
    root = Path(ledger_dir)
    if not root.is_dir():
        return out
    for path in sorted(root.glob("*.jsonl")):
        try:
            summary = _scan_file(path)
        except OSError:
            continue
        if summary is not None:
            out.append({"key": path.stem, **summary})
    return out


def _scan_file(path: Path) -> dict[str, Any] | None:
    entries = requests = 0
    first_ts = last_ts = None
    session_key = session_source = None
    run_label = ""
    with path.open("rb") as handle:
        for raw_line in handle:
            try:
                entry = json.loads(raw_line)
            except ValueError:
                continue
            if not isinstance(entry, dict):
                continue
            entries += 1
            ts = entry.get("ts")
            if first_ts is None:
                first_ts = ts
                session_key = entry.get("session_key")
                session_source = entry.get("session_source")
                run_label = entry.get("run_label") or ""
            if ts:
                last_ts = ts
            if entry.get("type") == "request":
                requests += 1
    if entries == 0:
        return None
    size_bytes = path.stat().st_size
    return {
        "session_key": session_key,
        "session_source": session_source,
        "run_label": run_label,
        "requests": requests,
        "entries": entries,
        "size_bytes": size_bytes,
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def read_session(ledger_dir: Path | str, key: str) -> dict[str, Any] | None:
    """One session's entries grouped by round trip, keyed by the file stem."""
    if not _KEY_OK.match(key or "") or key.strip(".-") == "":
        return None
    path = Path(ledger_dir) / f"{key}.jsonl"
    if not path.is_file():
        return None
    entries: list[dict[str, Any]] = []
    session_key = session_source = None
    run_label = ""
    try:
        with path.open("rb") as handle:
            for raw_line in handle:
                try:
                    entry = json.loads(raw_line)
                except ValueError:
                    continue
                if not isinstance(entry, dict):
                    continue
                if session_key is None and entry.get("session_key"):
                    session_key = entry.get("session_key")
                    session_source = entry.get("session_source")
                if run_label == "" and entry.get("run_label"):
                    run_label = entry["run_label"]
                entries.append(entry)
    except OSError:
        return None
    return {"key": key, "session_key": session_key, "session_source": session_source,
            "run_label": run_label, "entries": entries,
            "round_trips": group_round_trips(entries), "summary": summarize(entries)}


def group_round_trips(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Entries bucketed per request_id, in first-appearance order."""
    trips: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for entry in entries:
        request_id = str(entry.get("request_id") or "")
        trip = trips.get(request_id)
        if trip is None:
            trip = {"request_id": request_id, "request": None, "stages": [], "response": None, "errors": []}
            trips[request_id] = trip
            order.append(request_id)
        kind = entry.get("type")
        if kind == "request":
            trip["request"] = entry
        elif kind == "stage":
            trip["stages"].append(entry)
        elif kind == "response" and trip["response"] is None:
            trip["response"] = entry
        elif kind == "error":
            trip["errors"].append(entry)
    return [trips[request_id] for request_id in order]


def _sum_bound(objects: list[dict[str, Any]], path: tuple[str, ...]) -> dict[str, float] | None:
    totals = {"min": 0.0, "mean": 0.0, "max": 0.0}
    found = False
    for obj in objects:
        node: Any = obj
        for part in path:
            node = (node or {}).get(part) if isinstance(node, dict) else None
        if isinstance(node, dict) and any(node.get(b) is not None for b in totals):
            found = True
            for bound in totals:
                totals[bound] += float(node.get(bound) or 0.0)
    return totals if found else None


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals over one session's ledger: tokens, duration, compression, green, router."""
    requests = errors = 0
    models: dict[str, dict[str, int]] = {}
    compression = {"tokens_before": 0, "tokens_after": 0, "tokens_saved": 0}
    first_ts = last_ts = None
    run_label: str | None = None
    responses = []
    request_models = _request_model_map(entries)
    for entry in entries:
        kind = entry.get("type")
        if kind == "request":
            requests += 1
            model = entry.get("model") or ""
            models.setdefault(model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
        elif kind == "response":
            responses.append(entry)
            model = request_models.get(str(entry.get("request_id")), "")
            if model:
                bucket = models.setdefault(model, {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0})
                usage = entry.get("usage") or {}
                for key in bucket:
                    bucket[key] += int(usage.get(key) or 0)
        elif kind == "stage" and entry.get("feature") == "compression":
            provenance = entry.get("provenance") or {}
            compression["tokens_before"] += int(provenance.get("tokens_before") or 0)
            compression["tokens_after"] += int(provenance.get("tokens_after") or 0)
            compression["tokens_saved"] += int(provenance.get("tokens_saved") or 0)
        elif kind == "error":
            errors += 1
        ts = entry.get("ts")
        if isinstance(ts, str) and ts:
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        if run_label is None and entry.get("run_label"):
            run_label = entry["run_label"]

    wall_clock = None
    if first_ts and last_ts:
        try:
            wall_clock = round((datetime.fromisoformat(last_ts) - datetime.fromisoformat(first_ts)).total_seconds(), 3)
        except ValueError:
            wall_clock = None

    green = None
    with_impacts = [e for e in responses if isinstance(e.get("impacts"), dict)]
    if with_impacts:
        def _sum_scalar(entries: list[dict[str, Any]], path: tuple[str, ...]) -> float:
            total = 0.0
            for entry in entries:
                node: Any = entry
                for part in path:
                    node = node.get(part) if isinstance(node, dict) else None
                if isinstance(node, (int, float)):
                    total += float(node)
            return total

        green = {
            "energy_kwh": _sum_bound(with_impacts, ("impacts", "energy_kwh")),
            "gwp_kgco2e": _sum_bound(with_impacts, ("impacts", "gwp_kgco2e")),
            # Single-value categories (mean bound); always shown alongside provenance.
            "water_l": _sum_scalar(with_impacts, ("impacts", "water_l")),
            "adpe_kg_sbeq": _sum_scalar(with_impacts, ("impacts", "adpe_kg_sbeq")),
            "primary_mj": _sum_scalar(with_impacts, ("impacts", "primary_mj")),
            "statuses": sorted({
                str(e["impacts"]["impact_status"])
                for e in with_impacts
                if isinstance(e["impacts"].get("impact_status"), str)
            }),
        }
    # Router stage rollup (tier decisions across the session).
    router = {"decisions": 0, "by_reason": {}, "by_tier": {},
              "warmth_applied": 0, "stickiness_applied": 0, "escalations": 0,
              "fallbacks": 0}
    classify_latencies: list[float] = []
    confidences: list[float] = []
    for entry in entries:
        if entry.get("type") != "stage" or entry.get("feature") != "router":
            continue
        provenance = entry.get("provenance") or {}
        router["decisions"] += 1
        reason = str(provenance.get("routing_reason") or "unknown")
        tier = str(provenance.get("final_tier") or "unknown")
        router["by_reason"][reason] = router["by_reason"].get(reason, 0) + 1
        router["by_tier"][tier] = router["by_tier"].get(tier, 0) + 1
        if provenance.get("warmth_applied"):
            router["warmth_applied"] += 1
        if provenance.get("stickiness_applied"):
            router["stickiness_applied"] += 1
        if reason in ("confidence_escalation", "long_context_floor"):
            router["escalations"] += 1
        if provenance.get("fallback_used"):
            router["fallbacks"] += 1
        if isinstance(provenance.get("classify_latency_ms"), (int, float)):
            classify_latencies.append(float(provenance["classify_latency_ms"]))
        if isinstance(provenance.get("confidence"), (int, float)):
            confidences.append(float(provenance["confidence"]))
    if classify_latencies:
        router["classify_latency_ms_p50"] = round(_percentile(sorted(classify_latencies), 0.5), 1)
        router["classify_latency_ms_mean"] = round(sum(classify_latencies) / len(classify_latencies), 1)
    if confidences:
        router["confidence_mean"] = round(sum(confidences) / len(confidences), 4)

    # Timing rollup over response events: observed stream TTFT percentiles,
    # with honesty counts (how many responses never reported one).
    ttfts = sorted(float(e["ttft_ms"]) for e in responses if isinstance(e.get("ttft_ms"), (int, float)))
    timing = {
        "ttft_p50_ms": round(_percentile(ttfts, 0.5), 1) if ttfts else None,
        "responses_with_ttft": len(ttfts),
        "responses_total": len(responses),
    }
    return {
        "requests": requests,
        "models": {model: counts for model, counts in models.items() if counts and any(counts.values())},
        "wall_clock_seconds": wall_clock,
        "compression": compression,
        "green": green,
        "timing": timing,
        "router": router,
        "errors": errors,
        "run_label": run_label or "",
    }


def _percentile(sorted_values: list[float], q: float) -> float | None:
    """Nearest-rank percentile of an already sorted list (trajectory timing module)."""
    if not sorted_values:
        return None
    import math

    return sorted_values[max(0, math.ceil(q * len(sorted_values)) - 1)]


def _request_model_map(entries: list[dict[str, Any]]) -> dict[str, str]:
    """request_id -> client-facing model, one pass (summarize is O(n), not O(n^2))."""
    return {str(e["request_id"]): str(e["model"])
            for e in entries if e.get("type") == "request" and e.get("request_id") and e.get("model")}


