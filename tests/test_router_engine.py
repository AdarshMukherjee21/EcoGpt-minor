"""Router engine: rules, state building, and the generic SystemOne client.

All network paths run against httpx.MockTransport — no live calls, ever.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from proxy import router_engine as re_mod
from proxy import router_settings as rs
from proxy.router_engine import (
    ClassifyError,
    ClassifierConfig,
    RouterClassifier,
    apply_rules,
    build_state,
    build_prompts,
    estimate_tokens,
    fallback_tier_index,
)


TIERS = [
    {"id": "goldfish", "description": "execution work", "model": "g-1"},
    {"id": "elephant", "description": "planning work", "model": "e-1"},
]
VALUES = rs.defaults()


def answer(choice, probabilities=None, confidence=0.9, **extra):
    data = {"choice": choice}
    if probabilities is not None:
        data["probabilities"] = probabilities
    if confidence is not None:
        data["confidence"] = confidence
    data.update(extra)
    return data


# --- token estimation ---------------------------------------------------------

def test_estimate_tokens_nonempty():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello world, this is a tokenizer test.") > 1


# --- prompt flattening ----------------------------------------------------------

def test_build_prompts_system_string_and_history():
    body = {
        "system": "You are a senior systems engineer.",
        "messages": [
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": [{"type": "text", "text": "now this"}]},
        ],
        "tools": [{"name": "run_tests", "type": "custom"}],
    }
    system_text, user_text = build_prompts(body)
    assert "You are a senior systems engineer." in system_text
    assert "[AVAILABLE TOOLS (1)]: run_tests" in system_text
    assert "[USER]: earlier question" in user_text
    assert user_text.rstrip().endswith("now this")


def test_build_prompts_flattens_blocks_and_markers():
    body = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image", "source": {}},
                {"type": "tool_use", "name": "grep", "input": {}},
                {"type": "tool_result", "content": "output"},
            ]},
        ],
    }
    _, user_text = build_prompts(body, max_history_turns=1)
    assert "[IMAGE_ATTACHMENT]" in user_text
    assert "[TOOL_USE: grep]" in user_text
    assert "[TOOL_RESULT]" in user_text
    assert "look at this" in user_text


# --- state building / truncation -------------------------------------------------

def test_build_state_short_inputs_pass_through():
    state = build_state("be brief", "what is 2+2?", token_budget=1900)
    assert "[SYSTEM PROMPT]" in state
    assert "[USER PROMPT]" in state
    assert "what is 2+2?" in state


def test_build_state_truncates_user_keeping_tail():
    filler = "\n".join(f"line {i} of the long prompt" for i in range(400))
    state = build_state("sys", filler, token_budget=600)
    assert "[USER PROMPT TRUNCATED" in state
    assert "line 0 of" not in state           # head lines dropped
    assert "line 399" in state                # tail (latest) kept


def test_build_state_truncates_system_head():
    system = " ".join(f"sysword{i}" for i in range(2000))
    state = build_state(system, "tiny question", token_budget=900)
    assert "[SYSTEM TRUNCATED]" in state
    assert estimate_tokens(state) < 900 + 80  # small slack for markers


# --- rules ---------------------------------------------------------------------

def test_rule_direct_classification_exact_confident():
    out = apply_rules(answer("goldfish", confidence=0.9), tiers=TIERS,
                      values=VALUES, full_prompt_tokens=10)
    assert out["final_idx"] == 0
    assert out["routing_reason"] == "direct_classification"
    assert not out["warmth_applied"] and not out["stickiness_applied"]


def test_rule_confidence_floor_escalates():
    out = apply_rules(answer("goldfish", confidence=0.20), tiers=TIERS,
                      values=VALUES, full_prompt_tokens=10)
    assert out["final_idx"] == 1
    assert out["routing_reason"] == "confidence_escalation"


def test_rule_confidence_floor_no_escalation_at_top():
    out = apply_rules(answer("elephant", confidence=0.10), tiers=TIERS,
                      values=VALUES, full_prompt_tokens=10)
    assert out["final_idx"] == 1
    assert out["routing_reason"] == "direct_classification"


def test_rule_long_context_floor():
    VALUES_LT = dict(VALUES, long_context_token_threshold=1000, min_tier_for_long_context=1)
    out = apply_rules(answer("goldfish", confidence=0.9), tiers=TIERS,
                      values=VALUES_LT, full_prompt_tokens=5000)
    assert out["final_idx"] == 1
    assert out["routing_reason"] == "long_context_floor"


def test_rule_warmth_holds_higher_tier_within_window():
    out = apply_rules(answer("goldfish", confidence=0.95), tiers=TIERS, values=VALUES,
                      full_prompt_tokens=10,
                      warmth={"tier_idx": 1, "model": "e-1", "elapsed": 30.0})
    assert out["final_idx"] == 1
    assert out["warmth_applied"] is True
    assert out["routing_reason"].startswith("warmth_hold")


def test_rule_warmth_outside_window_downgrades():
    out = apply_rules(answer("goldfish", confidence=0.95), tiers=TIERS, values=VALUES,
                      full_prompt_tokens=10,
                      warmth={"tier_idx": 1, "model": "e-1", "elapsed": 500.0})
    assert out["final_idx"] == 0
    assert out["warmth_applied"] is False


def test_rule_warmth_requires_known_elapsed():
    out = apply_rules(answer("goldfish", confidence=0.95), tiers=TIERS, values=VALUES,
                      full_prompt_tokens=10,
                      warmth={"tier_idx": 1, "model": "e-1", "elapsed": None})
    assert out["final_idx"] == 0


def test_rule_stickiness_suppresses_borderline_upflip():
    # goldfish chosen with mid-band confidence (0.55 < 0.5+0.10), 200s after an
    # elephant turn: warmth window passed, stickiness window not -> keep elephant.
    out = apply_rules(answer("goldfish", confidence=0.55), tiers=TIERS, values=VALUES,
                      full_prompt_tokens=10,
                      warmth={"tier_idx": 1, "model": "e-1", "elapsed": 200.0})
    assert out["final_idx"] == 1
    assert out["stickiness_applied"] is True
    assert out["routing_reason"] == "cache_stickiness"


def test_rule_stickiness_allows_clear_flips():
    out = apply_rules(answer("goldfish", confidence=0.9), tiers=TIERS, values=VALUES,
                      full_prompt_tokens=10,
                      warmth={"tier_idx": 1, "model": "e-1", "elapsed": 200.0})
    assert out["final_idx"] == 0
    assert out["stickiness_applied"] is False


def test_rule_warmth_disabled_disables_stickiness_too():
    values = dict(VALUES, warmth_enabled=False)
    out = apply_rules(answer("goldfish", confidence=0.55), tiers=TIERS, values=values,
                      full_prompt_tokens=10,
                      warmth={"tier_idx": 1, "model": "e-1", "elapsed": 200.0})
    assert out["final_idx"] == 0
    assert not out["warmth_applied"] and not out["stickiness_applied"]


def test_rule_unknown_choice_falls_back_to_argmax():
    out = apply_rules(answer("hallucinated-tier",
                             probabilities={"goldfish": 0.2, "elephant": 0.8},
                             confidence=None),
                      tiers=TIERS, values=VALUES, full_prompt_tokens=10)
    assert out["final_idx"] == 1


def test_rule_confidence_fallback_formula():
    out = apply_rules(answer("elephant", None, confidence=None),
                      tiers=TIERS, values=VALUES, full_prompt_tokens=10)
    # fallback: top1*0.7 + (top1-top2)*0.3 with top1=1.0, top2=0.0 -> 0.7
    assert out["confidence"] == 0.7


def test_fallback_tier_index_postures():
    assert fallback_tier_index({"on_classify_failure": "planner"}) == -1
    assert fallback_tier_index({"on_classify_failure": "goldfish"}) == 0


def test_decision_record_is_metadata_only():
    dec = re_mod.Decision(endpoint="http://127.0.0.1:9/v1/systemone", router_model="tev1:4b",
                          final_tier="goldfish", target_model="g-1", confidence=0.8,
                          probabilities={"goldfish": 0.8, "elephant": 0.2})
    rec = dec.record()
    blob = json.dumps(rec)
    assert "goldfish" in blob and "tev1:4b" in blob
    assert "endpoint_host" in rec  # host only, never a URL with credentials
    assert "@" not in blob


# --- warmth from ledger ----------------------------------------------------------

def test_warmth_from_ledger_seeds_recent_decision(tmp_path):
    from proxy import ledger

    sessions = tmp_path / "sessions"
    ledger.request_event(
        sessions, session_key="claude:abc:fp", session_source="claude",
        request_id="r1", run_label="", requested_model="eco-auto",
        model="eco-auto", route="anthropic", stream=False,
        body={"messages": [{"role": "user", "content": "hi"}]},
    )
    ledger.stage_event(
        sessions, session_key="claude:abc:fp", session_source="claude",
        request_id="r1", run_label="", feature="router",
        provenance={"final_tier": "elephant", "target_model": "e-1"},
    )
    now_wall = datetime.now(timezone.utc)
    warmth = re_mod.warmth_from_ledger(sessions, "claude:abc:fp", TIERS, now_wall=now_wall)
    assert warmth["tier_idx"] == 1
    assert warmth["model"] == "e-1"
    assert 0 <= warmth["elapsed"] <= 5.0


def test_warmth_from_ledger_stale_is_aged_out(tmp_path):
    from proxy import ledger

    sessions = tmp_path / "sessions"
    ledger.request_event(
        sessions, session_key="claude:old:fp", session_source="claude",
        request_id="r2", run_label="", requested_model="eco-auto",
        model="eco-auto", route="anthropic", stream=False,
        body={"messages": [{"role": "user", "content": "hi"}]},
    )
    event = ledger.stage_event(
        sessions, session_key="claude:old:fp", session_source="claude",
        request_id="r2", run_label="", feature="router",
        provenance={"final_tier": "elephant", "target_model": "e-1"},
    )
    # age the event by rewriting its ts far into the past
    path = ledger.session_path(sessions, "claude:old:fp")
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in lines:
        if line.get("type") == "stage":
            line["ts"] = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in lines) + "\n", encoding="utf-8")
    warmth = re_mod.warmth_from_ledger(sessions, "claude:old:fp", TIERS)
    assert warmth is not None
    assert warmth["elapsed"] > 300.0


def test_warmth_from_ledger_absent_file(tmp_path):
    assert re_mod.warmth_from_ledger(tmp_path / "none", "k", TIERS) is None



def make_classifier(handler, endpoint="http://127.0.0.1:11434/v1/systemone",
                    auth_scheme="none", api_key="", router_model="tev1:4b"):
    return RouterClassifier(
        ClassifierConfig(endpoint_url=endpoint, auth_scheme=auth_scheme, api_key=api_key,
                         router_model=router_model, timeout_ms=2000),
        transport=httpx.MockTransport(handler))


def test_classify_round_trip_and_payload_shape():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["json"] = json.loads(request.content)
        return httpx.Response(200, json={
            "model": "tev1:4b",
            "answers": {"difficulty": {"choice": "elephant",
                                       "probabilities": {"goldfish": 0.1, "elephant": 0.9},
                                       "confidence": 0.87}},
            "usage": {"input_tokens": 42, "output_tokens": 1},
        })

    async def run():
        classifier = make_classifier(handler)
        try:
            return await classifier.classify(
                "[SYSTEM PROMPT]\n...", {"goldfish": "execution", "elephant": "planning"},
                "Which tier?")
        finally:
            await classifier.aclose()

    result = asyncio.run(run())
    # Jev envelope, generic endpoint
    assert seen["json"]["model"] == "tev1:4b"
    assert seen["json"]["state"].startswith("[SYSTEM PROMPT]")
    assert seen["json"]["questions"]["difficulty"]["criteria"] == {"goldfish": "execution", "elephant": "planning"}
    assert seen["json"]["questions"]["difficulty"]["type"] == "choice"
    assert seen["headers"]["content-type"] == "application/json"
    assert "authorization" not in {k.lower() for k in seen["headers"]}
    assert result["choice"] == "elephant"
    assert result["_latency_ms"] >= 0
    assert result["_usage"]["input_tokens"] == 42


def test_classify_bearer_auth_header_from_server_side_config():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"answers": {"difficulty": {"choice": "goldfish"}}})

    async def run():
        classifier = make_classifier(handler, endpoint="https://openrouter.example/api/alpha/decisions",
                                     auth_scheme="bearer", api_key="sk-internal-test")
        try:
            return await classifier.classify("state", {}, "q")
        finally:
            await classifier.aclose()

    result = asyncio.run(run())
    assert seen["auth"] == "Bearer sk-internal-test"
    assert result["choice"] == "goldfish"


def test_classify_x_api_key_scheme():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["x-api-key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"answers": {"difficulty": {"choice": "goldfish"}}})

    async def run():
        classifier = make_classifier(handler, auth_scheme="x-api-key", api_key="k2")
        try:
            return await classifier.classify("state", {}, "q")
        finally:
            await classifier.aclose()

    asyncio.run(run())
    assert seen["x-api-key"] == "k2"


def test_classify_retries_429_then_succeeds():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"answers": {"difficulty": {"choice": "goldfish"}}})

    async def run():
        classifier = make_classifier(handler)
        try:
            return await classifier.classify("state", {}, "q")
        finally:
            await classifier.aclose()

    result = asyncio.run(run())
    assert calls["n"] == 2
    assert result["choice"] == "goldfish"


def test_classify_http_error_raises_classify_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad model")

    async def run():
        classifier = make_classifier(handler)
        try:
            await classifier.classify("state", {}, "q")
        finally:
            await classifier.aclose()

    with pytest.raises(ClassifyError, match="HTTP 400"):
        asyncio.run(run())


def test_classify_unreachable_fails_fast():
    # Transport-level failure surfaces immediately (no retry storm).
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async def run():
        classifier = make_classifier(handler)
        try:
            await classifier.classify("state", {}, "q")
        finally:
            await classifier.aclose()

    with pytest.raises(ClassifyError, match="endpoint unreachable"):
        asyncio.run(run())


def test_classify_missing_answers_shape():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"hello": "world"})

    async def run():
        classifier = make_classifier(handler)
        try:
            await classifier.classify("state", {}, "q")
        finally:
            await classifier.aclose()

    with pytest.raises(ClassifyError, match="missing answers.difficulty"):
        asyncio.run(run())


def test_classify_no_endpoint_configured():
    async def run():
        classifier = RouterClassifier(ClassifierConfig(
            endpoint_url="", auth_scheme="none", api_key="", router_model="m", timeout_ms=1000))
        try:
            await classifier.classify("state", {}, "q")
        finally:
            await classifier.aclose()

    with pytest.raises(ClassifyError, match="no classifier endpoint"):
        asyncio.run(run())
