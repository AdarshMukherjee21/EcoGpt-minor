"""System One tier-routing engine: classifier client, state builder, decision rules.

Composed as a library into the proxy pipeline (no extra localhost hop — the
final_router FastAPI service from the survey is reference work only). One
generic client talks to ANY SystemOne (Jev-API) compatible decision endpoint:

    POST <endpoint_url>
    {"model": "<router_model>",
     "state": "[SYSTEM PROMPT]\\n...\\n\\n[USER PROMPT]\\n...",
     "questions": {"difficulty": {"type": "choice",
                                  "instructions": "<routing question>",
                                  "criteria": {"<tier id>": "<tier description>"}}}}
    -> {"answers": {"difficulty": {"choice": "...", "probabilities": {...},
                                   "confidence": 0.0-1.0}}, ...}

Decision rules, in order (from the approved plan):
    1. direct classification (model choice / argmax of probabilities)
    2. confidence-floor escalation (+1 tier when confidence < confidence_floor)
    3. long-context floor (full prompt over token threshold -> minimum tier)
    4. warmth hold (no downgrade within the warmth window) and cache
       stickiness (no borderline flips within the stickiness window)

Classification failure is never fatal: callers fall back per the
``on_classify_failure`` setting (:func:`fallback_tier_index`).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx

from . import ledger
from .model_cache import TIKTOKEN_ENCODING

try:  # tiktoken is the token counter; a missing cache/import falls back
    import tiktoken as _tiktoken
except Exception:  # pragma: no cover - exercised only without tiktoken installed
    _tiktoken = None


# ---------------------------------------------------------------------------
# Token estimation (o200k_base — the same vocabulary the Kompress cache warms)
# ---------------------------------------------------------------------------

_ENCODER: Any = None
_ENCODER_FAILED = False


def _encoding() -> Any:
    global _ENCODER, _ENCODER_FAILED
    if _ENCODER is None and not _ENCODER_FAILED and _tiktoken is not None:
        try:
            _ENCODER = _tiktoken.get_encoding(TIKTOKEN_ENCODING)
        except Exception:  # pragma: no cover - offline cache misses
            _ENCODER_FAILED = True
    return _ENCODER


def estimate_tokens(text: str) -> int:
    """tiktoken (o200k_base) count when available, else a chars/4 estimate."""
    if not text:
        return 0
    encoder = _encoding()
    if encoder is not None:
        try:
            return len(encoder.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Prompt (normalized Anthropic Messages body) flattening
# ---------------------------------------------------------------------------

def _flatten_content(content: Any) -> str:
    """Anthropic content (string or block list) -> plain text for the classifier."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                btype = str(block.get("type", ""))
                if btype in ("text", "input_text") and block.get("text") is not None:
                    parts.append(str(block["text"]))
                elif btype == "tool_result":
                    parts.append("[TOOL_RESULT]")
                elif btype in ("tool_use", "tool_call"):
                    name = str(block.get("name") or "").strip()
                    parts.append(f"[TOOL_USE: {name}]" if name else "[TOOL_USE]")
                elif btype in ("image", "input_image"):
                    parts.append("[IMAGE_ATTACHMENT]")
                elif block.get("content") is not None:
                    parts.append(str(block["content"]))
        return "\n".join(p for p in parts if p)
    return str(content)


def build_prompts(body: dict[str, Any], max_history_turns: int = 6) -> tuple[str, str]:
    """(system_text, user_text) from a normalized Anthropic Messages body.

    System/developer framing in ``system``; the recent conversation window in
    the user side (latest turn last, unmarked); a tool-schema summary is
    appended because tool orchestration raises tier need.
    """
    system_parts: list[str] = []
    system_bits = body.get("system")
    if isinstance(system_bits, str) and system_bits.strip():
        system_parts.append(system_bits)
    elif isinstance(system_bits, list):
        text = _flatten_content(system_bits)
        if text.strip():
            system_parts.append(text)

    tools = body.get("tools") or []
    names = [str(t.get("name") or "") for t in tools if isinstance(t, dict)]
    names = [n for n in names if n]
    if names:
        system_parts.append(f"[AVAILABLE TOOLS ({len(names)})]: {', '.join(names)}")

    convo: list[tuple[str, str]] = []
    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        text = _flatten_content(msg.get("content")).strip()
        if not text:
            continue
        convo.append((str(msg.get("role", "user")).strip().lower(), text))

    recent = convo[-max_history_turns:]
    user_parts = [f"[{role.upper()}]: {text}" for role, text in recent[:-1]]
    if recent:
        user_parts.append(recent[-1][1])
    return "\n".join(p for p in system_parts if p), "\n".join(p for p in user_parts if p)


def build_state(system_text: str, user_text: str, token_budget: int) -> str:
    """Compose classification text, trimmed to the endpoint's input budget.

    System context is truncated from the head (the early framing matters most);
    the user side keeps its tail (the latest instruction) when long, with
    explicit truncation markers. ``token_budget`` is the endpoint's *input*
    budget in tokens (small local decision models accept ~2K).
    """
    state_parts: list[str] = []
    if system_text:
        cap = max(200, token_budget // 3)
        trimmed = system_text
        if estimate_tokens(trimmed) > cap:
            words = system_text.split()
            # Shrink in 20% steps until the estimate fits; budgets are tiny.
            while words and estimate_tokens(" ".join(words)) > cap:
                words = words[: max(1, int(len(words) * 0.8))]
            trimmed = " ".join(words) + "\n[SYSTEM TRUNCATED]"
        state_parts.append(f"[SYSTEM PROMPT]\n{trimmed}")

    if user_text:
        system_spend = estimate_tokens(state_parts[0]) if state_parts else 0
        reserve = max(100, token_budget - system_spend - 32)
        trimmed_user = user_text
        if estimate_tokens(trimmed_user) > reserve:
            lines = user_text.split("\n")
            tail: list[str] = []
            while lines and estimate_tokens("\n".join([lines[-1], *tail])) <= reserve:
                tail.insert(0, lines.pop())
            if lines and not tail:
                tail = [lines[-1]]
            trimmed_user = ("[USER PROMPT TRUNCATED — latest context kept]\n"
                            + "\n".join(tail))
        state_parts.append(f"[USER PROMPT]\n{trimmed_user}")

    return "\n\n".join(state_parts)


# ---------------------------------------------------------------------------
# Decision record + deterministic rules (pure, unit-testable)
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    endpoint: str = ""
    router_model: str = ""
    final_tier: str = ""
    target_model: str = ""          # tier model id (aliasing / overrides apply later)
    raw_choice: str = ""
    routing_reason: str = "direct_classification"
    warmth_applied: bool = False
    stickiness_applied: bool = False
    confidence: float = 0.0
    probabilities: dict[str, float] = field(default_factory=dict)
    full_prompt_tokens: int = 0
    state_tokens: int = 0
    router_input_tokens: int = 0
    router_output_tokens: int = 0
    classify_latency_ms: float = 0.0
    duration_ms: float = 0.0
    fallback_used: bool = False
    fallback_reason: str = ""

    def record(self) -> dict[str, Any]:
        """Ledger/usage-safe projection (metadata only — never prompt content)."""
        top = max(self.probabilities, key=lambda k: self.probabilities[k]) if self.probabilities else ""
        return {
            "endpoint_host": urlsplit(self.endpoint).hostname or self.endpoint,
            "router_model": self.router_model,
            "final_tier": self.final_tier,
            "target_model": self.target_model,
            "raw_choice": self.raw_choice,
            "routing_reason": self.routing_reason,
            "warmth_applied": self.warmth_applied,
            "stickiness_applied": self.stickiness_applied,
            "confidence": self.confidence,
            "top_probability_tier": top,
            "full_prompt_tokens": self.full_prompt_tokens,
            "state_tokens": self.state_tokens,
            "router_input_tokens": self.router_input_tokens,
            "router_output_tokens": self.router_output_tokens,
            "classify_latency_ms": self.classify_latency_ms,
            "duration_ms": self.duration_ms,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


def fallback_tier_index(values: dict[str, Any]) -> int:
    """Tier index used when classification fails, per ``on_classify_failure``."""
    posture = str(values.get("on_classify_failure") or "planner")
    if posture == "goldfish":
        return 0
    return -1  # planner (default) and passthrough: the most capable tier


def apply_rules(
    answer: dict[str, Any] | None,
    *,
    tiers: list[dict[str, Any]],
    values: dict[str, Any],
    full_prompt_tokens: int,
    warmth: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deterministic post-classification rules.

    ``warmth`` (from server-side session state or a ledger seed) is
    ``{"tier_idx": int | None, "model": str, "elapsed": float | None}`` with
    elapsed = seconds since the previous turn's dispatch in the same session
    (None = unknown duration; neither time rule then applies).

    Returns ``{"final_idx", "routing_reason", "warmth_applied",
    "stickiness_applied", "confidence", "probabilities"}``.
    """
    tier_ids = [str(t["id"]) for t in tiers]

    probabilities = {t: round(float((answer.get("probabilities") or {}).get(t, 0.0)), 4)
                     for t in tier_ids}
    raw_choice = str(answer.get("choice") or "").strip()
    if raw_choice not in tier_ids:
        # Moodling/unknown label: fall back to the argmax option, else the top tier.
        raw_choice = (max(probabilities, key=lambda t: probabilities[t])
                      if any(probabilities.values()) else tier_ids[-1])
    idx = tier_ids.index(raw_choice)

    conf = answer.get("confidence")
    if isinstance(conf, (int, float)) and 0.0 <= float(conf) <= 1.0:
        confidence = round(float(conf), 4)
    elif not (answer.get("probabilities") or {}):
        # Off-spec decision endpoint: no distribution AND no confidence was
        # returned. The stated choice is treated as certain, per the documented
        # fallback (top1=1.0, top2=0.0 -> top1*0.7 + (top1-top2)*0.3 = 0.7).
        confidence = 0.7
    else:
        # Backend omitted confidence (schema-optional) but did return a
        # distribution: grade top-probability strength + top1/top2 separation.
        sorted_probs = sorted(probabilities.values(), reverse=True)
        top1 = sorted_probs[0] if sorted_probs else 0.0
        top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        confidence = round(min(1.0, top1 * 0.7 + (top1 - top2) * 0.3), 4)

    reason = "direct_classification"
    warmth_applied = stickiness_applied = False

    # Rule 1: confidence-floor escalation (+1 tier as a safety margin)
    floor = float(values.get("confidence_floor", 0.45))
    if confidence < floor and idx < len(tier_ids) - 1:
        idx += 1
        reason = "confidence_escalation"

    # Rule 2: long-context floor guard
    threshold = int(values.get("long_context_token_threshold", 256_000))
    floor_idx = int(values.get("min_tier_for_long_context", 1))
    if (full_prompt_tokens > threshold
            and 0 <= floor_idx < len(tier_ids)
            and idx < floor_idx):
        idx = floor_idx
        reason = "long_context_floor"

    prev_idx_raw = (warmth or {}).get("tier_idx")
    prev_idx = int(prev_idx_raw) if isinstance(prev_idx_raw, int) else None
    elapsed = (warmth or {}).get("elapsed")
    warm_on = bool(values.get("warmth_enabled", True))
    warmth_window = float(values.get("warmth_window_seconds", 120.0))
    stickiness_window = float(values.get("stickiness_window_seconds", 300.0))
    sticky_margin = float(values.get("stickiness_confidence_margin", 0.10))

    def in_window(seconds: float) -> bool:
        return elapsed is not None and isinstance(elapsed, (int, float)) and 0.0 <= float(elapsed) <= seconds

    # Rule 3: warmth hold — never downgrade while the previous model is warm
    if (warm_on and prev_idx is not None and in_window(warmth_window) and prev_idx > idx):
        idx = prev_idx
        warmth_applied = True
        reason = f"warmth_hold({float(elapsed):.1f}s<={warmth_window:.0f}s)"

    # Rule 4: cache stickiness — suppress borderline tier flips (either
    # direction) so consecutive requests reuse the provider's prompt cache.
    if (warm_on and prev_idx is not None and prev_idx != idx
            and in_window(stickiness_window)
            and confidence < 0.5 + sticky_margin):
        idx = prev_idx
        stickiness_applied = True
        reason = "cache_stickiness"

    return {
        "final_idx": idx,
        "routing_reason": reason,
        "warmth_applied": warmth_applied,
        "stickiness_applied": stickiness_applied,
        "confidence": confidence,
        "probabilities": probabilities,
    }


# ---------------------------------------------------------------------------
# Warmth state: server-side, seeded from the per-session JSONL ledger
# ---------------------------------------------------------------------------

def warmth_from_ledger(
    ledger_dir, session_key: str | None, tiers: list[dict[str, Any]],
    now_wall: datetime | None = None,
) -> dict[str, Any] | None:
    """Seed warmth for a session from its JSONL ledger (restart resilience).

    Reads the newest ``router`` stage event and returns
    ``{"tier_idx", "model", "elapsed"}`` — elapsed measured against the event's
    wall-clock timestamp — or None when there is no usable event.
    """
    path = ledger.session_path(ledger_dir, session_key)
    try:
        if not path.is_file():
            return None
        last: dict[str, Any] | None = None
        with path.open("rb") as handle:
            for raw_line in handle:
                try:
                    entry = json.loads(raw_line)
                except ValueError:
                    continue
                if isinstance(entry, dict) and entry.get("type") == "stage" \
                        and entry.get("feature") == "router":
                    last = entry
        if last is None:
            return None
        provenance = last.get("provenance") or {}
        tier = str(provenance.get("final_tier") or "")
        tier_ids = [str(t["id"]) for t in tiers]
        if tier not in tier_ids:
            return None
        ts = last.get("ts")
        elapsed: float | None = None
        if isinstance(ts, str) and ts:
            try:
                event_dt = datetime.fromisoformat(ts)
                now = now_wall or datetime.now(timezone.utc)
                if event_dt.tzinfo is None:
                    event_dt = event_dt.replace(tzinfo=timezone.utc)
                elapsed = (now - event_dt).total_seconds()
                if elapsed < 0:
                    elapsed = 0.0
            except ValueError:
                elapsed = None
        return {"tier_idx": tier_ids.index(tier),
                "model": str(provenance.get("target_model") or ""),
                "elapsed": elapsed}
    except OSError:
        return None


# ---------------------------------------------------------------------------
# The async classifier client (one generic SystemOne endpoint)
# ---------------------------------------------------------------------------

class ClassifyError(Exception):
    def __init__(self, reason: str, attempts: int = 1) -> None:
        super().__init__(reason)
        self.reason = reason
        self.attempts = attempts


@dataclass
class ClassifierConfig:
    endpoint_url: str
    auth_scheme: str
    api_key: str
    router_model: str
    timeout_ms: int

    @classmethod
    def from_values(cls, values: dict[str, Any]) -> "ClassifierConfig":
        return cls(
            endpoint_url=str(values.get("endpoint_url") or "").strip(),
            auth_scheme=str(values.get("auth_scheme") or "none").strip(),
            api_key=str(values.get("api_key") or "").strip(),
            router_model=str(values.get("router_model") or "").strip(),
            timeout_ms=int(values.get("classify_timeout_ms") or 8000),
        )

    def fingerprint(self) -> tuple[str, str, str, str, int]:
        """Cheap identity used by the app layer to rebuild the shared client."""
        return (self.endpoint_url, self.auth_scheme, self.api_key,
                self.router_model, self.timeout_ms)


def classify_request_payload(config: ClassifierConfig, state_text: str,
                             criteria: dict[str, str],
                             routing_question: str) -> dict[str, Any]:
    """The exact Jev envelope the client sends (exposed for tests)."""
    return {
        "model": config.router_model,
        "state": state_text,
        "questions": {
            "difficulty": {
                "type": "choice",
                "instructions": routing_question,
                "criteria": dict(criteria),
            }
        },
    }


class RouterClassifier:
    """Async client for one SystemOne decision endpoint (shared per app).

    Retry policy: transport-level failures fail fast (a retry cannot fix an
    unreachable endpoint); one extra attempt absorbs slow local cold starts
    when the configured timeout elapses; upstream 429/502/503/504 retry with
    exponential backoff capped at 4s, honoring ``Retry-After``.
    """

    RETRYABLE = (429, 502, 503, 504)
    MAX_ATTEMPTS = 4

    def __init__(self, config: ClassifierConfig, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(max(0.5, self.config.timeout_ms / 1000.0), connect=5.0),
                transport=self._transport,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        config = self.config
        if config.auth_scheme == "bearer" and config.api_key:
            # Credential stays server-side: it travels only in the
            # proxy -> decision-endpoint header, never in any client body.
            headers["authorization"] = f"Bearer {config.api_key}"
        elif config.auth_scheme == "x-api-key" and config.api_key:
            headers["x-api-key"] = config.api_key
        return headers

    async def classify(self, state_text: str, criteria: dict[str, str],
                       routing_question: str) -> dict[str, Any]:
        """One classification pass; returns the raw ``answers.difficulty`` mapping.

        ``state_text`` is the caller-composed (already truncated)
        :func:`build_state` output. Raises :class:`ClassifyError` on
        network/HTTP/shape failures after the retry policy; the answer is
        annotated with ``_latency_ms`` and ``_usage``.
        """
        config = self.config
        if not config.endpoint_url:
            raise ClassifyError("no classifier endpoint configured", 0)
        payload = classify_request_payload(config, state_text, criteria, routing_question)
        client = await self._ensure_client()
        attempts = 0
        last_reason = "unknown error"
        while attempts < self.MAX_ATTEMPTS:
            attempts += 1
            started = time.monotonic()
            try:
                resp = await client.post(config.endpoint_url, json=payload, headers=self._headers())
            except httpx.HTTPError as exc:
                raise ClassifyError(f"endpoint unreachable: {type(exc).__name__}: {exc}", attempts) from exc
            latency_ms = (time.monotonic() - started) * 1000.0
            if resp.status_code in self.RETRYABLE:
                wait_s = min(1.0 * (2 ** (attempts - 1)), 4.0)
                retry_after = resp.headers.get("Retry-After")
                if retry_after and str(retry_after).strip().isdigit():
                    wait_s = max(wait_s, min(float(retry_after), 30.0))
                await asyncio.sleep(wait_s)
                last_reason = f"HTTP {resp.status_code} after {attempts} attempts"
                continue
            if resp.status_code >= 400:
                raise ClassifyError(f"HTTP {resp.status_code}: {resp.text[:300]}", attempts)
            try:
                data = resp.json()
            except ValueError as exc:
                raise ClassifyError(f"non-JSON response: {exc}", attempts) from exc
            if not isinstance(data, dict):
                raise ClassifyError("unexpected (non-object) response", attempts)
            answers = data.get("answers")
            answer = answers.get("difficulty") if isinstance(answers, dict) else None
            if not isinstance(answer, dict) or not answer:
                raise ClassifyError(f"missing answers.difficulty: {str(data)[:200]}", attempts)
            annotated = dict(answer)
            annotated.setdefault("_latency_ms", round(latency_ms, 2))
            annotated.setdefault("_usage", data.get("usage") or {})
            annotated.setdefault("_model", data.get("model") or config.router_model)
            return annotated
        raise ClassifyError(last_reason, attempts)
