from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

from . import claude_config
from . import claude_desktop
from . import features as features_settings
from . import green as green_metrics
from . import green_impact
from . import headroom_settings
from . import ledger
from . import model_cache
from . import usage as usage_log
from .config import (
    CONFIG_PATH,
    DASHBOARD_DIR,
    LEDGER_DIR,
    LISTEN_HOST,
    LISTEN_PORT,
    SIDECAR_LOG_PATH,
    USAGE_DB_PATH,
    Settings,
    load_settings,
    save_settings,
)
from .session_key import extract_session
from .sidecar import Sidecar
from .sidecar_client import CompressError, Outcome, SidecarClient
from .translate_request import to_responses_request
from .translate_response import StreamConverter, reported_cost, to_anthropic_message, usage_from_responses
from .translate_chat import (
    AnthropicToChat,
    ChatToAnthropic,
    chat_usage_to_anthropic,
    from_chat_completions,
    to_anthropic_message_from_chat,
    to_chat_completions,
    to_chat_completions_request,
)
from .translate_responses_ingress import (
    AnthropicToResponses,
    from_responses_request,
    to_responses_response,
)
from .routing import ROUTE_ANTHROPIC, ROUTE_RESPONSES, ROUTE_CHAT, VALID_ROUTES, normalize_model, route_for, upstream_base
from .green import GREEN_OVERRIDES_PATH
from .usage import StreamUsage

logger = logging.getLogger("token_proxy")


class SettingsUpdate(BaseModel):
    upstream_url: Optional[str] = None
    api_key: Optional[str] = None
    exposed_models: Optional[list[str]] = None
    route_overrides: Optional[dict[str, str]] = None
    model_overrides: Optional[dict[str, str]] = None
    providers: Optional[list[dict[str, Any]]] = None
    model_providers: Optional[dict[str, str]] = None


class GreenFactorsUpdate(BaseModel):
    overrides: dict[str, Optional[dict[str, Any]]]


class HeadroomUpdate(BaseModel):
    values: dict[str, Any]


class DesktopApply(BaseModel):
    base_url: Optional[str] = None
    api_key: Optional[str] = None


class ClaudeApply(BaseModel):
    model_classes: Optional[dict[str, str]] = None


@dataclass
class RequestContext:
    """State threaded through the pipeline stages of one /v1/messages round trip."""

    request_id: str
    requested_model: str
    model: str
    route: str
    stream: bool
    body: dict[str, Any]
    features: dict[str, Any]
    session_key: str | None = None
    session_source: str | None = None
    upstream_model: str = ""


def error_json(status: int, message: str, kind: str = "api_error") -> JSONResponse:
    return JSONResponse(status_code=status, content={"type": "error", "error": {"type": kind, "message": message}})


def safe_json(resp: httpx.Response) -> dict[str, Any]:
    try:
        data = resp.json()
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def upstream_error(resp: httpx.Response) -> JSONResponse:
    message = resp.text[:1000]
    err = safe_json(resp).get("error")
    if isinstance(err, dict):
        message = str(err.get("message", message))
    elif isinstance(err, str):
        message = err
    return error_json(resp.status_code, message)


def header_cost(headers: httpx.Headers) -> str | None:
    for name, value in headers.items():
        if "cost" in name.lower():
            return f"{name}: {value}"
    return None


def _sidecar_settings(settings: Settings) -> dict[str, Any]:
    return settings.headroom


def create_app(
    settings_path: Path = CONFIG_PATH,
    db_path: Path = USAGE_DB_PATH,
    claude_settings_path: Path = claude_config.DEFAULT_SETTINGS_PATH,
    desktop_app_dir: Path = claude_desktop.DEFAULT_APP_DIR,
    desktop_backup_dir: Path = claude_desktop.DEFAULT_BACKUP_DIR,
    ledger_dir: Path = LEDGER_DIR,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    sidecar_transport: httpx.AsyncBaseTransport | None = None,
    manage_sidecar: bool = True,
    sidecar_log_path: Path = SIDECAR_LOG_PATH,
) -> FastAPI:
    settings = load_settings(settings_path)
    usage_log.init_db(db_path)
    client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0), transport=upstream_transport)
    sidecar = Sidecar(sidecar_log_path)
    state: dict[str, Any] = {"sidecar_client": None, "gateway_meta": {"ts": 0.0, "models": {}}}
    _background_tasks: set["asyncio.Task[None]"] = set()

    def build_sidecar_client() -> SidecarClient:
        h = _sidecar_settings(settings)
        return SidecarClient(
            settings.sidecar_url(),
            timeout_ms=int(h["request_timeout_ms"]),
            retries=int(h["retries"]),
            backoff_ms=int(h["backoff_ms"]),
            token=str(h.get("HEADROOM_PROXY_TOKEN") or ""),
            transport=sidecar_transport,
        )

    async def rebuild_sidecar_client() -> None:
        old = state["sidecar_client"]
        state["sidecar_client"] = build_sidecar_client()
        if old is not None:
            await old.aclose()

    def sidecar_client() -> SidecarClient:
        if state["sidecar_client"] is None:
            state["sidecar_client"] = build_sidecar_client()
        return state["sidecar_client"]

    def sidecar_env() -> dict[str, str]:
        return headroom_settings.sidecar_env(settings.headroom)

    def start_sidecar() -> None:
        if manage_sidecar and settings.headroom["sidecar_enabled"]:
            sidecar.start(sidecar_env(), int(settings.headroom["sidecar_port"]))
            prefetch_model_cache()

    def prefetch_model_cache() -> None:
        """The sidecar itself is offline, so this process pre-heats its model cache.

        Missing Kompress artefacts are downloaded in the background; when they
        land, a running managed sidecar is restarted so ML compression loads
        instead of the structural-only fallback. Never blocks startup or requests.
        """
        if not manage_sidecar or not settings.headroom["sidecar_enabled"]:
            return
        if settings.headroom.get("HEADROOM_DISABLE_KOMPRESS"):
            return
        model_cache.ensure(background=True, on_ready=on_model_cache_ready)
        logger.info("Kompress model cache: %s", model_cache.status().get("status"))

    def on_model_cache_ready() -> None:
        if manage_sidecar and sidecar.running and settings.headroom["sidecar_enabled"]:
            logger.info("Kompress model cache ready; restarting the sidecar to enable ML compression")
            sidecar.restart(sidecar_env(), int(settings.headroom["sidecar_port"]))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        start_sidecar()
        yield
        if state["sidecar_client"] is not None:
            await state["sidecar_client"].aclose()
        if manage_sidecar:
            sidecar.stop()
        await client.aclose()

    app = FastAPI(title="ecoGPT Proxy", lifespan=lifespan)

    @app.exception_handler(httpx.HTTPError)
    async def upstream_unreachable(_request: Request, exc: httpx.HTTPError) -> JSONResponse:
        return error_json(502, f"Could not reach the gateway at {settings.upstream_url}: {exc}")

    def base(provider: dict[str, str] | None = None) -> str:
        # effective_providers() already normalized the URL through upstream_base().
        return provider["url"] if provider else upstream_base(settings.upstream_url)

    def upstream_headers(request: Request, passthrough: bool, provider: dict[str, str] | None = None) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        key = settings.effective_key(provider)
        if key:
            # Credential substitution: the gateway always speaks with its own
            # upstream key and never leaks or honors the client's dummy token.
            headers["authorization"] = f"Bearer {key}"
        else:
            # Transparent mode (no configured key): forward client credentials.
            for name in ("authorization", "x-api-key"):
                if name in request.headers:
                    headers[name] = request.headers[name]
        if passthrough:
            headers["anthropic-version"] = request.headers.get("anthropic-version", "2023-06-01")
            if "anthropic-beta" in request.headers:
                headers["anthropic-beta"] = request.headers["anthropic-beta"]
        return headers

    def check_exposed(model: str) -> JSONResponse | None:
        if model in settings.exposed_models:
            return None
        return error_json(
            404,
            f"Model '{model}' is not exposed by the proxy. Enable it in the dashboard at "
            f"http://{LISTEN_HOST}:{LISTEN_PORT}/",
            "not_found_error",
        )

    def log_usage(
        *,
        model: str,
        route: str,
        stream: bool,
        status: int,
        usage: dict[str, Any] | None,
        cost: str | None,
        duration_ms: float,
        compression: dict[str, Any] | None,
        impacts: dict[str, Any] | None = None,
        ttft_ms: float | None = None,
        tps: float | None = None,
    ) -> None:
        usage_log.record(
            db_path,
            model=model,
            route=route,
            stream=stream,
            status=status,
            usage=usage,
            cost=cost,
            duration_ms=int(duration_ms),
            compression=compression,
            impacts=impacts,
            ttft_ms=int(ttft_ms) if ttft_ms is not None else None,
            tps=tps,
        )

    # --- Ledger observers (tracing flag gated; every write fails open) ---------------

    def ledger_request(ctx: RequestContext) -> None:
        if not ctx.features["tracing"]:
            return
        ledger.request_event(
            ledger_dir,
            session_key=ctx.session_key,
            session_source=ctx.session_source,
            request_id=ctx.request_id,
            run_label=ctx.features["run_label"],
            requested_model=ctx.requested_model,
            model=ctx.model,
            route=ctx.route,
            stream=ctx.stream,
            body=ctx.body,
            upstream_model=ctx.upstream_model,
        )

    def ledger_stage(ctx: RequestContext, record: dict[str, Any], duration_ms: float, *, error_message: str | None = None) -> None:
        if not ctx.features["tracing"]:
            return
        provenance = dict(record)
        provenance["duration_ms"] = duration_ms
        ledger.stage_event(
            ledger_dir,
            session_key=ctx.session_key,
            session_source=ctx.session_source,
            request_id=ctx.request_id,
            run_label=ctx.features["run_label"],
            feature="compression",
            provenance=provenance,
        )
        if error_message:
            ledger.error_event(
                ledger_dir,
                session_key=ctx.session_key,
                session_source=ctx.session_source,
                request_id=ctx.request_id,
                run_label=ctx.features["run_label"],
                stage="compression",
                message=error_message,
            )

    def ledger_error(ctx: RequestContext, stage: str, message: str) -> None:
        if not ctx.features["tracing"]:
            return
        ledger.error_event(
            ledger_dir,
            session_key=ctx.session_key,
            session_source=ctx.session_source,
            request_id=ctx.request_id,
            run_label=ctx.features["run_label"],
            stage=stage,
            message=message,
        )

    def persist_settings(*, keep: set[str] = frozenset()) -> None:
        """Save the in-memory settings, refreshing blocks from disk first except the ones being saved.

        The ablation harness may rewrite config.json at any time, so a dashboard save
        must not clobber a newer on-disk ``features`` or ``green`` block that this
        dashboard process has not seen.
        """
        if "features" not in keep:
            settings.features = features_settings.load_features(settings_path)
        if "green" not in keep:
            settings.green = _green_settings_from_disk()
        save_settings(settings, settings_path)

    def _green_settings_from_disk() -> dict[str, Any]:
        try:
            raw = json.loads(Path(settings_path).read_text(encoding="utf-8"))
            stored = raw.get("green") if isinstance(raw, dict) else {}
        except (OSError, ValueError):
            stored = {}
        if not isinstance(stored, dict):
            stored = {}
        known = {entry["key"] for entry in green_metrics.GREEN_SCHEMA}
        return green_metrics.normalize({k: v for k, v in stored.items() if k in known})

    _green_unknown_models: set[tuple[str, str]] = set()

    def compute_impacts(
        ctx: RequestContext,
        usage: dict[str, Any] | None,
        *,
        duration_ms: float = 0.0,
        ttft_ms: float | None = None,
    ) -> dict[str, Any] | None:
        """Green metrics observer: flag-gated, fail-open, never mutates the request."""
        if not ctx.features["green_metrics"] or not usage:
            return None
        # TPS follows the same convention as the reference demo: tokens after the
        # first over generation time; only meaningful when TTFT was observed.
        tps = None
        if ttft_ms and ttft_ms > 0 and duration_ms > ttft_ms and (usage.get("output_tokens") or 0) > 1:
            generation_s = (duration_ms - ttft_ms) / 1000.0
            if generation_s > 0:
                tps = (usage["output_tokens"] - 1) / generation_s
        try:
            # Attribute impacts to the *effective* upstream model (tokens are
            # consumed by it, even when an alias name faces the client).
            impacts = green_impact.compute(
                ctx.upstream_model or ctx.model,
                usage,
                settings.green,
                duration_ms=duration_ms,
                ttft_ms=ttft_ms,
                tps=tps,
            )
        except Exception as exc:  # noqa: BLE001 - green must never break a response
            logger.error("green metrics failed: %s", exc)
            ledger_error(ctx, "green", f"{type(exc).__name__}: {exc}")
            return None
        # Observed stream timings are part of the impact record (usage DB + ledger).
        if ttft_ms is not None:
            impacts["ttft_ms"] = ttft_ms
        if tps is not None:
            impacts["tps"] = tps
        status = impacts.get("impact_status")
        if status == "unconfigured_model":
            marker = (ctx.session_key or ledger.NO_SESSION_KEY, ctx.upstream_model or ctx.model)
            if marker not in _green_unknown_models:
                _green_unknown_models.add(marker)
                ledger_error(ctx, "green_unknown_model",
                             f"no green parameters for model '{ctx.upstream_model or ctx.model}'; "
                             "impacts recorded as null")
        elif status == "missing_usage":
            marker = (ctx.session_key or ledger.NO_SESSION_KEY, "missing_usage")
            if marker not in _green_unknown_models:
                _green_unknown_models.add(marker)
                ledger_error(ctx, "green_missing_usage",
                             str(impacts.get("impact_provenance") or "no reliable usage for green estimation"))
        return impacts

    def observe_response(
        ctx: RequestContext,
        record: dict[str, Any],
        *,
        route: str,
        stream: bool,
        status: int,
        usage: dict[str, Any] | None,
        cost: str | None,
        duration_ms: float,
        stop_reason: str | None = None,
        content: Any = None,
        ttft_ms: float | None = None,
    ) -> None:
        """Log the usage row, write the ledger response event, and schedule the sidecar relay."""
        impacts = compute_impacts(ctx, usage, duration_ms=duration_ms, ttft_ms=ttft_ms)
        tps = impacts.get("tps") if isinstance(impacts, dict) else None
        log_usage(model=ctx.model, route=route, stream=stream, status=status, usage=usage,
                  cost=cost, duration_ms=duration_ms, compression=record, impacts=impacts,
                  ttft_ms=ttft_ms, tps=tps)
        if ctx.features["tracing"]:
            ledger.response_event(
                ledger_dir,
                session_key=ctx.session_key,
                session_source=ctx.session_source,
                request_id=ctx.request_id,
                run_label=ctx.features["run_label"],
                status=status,
                duration_ms=duration_ms,
                usage=usage,
                stop_reason=stop_reason,
                content=content,
                impacts=impacts,
            )
        schedule_relay(record, status=status, latency_ms=duration_ms, usage=usage or {})

    def observe_upstream_error(ctx: RequestContext, record: dict[str, Any], *, status: int, duration_ms: float) -> None:
        """Ledger response event + sidecar relay cleanup for failed upstream calls."""
        if ctx.features["tracing"]:
            ledger.response_event(
                ledger_dir,
                session_key=ctx.session_key,
                session_source=ctx.session_source,
                request_id=ctx.request_id,
                run_label=ctx.features["run_label"],
                status=status,
                duration_ms=duration_ms,
                usage=None,
            )
        schedule_relay(record, status=status, latency_ms=duration_ms, usage={})

    def compression_record(outcome: Outcome | None, *, fallback: str | None, attempts: int) -> dict[str, Any]:
        if outcome is None:
            return {"compressed": False, "fallback": fallback, "attempts": attempts}
        return {
            "compressed": outcome.tokens_saved > 0,
            "tokens_before": outcome.tokens_before,
            "tokens_after": outcome.tokens_after,
            "tokens_saved": outcome.tokens_saved,
            "session_source": outcome.session_source,
            "session_replayed": outcome.session_replayed,
            "attempts": outcome.attempts,
            "fallback": fallback,
            "turn_id": outcome.turn_id,
            "relayed": False,
        }

    def extract_identity(request: Request, ctx: RequestContext) -> None:
        h = settings.headroom
        session = extract_session(
            dict(request.headers),
            ctx.body,
            {
                "claude_header": h["session_claude_header"],
                "claude_metadata": h["session_claude_metadata"],
                "codex_header": h["session_codex_header"],
                "codex_prompt_cache_key": h["session_codex_prompt_cache_key"],
            },
        )
        if session is not None:
            ctx.session_key = session.key
            ctx.session_source = session.source

    async def compression_stage(ctx: RequestContext) -> tuple[dict[str, Any], Outcome | None]:
        """Flag-gated compression stage. Returns (compression record, outcome); mutates ctx.body."""
        h = settings.headroom
        if not ctx.features["compression"]:
            record = compression_record(None, fallback="compression_disabled", attempts=0)
            ledger_stage(ctx, record, 0)
            return record, None
        if not h["sidecar_enabled"]:
            record = compression_record(None, fallback="sidecar_disabled", attempts=0)
            ledger_stage(ctx, record, 0)
            return record, None
        if ctx.session_key is None:
            record = compression_record(None, fallback="no_session_key", attempts=0)
            ledger_stage(ctx, record, 0)
            return record, None
        started = time.monotonic()
        try:
            outcome = await sidecar_client().compress(
                ctx.body,
                model=ctx.model,
                token_budget=int(h["token_budget"]),
                session_key=ctx.session_key,
                session_source=ctx.session_source,
                values=h,
            )
        except CompressError as exc:
            duration_ms = int((time.monotonic() - started) * 1000)
            if h["failure_policy"] == "fail_request":
                raise _CompressionFailed(exc) from exc
            logger.warning("sidecar compression failed after %s attempts (%s); forwarding original", exc.attempts, exc.reason)
            record = compression_record(None, fallback=f"forwarded_original: {exc.reason}", attempts=exc.attempts)
            ledger_stage(ctx, record, duration_ms, error_message=exc.reason)
            return record, None
        duration_ms = int((time.monotonic() - started) * 1000)
        ctx.body = outcome.body
        record = compression_record(outcome, fallback=None, attempts=outcome.attempts)
        ledger_stage(ctx, record, duration_ms)
        return record, outcome

    def schedule_relay(record: dict[str, Any], *, status: int, latency_ms: float, usage: dict[str, Any]) -> None:
        turn_id = record.get("turn_id")
        if not turn_id or not settings.headroom["relay_usage"] or not record.get("relay_pending"):
            return
        record["relay_pending"] = False
        task = asyncio.create_task(_relay_turn(turn_id, status, latency_ms, usage))
        _background_tasks.add(task)
        task.add_done_callback(_relay_done)

    def _relay_done(task: "asyncio.Task[None]") -> None:
        _background_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("usage relay task failed: %s", task.exception())

    async def _relay_turn(turn_id: str, status: int, latency_ms: float, usage: dict[str, Any]) -> None:
        payload_usage = {
            "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0,
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0) or 0,
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) or 0,
        }
        ok = await sidecar_client().relay(turn_id, status=status, latency_ms=latency_ms, usage=payload_usage)
        if ok:
            usage_log.mark_relayed(db_path, turn_id)

    async def passthrough(request: Request, ctx: RequestContext, record: dict[str, Any], outcome: Outcome | None) -> Response:
        provider = settings.provider_for(ctx.model)
        url = f"{base(provider)}/v1/messages"
        headers = upstream_headers(request, passthrough=True, provider=provider)
        stream = ctx.stream
        started = time.monotonic()
        payload = json.dumps(ctx.body).encode("utf-8")
        if outcome is not None and "relay_usage" in outcome.obligations:
            record["relay_pending"] = True

        if not stream:
            resp = await client.post(url, content=payload, headers=headers)
            data = safe_json(resp)
            usage = data.get("usage") if resp.status_code < 400 else None
            observe_response(ctx, record, route=ROUTE_ANTHROPIC, stream=False, status=resp.status_code,
                             usage=usage, cost=header_cost(resp.headers),
                             duration_ms=(time.monotonic() - started) * 1000,
                             stop_reason=data.get("stop_reason") if isinstance(data.get("stop_reason"), str) else None,
                             content=data.get("content"))
            return Response(content=resp.content, status_code=resp.status_code,
                            media_type=resp.headers.get("content-type", "application/json"))

        req = client.build_request("POST", url, content=payload, headers=headers)
        resp = await client.send(req, stream=True)
        media = resp.headers.get("content-type", "text/event-stream")
        if resp.status_code >= 400:
            await resp.aread()
            await resp.aclose()
            observe_upstream_error(ctx, record, status=resp.status_code,
                                   duration_ms=(time.monotonic() - started) * 1000)
            return Response(content=resp.content, status_code=resp.status_code, media_type=media)

        tracker = StreamUsage()

        async def relay():
            try:
                async for chunk in resp.aiter_bytes():
                    tracker.feed(chunk)
                    yield chunk
            finally:
                await resp.aclose()
                observe_response(ctx, record, route=ROUTE_ANTHROPIC, stream=True, status=resp.status_code,
                                 usage=tracker.usage, cost=header_cost(resp.headers),
                                 duration_ms=(time.monotonic() - started) * 1000,
                                 stop_reason=tracker.stop_reason,
                                 ttft_ms=tracker.ttft_ms(started))

        return StreamingResponse(relay(), status_code=resp.status_code, media_type=media)

    async def via_responses(request: Request, ctx: RequestContext, record: dict[str, Any], outcome: Outcome | None) -> Response:
        provider = settings.provider_for(ctx.model)
        upstream_body = to_responses_request(ctx.body, ctx.upstream_model)
        url = f"{base(provider)}/v1/responses"
        headers = upstream_headers(request, passthrough=False, provider=provider)
        stream = ctx.stream
        started = time.monotonic()
        if outcome is not None and "relay_usage" in outcome.obligations:
            record["relay_pending"] = True

        if not stream:
            resp = await client.post(url, json=upstream_body, headers=headers)
            if resp.status_code >= 400:
                observe_upstream_error(ctx, record, status=resp.status_code,
                                       duration_ms=(time.monotonic() - started) * 1000)
                return upstream_error(resp)
            data = safe_json(resp)
            usage = usage_from_responses(data.get("usage"))
            message = to_anthropic_message(data, ctx.requested_model)
            observe_response(ctx, record, route=ROUTE_RESPONSES, stream=False, status=resp.status_code,
                             usage=usage, cost=header_cost(resp.headers) or reported_cost(data.get("usage")),
                             duration_ms=(time.monotonic() - started) * 1000,
                             stop_reason=message.get("stop_reason"),
                             content=message.get("content"))
            return JSONResponse(message)

        req = client.build_request("POST", url, json=upstream_body, headers=headers)
        resp = await client.send(req, stream=True)
        if resp.status_code >= 400:
            await resp.aread()
            error = upstream_error(resp)
            await resp.aclose()
            observe_upstream_error(ctx, record, status=resp.status_code,
                                   duration_ms=(time.monotonic() - started) * 1000)
            return error

        converter = StreamConverter(ctx.requested_model)

        async def relay():
            data_lines: list[str] = []
            first_content: float | None = None
            try:
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        data_lines.append(line[5:].strip())
                    elif not line and data_lines:
                        raw = "\n".join(data_lines)
                        data_lines = []
                        try:
                            event = json.loads(raw)
                        except ValueError:
                            continue
                        if isinstance(event, dict):
                            # First content-bearing event ≈ TTFT (same convention as
                            # the reference demo; approximate for reasoning models).
                            if (first_content is None
                                    and event.get("type") in ("response.output_text.delta",
                                                             "response.reasoning_summary_text.delta")):
                                first_content = time.monotonic()
                            for chunk in converter.feed(event):
                                yield chunk
            finally:
                await resp.aclose()
                observe_response(ctx, record, route=ROUTE_RESPONSES, stream=True, status=resp.status_code,
                                 usage=converter.usage, cost=header_cost(resp.headers) or converter.cost,
                                 duration_ms=(time.monotonic() - started) * 1000,
                                 stop_reason=converter.stop_reason,
                                 ttft_ms=(first_content - started) * 1000 if first_content is not None else None)

        return StreamingResponse(relay(), media_type="text/event-stream")

    async def via_chat(request: Request, ctx: RequestContext, record: dict[str, Any], outcome: Outcome | None) -> Response:
        """Messages -> Chat Completions upstream, CC response/SS back to Messages."""
        provider = settings.provider_for(ctx.model)
        upstream_body = to_chat_completions_request(ctx.body, ctx.upstream_model)
        url = f"{base(provider)}/v1/chat/completions"
        headers = upstream_headers(request, passthrough=False, provider=provider)
        stream = ctx.stream
        started = time.monotonic()
        if outcome is not None and "relay_usage" in outcome.obligations:
            record["relay_pending"] = True

        if not stream:
            resp = await client.post(url, json=upstream_body, headers=headers)
            if resp.status_code >= 400:
                observe_upstream_error(ctx, record, status=resp.status_code,
                                       duration_ms=(time.monotonic() - started) * 1000)
                return upstream_error(resp)
            data = safe_json(resp)
            message = to_anthropic_message_from_chat(data, ctx.requested_model)
            observe_response(ctx, record, route=ROUTE_CHAT, stream=False, status=resp.status_code,
                             usage=message.get("usage"), cost=header_cost(resp.headers) or reported_cost(data.get("usage")),
                             duration_ms=(time.monotonic() - started) * 1000,
                             stop_reason=message.get("stop_reason"),
                             content=message.get("content"))
            return JSONResponse(message)

        req = client.build_request("POST", url, json=upstream_body, headers=headers)
        resp = await client.send(req, stream=True)
        if resp.status_code >= 400:
            await resp.aread()
            error = upstream_error(resp)
            await resp.aclose()
            observe_upstream_error(ctx, record, status=resp.status_code,
                                   duration_ms=(time.monotonic() - started) * 1000)
            return error

        converter = ChatToAnthropic(ctx.requested_model)

        async def relay():
            first_content: float | None = None
            try:
                async for chunk in resp.aiter_bytes():
                    output = converter.feed(chunk)
                    if output:
                        # First translated chunk approximates the first token (the
                        # chat upstream emits its role delta when generation starts);
                        # documented as approximate, like the demo's TTFT caveat.
                        if first_content is None:
                            first_content = time.monotonic()
                        yield output
            finally:
                await resp.aclose()
                stream_usage = chat_usage_to_anthropic(converter.usage_raw)
                if not stream_usage.get("output_tokens") and not stream_usage.get("input_tokens"):
                    stream_usage = None
                observe_response(ctx, record, route=ROUTE_CHAT, stream=True, status=resp.status_code,
                                 usage=stream_usage, cost=header_cost(resp.headers) or converter.cost,
                                 duration_ms=(time.monotonic() - started) * 1000,
                                 stop_reason=converter.stop_reason,
                                 ttft_ms=(first_content - started) * 1000 if first_content is not None else None)

        return StreamingResponse(relay(), media_type="text/event-stream")

    @app.post("/v1/messages")
    async def messages(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return error_json(400, "Request body is not valid JSON", "invalid_request_error")
        if not isinstance(body, dict):
            return error_json(400, "Request body must be a JSON object", "invalid_request_error")
        return await messages_roundtrip(request, body)

    async def messages_roundtrip(request: Request, body: dict[str, Any]) -> Response:
        """The full /v1/messages pipeline, shared by Messages and ingress formats."""
        requested = body.get("model", "")
        model = normalize_model(requested)
        blocked = check_exposed(model)
        if blocked is not None:
            return blocked
        body["model"] = model
        ctx = build_context(request, body)

        ledger_request(ctx)
        try:
            record, outcome = await compression_stage(ctx)
        except _CompressionFailed as failed:
            ledger_error(ctx, "compression", f"failed after {failed.inner.attempts} attempts: {failed.inner.reason}")
            return error_json(
                503,
                f"Headroom sidecar unavailable after {failed.inner.attempts} attempts: {failed.inner.reason}",
            )
        # Model aliasing: forward under the effective upstream model id. The
        # ledger has already recorded the request under the client-facing name.
        if ctx.upstream_model != ctx.model:
            ctx.body["model"] = ctx.upstream_model
        if ctx.route == ROUTE_RESPONSES:
            return await via_responses(request, ctx, record, outcome)
        if ctx.route == ROUTE_CHAT:
            return await via_chat(request, ctx, record, outcome)
        return await passthrough(request, ctx, record, outcome)
    def build_context(request: Request, body: dict[str, Any]) -> RequestContext:
        """Identity stage: read flags from config at request time, compute route, extract session."""
        flags = features_settings.load_features(settings_path)
        model = body["model"]
        ctx = RequestContext(
            request_id=str(uuid.uuid4()),
            requested_model=body.get("model", ""),
            model=model,
            route=route_for(model, settings.route_overrides),
            stream=bool(body.get("stream")),
            body=body,
            features=flags,
            upstream_model=settings.model_overrides.get(model, model),
        )
        extract_identity(request, ctx)
        return ctx

    @app.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return error_json(400, "Request body is not valid JSON", "invalid_request_error")
        if not isinstance(body, dict):
            return error_json(400, "Request body must be a JSON object", "invalid_request_error")
        model = normalize_model(body.get("model", ""))
        blocked = check_exposed(model)
        if blocked is not None:
            return blocked
        body["model"] = model
        route = route_for(model, settings.route_overrides)
        if route in (ROUTE_RESPONSES, ROUTE_CHAT):
            return JSONResponse({"input_tokens": max(1, len(json.dumps(body)) // 4)})
        provider = settings.provider_for(model)
        resp = await client.post(
            f"{base(provider)}/v1/messages/count_tokens",
            json=body,
            headers=upstream_headers(request, passthrough=True, provider=provider),
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "application/json"),
        )

    GATEWAY_META_TTL_SECONDS = 300.0

    def pretty_model_name(model_id: str) -> str:
        """Human-readable fallback when the gateway has no metadata for a model."""
        base_name, _, variant = model_id.partition(":")
        words = [w for w in re.split(r"[-_/.\s]+", base_name) if w]
        titled = " ".join(w if not w.islower() else w[:1].upper() + w[1:] for w in words)
        return f"{titled} ({variant})" if variant else titled

    async def gateway_model_metadata() -> dict[str, dict[str, Any]]:
        """Model metadata from the primary gateway (display_name/name, sizes), cached.

        Claude Code's model picker and Claude Desktop's gateway model discovery
        both render display names, so /v1/models forwards real names instead of
        the raw model id. Failures leave the previous cache in place.
        """
        entry = state["gateway_meta"]
        now = time.time()
        if entry["models"] and now - entry["ts"] < GATEWAY_META_TTL_SECONDS:
            return entry["models"]
        provider = settings.default_provider()
        headers = {}
        if settings.effective_key(provider):
            headers["authorization"] = f"Bearer {settings.effective_key(provider)}"
        try:
            resp = await client.get(f"{base(provider)}/v1/models", headers=headers, timeout=15.0)
        except httpx.HTTPError:
            return entry["models"]
        if resp.status_code < 400:
            meta: dict[str, dict[str, Any]] = {}
            for item in safe_json(resp).get("data", []):
                if isinstance(item, dict) and item.get("id"):
                    meta[str(item["id"])] = item
            entry["models"] = meta
            entry["ts"] = now
        return entry["models"]

    def display_name_for(model_id: str, meta: dict[str, dict[str, Any]]) -> str:
        """Friendly label for a model id: gateway metadata, else pretty fallback."""
        info = meta.get(model_id) or {}
        return str(info.get("display_name") or info.get("name") or "") or pretty_model_name(model_id)

    def model_row(model_id: str, meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
        info = meta.get(model_id) or {}
        display_name = display_name_for(model_id, meta)
        row: dict[str, Any] = {
            "type": "model",
            "id": model_id,
            "display_name": display_name,
            "created_at": "1970-01-01T00:00:00Z",
        }
        description = info.get("description")
        if description:
            row["description"] = str(description)[:100]
        context_length = info.get("context_length", info.get("max_input_tokens"))
        if isinstance(context_length, (int, float)) and context_length > 0:
            row["max_input_tokens"] = int(context_length)
        return row

    @app.get("/v1/models")
    async def list_models(limit: int | None = None, after_id: str | None = None):
        meta = await gateway_model_metadata()
        ids = list(settings.exposed_models)
        if after_id and after_id in ids:
            ids = ids[ids.index(after_id) + 1:]
        if isinstance(limit, int) and limit >= 0:
            ids = ids[:limit]
        data = [model_row(m, meta) for m in ids]
        return {
            "data": data,
            "has_more": False,
            "first_id": data[0]["id"] if data else None,
            "last_id": data[-1]["id"] if data else None,
        }

    def openai_error(status_code: int, message: str, kind: str = "api_error") -> JSONResponse:
        return JSONResponse({"error": {"message": message, "type": kind, "param": None, "code": kind}},
                            status_code=status_code)

    def object_body(resp: Response) -> dict[str, Any]:
        try:
            data = json.loads(resp.body)
        except (ValueError, AttributeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _wrap_ingress_sse(resp: StreamingResponse, converter) -> StreamingResponse:
        """Re-encode an Anthropic-SSE StreamingResponse into the ingress format."""

        async def relay():
            async for chunk in resp.body_iterator:
                output = converter.feed(chunk)
                if output:
                    yield output

        return StreamingResponse(relay(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return openai_error(400, "Request body is not valid JSON", "invalid_request_error")
        if not isinstance(body, dict):
            return openai_error(400, "Request body must be a JSON object", "invalid_request_error")
        stream = bool(body.get("stream"))
        requested_model = str(body.get("model", ""))
        messages_body = from_chat_completions(body)
        messages_body["model"] = normalize_model(requested_model)
        resp = await messages_roundtrip(request, messages_body)
        if resp.status_code >= 400:
            error = object_body(resp).get("error") or {}
            return openai_error(resp.status_code, str(error.get("message", "Upstream error")),
                                str(error.get("type", "api_error")))
        if not stream:
            return JSONResponse(to_chat_completions(object_body(resp), requested_model))
        return _wrap_ingress_sse(resp, AnthropicToChat(requested_model))

    @app.post("/v1/responses")
    async def responses_ingress(request: Request):
        try:
            body = await request.json()
        except ValueError:
            return openai_error(400, "Request body is not valid JSON", "invalid_request_error")
        if not isinstance(body, dict):
            return openai_error(400, "Request body must be a JSON object", "invalid_request_error")
        stream = bool(body.get("stream"))
        requested_model = str(body.get("model", ""))
        messages_body = from_responses_request(body)
        messages_body["model"] = normalize_model(requested_model)
        resp = await messages_roundtrip(request, messages_body)
        if resp.status_code >= 400:
            error = object_body(resp).get("error") or {}
            return openai_error(resp.status_code, str(error.get("message", "Upstream error")),
                                str(error.get("type", "api_error")))
        if not stream:
            return JSONResponse(to_responses_response(object_body(resp), requested_model))
        return _wrap_ingress_sse(resp, AnthropicToResponses(requested_model))

    def claude_state() -> dict[str, Any]:
        state_ = claude_config.status(claude_settings_path)
        state_["applied"] = state_.get("base_url") == settings.proxy_url()
        return state_

    def desktop_state() -> dict[str, Any]:
        state_ = claude_desktop.status(desktop_app_dir, desktop_backup_dir)
        state_["applied"] = state_.get("base_url") == settings.proxy_url()
        return state_

    @app.get("/", response_model=None)
    async def dashboard():
        return FileResponse(DASHBOARD_DIR / "index.html")

    @app.get("/api/state")
    async def state_view():
        meta = await gateway_model_metadata()

        def _display(model_id: str) -> str:
            # Desktop aliases (claude-eco-N) label as the real model they route to.
            return display_name_for(settings.model_overrides.get(model_id, model_id), meta)

        return {
            "proxy_url": settings.proxy_url(),
            "settings": {
                "upstream_url": settings.upstream_url,
                "api_key_set": bool(settings.effective_key()),
                "exposed_models": settings.exposed_models,
                "model_display_names": {m: _display(m) for m in settings.exposed_models},
                "route_overrides": settings.route_overrides,
                "model_overrides": settings.model_overrides,
                "providers": [
                    {**p, "api_key": None, "api_key_set": bool(p["api_key"] or "")}
                    for p in settings.effective_providers()
                ],
                "model_providers": settings.model_providers,
            },
            "claude": claude_state(),
            "desktop": desktop_state(),
        }

    @app.put("/api/settings")
    async def update_settings(update: SettingsUpdate):
        new_url = None
        if update.upstream_url is not None:
            new_url = upstream_base(update.upstream_url)
        if update.providers is not None:
            normalized_providers = []
            for index, provider in enumerate(update.providers[:3]):
                url = upstream_base(str(provider.get("url") or "").strip())
                if url:
                    parts = urlsplit(url)
                    if parts.scheme not in ("http", "https") or not parts.hostname:
                        return error_json(400, "Provider URLs must start with http:// or https://",
                                          "invalid_request_error")
                    if parts.port == LISTEN_PORT:
                        return error_json(400, f"A provider cannot be the proxy itself (port {LISTEN_PORT})",
                                          "invalid_request_error")
                name = str(provider.get("name") or "").strip() or ("primary" if index == 0 else f"provider-{index + 1}")
                key = str(provider.get("api_key") or "").strip()
                if not key:
                    # empty key field = keep the stored key for this name/url pair
                    for existing in settings.effective_providers():
                        if existing["name"] == name and existing["url"] == url and existing["api_key"]:
                            key = existing["api_key"]
                            break
                normalized_providers.append({"name": name, "url": url, "api_key": key})
            names = [p["name"] for p in normalized_providers if p["url"]]
            if not names:
                return error_json(400, "At least one provider needs a URL", "invalid_request_error")
            if len(set(names)) != len(names):
                return error_json(400, "Provider names must be unique", "invalid_request_error")
            if update.model_providers is not None:
                unknown = [v for v in update.model_providers.values() if v and v not in names]
                if unknown:
                    return error_json(400, f"Unknown provider name: {unknown[0]}", "invalid_request_error")
            settings.providers = normalized_providers
            # keep legacy top-level fields in sync (they back provider[0] if config is hand-edited)
            settings.upstream_url = next(p["url"] for p in normalized_providers if p["url"])
            settings.api_key = normalized_providers[0]["api_key"]
        elif update.model_providers is not None:
            names = [p["name"] for p in settings.effective_providers()]
            unknown = [v for v in update.model_providers.values() if v and v not in names]
            if unknown:
                return error_json(400, f"Unknown provider name: {unknown[0]}", "invalid_request_error")
        if update.model_providers is not None:
            settings.model_providers = {k: v for k, v in update.model_providers.items() if v}
        else:
            settings.model_providers = {k: v for k, v in settings.model_providers.items()
                                        if v in {p["name"] for p in settings.effective_providers()}}
        if new_url is not None and update.providers is None:
            parts = urlsplit(new_url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                return error_json(400, "Upstream URL must start with http:// or https://", "invalid_request_error")
            if parts.port == LISTEN_PORT:
                return error_json(400, f"Upstream cannot be the proxy itself (port {LISTEN_PORT})",
                                  "invalid_request_error")
            settings.upstream_url = new_url
        if update.api_key is not None and update.providers is None:
            settings.api_key = update.api_key.strip()
        if update.route_overrides is not None and any(v not in VALID_ROUTES for v in update.route_overrides.values()):
            return error_json(400, "Route must be 'anthropic', 'responses' or 'chat'", "invalid_request_error")
        if update.model_overrides is not None:
            aliases = {str(k).strip(): str(v).strip() for k, v in update.model_overrides.items()}
            if any(not k or not v for k, v in aliases.items()):
                return error_json(400, "Model overrides must map non-empty names to non-empty model ids",
                                  "invalid_request_error")
            settings.model_overrides = aliases
        if new_url is not None:
            settings.upstream_url = new_url
        if update.api_key is not None:
            settings.api_key = update.api_key.strip()
        if update.exposed_models is not None:
            settings.exposed_models = list(dict.fromkeys(update.exposed_models))
        if update.route_overrides is not None:
            settings.route_overrides = dict(update.route_overrides)
        persist_settings()
        return {"ok": True}

    @app.get("/api/features")
    async def features_view():
        return {
            "schema": features_settings.FEATURES,
            "values": features_settings.load_features(settings_path),
        }

    @app.put("/api/features")
    async def features_update(update: HeadroomUpdate):
        current = features_settings.load_features(settings_path)
        try:
            normalized = features_settings.normalize({**current, **update.values})
        except ValueError as exc:
            return error_json(400, str(exc), "invalid_request_error")
        settings.features = normalized
        persist_settings(keep={"features"})
        return {"ok": True}

    @app.get("/api/green")
    async def green_view():
        mix, mix_source = green_impact.electricity(settings.green.get("region_country", "WOR"))
        return {
            "schema": green_metrics.GREEN_SCHEMA,
            "values": settings.green,
            "attribution": green_metrics.attribution(),
            "electricity": {**mix, "source": mix_source},
        }

    @app.get("/api/green/factors")
    async def green_factors_view():
        factors = green_metrics.load_factors()
        overrides = green_metrics.load_overrides()
        rows = (factors.get("models") or {}).items()
        models = []
        for key, row in sorted(rows):
            params = green_impact._params_from_row(row)
            models.append(
                {
                    "model": key,
                    "basis": row.get("basis"),
                    "arch": row.get("arch"),
                    "wh_per_output_token": row.get("wh_per_output_token") or {},
                    "overridden": key in overrides,
                    "mode": green_impact._row_mode(row),
                    "params": {"active_b": params["active_b"], "total_b": params["total_b"]} if params else None,
                }
            )
        return {"models": models, "attribution": green_metrics.attribution()}

    @app.put("/api/green/factors")
    async def green_factors_update(update: GreenFactorsUpdate):
        merged = green_metrics.load_overrides()
        for key, value in update.overrides.items():
            if value is None:
                merged.pop(key, None)
                continue
            if not isinstance(value, dict):
                return error_json(400, f"{key}: expected an object", "invalid_request_error")
            parsed: dict[str, Any] = {"basis": "user-override"}
            wpt = value.get("wh_per_output_token")
            if wpt is not None:
                if not isinstance(wpt, dict):
                    return error_json(400, f"{key}: expected wh_per_output_token {{min, mean, max}}",
                                      "invalid_request_error")
                bounds = {}
                for bound in ("min", "mean", "max"):
                    try:
                        bounds[bound] = float(wpt[bound])
                    except (KeyError, TypeError, ValueError):
                        return error_json(400, f"{key}: wh_per_output_token.{bound} must be a number",
                                          "invalid_request_error")
                parsed["wh_per_output_token"] = bounds
            params: dict[str, float] = {}
            for field in ("active_b", "total_b"):
                raw = value.get(field)
                if raw is None or raw == "":
                    continue
                try:
                    params[field] = float(raw)
                except (TypeError, ValueError):
                    return error_json(400, f"{key}: {field} must be a number", "invalid_request_error")
                if params[field] <= 0:
                    return error_json(400, f"{key}: {field} must be positive", "invalid_request_error")
            if "total_b" in params and "active_b" in params and params["active_b"] > params["total_b"]:
                return error_json(400, f"{key}: active_b must not exceed total_b", "invalid_request_error")
            if params:
                parsed["params"] = params
            if not wpt and not params:
                return error_json(400, f"{key}: provide wh_per_output_token or active_b/total_b",
                                  "invalid_request_error")
            merged[key] = parsed
        GREEN_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
        GREEN_OVERRIDES_PATH.write_text(json.dumps({"models": merged}, indent=2), encoding="utf-8")
        return {"ok": True, "overrides": merged}

    @app.post("/api/green/models/match")
    async def green_models_match():
        """Re-read the vendored EcoLogits catalogue (drop the cache) and report matches."""
        green_impact._catalogue_cache.clear()
        factors = green_metrics.load_factors()
        keys = sorted((factors.get("models") or {}).keys())
        matches = {key: green_impact.catalogue_match(key) is not None for key in keys}
        revision = green_impact.ECOLOGITS_REVISION
        return {"ok": True, "revision": revision, "matches": matches}

    @app.put("/api/green")
    async def green_update(update: HeadroomUpdate):
        current = dict(settings.green)
        try:
            normalized = green_metrics.normalize({**current, **update.values})
        except ValueError as exc:
            return error_json(400, str(exc), "invalid_request_error")
        settings.green = normalized
        persist_settings(keep={"green"})
        return {"ok": True}

    @app.get("/api/green/panel")
    async def green_panel_view():
        return usage_log.green_panel(db_path)

    @app.get("/api/green/coverage")
    async def green_coverage_view():
        return usage_log.coverage(db_path)

    @app.get("/api/usage/export.csv")
    async def usage_export_csv():
        """Full request ledger as CSV (all columns, including green phases)."""

        def rows() -> Iterator[bytes]:
            all_rows = usage_log.recent(db_path, 10_000_000)
            if not all_rows:
                yield b""
                return
            fields = list(all_rows[0].keys())
            yield (",".join(fields) + "\n").encode()
            for row in reversed(all_rows):  # chronological order
                cells = []
                for field in fields:
                    value = row.get(field)
                    if value is None:
                        cells.append("")
                    else:
                        cell = str(value).replace('"', '""')
                        cells.append(f'"{cell}"' if ("," in cell or '"' in cell or "\n" in cell) else cell)
                yield (",".join(cells) + "\n").encode()

        return StreamingResponse(rows(), media_type="text/csv",
                                 headers={"Content-Disposition": 'attachment; filename="green_requests.csv"'})

    @app.get("/api/sessions")
    async def sessions_view():
        return {"sessions": ledger.list_sessions(ledger_dir)}

    @app.get("/api/sessions/{key}")
    async def session_detail(key: str):
        session = ledger.read_session(ledger_dir, key)
        if session is None:
            return error_json(404, "Session not found", "not_found_error")
        return session

    @app.get("/api/sessions/{key}/summary")
    async def session_summary(key: str):
        session = ledger.read_session(ledger_dir, key)
        if session is None:
            return error_json(404, "Session not found", "not_found_error")
        return {"key": key, "session_key": session.get("session_key"),
                "session_source": session.get("session_source"),
                "run_label": session.get("run_label"), **session.get("summary", {})}

    @app.get("/api/gateway/models")
    async def gateway_models(provider: str | None = None):
        selected = settings.default_provider()
        for candidate in settings.effective_providers():
            if candidate["name"] == provider:
                selected = candidate
                break
        headers = {}
        if settings.effective_key(selected):
            headers["authorization"] = f"Bearer {settings.effective_key(selected)}"
        try:
            resp = await client.get(f"{base(selected)}/v1/models", headers=headers)
        except httpx.HTTPError as exc:
            return error_json(502, f"Could not reach the gateway at {selected['url']}: {exc}")
        if resp.status_code >= 400:
            return upstream_error(resp)
        models = []
        for item in safe_json(resp).get("data", []):
            model_id = item.get("id", "")
            models.append(
                {
                    "id": model_id,
                    "display_name": item.get("display_name", item.get("name", model_id)),
                    "exposed": model_id in settings.exposed_models,
                    "route": route_for(model_id, settings.route_overrides),
                    "provider": selected["name"]
                    if model_id not in settings.model_providers
                    else settings.model_providers[model_id],
                }
            )
        return {"models": models}

    @app.post("/api/claude/apply")
    async def apply_claude(update: Optional[ClaudeApply] = None):
        try:
            # The gateway's own key goes into ANTHROPIC_AUTH_TOKEN: Claude Code
            # sends it as the bearer credential the proxy substitutes upstream.
            gateway_key = settings.effective_key(settings.default_provider())
            choices = (update.model_classes if update is not None else None) or {}
            exposed = set(settings.exposed_models)
            choices = {k: v for k, v in choices.items()
                       if v and v != "auto" and v in exposed}
            claude_config.apply(claude_settings_path, settings.proxy_url(), settings.exposed_models,
                                gateway_key=gateway_key, model_classes=choices)
        except (OSError, ValueError) as exc:
            return error_json(400, str(exc), "invalid_request_error")
        return claude_state()

    @app.post("/api/claude/restore")
    async def restore_claude():
        try:
            claude_config.restore(claude_settings_path)
        except (OSError, ValueError) as exc:
            return error_json(400, str(exc), "invalid_request_error")
        return claude_state()

    @app.get("/api/desktop/status")
    async def desktop_status():
        return desktop_state()

    @app.post("/api/desktop/apply")
    async def apply_desktop(update: Optional[DesktopApply] = None):
        update = update or DesktopApply()
        base_url = (update.base_url or settings.proxy_url()).strip().rstrip("/") or "/"
        api_key = update.api_key or (settings.effective_key() or "ecogpt-managed")
        # Claude Desktop only accepts Anthropic-shaped model names and sends
        # them VERBATIM to the gateway, so expose stable claude-eco-N aliases
        # (one per exposed model) and map each to its upstream model here.
        # Idempotent: existing aliases are not re-aliased.
        real_models = [m for m in settings.exposed_models if not claude_desktop.is_alias(m)]
        aliases = [claude_desktop.desktop_alias_name(i) for i in range(len(real_models))]
        # Picker labels should be human-readable: resolve gateway display names
        # for the real ids and pass them to the profile builder (which falls
        # back to the raw id when metadata is missing).
        meta = await gateway_model_metadata()
        display_names = [display_name_for(m, meta) for m in real_models]
        model_entries = claude_desktop.desktop_models(real_models, display_names)
        alias_map = dict(zip(aliases, real_models))
        if alias_map:
            merged_overrides = dict(settings.model_overrides)
            merged_overrides.update(alias_map)
            settings.model_overrides = merged_overrides
            settings.exposed_models = list(dict.fromkeys(aliases + real_models))
            persist_settings()
        try:
            claude_desktop.apply(desktop_app_dir, base_url, api_key, backup_dir=desktop_backup_dir,
                                 model_entries=model_entries)
        except (OSError, ValueError) as exc:
            return error_json(400, str(exc), "invalid_request_error")
        return desktop_state()

    @app.post("/api/desktop/restore")
    async def restore_desktop():
        try:
            claude_desktop.restore(desktop_app_dir, desktop_backup_dir)
        except (OSError, ValueError) as exc:
            return error_json(400, str(exc), "invalid_request_error")
        return desktop_state()


    @app.get("/api/usage")
    async def usage_rows(limit: int = 50):
        return {"rows": usage_log.recent(db_path, max(1, min(limit, 500)))}

    @app.get("/api/usage/daily")
    async def usage_daily(days: int = 30):
        return {"days": usage_log.daily(db_path, max(1, min(days, 90)))}

    @app.get("/api/usage/totals")
    async def usage_totals():
        return usage_log.totals(db_path)

    @app.get("/api/headroom")
    async def headroom_view():
        return {
            "schema": headroom_settings.SCHEMA,
            "values": settings.headroom,
            "sidecar": {**sidecar.status(), "healthy": await sidecar_client().health()},
            "sidecar_url": settings.sidecar_url(),
        }

    @app.put("/api/headroom")
    async def headroom_update(update: HeadroomUpdate):
        previous_env = headroom_settings.sidecar_env(settings.headroom)
        previous_port = settings.headroom["sidecar_port"]
        previous_client = (
            settings.headroom["request_timeout_ms"],
            settings.headroom["retries"],
            settings.headroom["backoff_ms"],
            settings.headroom["HEADROOM_PROXY_TOKEN"],
        )
        try:
            merged = {**settings.headroom, **update.values}
            normalized = headroom_settings.normalize(merged)
        except ValueError as exc:
            return error_json(400, str(exc), "invalid_request_error")
        settings.headroom = normalized
        persist_settings()
        env_changed = headroom_settings.sidecar_env(normalized) != previous_env or normalized["sidecar_port"] != previous_port
        client_changed = previous_client != (
            normalized["request_timeout_ms"],
            normalized["retries"],
            normalized["backoff_ms"],
            normalized["HEADROOM_PROXY_TOKEN"],
        )
        if client_changed or normalized["sidecar_port"] != previous_port:
            await rebuild_sidecar_client()
        restarted = False
        if env_changed and sidecar.running and manage_sidecar:
            sidecar.restart(sidecar_env(), int(normalized["sidecar_port"]))
            restarted = True
        if normalized["sidecar_enabled"] and not sidecar.running and manage_sidecar:
            sidecar.start(sidecar_env(), int(normalized["sidecar_port"]))
        return {"ok": True, "restarted": restarted, "restart_required": False}

    @app.post("/api/headroom/start")
    async def headroom_start():
        if not manage_sidecar:
            return error_json(400, "Sidecar management is disabled in this process", "invalid_request_error")
        sidecar.start(sidecar_env(), int(settings.headroom["sidecar_port"]))
        prefetch_model_cache()
        return sidecar.status()

    @app.post("/api/headroom/stop")
    async def headroom_stop():
        if not manage_sidecar:
            return error_json(400, "Sidecar management is disabled in this process", "invalid_request_error")
        sidecar.stop()
        return sidecar.status()

    @app.post("/api/headroom/restart")
    async def headroom_restart():
        if not manage_sidecar:
            return error_json(400, "Sidecar management is disabled in this process", "invalid_request_error")
        sidecar.restart(sidecar_env(), int(settings.headroom["sidecar_port"]))
        prefetch_model_cache()
        return sidecar.status()

    @app.get("/api/headroom/model")
    async def kompress_model_view():
        return model_cache.current_status()

    @app.post("/api/headroom/model")
    async def kompress_model_download():
        """Check the Kompress cache now; download in the background when missing.

        The sidecar restarts by itself once the cache is warm, so ML compression
        turns on without any further action. Repeated posts while a download is
        already running are no-ops.
        """
        if settings.headroom.get("HEADROOM_DISABLE_KOMPRESS"):
            return model_cache.current_status()
        model_cache.ensure(background=True, on_ready=on_model_cache_ready)
        return model_cache.current_status()

    @app.get("/api/headroom/stats")
    async def headroom_stats():
        try:
            return await sidecar_client().stats()
        except (httpx.HTTPError, ValueError) as exc:
            return error_json(502, f"Sidecar stats unavailable: {exc}")

    return app


class _CompressionFailed(Exception):
    def __init__(self, inner: CompressError) -> None:
        super().__init__(inner.reason)
        self.inner = inner
