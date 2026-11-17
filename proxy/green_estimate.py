"""IKP-based per-model size estimation — the reference demo's technique, free-tier friendly.

For models whose parameters are unknown (no override, no factor-table row, no
exact EcoLogits catalogue match) the same stratified factual-probe benchmark as
the green_metrics_proxy demo estimates a model's effective knowledge-equivalent
capacity: sample N/7 probes from each public IKP tier T1..T7, ask the target,
grade with a judge model, and convert the per-tier mean accuracy through the
public calibration (N_eff = 10^((acc - intercept)/slope), cohort n=93,
R^2=0.9105). The result is a knowledge-equivalent scenario, NOT a physical
weight count — it is applied only when the user explicitly accepts it
(``.../estimate/apply`` saves an explicit ``basis: "ikp_estimate"`` scenario).

Operational contract (owner requirements):
- pacing: ``pace_s`` sleeps between every upstream call (free-tier throttling);
- retries: each call retries up to 4 times with exponential backoff, honouring
  ``Retry-After`` on HTTP 429;
- resumability: the whole run state (per-probe answer, verdict, usage, error,
  progress, step label) persists to data/green_estimation.json after every step,
  so a paused/failed/crashed run resumes from the first unfinished probe with a
  manual CONTINUE — never rerun from scratch;
- states: ``running -> done`` | ``running -> paused`` (user stop / daemon
  restart / user interrupt) | ``running -> failed`` (fatal error); paused and
  failed runs are resumable; a run that was left ``running`` by a daemon crash
  is reset to ``paused`` on startup.

Chat calls always use ``/v1/chat/completions`` on the model's configured
provider (the same assumption the reference demo makes about gateways).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

from .config import upstream_base

PROJECT_DIR = Path(__file__).resolve().parent.parent
ESTIMATION_STATE_PATH = PROJECT_DIR / "data" / "green_estimation.json"
IKP_DIR = PROJECT_DIR / "data" / "ikp"
PROBES_PATH = IKP_DIR / "final_probe_set_v8.json"
CALIBRATION_PATH = IKP_DIR / "effective_params.json"

TIERS = tuple(f"T{i}" for i in range(1, 8))
CALL_ATTEMPTS = 4
BACKOFF_S = (2.0, 4.0, 8.0)

_lock = asyncio.Lock()
_tasks: dict[str, "asyncio.Task[None]"] = {}

# One model's estimation record shape:
# {"status", "step", "judge", "sample_n", "pace_s", "max_tokens", "complete_n",
#  "accuracy", "effective_b", "error", "rows"[{tier,question,gold,answer,verdict}], updated_at}


def load_state(path: Path | None = None) -> dict[str, dict[str, Any]]:
    try:
        data = json.loads(Path(path or ESTIMATION_STATE_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, dict) else {}


def save_state(models: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    path = Path(path or ESTIMATION_STATE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"models": models}, indent=2), encoding="utf-8")


def _touch(model: str, *, status: str | None = None, step: str | None = None,
           error: str | None = None, path: Path | None = None) -> dict[str, Any]:
    """Read-modify-write one model's run record; returns the fresh record."""
    models = load_state(path)
    record = models.get(model)
    if record is None:  # concurrently deleted
        record = {"status": "paused"}
    if status is not None:
        record["status"] = status
    if step is not None:
        record["step"] = step
    if error is not None:
        record["error"] = error
    record["updated_at"] = time.time()
    models[model] = record
    save_state(models, path)
    return record


def reset_stale_runs(path: Path | None = None) -> list[str]:
    """Daemon startup: runs left 'running' by a restart are now resumable pauses."""
    models = load_state(path)
    resumed = []
    changed = False
    for model, record in models.items():
        if record.get("status") == "running":
            record["status"] = "paused"
            record["step"] = "interrupted by daemon restart — press continue to resume"
            record["updated_at"] = time.time()
            changed = True
            resumed.append(model)
    if changed:
        save_state(models, path)
    return resumed


def is_running(model: str, path: Path | None = None) -> bool:
    task = _tasks.get(model)
    if task is not None and not task.done():
        return True
    return load_state(path).get(model, {}).get("status") == "running"


def status_of(model: str, path: Path | None = None) -> dict[str, Any]:
    return load_state(path).get(model, {})


def _sample_probes(sample_n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Stratified sample: sample_n must be a positive multiple of 7 (odd tiers even)."""
    if sample_n < 7 or sample_n % 7:
        raise ValueError("probe count must be a positive multiple of 7")
    probes = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
    per_tier = sample_n // 7
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for tier in TIERS:
        candidates = [p for p in probes if p.get("tier") == tier]
        if not candidates:
            raise ValueError(f"IKP probe set has no probes for tier {tier}")
        sample.extend(rng.sample(candidates, min(per_tier, len(candidates))))
    rows = [{"tier": p["tier"], "question": p["question"], "gold": p["answer"],
             "answer": None, "verdict": None, "error": None} for p in sample]
    return rows


def _calibration() -> tuple[float, float]:
    data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    cal = data.get("calibration") or {}
    return float(cal["slope"]), float(cal["intercept"])


_TIERS_DONE = ("CORRECT", "WRONG", "REFUSAL")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Per-tier mean accuracy through the public calibration -> N_eff in billions."""
    incomplete = [r for r in rows if r["verdict"] not in _TIERS_DONE]
    if incomplete:
        return None
    tiers = [t for t in TIERS if any(r["tier"] == t for r in rows)]
    if not tiers:
        return None
    per_tier: dict[str, float] = {}
    for tier in tiers:
        graded = [r for r in rows if r["tier"] == tier]
        if not graded:
            return None
        per_tier[tier] = sum(1 for r in graded if r["verdict"] == "CORRECT") / len(graded)
    accuracy = sum(per_tier.values()) / len(per_tier)
    slope, intercept = _calibration()
    effective = 10 ** ((accuracy - intercept) / slope) if accuracy > 0 else None
    return {
        "accuracy": accuracy,
        "effective_b": effective,
        "per_tier": per_tier,
        "warning": ("IKP measures factual knowledge-equivalent capacity, not physical "
                    "parameter count; treating it as total/active parameters is an "
                    "explicit dense-model scenario"),
    }


def _judge_prompt(question: str, gold: str, answer: str) -> str:
    reference = " OR ".join(x.strip() for x in str(gold).split(";"))
    return (f"You are a strict factual judge. Classify the model response.\n"
            f"Question: {question}\nCorrect answer: {reference}\nResponse: {answer}\n"
            "Rules: YEAR must match exactly; NUMBER within 1-2%; NAME same entity minor spelling okay; "
            "RESEARCH FIELD adjacent subfields accepted only if supported, unrelated multiple guesses wrong; "
            "refuses or doesn't know -> REFUSAL; otherwise if incorrect -> WRONG.\n"
            "Reply exactly one word: CORRECT, REFUSAL, or WRONG.")


async def _chat_call(client: httpx.AsyncClient, base: str, key: str, model: str,
                     messages: list[dict[str, str]], max_tokens: int) -> str:
    """One completions call with exponential backoff; 429 honours Retry-After."""
    last_error = ""
    for attempt in range(CALL_ATTEMPTS):
        try:
            response = await client.post(
                f"{base}/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": messages, "max_tokens": max_tokens,
                      "temperature": 0},
            )
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 429:
                retry_after = response.headers.get("retry-after")
                try:
                    delay = float(retry_after) if retry_after else BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
                except ValueError:
                    delay = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
                last_error = f"429 rate limited (waiting {delay:.1f}s before attempt {attempt + 2})"
                await asyncio.sleep(min(delay, 60.0))
                continue
            if response.status_code >= 400:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code in (400, 401, 403, 404, 422):  # not throttling — fatal
                    raise RuntimeError(last_error)
            else:
                try:
                    data = response.json()
                except ValueError:
                    last_error = "invalid JSON from upstream"
                else:
                    error = data.get("error")
                    if not error:
                        choice = (data.get("choices") or [{}])[0]
                        answer = ((choice.get("message") or {}).get("content") or "").strip()
                        if not answer:
                            last_error = "empty model output"
                        elif choice.get("finish_reason") == "length":
                            last_error = "model hit max_tokens; raise the estimator's output limit"
                        else:
                            return answer
        if attempt < CALL_ATTEMPTS - 1:
            delay = BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)]
            await asyncio.sleep(delay)
    raise RuntimeError(f"estimation call failed after {CALL_ATTEMPTS} attempts: {last_error}")


async def _run_loop(client: httpx.AsyncClient, base: str, key: str, target: str,
                    path: Path | None) -> None:
    record = status_of(target, path)
    rows = record.get("rows") or []
    max_tokens = int(record.get("max_tokens", 1024))
    pace = float(record.get("pace_s", 3.2))
    total = int(record.get("sample_n", len(rows)))
    try:
        for index, row in enumerate(rows):
            fresh = await asyncio.to_thread(status_of, target, path) or {}
            if fresh.get("status") != "running":
                return  # stopped or reset externally
            if row["verdict"] in ("CORRECT", "WRONG", "REFUSAL"):
                continue
            if row["answer"] is None:
                _touch(target, status="running", step=f"probing {index + 1}/{total} (tier {row['tier']})",
                       error=None, path=path)
                row["answer"] = await _chat_call(
                    client, base, key, target,
                    [{"role": "system",
                      "content": "Answer factual questions directly and concisely. If you don't know, say 'I don't know'."},
                     {"role": "user", "content": row["question"]}],
                    max_tokens)
                await _persist_row(target, rows, path)
                await asyncio.sleep(pace)
            _touch(target, status="running", step=f"judging {index + 1}/{total} (tier {row['tier']})",
                   path=path)
            raw = await _chat_call(client, base, key, record.get("judge") or target,
                                   [{"role": "user", "content":
                                     _judge_prompt(row["question"], row["gold"], row["answer"])}],
                                   max_tokens)
            verdict = raw.upper().strip().rstrip(".")
            if verdict not in ("CORRECT", "WRONG", "REFUSAL"):
                raise RuntimeError(f"judge returned an invalid label: {raw[:150]!r}")
            row["verdict"] = verdict
            await _persist_row(target, rows, path)
            await asyncio.sleep(pace)
        result = summarize(rows)
        if result is None:
            raise RuntimeError("estimation finished but produced no gradeable probes")
        final = _touch(target, status="done", step="estimation complete",
                       error=None, path=path)
        final.update({"accuracy": result["accuracy"], "effective_b": result["effective_b"],
                      "complete_n": len(rows), "per_tier": result["per_tier"],
                      "warning": result["warning"]})
        models = load_state(path)
        models[target] = final
        save_state(models, path)
    except asyncio.CancelledError:
        _touch(target, status="paused", step="stopped by user — press continue to resume", path=path)
        raise
    except Exception as exc:  # noqa: BLE001 - the job must persist its failure for CONTINUE
        _touch(target, status="failed", step="failed — press continue to retry from where it stopped",
               error=str(exc)[:300], path=path)


async def _persist_row(target: str, rows: list[dict[str, Any]], path: Path | None) -> None:
    models = load_state(path)
    record = models.get(target)
    if record is not None and record.get("status") == "running":
        record["rows"] = rows
        record["complete_n"] = sum(1 for r in rows if r["verdict"] in ("CORRECT", "WRONG", "REFUSAL"))
        record["updated_at"] = time.time()
        save_state(models, path)


async def start(model: str, *, judge: str, sample_n: int, pace_s: float, max_tokens: int,
                base: str, key: str, path: Path | None = None,
                transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    """Start a fresh estimation run (the controller checks there is none running)."""
    async with _lock:
        if is_running(model, path):
            raise RuntimeError(f"an estimation for '{model}' is already running")
        probes = _sample_probes(sample_n)  # raises ValueError on bad shapes (400 for the caller)
        slope, intercept = _calibration()
        models = load_state(path)
        models[model] = {
            "status": "running",
            "step": "starting",
            "judge": judge,
            "sample_n": len(probes),
            "pace_s": pace_s,
            "max_tokens": max_tokens,
            "complete_n": 0,
            "accuracy": None,
            "effective_b": None,
            "error": None,
            "calibration": {"slope": slope, "intercept": intercept},
            "rows": probes,
            "updated_at": time.time(),
        }
        save_state(models, path)
    await _spawn(model, base, key, path, transport)
    return status_of(model, path)


async def _spawn(model: str, base: str, key: str, path: Path | None,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
    client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=20), transport=transport)

    async def runner() -> None:
        try:
            await _run_loop(client, base, key, model, path)
        finally:
            await client.aclose()
            _tasks.pop(model, None)

    _tasks[model] = asyncio.create_task(runner())


async def continue_run(model: str, *, base: str, key: str, path: Path | None = None,
                       transport: httpx.AsyncBaseTransport | None = None) -> dict[str, Any]:
    """Resume a paused/failed run from its first unfinished probe (CONTINUE button)."""
    async with _lock:
        if is_running(model, path):
            raise RuntimeError(f"estimation for '{model}' is already running")
        record = load_state(path).get(model)
        if not record:
            raise LookupError(f"no estimation has been started for '{model}'")
        if record.get("status") == "done":
            raise RuntimeError(f"estimation for '{model}' is already complete")
        record["status"] = "running"
        record["step"] = "resuming"
        record["error"] = None
        record["updated_at"] = time.time()
        models = load_state(path)
        models[model] = record
        save_state(models, path)
    await _spawn(model, base, key, path, transport)
    return status_of(model, path)


async def stop(model: str, path: Path | None = None) -> dict[str, Any]:
    """User stop: cancel the task; the loop flips the record to paused."""
    task = _tasks.get(model)
    if task is not None and not task.done():
        task.cancel()
    else:
        _touch(model, status="paused", step="stopped — press continue to resume", path=path)
    for _ in range(50):
        task = _tasks.get(model)
        if task is None or task.done():
            break
        await asyncio.sleep(0.1)
    return status_of(model, path)
