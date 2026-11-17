from __future__ import annotations

import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

SCHEMA = """
create table if not exists usage (
    id integer primary key autoincrement,
    ts real not null,
    model text not null,
    route text not null,
    stream integer not null,
    status integer not null,
    input_tokens integer not null default 0,
    output_tokens integer not null default 0,
    cache_read_tokens integer not null default 0,
    cache_write_tokens integer not null default 0,
    cost_reported text,
    duration_ms integer not null default 0
)
"""

# Columns added after the first release. Existing rows keep their values; new ones default to 0/NULL.
MIGRATIONS = {
    "compressed": "integer not null default 0",
    "tokens_before": "integer not null default 0",
    "tokens_after": "integer not null default 0",
    "tokens_saved": "integer not null default 0",
    "session_source": "text",
    "session_replayed": "integer not null default 0",
    "attempts": "integer not null default 0",
    "fallback": "text",
    "turn_id": "text",
    "relayed": "integer not null default 0",
    # Green metrics (Phase 1). Nullable: rows recorded before the feature, or with
    # the flag off, or for unknown models read as NULL.
    "energy_kwh_mean": "real",
    "energy_kwh_min": "real",
    "energy_kwh_max": "real",
    "gwp_kgco2e_mean": "real",
    "gwp_kgco2e_min": "real",
    "gwp_kgco2e_max": "real",
    "model_basis": "text",
    # Green metrics (Phase 2): EcoLogits-native core. Water/ADPe/PE are mean-bound
    # single values (always displayed with impact_provenance in the dashboard);
    # energy/GWP keep min/mean/max intervals from both compute paths.
    # Legacy rows (energy set, impact_status NULL) read as estimated by the queries.
    "water_l": "real",
    "adpe_kg_sbeq": "real",
    "primary_mj": "real",
    "impact_status": "text",
    "impact_provenance": "text",
    "ttft_ms": "integer",
    "tps": "real",
}


def normalize_anthropic_usage(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    return {
        "input_tokens": usage.get("input_tokens") or 0,
        "output_tokens": usage.get("output_tokens") or 0,
        "cache_read_input_tokens": usage.get("cache_read_input_tokens") or 0,
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens") or 0,
    }


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(SCHEMA)
        existing = {row[1] for row in db.execute("pragma table_info(usage)")}
        for column, definition in MIGRATIONS.items():
            if column not in existing:
                db.execute(f"alter table usage add column {column} {definition}")


def record(
    path: Path,
    *,
    model: str,
    route: str,
    stream: bool,
    status: int,
    usage: dict[str, Any] | None,
    cost: str | None,
    duration_ms: int,
    compression: dict[str, Any] | None = None,
    impacts: dict[str, Any] | None = None,
    ttft_ms: int | None = None,
    tps: float | None = None,
) -> None:
    counts = normalize_anthropic_usage(usage)
    comp = compression or {}
    impacts = impacts or {}
    energy = impacts.get("energy_kwh") or {}
    gwp = impacts.get("gwp_kgco2e") or {}
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "insert into usage (ts, model, route, stream, status, input_tokens, output_tokens, "
            "cache_read_tokens, cache_write_tokens, cost_reported, duration_ms, compressed, "
            "tokens_before, tokens_after, tokens_saved, session_source, session_replayed, attempts, "
            "fallback, turn_id, relayed, energy_kwh_mean, energy_kwh_min, energy_kwh_max, "
            "gwp_kgco2e_mean, gwp_kgco2e_min, gwp_kgco2e_max, model_basis, "
            "water_l, adpe_kg_sbeq, primary_mj, impact_status, impact_provenance, ttft_ms, tps) "
            "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, "
            "?, ?, ?, ?, ?, ?, ?)",
            (
                time.time(),
                model,
                route,
                int(stream),
                status,
                counts["input_tokens"],
                counts["output_tokens"],
                counts["cache_read_input_tokens"],
                counts["cache_creation_input_tokens"],
                cost,
                duration_ms,
                int(comp.get("compressed", False)),
                comp.get("tokens_before", 0),
                comp.get("tokens_after", 0),
                comp.get("tokens_saved", 0),
                comp.get("session_source"),
                int(comp.get("session_replayed", False)),
                comp.get("attempts", 0),
                comp.get("fallback"),
                comp.get("turn_id"),
                int(comp.get("relayed", False)),
                energy.get("mean"),
                energy.get("min"),
                energy.get("max"),
                gwp.get("mean"),
                gwp.get("min"),
                gwp.get("max"),
                impacts.get("model_basis"),
                impacts.get("water_l"),
                impacts.get("adpe_kg_sbeq"),
                impacts.get("primary_mj"),
                impacts.get("impact_status"),
                impacts.get("impact_provenance"),
                ttft_ms,
                tps,
            ),
        )


def mark_relayed(path: Path, turn_id: str) -> None:
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("update usage set relayed = 1 where turn_id = ?", (turn_id,))


def recent(path: Path, limit: int) -> list[dict[str, Any]]:
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("select * from usage order by id desc limit ?", (limit,)).fetchall()
    return [dict(row) for row in rows]


def totals(path: Path) -> dict[str, Any]:
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        row = db.execute(
            "select count(*) as requests, "
            "coalesce(sum(compressed), 0) as compressed_requests, "
            "coalesce(sum(tokens_before), 0) as tokens_before, "
            "coalesce(sum(tokens_after), 0) as tokens_after, "
            "coalesce(sum(tokens_saved), 0) as tokens_saved, "
            "coalesce(sum(case when fallback is not null then 1 else 0 end), 0) as fallbacks, "
            "coalesce(sum(cache_read_tokens), 0) as cache_read_tokens, "
            "coalesce(sum(cache_write_tokens), 0) as cache_write_tokens "
            "from usage"
        ).fetchone()
    return dict(row)


def parse_cost(cost: str | None) -> float | None:
    """`header_cost` stores 'name: value' strings; extract the numeric part."""
    if not cost:
        return None
    text = str(cost).split(": ")[-1].strip().lstrip("$")
    try:
        return float(text)
    except ValueError:
        return None


def daily(path: Path, days: int = 30) -> list[dict[str, Any]]:
    """Per-UTC-day rollup for the dashboard charts: tokens, cost, green."""
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "select date(ts, 'unixepoch') as day, count(*) as requests, "
            "coalesce(sum(input_tokens), 0) as input_tokens, "
            "coalesce(sum(output_tokens), 0) as output_tokens, "
            "coalesce(sum(cache_read_tokens), 0) as cache_read_tokens, "
            "coalesce(sum(cache_write_tokens), 0) as cache_write_tokens, "
            "coalesce(sum(tokens_saved), 0) as tokens_saved, "
            "coalesce(sum(case when compressed then 1 else 0 end), 0) as compressed, "
            "coalesce(sum(energy_kwh_mean), 0.0) as energy_kwh_mean, "
            "coalesce(sum(energy_kwh_min), 0.0) as energy_kwh_min, "
            "coalesce(sum(energy_kwh_max), 0.0) as energy_kwh_max, "
            "coalesce(sum(gwp_kgco2e_mean), 0.0) as gwp_kgco2e_mean, "
            "coalesce(sum(gwp_kgco2e_min), 0.0) as gwp_kgco2e_min, "
            "coalesce(sum(gwp_kgco2e_max), 0.0) as gwp_kgco2e_max "
            "from usage group by day order by day asc"
        ).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        # cost_reported is not aggregated in SQL (text); re-query just costs per day.
        with closing(sqlite3.connect(path)) as db:
            raw = db.execute(
                "select cost_reported from usage where date(ts, 'unixepoch') = ?", (item["day"],)
            ).fetchall()
        item["cost"] = round(sum(c for r in raw if (c := parse_cost(r[0])) is not None), 6)
        out.append(item)
    return out[-days:]


# Columns and the estimated-row filter shared by the green queries. Legacy rows
# (Phase 1: energy set, impact_status NULL) also count as estimated.
_ESTIMATED_WHERE = "(impact_status = 'estimated' OR (impact_status IS NULL AND energy_kwh_mean IS NOT NULL))"
_GREEN_COLUMNS = (
    "coalesce(sum(energy_kwh_mean), 0.0) as energy_kwh_mean, "
    "coalesce(sum(energy_kwh_min), 0.0) as energy_kwh_min, "
    "coalesce(sum(energy_kwh_max), 0.0) as energy_kwh_max, "
    "coalesce(sum(gwp_kgco2e_mean), 0.0) as gwp_kgco2e_mean, "
    "coalesce(sum(gwp_kgco2e_min), 0.0) as gwp_kgco2e_min, "
    "coalesce(sum(gwp_kgco2e_max), 0.0) as gwp_kgco2e_max, "
    "coalesce(sum(water_l), 0.0) as water_l, "
    "coalesce(sum(adpe_kg_sbeq), 0.0) as adpe_kg_sbeq, "
    "coalesce(sum(primary_mj), 0.0) as primary_mj"
)


def green_panel(path: Path) -> dict[str, Any]:
    """Aggregations for the Green panel: totals, per-model and per-day (UTC) rollups,
    plus recent per-request rows for the impact-over-time chart."""
    def query(sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(path)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(sql, args)]

    estimate_where = f"where {_ESTIMATED_WHERE}"
    total = query(f"select count(*) as requests, {_GREEN_COLUMNS} from usage {estimate_where}")
    by_model = query(
        "select model, model_basis, count(*) as requests, "
        f"{_GREEN_COLUMNS} from usage {estimate_where} group by model order by energy_kwh_mean desc"
    )
    by_day = query(
        "select date(ts, 'unixepoch') as day, count(*) as requests, "
        f"{_GREEN_COLUMNS} from usage {estimate_where} group by day order by day desc"
    )
    recent = [
        {
            "ts": row["ts"],
            "model": row["model"],
            "energy_kwh_mean": row["energy_kwh_mean"],
            "gwp_kgco2e_mean": row["gwp_kgco2e_mean"],
            "water_l": row["water_l"],
            "primary_mj": row["primary_mj"],
            "impact_status": row["impact_status"] or "estimated",
            "legacy": row["impact_status"] is None,
        }
        for row in query(
            "select ts, model, energy_kwh_mean, gwp_kgco2e_mean, water_l, primary_mj, impact_status "
            f"from usage {estimate_where} order by id desc limit 200"
        )
    ]
    return {
        "totals": total[0] if total else {},
        "by_model": by_model,
        "by_day": by_day,
        "recent": list(reversed(recent)),
    }


def coverage(path: Path) -> dict[str, Any]:
    """Coverage accounting for the dashboard caption: only 'estimated' rows carry
    numbers, so totals are partial whenever any request was unconfigured, missing
    usage, or recorded before green metrics existed."""
    def query(sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with closing(sqlite3.connect(path)) as db:
            db.row_factory = sqlite3.Row
            return [dict(row) for row in db.execute(sql, args)]

    rows = query(
        # aliased 'impact_state' (not 'status'): group by must not collide with the
        # real HTTP status column that SQLite would otherwise prefer.
        "select coalesce(impact_status, 'legacy_estimated') as impact_state, count(*) as requests "
        "from usage where energy_kwh_mean is not null or impact_status is not null group by impact_state"
    )
    total_rows = query("select count(*) as requests from usage")[0]["requests"]
    by_status = {row["impact_state"]: row["requests"] for row in rows}
    estimated = by_status.get("estimated", 0) + by_status.get("legacy_estimated", 0)
    return {
        "requests_total": total_rows,
        "estimated": estimated,
        "legacy_estimated": by_status.get("legacy_estimated", 0),
        "by_status": by_status,
        "ratio": round(estimated / total_rows, 4) if total_rows else 1.0,
    }


class StreamUsage:
    """Collects Anthropic usage from a passthrough SSE stream without altering it.

    Also observes the first content-bearing event (monotonic clock) so the green
    observer can pass TTFT/TPS to the EcoLogits compute core. Approximate by
    nature: the first content event is not always the first generated token,
    especially with reasoning models.
    """

    CONTENT_TYPES = ("content_block_start", "content_block_delta")

    def __init__(self) -> None:
        self._buffer = b""
        self.usage: dict[str, Any] = {}
        self.stop_reason: str | None = None
        self.first_content: float | None = None

    def note_content(self) -> None:
        if self.first_content is None:
            self.first_content = time.monotonic()

    def ttft_ms(self, started: float) -> float | None:
        if self.first_content is None or started is None or self.first_content < started:
            return None
        return (self.first_content - started) * 1000.0

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            try:
                event = json.loads(line[5:])
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "message_start":
                self._merge(event.get("message", {}).get("usage"))
            elif event.get("type") == "message_delta":
                self._merge(event.get("usage"))
                stop = (event.get("delta") or {}).get("stop_reason")
                if isinstance(stop, str):
                    self.stop_reason = stop
            if event.get("type") in self.CONTENT_TYPES:
                self.note_content()

    def _merge(self, usage: Any) -> None:
        if isinstance(usage, dict):
            self.usage.update({key: value for key, value in usage.items() if value is not None})
